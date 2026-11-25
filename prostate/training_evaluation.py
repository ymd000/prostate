"""Final-checkpoint training diagnostics with the same inference as CV or holdout."""

import gc
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from tqdm import tqdm

from prostate.archive_metrics import archive_prediction_metadata, summarize_archive_metrics
from prostate.data import CLASS_NAMES, Sample
from prostate.folds import (
    evaluation_folds,
    prediction_split,
    read_assignments,
    validate_checkpoint_split,
)
from prostate.inference import load_checkpoint_model, predict_sample
from prostate.metrics import confusion_matrix, scores, summarize_predictions
from prostate.train import resolve_device

EVALUATION_PROTOCOL = "eval_unaugmented_halo_v1"
ARCHIVE_EVALUATION_PROTOCOL = "archive_balanced_tiles_v1"


def _evaluation_protocol(settings: dict) -> str:
    return (
        ARCHIVE_EVALUATION_PROTOCOL
        if settings.get("training_recipe") == "archive_random_v1"
        else EVALUATION_PROTOCOL
    )


def _write_json(path: Path, content: dict | list):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _fingerprint(checkpoint_path: Path) -> dict:
    stat = checkpoint_path.stat()
    return {
        "path": str(checkpoint_path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def read_run(run_dir: Path):
    settings = json.loads((run_dir / "config.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if tuple(manifest["classes"]) != CLASS_NAMES:
        raise ValueError("Manifest class order does not match training diagnostics")
    samples = [Sample(**record) for record in manifest["samples"]]
    assignments = read_assignments(run_dir, settings, samples)
    return settings, samples, assignments


def training_report_current(checkpoint_path: Path, output_dir: Path) -> bool:
    if not checkpoint_path.is_file() or not (output_dir / "train_images.json").is_file():
        return False
    try:
        report = json.loads((output_dir / "train_metrics.json").read_text())
        return (
            report["complete"]
            and report["split"] == "train"
            and report["evaluation_protocol"] in (EVALUATION_PROTOCOL, ARCHIVE_EVALUATION_PROTOCOL)
            and report["checkpoint_fingerprint"] == _fingerprint(checkpoint_path)
        )
    except (OSError, ValueError, KeyError):
        return False


def evaluate_training_fold(
    model: torch.nn.Module,
    samples: list[Sample],
    checkpoint: dict,
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
    encoder_bundle: tuple | None = None,
) -> dict:
    """Evaluate all original training pixels; keep only one temporary prediction on disk."""
    fold = checkpoint["fold"]
    validate_checkpoint_split(checkpoint, samples, fold)
    if not samples:
        raise ValueError("Training diagnostics require at least one training image")
    if checkpoint["architecture"] == "token_decoder_v1" and encoder_bundle is None:
        raise ValueError("Token decoder training diagnostics require the frozen encoder")
    settings = checkpoint["config"]
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with TemporaryDirectory(prefix="train-eval-", dir=output_dir) as directory:
        output = Path(directory) / "prediction.npy"
        for sample in tqdm(samples, desc=f"Fold {fold} training evaluation"):
            prediction = predict_sample(model, sample, output, settings, device, encoder_bundle)
            target = np.load(sample.label_cache, mmap_mode="r")
            matrix = confusion_matrix(target, prediction)
            archive_metadata = archive_prediction_metadata(output)
            del target, prediction
            output.unlink()
            records.append(
                {
                    "image_id": sample.image_id,
                    "case_id": sample.case_id,
                    "fold": fold,
                    "confusion_matrix": matrix.tolist(),
                    **scores(matrix),
                    **archive_metadata,
                }
            )
    report = summarize_predictions(records, [sample.image_id for sample in samples])
    report.update(
        {
            "split": "train",
            "fold": fold,
            "epoch": settings["epochs"],
            "evaluation_protocol": _evaluation_protocol(settings),
            "checkpoint_fingerprint": _fingerprint(checkpoint_path),
            "patch_size": settings["patch_size"],
            "halo": settings["halo"],
            "amp": settings["amp"],
            "amp_dtype": settings.get("amp_dtype", "float16"),
            "predictions_saved": False,
        }
    )
    if settings.get("training_recipe") == "archive_random_v1":
        report["archive_metrics"] = summarize_archive_metrics(records)
        report["validation_mode"] = settings["validation_mode"]
        report["inference_tile_size"] = settings["inference_tile_size"]
    _write_json(output_dir / "train_images.json", records)
    _write_json(output_dir / "train_metrics.json", report)
    return report


def compare_training_heldout(training: dict, held_out: dict | None, split: str) -> dict:
    train_mean = training["primary_case_mean"]
    held_out_mean = held_out["primary_case_mean"] if held_out else None
    comparison = {
        "train_images_evaluated": training["images_evaluated"],
        "train_cases_evaluated": training["cases_evaluated"],
        "train_primary_case_mean": train_mean,
        "train_pooled_pixels": training["pooled_pixels"],
        f"{split}_primary_case_mean": held_out_mean,
        f"{split}_pooled_pixels": held_out["pooled_pixels"] if held_out else None,
        "foreground_macro_dice_gap": train_mean["foreground_macro_dice"]
        - held_out_mean["foreground_macro_dice"]
        if held_out_mean
        and train_mean["foreground_macro_dice"] is not None
        and held_out_mean["foreground_macro_dice"] is not None
        else None,
    }
    if "archive_metrics" in training:
        comparison["train_archive_metrics"] = training["archive_metrics"]
        comparison[f"{split}_archive_metrics"] = held_out["archive_metrics"] if held_out else None
    return comparison


def collect_training_reports(run_dir: Path, output_dir: Path | None = None) -> dict:
    settings, samples, assignments = read_run(run_dir)
    destination = output_dir if output_dir is not None else run_dir
    destination.mkdir(parents=True, exist_ok=True)
    folds = {}
    by_id = {sample.image_id: sample for sample in samples}
    split = prediction_split(settings)
    for fold in evaluation_folds(settings):
        folder = destination / f"fold_{fold}"
        if not training_report_current(run_dir / f"fold_{fold}/checkpoint.pt", folder):
            continue
        training = json.loads((folder / "train_metrics.json").read_text())
        held_out = None
        evaluation_path = run_dir / f"fold_{fold}/evaluation.json"
        if evaluation_path.is_file():
            records = json.loads(evaluation_path.read_text())
            expected = [key for key, value in assignments.items() if value == fold]
            if any(
                record["image_id"] not in by_id
                or record["fold"] != fold
                or record["case_id"] != by_id[record["image_id"]].case_id
                for record in records
            ):
                raise ValueError(f"Saved {split} evaluation does not match the manifest")
            held_out = summarize_predictions(records, expected)
            if not held_out["complete"]:
                raise ValueError(f"Incomplete saved {split} evaluation in fold {fold}")
            if settings.get("training_recipe") == "archive_random_v1":
                held_out["archive_metrics"] = summarize_archive_metrics(records)
        folds[str(fold)] = compare_training_heldout(training, held_out, split)
    missing = [fold for fold in evaluation_folds(settings) if str(fold) not in folds]
    summary = {
        "complete": not missing,
        "split": "train",
        "evaluation_split": split,
        "evaluation_protocol": _evaluation_protocol(settings),
        "run_dir": str(run_dir.resolve()),
        "output_dir": str(destination.resolve()),
        "folds_evaluated": [int(fold) for fold in folds],
        "missing_folds": missing,
        "folds": folds,
    }
    _write_json(destination / "train_metrics.json", summary)
    return summary


def evaluate_training_run(
    run_dir: Path,
    requested_device: str = "auto",
    selected_folds: list[int] | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Backfill final training metrics from checkpoints, including older sampling schemes."""
    settings, samples, assignments = read_run(run_dir)
    available = evaluation_folds(settings)
    selected = list(available) if selected_folds is None else selected_folds
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(fold not in available for fold in selected)
    ):
        raise ValueError("Choose distinct, valid fold IDs")
    if selected_folds is None:
        selected = [fold for fold in selected if (run_dir / f"fold_{fold}/checkpoint.pt").is_file()]
    if not selected:
        raise FileNotFoundError("No saved fold checkpoints are available")
    device = resolve_device(requested_device)
    torch.set_num_threads(settings.get("cpu_threads", 4))
    destination = output_dir if output_dir is not None else run_dir
    encoder_bundle = None
    for fold in selected:
        path = run_dir / f"fold_{fold}/checkpoint.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        training = [sample for sample in samples if assignments[sample.image_id] != fold]
        validate_checkpoint_split(checkpoint, training, fold)
        if checkpoint["architecture"] == "token_decoder_v1" and encoder_bundle is None:
            from prostate.extractor import load_gigapath_flash

            encoder_bundle = load_gigapath_flash(device)
        model = load_checkpoint_model(checkpoint, device)
        bundle = encoder_bundle if checkpoint["architecture"] == "token_decoder_v1" else None
        evaluate_training_fold(
            model, training, checkpoint, path, destination / f"fold_{fold}", device, bundle
        )
        del model, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return collect_training_reports(run_dir, destination)
