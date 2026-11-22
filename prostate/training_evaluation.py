"""Final-checkpoint training diagnostics using the same inference protocol as OOF."""

import csv
import gc
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from tqdm import tqdm

from prostate.data import CLASS_NAMES, Sample
from prostate.inference import load_checkpoint_model, predict_sample
from prostate.metrics import confusion_matrix, scores, summarize_predictions
from prostate.train import resolve_device

EVALUATION_PROTOCOL = "eval_unaugmented_halo_v1"


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


def _read_run(run_dir: Path):
    settings = json.loads((run_dir / "config.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if tuple(manifest["classes"]) != CLASS_NAMES:
        raise ValueError("Manifest class order does not match training diagnostics")
    samples = [Sample(**record) for record in manifest["samples"]]
    with (run_dir / "folds.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assignments = {row["image_id"]: int(row["fold"]) for row in rows}
    ids = {sample.image_id for sample in samples}
    if len(ids) != len(samples) or len(rows) != len(ids) or set(assignments) != ids:
        raise ValueError("Saved fold assignments do not match the manifest")
    case_folds = {}
    for sample in samples:
        fold = assignments[sample.image_id]
        if not 0 <= fold < settings["n_splits"]:
            raise ValueError(f"Invalid saved fold: {fold}")
        if case_folds.setdefault(sample.case_id, fold) != fold:
            raise ValueError(f"Case is split across folds: {sample.case_id}")
    return settings, samples, assignments


def _validate_checkpoint(checkpoint: dict, samples: list[Sample], fold: int):
    expected_ids = {sample.image_id for sample in samples}
    expected_cases = {sample.case_id for sample in samples}
    if checkpoint["fold"] != fold:
        raise ValueError("Checkpoint fold mismatch in training diagnostics")
    if (
        set(checkpoint["train_image_ids"]) != expected_ids
        or len(checkpoint["train_image_ids"]) != len(samples)
        or set(checkpoint["train_case_ids"]) != expected_cases
    ):
        raise ValueError("Checkpoint training split does not match the saved fold assignments")


def training_report_current(checkpoint_path: Path, output_dir: Path) -> bool:
    if not checkpoint_path.is_file() or not (output_dir / "train_images.json").is_file():
        return False
    try:
        report = json.loads((output_dir / "train_metrics.json").read_text())
        return (
            report["complete"]
            and report["split"] == "train"
            and report["evaluation_protocol"] == EVALUATION_PROTOCOL
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
    _validate_checkpoint(checkpoint, samples, fold)
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
            del target, prediction
            output.unlink()
            records.append(
                {
                    "image_id": sample.image_id,
                    "case_id": sample.case_id,
                    "fold": fold,
                    "confusion_matrix": matrix.tolist(),
                    **scores(matrix),
                }
            )
    report = summarize_predictions(records, [sample.image_id for sample in samples])
    report.update(
        {
            "split": "train",
            "fold": fold,
            "epoch": settings["epochs"],
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "checkpoint_fingerprint": _fingerprint(checkpoint_path),
            "patch_size": settings["patch_size"],
            "halo": settings["halo"],
            "amp": settings["amp"],
            "amp_dtype": settings.get("amp_dtype", "float16"),
            "predictions_saved": False,
        }
    )
    _write_json(output_dir / "train_images.json", records)
    _write_json(output_dir / "train_metrics.json", report)
    return report


def collect_training_reports(run_dir: Path, output_dir: Path | None = None) -> dict:
    settings, samples, assignments = _read_run(run_dir)
    destination = output_dir if output_dir is not None else run_dir
    destination.mkdir(parents=True, exist_ok=True)
    folds = {}
    by_id = {sample.image_id: sample for sample in samples}
    for fold in range(settings["n_splits"]):
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
                raise ValueError("Saved OOF evaluation does not match the manifest")
            held_out = summarize_predictions(records, expected)
            if not held_out["complete"]:
                raise ValueError(f"Incomplete saved OOF evaluation in fold {fold}")
        train_mean = training["primary_case_mean"]
        oof_mean = held_out["primary_case_mean"] if held_out else None
        folds[str(fold)] = {
            "train_images_evaluated": training["images_evaluated"],
            "train_cases_evaluated": training["cases_evaluated"],
            "train_primary_case_mean": train_mean,
            "train_pooled_pixels": training["pooled_pixels"],
            "oof_primary_case_mean": oof_mean,
            "oof_pooled_pixels": held_out["pooled_pixels"] if held_out else None,
            "foreground_macro_dice_gap": train_mean["foreground_macro_dice"]
            - oof_mean["foreground_macro_dice"]
            if oof_mean
            and train_mean["foreground_macro_dice"] is not None
            and oof_mean["foreground_macro_dice"] is not None
            else None,
        }
    missing = [fold for fold in range(settings["n_splits"]) if str(fold) not in folds]
    summary = {
        "complete": not missing,
        "split": "train",
        "evaluation_protocol": EVALUATION_PROTOCOL,
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
    settings, samples, assignments = _read_run(run_dir)
    selected = list(range(settings["n_splits"])) if selected_folds is None else selected_folds
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(not 0 <= fold < settings["n_splits"] for fold in selected)
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
        _validate_checkpoint(checkpoint, training, fold)
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
