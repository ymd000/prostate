"""Collect held-out metrics and recover missing soft scores from final checkpoints."""

import gc
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from prostate.archive_metrics import (
    archive_prediction_current,
    archive_prediction_metadata,
    summarize_archive_metrics,
)
from prostate.data import Sample
from prostate.folds import evaluation_folds, prediction_split, validate_checkpoint_split
from prostate.inference import load_checkpoint_model, predict_sample, save_label_png
from prostate.metrics import confusion_matrix, scores, summarize_oof, summarize_predictions
from prostate.train import resolve_device
from prostate.training_evaluation import read_run


def _write_json(path: Path, value: dict | list):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def evaluate_prediction(sample: Sample, prediction_path: Path, fold: int) -> dict:
    target = np.load(sample.label_cache, mmap_mode="r")
    prediction = np.load(prediction_path, mmap_mode="r")
    matrix = confusion_matrix(target, prediction)
    return {
        "image_id": sample.image_id,
        "case_id": sample.case_id,
        "fold": fold,
        "prediction": str(prediction_path.resolve()),
        "confusion_matrix": matrix.tolist(),
        **scores(matrix),
        **archive_prediction_metadata(prediction_path),
    }


def _infer_missing(
    run_dir: Path, settings: dict, samples: list[Sample], assignments: dict, requested_device: str
):
    """Use saved splits/settings directly, including runs with older training recipes."""
    split = prediction_split(settings)
    device, encoder_bundle = None, None
    available = 0
    for fold in evaluation_folds(settings):
        folder = run_dir / f"fold_{fold}"
        checkpoint_path = folder / "checkpoint.pt"
        evaluation_path = folder / "evaluation.json"
        if not checkpoint_path.is_file():
            if evaluation_path.is_file():
                raise FileNotFoundError(f"Missing final checkpoint: {checkpoint_path}")
            print(f"Fold {fold}: no final checkpoint; skipping inference")
            continue
        available += 1
        held_out = [sample for sample in samples if assignments[sample.image_id] == fold]
        missing = [
            sample
            for sample in held_out
            if not archive_prediction_current(
                run_dir / split / f"{sample.image_id}.npy", settings, (sample.height, sample.width)
            )
        ]
        if not missing and evaluation_path.is_file():
            continue
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        training = [sample for sample in samples if assignments[sample.image_id] != fold]
        validate_checkpoint_split(checkpoint, training, fold)
        saved, checkpoint_settings = dict(settings), dict(checkpoint["config"])
        for config in (saved, checkpoint_settings):
            config.setdefault("class_weighting", "none")
        if saved != checkpoint_settings:
            raise ValueError("Checkpoint config does not match the saved experiment")
        if missing:
            if device is None:
                device = resolve_device(requested_device)
                torch.set_num_threads(settings.get("cpu_threads", 4))
            print(
                f"Fold {fold}: inferring {len(missing)} image(s) for missing archive metrics on {device}; training will not run"
            )
            if checkpoint["architecture"] == "token_decoder_v1" and encoder_bundle is None:
                from prostate.extractor import load_gigapath_flash

                encoder_bundle = load_gigapath_flash(device)
            bundle = encoder_bundle if checkpoint["architecture"] == "token_decoder_v1" else None
            model = load_checkpoint_model(checkpoint, device)
            for sample in tqdm(missing, desc=f"Fold {fold} {split} inference"):
                output = run_dir / split / f"{sample.image_id}.npy"
                prediction = predict_sample(
                    model, sample, output, checkpoint["config"], device, bundle
                )
                save_label_png(prediction, output.with_suffix(".png"))
                del prediction
            del model
        records = [
            evaluate_prediction(sample, run_dir / split / f"{sample.image_id}.npy", fold)
            for sample in held_out
        ]
        _write_json(evaluation_path, records)
        del checkpoint
        gc.collect()
        if device is not None and device.type == "cuda":
            torch.cuda.empty_cache()
    if not available:
        raise FileNotFoundError(
            "No final checkpoints are available; evaluation cannot train a model"
        )


def evaluate_run(
    run_dir: Path, *, infer_missing: bool = False, requested_device: str = "auto"
) -> dict:
    settings, sample_list, assignments = read_run(run_dir)
    if infer_missing:
        _infer_missing(run_dir, settings, sample_list, assignments, requested_device)
    samples = {sample.image_id: sample for sample in sample_list}
    split = prediction_split(settings)
    records = []
    for fold in evaluation_folds(settings):
        path = run_dir / f"fold_{fold}" / "evaluation.json"
        if not path.exists():
            continue
        checkpoint = torch.load(
            path.parent / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        training = [sample for key, sample in samples.items() if assignments[key] != fold]
        validate_checkpoint_split(checkpoint, training, fold)
        items = json.loads(path.read_text())
        expected = {key for key, value in assignments.items() if value == fold}
        if {item["image_id"] for item in items} != expected or len(items) != len(expected):
            raise ValueError(f"Incomplete fold evaluation: {path}")
        for item in items:
            sample = samples[item["image_id"]]
            if item["case_id"] != sample.case_id or item["fold"] != fold:
                raise ValueError("Saved evaluation does not match the manifest")
            prediction_path = run_dir / split / f"{sample.image_id}.npy"
            if not prediction_path.with_suffix(".png").exists():
                raise ValueError(f"Missing {split} PNG: {sample.image_id}")
            records.append(evaluate_prediction(sample, prediction_path, fold))
    if split == "holdout":
        expected = sorted(key for key, value in assignments.items() if value == 0)
        summary = summarize_predictions(records, expected)
        summary.update(split=split, holdout_case_id=settings["holdout_case_id"])
    else:
        expected = sorted(samples)
        summary = summarize_oof(records, expected, settings["n_splits"])
    if settings.get("training_recipe") == "archive_random_v1" and any(
        "archive_metrics" not in record for record in records
    ):
        raise ValueError("Missing archive metric sidecars; regenerate the predictions")
    summary["archive_metrics"] = summarize_archive_metrics(records, expected)
    _write_json(run_dir / f"{split}_metrics.json", summary)
    _write_json(run_dir / f"{split}_images.json", records)
    return summary
