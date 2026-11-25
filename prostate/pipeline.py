"""Prepare immutable CV or holdout experiments and collect held-out predictions."""

import csv
import gc
import io
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from prostate.archive_metrics import archive_prediction_metadata, summarize_archive_metrics
from prostate.config import ArchiveConfig, ArchiveEvaluationConfig, Config
from prostate.data import CLASS_NAMES, Sample, prepare_dataset
from prostate.folds import (
    assign_folds,
    assign_holdout,
    evaluation_folds,
    prediction_split,
    read_assignments,
    validate_checkpoint_split,
)
from prostate.inference import load_checkpoint_model, predict_sample, save_label_png
from prostate.metrics import confusion_matrix, scores, summarize_oof, summarize_predictions
from prostate.precision import resolve_precision
from prostate.train import resolve_device, train_fold, train_token_fold
from prostate.training_evaluation import (
    collect_training_reports,
    evaluate_training_fold,
    training_report_current,
)


def write_json(path: Path, value: dict | list):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _save_fixed(path: Path, content: str):
    if path.exists() and path.read_text() != content:
        raise ValueError(f"Experiment differs from {path}. Use a new --run-dir.")
    if not path.exists():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(content)
        temporary.replace(path)


def _save_training_config(config: Config | ArchiveConfig):
    path = config.run_dir / "config.json"
    if path.exists():
        saved = json.loads(path.read_text())
        # Runs made before class weighting used ordinary CE. Preserve their config
        # while allowing the same unweighted objective to resume.
        saved.setdefault("class_weighting", "none")
        if saved != config.to_dict():
            raise ValueError(f"Experiment differs from {path}. Use a new --run-dir.")
    else:
        _save_fixed(path, json.dumps(config.to_dict(), indent=2) + "\n")


def prepare_run(config: Config | ArchiveConfig) -> tuple[list[Sample], dict[str, int]]:
    samples = prepare_dataset(config.data_dir, config.cache_dir)
    holdout = config.holdout_case_id is not None
    assignment = (
        assign_holdout(samples, config.holdout_case_id)
        if holdout
        else assign_folds(samples, config.n_splits, config.seed)
    )
    config.run_dir.mkdir(parents=True, exist_ok=True)
    incompatible = "folds.csv" if holdout else "split.csv"
    if (config.run_dir / incompatible).exists():
        raise ValueError("Experiment split differs. Use a new --run-dir.")
    if holdout:
        _save_training_config(config)
    manifest = {"classes": CLASS_NAMES, "samples": [sample.to_dict() for sample in samples]}
    _save_fixed(config.run_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    table = io.StringIO()
    writer = csv.writer(table, lineterminator="\n")
    writer.writerow(
        ["image_id", "case_id", "split" if holdout else "fold", "width", "height", *CLASS_NAMES]
    )
    for sample in samples:
        writer.writerow(
            [
                sample.image_id,
                sample.case_id,
                ("holdout" if assignment[sample.image_id] == 0 else "train")
                if holdout
                else assignment[sample.image_id],
                sample.width,
                sample.height,
                *sample.class_counts,
            ]
        )
    _save_fixed(config.run_dir / ("split.csv" if holdout else "folds.csv"), table.getvalue())
    for fold in evaluation_folds(config.to_dict()):
        held_out = [sample for sample in samples if assignment[sample.image_id] == fold]
        label = "Holdout" if holdout else f"Fold {fold}"
        print(f"{label}: {len(held_out)} images, {len({s.case_id for s in held_out})} cases")
    if holdout:
        training = [sample for sample in samples if assignment[sample.image_id] == -1]
        print(f"Training: {len(training)} images, {len({s.case_id for s in training})} cases")
    return samples, assignment


def _evaluate_image(sample: Sample, prediction_path: Path, fold: int) -> dict:
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


def evaluate_run(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    sample_list = [Sample(**item) for item in manifest["samples"]]
    assignments = read_assignments(run_dir, config, sample_list)
    samples = {sample.image_id: sample for sample in sample_list}
    split = prediction_split(config)
    records = []
    for fold in evaluation_folds(config):
        path = run_dir / f"fold_{fold}" / "evaluation.json"
        if not path.exists():
            continue
        checkpoint_path = path.parent / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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
            records.append(_evaluate_image(sample, prediction_path, fold))
    if split == "holdout":
        expected = sorted(key for key, value in assignments.items() if value == 0)
        summary = summarize_predictions(records, expected)
        summary.update(split=split, holdout_case_id=config["holdout_case_id"])
    else:
        summary = summarize_oof(records, sorted(samples), config["n_splits"])
    if config.get("training_recipe") == "archive_random_v1":
        if any("archive_metrics" not in record for record in records):
            raise ValueError("Missing archive metric sidecars; regenerate the predictions")
        summary["archive_metrics"] = summarize_archive_metrics(records)
    write_json(run_dir / f"{split}_metrics.json", summary)
    write_json(run_dir / f"{split}_images.json", records)
    return summary


def extract_run(config: Config, batch_size: int = 8):
    from prostate.extractor import extract_dataset

    samples = prepare_dataset(config.data_dir, config.cache_dir)
    device = resolve_device(config.device)
    extract_dataset(samples, config.token_dir, config.patch_size, batch_size, device)


def run_cv(config: Config, selected_folds: list[int] | None = None, resume: bool = False) -> dict:
    if config.holdout_case_id is not None:
        raise ValueError("Use run_holdout for a fixed holdout case")
    selected = list(range(config.n_splits)) if selected_folds is None else selected_folds
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Choose each fold at most once")
    if any(fold < 0 or fold >= config.n_splits for fold in selected):
        raise ValueError(f"Fold IDs must be in 0..{config.n_splits - 1}")
    return _run_splits(config, selected, resume)


def run_holdout(config: Config | ArchiveConfig, resume: bool = False) -> dict:
    if config.holdout_case_id is None:
        raise ValueError("A holdout case ID is required")
    return _run_splits(config, [0], resume)


def evaluate_archive_checkpoint(config: ArchiveEvaluationConfig) -> dict:
    from prostate.archive_evaluation import evaluate_archive_checkpoint as evaluate

    return evaluate(config)


def _run_splits(config: Config | ArchiveConfig, selected: list[int], resume: bool) -> dict:
    device = resolve_device(config.device)
    amp_enabled, amp_dtype = resolve_precision(device, config.amp, config.amp_dtype)
    print(f"Device: {device}; AMP: {amp_enabled}; precision: {amp_dtype}")
    samples, assignment = prepare_run(config)
    _save_training_config(config)
    split = prediction_split(config.to_dict())

    encoder_bundle = None
    if config.encoder == "gigapath-flash":
        from prostate.extractor import load_gigapath_flash

        missing = [
            sample.image_id
            for sample in samples
            if split != "holdout" or assignment[sample.image_id] != 0
            if not (config.token_dir / f"{sample.image_id}.h5").exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} token file(s) missing (e.g. {missing[0]}.h5). "
                "Run 'python -m prostate extract' first."
            )
        encoder_bundle = load_gigapath_flash(device)

    for fold in selected:
        training = [sample for sample in samples if assignment[sample.image_id] != fold]
        held_out = [sample for sample in samples if assignment[sample.image_id] == fold]
        train_cases = {sample.case_id for sample in training}
        if train_cases & {sample.case_id for sample in held_out}:
            raise ValueError("Training and held-out cases overlap")
        checkpoint_path = config.run_dir / f"fold_{fold}/checkpoint.pt"
        evaluation_path = config.run_dir / f"fold_{fold}" / "evaluation.json"
        if evaluation_path.exists():
            if resume:
                print(f"Fold {fold}: already complete; skipping")
                if not training_report_current(checkpoint_path, evaluation_path.parent):
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                    validate_checkpoint_split(checkpoint, training, fold)
                    model = load_checkpoint_model(checkpoint, device)
                    evaluate_training_fold(
                        model,
                        training,
                        checkpoint,
                        checkpoint_path,
                        evaluation_path.parent,
                        device,
                        encoder_bundle,
                    )
                    del model, checkpoint
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                continue
            raise ValueError(f"Fold {fold} already exists. Use --resume or a new --run-dir.")

        if checkpoint_path.exists():
            if not resume:
                raise ValueError(
                    f"Final checkpoint already exists in fold {fold}. "
                    "Use --resume to finish inference without retraining, "
                    "or archive-evaluate to change archive inference settings."
                )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            validate_checkpoint_split(checkpoint, training, fold)
            print(f"Fold {fold}: final checkpoint found; resuming inference without training")
            model = load_checkpoint_model(checkpoint, device)
            del checkpoint
        elif isinstance(config, ArchiveConfig):
            from prostate.archive_train import train_archive_fold

            model = train_archive_fold(training, config, fold, device)
        elif config.encoder == "gigapath-flash":
            model = train_token_fold(training, config, fold, device)
        else:
            model = train_fold(training, config, fold, device)

        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        records = []
        for sample in tqdm(held_out, desc=f"Fold {fold} {split}"):
            output = config.run_dir / split / f"{sample.image_id}.npy"
            labels = predict_sample(model, sample, output, config.to_dict(), device, encoder_bundle)
            save_label_png(labels, output.with_suffix(".png"))
            records.append(_evaluate_image(sample, output, fold))
        write_json(evaluation_path, records)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        evaluate_training_fold(
            model,
            training,
            checkpoint,
            checkpoint_path,
            evaluation_path.parent,
            device,
            encoder_bundle,
        )
        del model, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = evaluate_run(config.run_dir)
    collect_training_reports(config.run_dir)
    return summary
