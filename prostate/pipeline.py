"""Prepare immutable experiments, train grouped CV and collect held-out predictions."""

import csv
import gc
import io
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from prostate.config import Config
from prostate.data import CLASS_NAMES, Sample, prepare_dataset
from prostate.folds import assign_folds
from prostate.inference import load_checkpoint_model, predict_sample, save_label_png
from prostate.metrics import confusion_matrix, scores, summarize_oof
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


def _save_training_config(config: Config):
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


def prepare_run(config: Config) -> tuple[list[Sample], dict[str, int]]:
    samples = prepare_dataset(config.data_dir, config.cache_dir)
    assignment = assign_folds(samples, config.n_splits, config.seed)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"classes": CLASS_NAMES, "samples": [sample.to_dict() for sample in samples]}
    _save_fixed(config.run_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    table = io.StringIO()
    writer = csv.writer(table, lineterminator="\n")
    writer.writerow(["image_id", "case_id", "fold", "width", "height", *CLASS_NAMES])
    for sample in samples:
        writer.writerow(
            [
                sample.image_id,
                sample.case_id,
                assignment[sample.image_id],
                sample.width,
                sample.height,
                *sample.class_counts,
            ]
        )
    _save_fixed(config.run_dir / "folds.csv", table.getvalue())
    for fold in range(config.n_splits):
        held_out = [sample for sample in samples if assignment[sample.image_id] == fold]
        print(f"Fold {fold}: {len(held_out)} images, {len({s.case_id for s in held_out})} cases")
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
    }


def evaluate_run(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    samples = {item["image_id"]: Sample(**item) for item in manifest["samples"]}
    with (run_dir / "folds.csv").open(newline="") as stream:
        assignments = {row["image_id"]: int(row["fold"]) for row in csv.DictReader(stream)}
    records = []
    for fold in range(config["n_splits"]):
        path = run_dir / f"fold_{fold}" / "evaluation.json"
        if not path.exists():
            continue
        checkpoint_path = path.parent / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["fold"] != fold:
            raise ValueError(f"Checkpoint fold mismatch: {checkpoint_path}")
        items = json.loads(path.read_text())
        expected = {key for key, value in assignments.items() if value == fold}
        if {item["image_id"] for item in items} != expected or len(items) != len(expected):
            raise ValueError(f"Incomplete fold evaluation: {path}")
        for item in items:
            sample = samples[item["image_id"]]
            if sample.case_id in checkpoint["train_case_ids"]:
                raise ValueError(f"OOF case leakage: {sample.case_id}")
            prediction_path = run_dir / "oof" / f"{sample.image_id}.npy"
            if not prediction_path.with_suffix(".png").exists():
                raise ValueError(f"Missing OOF PNG: {sample.image_id}")
            records.append(_evaluate_image(sample, prediction_path, fold))
    summary = summarize_oof(records, sorted(samples), config["n_splits"])
    write_json(run_dir / "oof_metrics.json", summary)
    write_json(run_dir / "oof_images.json", records)
    return summary


def extract_run(config: Config, batch_size: int = 8):
    from prostate.extractor import extract_dataset

    samples = prepare_dataset(config.data_dir, config.cache_dir)
    device = resolve_device(config.device)
    extract_dataset(samples, config.token_dir, config.patch_size, batch_size, device)


def run_cv(config: Config, selected_folds: list[int] | None = None, resume: bool = False) -> dict:
    selected = list(range(config.n_splits)) if selected_folds is None else selected_folds
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Choose each fold at most once")
    if any(fold < 0 or fold >= config.n_splits for fold in selected):
        raise ValueError(f"Fold IDs must be in 0..{config.n_splits - 1}")
    device = resolve_device(config.device)
    amp_enabled, amp_dtype = resolve_precision(device, config.amp, config.amp_dtype)
    print(f"Device: {device}; AMP: {amp_enabled}; precision: {amp_dtype}")
    samples, assignment = prepare_run(config)
    _save_training_config(config)

    encoder_bundle = None
    if config.encoder == "gigapath-flash":
        from prostate.extractor import load_gigapath_flash

        missing = [
            sample.image_id
            for sample in samples
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
            raise ValueError("Training and OOF cases overlap")
        checkpoint_path = config.run_dir / f"fold_{fold}/checkpoint.pt"
        evaluation_path = config.run_dir / f"fold_{fold}" / "evaluation.json"
        if evaluation_path.exists():
            if resume:
                print(f"Fold {fold}: already complete; skipping")
                if not training_report_current(checkpoint_path, evaluation_path.parent):
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                    if checkpoint["fold"] != fold:
                        raise ValueError(f"Checkpoint fold mismatch: {checkpoint_path}")
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

        if config.encoder == "gigapath-flash":
            model = train_token_fold(training, config, fold, device)
        else:
            model = train_fold(training, config, fold, device)

        records = []
        for sample in tqdm(held_out, desc=f"Fold {fold} OOF"):
            output = config.run_dir / "oof" / f"{sample.image_id}.npy"
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
