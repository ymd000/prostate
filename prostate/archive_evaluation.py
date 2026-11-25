"""Evaluate a final archive checkpoint with independent inference settings."""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from prostate.archive_metrics import archive_prediction_metadata, summarize_archive_metrics
from prostate.config import ArchiveEvaluationConfig
from prostate.folds import validate_checkpoint_split
from prostate.inference import load_checkpoint_model, predict_sample, save_label_png
from prostate.metrics import confusion_matrix, scores, summarize_predictions
from prostate.train import resolve_device
from prostate.training_evaluation import (
    ARCHIVE_EVALUATION_PROTOCOL,
    compare_training_heldout,
    evaluate_training_fold,
    read_run,
)


def _write_json(path: Path, content: dict | list):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def evaluate_archive_checkpoint(config: ArchiveEvaluationConfig) -> dict:
    """Never train, change source artifacts, or download pretrained model weights."""
    source = config.run_dir.resolve()
    settings, samples, assignments = read_run(source)
    if settings.get("training_recipe") != "archive_random_v1":
        raise ValueError("archive-evaluate requires an archive-holdout experiment")
    checkpoint_path = source / "fold_0" / "checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No final checkpoint is available: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    training = [sample for sample in samples if assignments[sample.image_id] == -1]
    held_out = [sample for sample in samples if assignments[sample.image_id] == 0]
    validate_checkpoint_split(checkpoint, training, 0)
    if checkpoint.get("architecture") != "archive_unet_v1" or checkpoint["config"] != settings:
        raise ValueError("Checkpoint architecture/config does not match the saved experiment")

    device = resolve_device(config.device)
    effective = dict(settings)
    if config.inference_tile_size is not None:
        effective["inference_tile_size"] = config.inference_tile_size
    if config.validation_mode is not None:
        effective["validation_mode"] = config.validation_mode
    effective["device"] = str(device)
    side, mode = effective["inference_tile_size"], effective["validation_mode"]
    destination = (
        config.output_dir.resolve()
        if config.output_dir is not None
        else source / f"evaluation-tile{side}-{mode}-{device.type}"
    )
    if destination == source or checkpoint_path.is_relative_to(destination):
        raise ValueError("Evaluation output must be separate from the source checkpoint directory")
    stat = checkpoint_path.stat()
    metadata = {
        "source_run_dir": str(source),
        "checkpoint_fingerprint": {
            "path": str(checkpoint_path),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "training_config": settings,
        "inference_settings": {
            "inference_tile_size": side,
            "validation_mode": mode,
            "device": str(device),
            "amp": False,
        },
        "evaluation_protocol": ARCHIVE_EVALUATION_PROTOCOL,
    }
    metadata_path = destination / "evaluation_config.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("Evaluation settings/checkpoint differ. Use a new --output-dir.")
    elif destination.exists() and any(destination.iterdir()):
        raise ValueError("Evaluation output directory is not empty. Use a new --output-dir.")
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(metadata_path, metadata)
    torch.set_num_threads(settings.get("cpu_threads", 4))
    print(f"Loading final checkpoint; training will not run: {checkpoint_path}")
    print(f"Evaluation: device={device}, tile={side}, mode={mode}, output={destination}")
    model = load_checkpoint_model(checkpoint, device)

    records = []
    for sample in tqdm(held_out, desc="Archive checkpoint holdout"):
        output = destination / "holdout" / f"{sample.image_id}.npy"
        prediction = predict_sample(model, sample, output, effective, device)
        save_label_png(prediction, output.with_suffix(".png"))
        target = np.load(sample.label_cache, mmap_mode="r")
        matrix = confusion_matrix(target, prediction)
        records.append(
            {
                "image_id": sample.image_id,
                "case_id": sample.case_id,
                "fold": 0,
                "prediction": str(output),
                "confusion_matrix": matrix.tolist(),
                **scores(matrix),
                **archive_prediction_metadata(output),
            }
        )
        del target, prediction
    summary = summarize_predictions(records, [sample.image_id for sample in held_out])
    summary.update(
        split="holdout",
        holdout_case_id=settings["holdout_case_id"],
        source_run_dir=str(source),
        output_dir=str(destination),
        evaluation_protocol=ARCHIVE_EVALUATION_PROTOCOL,
        inference_settings=metadata["inference_settings"],
        checkpoint_fingerprint=metadata["checkpoint_fingerprint"],
        archive_metrics=summarize_archive_metrics(records),
    )
    _write_json(destination / "holdout_images.json", records)
    _write_json(destination / "holdout_metrics.json", summary)

    # Only this in-memory copy receives overrides; the actual checkpoint and
    # the training experiment's config/history remain untouched.
    evaluation_checkpoint = checkpoint | {"config": effective}
    training_report = evaluate_training_fold(
        model, training, evaluation_checkpoint, checkpoint_path, destination / "fold_0", device
    )
    comparison = {
        "complete": summary["complete"] and training_report["complete"],
        "split": "train",
        "evaluation_split": "holdout",
        "evaluation_protocol": ARCHIVE_EVALUATION_PROTOCOL,
        "run_dir": str(source),
        "output_dir": str(destination),
        "inference_settings": metadata["inference_settings"],
        "folds_evaluated": [0],
        "missing_folds": [],
        "folds": {"0": compare_training_heldout(training_report, summary, "holdout")},
    }
    _write_json(destination / "train_metrics.json", comparison)
    return summary
