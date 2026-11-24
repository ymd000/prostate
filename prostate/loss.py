"""Cross-entropy setup and auditable pixel-frequency weights from the training split."""

import json

import numpy as np
import torch
from torch import nn

from prostate.config import ArchiveConfig, Config
from prostate.data import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, Sample


def class_weights(counts: np.ndarray, mode: str) -> np.ndarray | None:
    """Normalize inverse square-root frequencies to mean 1 over present classes."""
    counts = np.asarray(counts)
    if (
        counts.shape != (NUM_CLASSES,)
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts < 0)
        or counts.sum() <= 0
    ):
        raise ValueError("Expected five nonnegative integer class counts with a positive total")
    if mode == "none":
        return None
    if mode != "inverse-sqrt":
        raise ValueError("class_weighting must be none or inverse-sqrt")
    present = counts > 0
    weights = np.zeros(NUM_CLASSES, dtype=np.float64)
    weights[present] = np.sqrt(counts.sum() / counts[present])
    weights[present] /= weights[present].mean()
    return weights.astype(np.float32)


def prepare_cross_entropy(
    samples: list[Sample], config: Config | ArchiveConfig, fold: int, device: torch.device
) -> tuple[nn.CrossEntropyLoss, dict]:
    """Create FP32 CE and save the exact training-only counts and weights before training."""
    if not samples or len({sample.image_id for sample in samples}) != len(samples):
        raise ValueError("Loss setup requires a nonempty, unique training image list")
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for sample in samples:
        values = np.asarray(sample.class_counts)
        if (
            values.shape != (NUM_CLASSES,)
            or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0)
            or values.sum() != sample.height * sample.width
        ):
            raise ValueError(f"Invalid cached class counts: {sample.image_id}")
        counts += values
    weights = class_weights(counts, config.class_weighting)
    tensor = None if weights is None else torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=tensor, ignore_index=IGNORE_INDEX)
    report = {
        "name": "cross_entropy",
        "class_weighting": config.class_weighting,
        "fold": fold,
        "classes": list(CLASS_NAMES),
        "count_source": "training_images_unpadded_pixels",
        "class_counts": {name: int(count) for name, count in zip(CLASS_NAMES, counts, strict=True)},
        "class_frequencies": {
            name: float(count / counts.sum())
            for name, count in zip(CLASS_NAMES, counts, strict=True)
        },
        "class_weights": None
        if weights is None
        else {name: float(weight) for name, weight in zip(CLASS_NAMES, weights, strict=True)},
        "weight_formula": None if weights is None else "1 / sqrt(training_pixel_frequency)",
        "weight_normalization": None if weights is None else "mean_present_class_weight_is_1",
        "absent_class_policy": "weight_zero" if weights is not None else "unweighted",
        "ignore_index": IGNORE_INDEX,
        "reduction": "mean",
        "loss_dtype": "float32",
        "train_image_ids": [sample.image_id for sample in samples],
        "train_case_ids": sorted({sample.case_id for sample in samples}),
    }
    destination = config.run_dir / f"fold_{fold}"
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / "loss.tmp.json"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination / "loss.json")
    print(
        f"Fold {fold}: CE class weighting={config.class_weighting}; weights={report['class_weights']}"
    )
    return criterion, report
