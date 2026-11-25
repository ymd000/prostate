"""The archived soft jac and accuracy-derived pjac, kept separate from class IoU."""

import json
from pathlib import Path

import numpy as np
import torch

from prostate.data import IGNORE_INDEX

ARCHIVE_METRICS = ("dice", "jac", "pdice", "pjac")
DEFINITIONS = {
    "dice": "(2 * sum_probability_of_true_class + 1) / (2 * pixels + 1)",
    "jac": "(sum_probability_of_true_class + 1) / (2 * pixels - sum_probability_of_true_class + 1)",
    "pdice": "(2 * correct_pixels + 1) / (2 * pixels + 1)",
    "pjac": "(correct_pixels + 1) / (2 * pixels - correct_pixels + 1); includes background",
    "classes": "All five classes, including non_gland (background); IGNORE_INDEX is excluded",
    "aggregation": "Arithmetic mean over batches/tiles, then arithmetic mean over images for evaluation",
}


class ArchiveMetrics:
    def __init__(self):
        self.values = []
        self.pixels = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        probabilities = logits.detach().float().softmax(dim=1)
        valid = targets != IGNORE_INDEX
        pixels = int(valid.sum())
        if not pixels:
            raise ValueError("Archive metrics require valid target pixels")
        self.pixels += pixels
        safe = targets.masked_fill(~valid, 0)
        soft_correct = probabilities.gather(1, safe.unsqueeze(1)).squeeze(1)[valid].sum().item()
        hard_correct = int(((probabilities.argmax(dim=1) == targets) & valid).sum())
        self.values.append(
            {
                "dice": (2 * soft_correct + 1) / (2 * pixels + 1),
                "jac": (soft_correct + 1) / (2 * pixels - soft_correct + 1),
                "pdice": (2 * hard_correct + 1) / (2 * pixels + 1),
                "pjac": (hard_correct + 1) / (2 * pixels - hard_correct + 1),
            }
        )

    def report(self, unit: str) -> dict:
        return {
            "protocol": f"archive_mean_{unit}_v1",
            "units": len(self.values),
            "pixels": self.pixels,
            "mean": {
                name: float(np.mean([v[name] for v in self.values])) if self.values else None
                for name in ARCHIVE_METRICS
            },
            "definitions": DEFINITIONS,
        }


def archive_prediction_metadata(path: Path) -> dict:
    sidecar = path.with_suffix(".archive_metrics.json")
    return {"archive_metrics": json.loads(sidecar.read_text())} if sidecar.is_file() else {}


def save_archive_prediction_metrics(path: Path, metrics: ArchiveMetrics, **metadata):
    sidecar = path.with_suffix(".archive_metrics.json")
    temporary = sidecar.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({**metrics.report("tile"), **metadata}, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(sidecar)


def archive_prediction_current(path: Path, settings: dict, shape: tuple[int, int]) -> bool:
    """Soft scores need a saved forward pass; hard predictions cannot reconstruct them."""
    if not path.is_file() or not path.with_suffix(".png").is_file():
        return False
    try:
        report = archive_prediction_metadata(path)["archive_metrics"]
        if report["protocol"] != "archive_mean_tile_v1" or not all(
            np.isfinite(report["mean"][name]) for name in ARCHIVE_METRICS
        ):
            return False
        if settings.get("training_recipe") == "archive_random_v1":
            return all(
                report[key] == settings[key] for key in ("validation_mode", "inference_tile_size")
            )
        return (
            report["tile_region"] == "retained_core"
            and report["pixels"] == shape[0] * shape[1]
            and report["image_shape"] == list(shape)
            and all(report[key] == settings[key] for key in ("patch_size", "halo"))
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def summarize_archive_metrics(records: list[dict], expected_ids: list[str] | None = None) -> dict:
    available = [record for record in records if "archive_metrics" in record]
    expected = set(expected_ids) if expected_ids is not None else {r["image_id"] for r in records}
    missing = sorted(expected - {record["image_id"] for record in available})

    def mean(items):
        return {
            name: float(np.mean([r["archive_metrics"]["mean"][name] for r in items]))
            if items
            else None
            for name in ARCHIVE_METRICS
        }

    return {
        "protocol": "archive_image_mean_of_tile_means_v1",
        "complete": not missing,
        "missing_image_ids": missing,
        "images_evaluated": len(available),
        "image_mean": mean(available),
        "fold_image_means": {
            str(fold): mean([r for r in available if r["fold"] == fold])
            for fold in sorted({r["fold"] for r in available})
        },
        "tile_regions": sorted(
            {r["archive_metrics"].get("tile_region", "unpadded_tile") for r in available}
        ),
        "definitions": DEFINITIONS,
    }
