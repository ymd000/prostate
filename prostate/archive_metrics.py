"""The archived soft jac and accuracy-derived pjac, kept separate from class IoU."""

import json
from pathlib import Path

import numpy as np
import torch

from prostate.data import IGNORE_INDEX

ARCHIVE_METRICS = ("dice", "jac", "pdice", "pjac")
DEFINITIONS = {
    "jac": "(sum_probability_of_true_class + 1) / (2 * pixels - sum_probability_of_true_class + 1)",
    "pjac": "(correct_pixels + 1) / (2 * pixels - correct_pixels + 1); includes background",
    "aggregation": "Arithmetic mean over batches/tiles, then arithmetic mean over images for evaluation",
}


class ArchiveMetrics:
    def __init__(self):
        self.values = []

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        probabilities = logits.detach().float().softmax(dim=1)
        valid = targets != IGNORE_INDEX
        pixels = int(valid.sum())
        if not pixels:
            raise ValueError("Archive metrics require valid target pixels")
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
            "mean": {
                name: float(np.mean([v[name] for v in self.values])) for name in ARCHIVE_METRICS
            },
            "definitions": DEFINITIONS,
        }


def archive_prediction_metadata(path: Path) -> dict:
    sidecar = path.with_suffix(".archive_metrics.json")
    return {"archive_metrics": json.loads(sidecar.read_text())} if sidecar.is_file() else {}


def summarize_archive_metrics(records: list[dict]) -> dict:
    values = [
        record["archive_metrics"]["mean"] for record in records if "archive_metrics" in record
    ]
    return {
        "protocol": "archive_image_mean_of_tile_means_v1",
        "images_evaluated": len(values),
        "image_mean": {
            name: float(np.mean([value[name] for value in values])) if values else None
            for name in ARCHIVE_METRICS
        },
        "definitions": DEFINITIONS,
    }
