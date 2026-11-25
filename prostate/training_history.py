"""Epoch metrics from existing training forwards, with an explicit online protocol."""

import csv
import json
from pathlib import Path

import torch

from prostate.archive_metrics import ARCHIVE_METRICS, ArchiveMetrics
from prostate.data import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES
from prostate.metrics import scores

ONLINE_PROTOCOL = "online_training_batches_v1"
METRICS = ("dice", "iou", "precision", "recall")


class EpochMetrics:
    def __init__(self, device: torch.device):
        self.matrix = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)
        self.archive_metrics = ArchiveMetrics()

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        self.archive_metrics.update(logits, targets)
        predictions = logits.detach().argmax(dim=1)
        valid = targets != IGNORE_INDEX
        codes = targets[valid] * NUM_CLASSES + predictions[valid]
        self.matrix += torch.bincount(codes, minlength=NUM_CLASSES**2).reshape(
            NUM_CLASSES, NUM_CLASSES
        )

    def report(self) -> dict:
        matrix = self.matrix.cpu().numpy()
        return {
            "confusion_matrix": matrix.tolist(),
            **scores(matrix),
            "archive_metrics": self.archive_metrics.report("batch"),
        }


class TrainingHistory:
    """Write scalar CSV columns plus complete epoch confusion matrices as JSONL."""

    def __init__(self, fold_dir: Path):
        self.fold_dir = fold_dir

    def __enter__(self):
        self.csv_stream = (self.fold_dir / "history.csv").open("w", newline="")
        self.json_stream = (self.fold_dir / "history_metrics.jsonl").open("w")
        fields = [
            "epoch",
            "train_loss",
            "seconds",
            "patches",
            "batches",
            "metrics_protocol",
            "online_foreground_macro_dice",
            "online_foreground_macro_iou",
            "online_pixel_accuracy",
        ]
        fields += [f"online_{metric}_{name}" for metric in METRICS for name in CLASS_NAMES]
        fields += [
            "archive_metrics_protocol",
            *[f"online_archive_{name}" for name in ARCHIVE_METRICS],
        ]
        self.writer = csv.DictWriter(self.csv_stream, fieldnames=fields)
        self.writer.writeheader()
        return self

    def log(
        self,
        epoch: int,
        loss: float,
        seconds: float,
        patches: int,
        batches: int,
        metrics: EpochMetrics,
    ):
        report = metrics.report()
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "seconds": seconds,
            "patches": patches,
            "batches": batches,
            "metrics_protocol": ONLINE_PROTOCOL,
            "archive_metrics_protocol": report["archive_metrics"]["protocol"],
        }
        row.update(
            {
                f"online_archive_{name}": value
                for name, value in report["archive_metrics"]["mean"].items()
            }
        )
        row.update(
            {
                f"online_{metric}": report[metric]
                for metric in ("foreground_macro_dice", "foreground_macro_iou", "pixel_accuracy")
            }
        )
        row.update(
            {
                f"online_{metric}_{name}": report[metric][name]
                for metric in METRICS
                for name in CLASS_NAMES
            }
        )
        self.writer.writerow(row)
        record = {
            "epoch": epoch,
            "train_loss": loss,
            "seconds": seconds,
            "patches": patches,
            "batches": batches,
            "metrics_protocol": ONLINE_PROTOCOL,
            **report,
        }
        self.json_stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.csv_stream.flush()
        self.json_stream.flush()

    def __exit__(self, *exc):
        self.csv_stream.close()
        self.json_stream.close()
