import csv
import json

import pytest
import torch
from torch.nn import functional as F

from prostate.training_history import ONLINE_PROTOCOL, EpochMetrics, TrainingHistory


def logits(predictions):
    return (
        F.one_hot(torch.tensor(predictions), num_classes=5)
        .permute(0, 3, 1, 2)
        .float()
        .requires_grad_()
    )


def test_epoch_metrics_ignore_padding_accumulate_batches_and_do_not_attach_gradients():
    metrics = EpochMetrics(torch.device("cpu"))
    first = logits([[[4, 0, 4], [4, 1, 4]]])
    metrics.update(first, torch.tensor([[[4, 4, 0], [4, 1, 255]]]))
    second = logits([[[0, 4]]])
    metrics.update(second, torch.tensor([[[4, 4]]]))
    report = metrics.report()
    assert sum(map(sum, report["confusion_matrix"])) == 7
    assert report["target_pixels"]["GP5"] == 5
    assert report["predicted_pixels"]["GP5"] == 4
    assert report["true_positive_pixels"]["GP5"] == 3
    assert report["dice"]["GP5"] == pytest.approx(2 / 3)
    assert report["precision"]["GP5"] == 0.75
    assert report["recall"]["GP5"] == 0.6
    assert not metrics.matrix.requires_grad
    assert first.grad is None and second.grad is None
    old = report["archive_metrics"]
    assert old["pixels"] == 7 and old["units"] == 2
    # Three of five valid pixels, then one of two: ignore padding and average batches equally.
    assert old["mean"]["pjac"] == pytest.approx(
        ((3 + 1) / (10 - 3 + 1) + (1 + 1) / (4 - 1 + 1)) / 2
    )
    true_class_probability = torch.exp(torch.tensor(1.0)).item() / (
        torch.exp(torch.tensor(1.0)).item() + 4
    )
    other_class_probability = 1 / (torch.exp(torch.tensor(1.0)).item() + 4)
    soft_first = 3 * true_class_probability + 2 * other_class_probability
    soft_second = true_class_probability + other_class_probability
    assert old["mean"]["jac"] == pytest.approx(
        ((soft_first + 1) / (10 - soft_first + 1) + (soft_second + 1) / (4 - soft_second + 1)) / 2
    )


def test_history_records_each_epoch_with_protocol_and_undefined_metrics(tmp_path):
    first = EpochMetrics(torch.device("cpu"))
    first.update(logits([[[4, 0]]]), torch.tensor([[[4, 4]]]))
    second = EpochMetrics(torch.device("cpu"))
    second.update(logits([[[0, 0]]]), torch.tensor([[[0, 0]]]))
    with TrainingHistory(tmp_path) as history:
        history.log(1, 0.4, 2.0, 2, 1, first)
        history.log(2, 0.2, 3.0, 2, 1, second)
        records = [
            json.loads(line)
            for line in (tmp_path / "history_metrics.jsonl").read_text().splitlines()
        ]
        assert len(records) == 2  # Flushed before the context closes.
    with (tmp_path / "history.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["epoch"] for row in rows] == ["1", "2"]
    assert rows[0]["metrics_protocol"] == ONLINE_PROTOCOL
    assert float(rows[0]["online_recall_GP5"]) == 0.5
    assert rows[1]["online_recall_GP5"] == ""
    assert records[0]["recall"]["GP5"] == 0.5
    assert records[1]["recall"]["GP5"] is None
    assert records[1]["target_pixels"]["GP5"] == 0
    assert all(record["metrics_protocol"] == ONLINE_PROTOCOL for record in records)
    assert rows[0]["archive_metrics_protocol"] == "archive_mean_batch_v1"
    for name in ("dice", "jac", "pdice", "pjac"):
        assert (
            float(rows[0][f"online_archive_{name}"]) == records[0]["archive_metrics"]["mean"][name]
        )
