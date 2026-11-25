import json

import numpy as np
import pytest
import torch
from torch import nn

from prostate.archive_metrics import (
    ARCHIVE_METRICS,
    archive_prediction_current,
    summarize_archive_metrics,
)
from prostate.inference import (
    predict_tiled,
    predict_tiled_with_encoder,
    save_label_png,
    tile_positions,
)


class FixedTileModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        rows, columns = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
        self.register_buffer(
            "logits", torch.stack([torch.sin(rows / (k + 1) + columns) * 3 for k in range(5)])[None]
        )

    def forward(self, inputs):
        self.calls += 1
        return self.logits


class CountingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward_features(self, inputs):
        assert not torch.is_grad_enabled()
        self.calls += 1
        return inputs.new_zeros((1, 5, 384))


@pytest.mark.parametrize("with_encoder", [False, True])
def test_cv_soft_metrics_use_only_retained_pixels_and_existing_forwards(tmp_path, with_encoder):
    truth = (np.indices((43, 57)).sum(0) % 5).astype(np.uint8)
    rgb = np.zeros((*truth.shape, 3), np.uint8)
    model = FixedTileModel().train()
    encoder = CountingEncoder()
    output = tmp_path / "prediction.npy"
    args = (model, rgb, output, 32, 5, torch.device("cpu"))
    if with_encoder:
        prediction = predict_tiled_with_encoder(encoder, [0, 0, 0], [1, 1, 1], *args, target=truth)
    else:
        prediction = predict_tiled(*args, target=truth)
    save_label_png(prediction, output.with_suffix(".png"))
    report = json.loads(output.with_suffix(".archive_metrics.json").read_text())
    assert model.training
    assert report["pixels"] == truth.size
    assert report["tile_region"] == "retained_core"
    assert report["image_shape"] == list(truth.shape)
    assert report["units"] == model.calls == 6
    assert encoder.calls == (6 if with_encoder else 0)

    probabilities = model.logits.double().softmax(1).numpy()[0]
    hard = probabilities.argmax(0)
    expected = []
    soft_total = 0
    for top, left, height, width in tile_positions(*truth.shape, 32, 5):
        target = truth[top : top + height, left : left + width]
        y, x = np.indices(target.shape) + 5
        soft = probabilities[target, y, x].sum()
        correct = (hard[y, x] == target).sum()
        pixels = target.size
        soft_total += soft
        expected.append(
            [
                (2 * soft + 1) / (2 * pixels + 1),
                (soft + 1) / (2 * pixels - soft + 1),
                (2 * correct + 1) / (2 * pixels + 1),
                (correct + 1) / (2 * pixels - correct + 1),
            ]
        )
        np.testing.assert_array_equal(
            prediction[top : top + height, left : left + width], hard[y, x]
        )
    assert report["mean"] == pytest.approx(
        dict(zip(ARCHIVE_METRICS, np.mean(expected, axis=0))), rel=1e-6
    )
    pooled_jac = (soft_total + 1) / (2 * truth.size - soft_total + 1)
    assert abs(report["mean"]["jac"] - pooled_jac) > 1e-5
    settings = {"patch_size": 32, "halo": 5}
    assert archive_prediction_current(output, settings, truth.shape)
    assert not archive_prediction_current(output, {**settings, "halo": 4}, truth.shape)
    assert not archive_prediction_current(output, settings, (43, 58))


def test_archive_aggregation_keeps_image_weights_fold_means_and_missing_soft_scores():
    records = [
        {
            "image_id": image,
            "case_id": case,
            "fold": fold,
            "archive_metrics": {
                "mean": dict.fromkeys(ARCHIVE_METRICS, value),
                "tile_region": "retained_core",
            },
        }
        for image, case, fold, value in (
            ("a1", "a", 0, 0.1),
            ("a2", "a", 0, 0.3),
            ("b1", "b", 1, 0.9),
        )
    ]
    records.append({"image_id": "c1", "case_id": "c", "fold": 1})
    report = summarize_archive_metrics(records, ["a1", "a2", "b1", "c1", "d1"])
    assert not report["complete"] and report["missing_image_ids"] == ["c1", "d1"]
    assert report["images_evaluated"] == 3
    assert report["image_mean"]["jac"] == pytest.approx(1.3 / 3)
    assert report["fold_image_means"]["0"]["jac"] == pytest.approx(0.2)
    assert report["fold_image_means"]["1"]["jac"] == pytest.approx(0.9)
    assert report["tile_regions"] == ["retained_core"]
    missing = summarize_archive_metrics([records[-1]])
    assert not missing["complete"] and missing["image_mean"] == dict.fromkeys(ARCHIVE_METRICS)
