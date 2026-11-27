import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from prostate.archive_data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ArchivePatchDataset,
    archive_image_tensor,
)
from prostate.archive_inference import archive_tile_positions, predict_archive_tiled
from prostate.config import ArchiveConfig
from prostate.data import Sample


def cached_sample(folder, name, height, width, empty=False):
    labels = (np.indices((height, width)).sum(axis=0) % 5).astype(np.uint8)
    rgb = np.repeat((labels * 40)[..., None], 3, axis=2)
    rgb_path, label_path, annotation = (
        folder / f"{name}_rgb.npy",
        folder / f"{name}_labels.npy",
        folder / f"{name}.png",
    )
    # Give each synthetic image its own cache folder, like prepare_dataset.
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(rgb_path, rgb)
    np.save(label_path, labels)
    Image.new("RGBA", (width, height), (0, 0, 0, 0) if empty else (255, 255, 255, 255)).save(
        annotation
    )
    return Sample(
        name,
        name.split("_")[0],
        "unused.jpg",
        str(annotation),
        str(rgb_path),
        str(label_path),
        width,
        height,
        tuple(np.bincount(labels.ravel(), minlength=5).tolist()),
        {},
    )


@pytest.mark.parametrize("step", [1, 16])
def test_random_crop_count_probabilities_replay_and_label_alignment(tmp_path, step):
    samples = [
        cached_sample(tmp_path / "A", "A_1", 131, 195),
        cached_sample(tmp_path / "B", "B_1", 85, 90),
    ]
    config = ArchiveConfig(patch_size=64, batch_size=2, num_workers=0, crop_start_step=step)
    dataset = ArchivePatchDataset(samples, config)
    assert len(dataset) == sum(int((s.height / 64) * (s.width / 64)) for s in samples) * 8
    expected = np.sqrt([(s.height - 64) * (s.width - 64) for s in samples])
    np.testing.assert_allclose(dataset.probabilities, expected / expected.sum())
    for epoch in (0, 1):
        dataset.epoch = epoch
        for index in (0, 2, 7):
            image, target = dataset[index]
            assert image.shape == (3, 64, 64)
            recovered = torch.round(
                (image[0] * IMAGENET_STD[0] + IMAGENET_MEAN[0]) * 255 / 40
            ).long()
            torch.testing.assert_close(recovered, target)
            replay_image, replay_target = dataset[index]
            torch.testing.assert_close(image, replay_image)
            torch.testing.assert_close(target, replay_target)
    dataset.epoch = 0
    first = [dataset[index][1] for index in range(8)]
    dataset.epoch = 1
    assert any(not torch.equal(first[index], dataset[index][1]) for index in range(8))


@pytest.mark.parametrize("step", [1, 16])
@pytest.mark.parametrize("extra_height,extra_width", [(35, 49), (16, 17), (1, 1)])
def test_crop_start_grid_keeps_exact_rgb_label_coordinates_and_exclusive_bounds(
    tmp_path, step, extra_height, extra_width
):
    side = 64
    sample = cached_sample(tmp_path, "A_1", side + extra_height, side + extra_width)
    rows, columns = np.indices((sample.height, sample.width))
    labels = (rows + columns) % 5
    # Recover the original coordinates even after rotation/flip and BGR normalization.
    np.save(sample.rgb_cache, np.stack((columns, rows, labels * 40), axis=-1).astype(np.uint8))
    dataset = ArchivePatchDataset(
        [sample], ArchiveConfig(patch_size=side, samples_multiplier=128, crop_start_step=step)
    )
    starts = set()
    for index in range(len(dataset)):
        image, target = dataset[index]
        bgr = torch.round(
            (
                image * image.new_tensor(IMAGENET_STD)[:, None, None]
                + image.new_tensor(IMAGENET_MEAN)[:, None, None]
            )
            * 255
        ).long()
        left, top = int(bgr[2].min()), int(bgr[1].min())
        starts.add((left, top))
        assert image.shape == (3, side, side) and target.shape == (side, side)
        assert 0 <= left < extra_width and 0 <= top < extra_height
        assert left % step == top % step == 0
        torch.testing.assert_close(target, (bgr[2] + bgr[1]) % 5)
        torch.testing.assert_close(target, bgr[0] // 40)
    if step == 16:
        assert starts == {
            (x, y) for x in range(0, extra_width, 16) for y in range(0, extra_height, 16)
        }
    elif min(extra_height, extra_width) > 1:
        assert any(left % 16 or top % 16 for left, top in starts)


def test_archived_input_preserves_bgr_channel_order():
    rgb = np.array([[[255, 128, 0]]], dtype=np.uint8)
    normalized = archive_image_tensor(rgb)[:, 0, 0]
    recovered = normalized * torch.tensor(IMAGENET_STD) + torch.tensor(IMAGENET_MEAN)
    torch.testing.assert_close(recovered, torch.tensor([0, 128 / 255, 1]))


def test_empty_annotations_and_undersized_images_fail_without_an_infinite_loop(tmp_path):
    config = ArchiveConfig(patch_size=64)
    blank = cached_sample(tmp_path / "blank", "blank_1", 65, 65, empty=True)
    with pytest.raises(ValueError, match="No nonzero"):
        ArchivePatchDataset([blank], config)
    small = cached_sample(tmp_path / "small", "small_1", 64, 100)
    with pytest.raises(ValueError, match="dimensions > patch_size"):
        ArchivePatchDataset([small], config)


class PointwiseArchiveModel(nn.Module):
    def forward(self, x):
        values = torch.round((x[:, :1] * IMAGENET_STD[0] + IMAGENET_MEAN[0]) * 255)
        return torch.cat([-torch.square(values - value) for value in range(5)], dim=1)


def test_archived_balanced_tiles_zero_padding_stitching_and_metric_sidecar(tmp_path):
    truth = (np.indices((43, 57)).sum(axis=0) % 5).astype(np.uint8)
    rgb = np.repeat(truth[..., None], 3, axis=2)
    coverage = np.zeros_like(truth)
    positions = list(archive_tile_positions(43, 57, 20))
    assert len(positions) == 9
    assert max(h for _, _, h, _ in positions) - min(h for _, _, h, _ in positions) == 1
    for top, left, h, w in positions:
        coverage[top : top + h, left : left + w] += 1
    np.testing.assert_array_equal(coverage, 1)
    model = PointwiseArchiveModel().train()
    output = tmp_path / "prediction.npy"
    settings = ArchiveConfig(inference_tile_size=20).to_dict()
    prediction = predict_archive_tiled(model, rgb, output, settings, torch.device("cpu"), truth)
    np.testing.assert_array_equal(prediction, truth)
    assert model.training
    report = json.loads(output.with_suffix(".archive_metrics.json").read_text())
    assert report["units"] == 9 and report["mean"]["pjac"] == 1
    assert report["validation_mode"] == "eval"


def test_historical_train_mode_batchnorm_does_not_mutate_the_saved_model(tmp_path):
    model = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(3, 5, 1)).eval()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    settings = ArchiveConfig(validation_mode="archive-train", inference_tile_size=20).to_dict()
    rgb = np.random.default_rng(0).integers(0, 255, (43, 57, 3), dtype=np.uint8)
    predict_archive_tiled(model, rgb, tmp_path / "train_bn.npy", settings, torch.device("cpu"))
    assert not model.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
