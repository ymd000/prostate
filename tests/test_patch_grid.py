from collections import Counter

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from prostate.config import Config
from prostate.data import IGNORE_INDEX, PatchDataset, Sample, grid_positions
from prostate.token_data import TokenPatchDataset
from prostate.train import training_loader


@pytest.fixture
def grid_samples(tmp_path):
    samples = []
    token_dir = tmp_path / "tokens"
    token_dir.mkdir()
    for number, (image_id, height, width) in enumerate(
        [("caseA_1", 35, 70), ("caseA_2", 9, 17), ("caseB_1", 32, 32), ("heldout_1", 33, 33)]
    ):
        labels = (np.indices((height, width)).sum(axis=0) % 5).astype(np.uint8)
        rgb = np.repeat((labels * 40 + 5)[..., None], 3, axis=2)
        rgb_path, label_path = tmp_path / f"{image_id}_rgb.npy", tmp_path / f"{image_id}_labels.npy"
        np.save(rgb_path, rgb)
        np.save(label_path, labels)
        samples.append(
            Sample(
                image_id,
                image_id.split("_")[0],
                "unused.jpg",
                "unused.png",
                str(rgb_path),
                str(label_path),
                width,
                height,
                tuple(np.bincount(labels.ravel(), minlength=5).tolist()),
                {},
            )
        )
        positions = list(grid_positions(height, width, 32))
        tokens = np.stack(
            [
                np.full((2, 2, 384), 100 * (number + 1) + index, dtype=np.float16)
                for index in range(len(positions))
            ]
        )
        with h5py.File(token_dir / f"{image_id}.h5", "w") as cache:
            cache.attrs["patch_size"] = 32
            cache.create_dataset("positions", data=np.array(positions, dtype=np.int32))
            cache.create_dataset("tokens", data=tokens)
    return samples, token_dir


def test_rgb_grid_covers_every_training_pixel_once_and_keeps_augmented_labels_aligned(grid_samples):
    samples, _ = grid_samples
    training = samples[:-1]
    dataset = PatchDataset(training, patch_size=32, seed=42)
    assert len(dataset) == 8  # Six tiles + one small image + one exact-size image.
    coverage = {
        sample.image_id: np.zeros((sample.height, sample.width), dtype=np.uint8)
        for sample in training
    }
    for index, (sample, top, left) in enumerate(dataset.patches):
        assert sample.case_id != "heldout"
        coverage[sample.image_id][top : top + 32, left : left + 32] += 1
        for epoch in (0, 1):
            dataset.epoch = epoch
            image, target = dataset[index]
            valid = target != IGNORE_INDEX
            assert valid.sum() == min(32, sample.height - top) * min(32, sample.width - left)
            decoded = torch.round((image[0] * 255 - 5) / 40).long()
            torch.testing.assert_close(decoded[valid], target[valid])
    for counts in coverage.values():
        np.testing.assert_array_equal(counts, 1)


def test_token_grid_visits_every_cached_patch_and_pairs_correct_labels(grid_samples):
    samples, token_dir = grid_samples
    dataset = TokenPatchDataset(samples[:-1], token_dir, patch_size=32, seed=42)
    assert len(dataset) == 8
    expected_ids = []
    for index, (sample, _, patch_index, top, left) in enumerate(dataset.patches):
        assert sample.case_id != "heldout"
        tokens, target = dataset[index]
        number = samples.index(sample)
        expected_id = 100 * (number + 1) + patch_index
        expected_ids.append(expected_id)
        assert tokens.shape == (384, 2, 2)
        assert tokens.dtype == torch.float32
        assert torch.all(tokens == expected_id)
        truth = np.load(sample.label_cache)[top : top + 32, left : left + 32]
        valid = target[target != IGNORE_INDEX].numpy()
        np.testing.assert_array_equal(
            np.bincount(valid, minlength=5), np.bincount(truth.ravel(), minlength=5)
        )
        assert target.dtype == torch.int64

    loader = training_loader(dataset, Config(batch_size=3), fold=0, device=torch.device("cpu"))
    for epoch in (0, 1):
        dataset.epoch = epoch
        seen = [int(value) for tokens, _ in loader for value in tokens[:, 0, 0, 0]]
        assert Counter(seen) == Counter(expected_ids)


def test_loader_shuffles_each_epoch_reproducibly_and_keeps_incomplete_last_batch():
    dataset = TensorDataset(torch.arange(7))
    config = Config(seed=23, batch_size=3)

    def orders():
        loader = training_loader(dataset, config, fold=3, device=torch.device("cpu"))
        result = []
        for _ in range(2):
            batches = [batch[0].tolist() for batch in loader]
            assert [len(batch) for batch in batches] == [3, 3, 1]
            values = [value for batch in batches for value in batch]
            assert sorted(values) == list(range(7))
            result.append(values)
        return result

    first, replay = orders(), orders()
    assert first == replay
    assert first[0] != first[1]


@pytest.mark.parametrize(
    "problem", ["duplicate", "missing_position", "missing_token", "patch_size"]
)
def test_incompatible_or_incomplete_token_cache_is_rejected(grid_samples, problem):
    samples, token_dir = grid_samples
    with h5py.File(token_dir / f"{samples[0].image_id}.h5", "r+") as cache:
        if problem == "patch_size":
            cache.attrs["patch_size"] = 256
        elif problem == "duplicate":
            cache["positions"][1] = cache["positions"][0]
        else:
            name = "positions" if problem == "missing_position" else "tokens"
            values = cache[name][:-1]
            del cache[name]
            cache.create_dataset(name, data=values)
    with pytest.raises(ValueError, match="patch_size mismatch|complete patch grid"):
        TokenPatchDataset(samples[:-1], token_dir, patch_size=32, seed=42)


def test_empty_training_sets_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="at least one case"):
        PatchDataset([], patch_size=32, seed=42)
    with pytest.raises(ValueError, match="at least one case"):
        TokenPatchDataset([], tmp_path, patch_size=32, seed=42)
