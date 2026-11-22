from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from prostate.data import IGNORE_INDEX, PatchDataset, decode_annotation, prepare_dataset


def test_color_mapping_matches_legacy_bgr_and_alpha_semantics():
    rgba = np.array(
        [
            [
                [0, 0, 0, 255],
                [255, 0, 0, 255],
                [0, 255, 0, 255],
                [255, 255, 0, 255],
                [0, 0, 255, 255],
                [255, 0, 255, 255],
                [0, 255, 255, 255],
                [255, 255, 255, 255],
                [187, 32, 204, 255],
                [255, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(decode_annotation(rgba), [[1, 4, 3, 3, 2, 1, 2, 0, 0, 0, 0]])


def make_pair(directory: Path, name: str, size=(19, 23)):
    directory.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((*size, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    Image.fromarray(rgb).save(directory / f"{name}.jpg", quality=100, subsampling=0)
    Image.fromarray(rgb).save(directory / f"{name}.png")


def test_pair_cache_and_padded_patch_preserve_alignment(tmp_path):
    source = tmp_path / "source"
    make_pair(source, "caseA_1")
    samples = prepare_dataset(source, tmp_path / "cache")
    sample = samples[0]
    assert sample.case_id == "caseA"
    assert sample.class_counts == (0, 0, 0, 0, 19 * 23)
    assert np.load(sample.rgb_cache, mmap_mode="r").shape == (19, 23, 3)
    image, label = PatchDataset(samples, patch_size=32, seed=1)[0]
    assert tuple(image.shape) == (3, 32, 32)
    assert tuple(label.shape) == (32, 32)
    assert int((label == 4).sum()) == 19 * 23
    assert int((label == IGNORE_INDEX).sum()) == 32 * 32 - 19 * 23
    assert image[0].min() > 0.9


def test_missing_pair_and_geometry_mismatch_are_errors(tmp_path):
    make_pair(tmp_path, "caseA_1")
    (tmp_path / "caseA_1.png").unlink()
    with pytest.raises(ValueError, match="Unpaired"):
        prepare_dataset(tmp_path, tmp_path / "cache")
    Image.new("RGB", (8, 8)).save(tmp_path / "caseA_1.png")
    with pytest.raises(ValueError, match="size mismatch"):
        prepare_dataset(tmp_path, tmp_path / "cache")


def test_cache_fingerprints_do_not_overwrite_previous_labels(tmp_path):
    source = tmp_path / "source"
    make_pair(source, "caseA_1")
    first = prepare_dataset(source, tmp_path / "cache")[0]
    Image.new("RGBA", (23, 19), (0, 0, 255, 255)).save(source / "caseA_1.png")
    second = prepare_dataset(source, tmp_path / "cache")[0]
    assert first.label_cache != second.label_cache
    assert np.all(np.load(first.label_cache) == 4)
    assert np.all(np.load(second.label_cache) == 2)
