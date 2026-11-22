import json

import numpy as np
import pytest
from PIL import Image
from PIL.JpegImagePlugin import get_sampling

from prostate.__main__ import main
from prostate.data import CLASS_NAMES
from prostate.thumbnails import (
    DISPLAY_COLORS,
    _error_density,
    _mask_preview,
    _overlay,
    generate_thumbnails,
)


def make_run(tmp_path):
    run_dir = tmp_path / "experiment"
    (run_dir / "oof").mkdir(parents=True)
    samples, records = [], []
    for index, score in enumerate((0.9, 0.1, None)):
        image_id = f"case{index}_1"
        source = tmp_path / f"{image_id}.jpg"
        Image.new("RGB", (21, 11), (180, 80, 150)).save(source)
        target = np.tile(np.arange(21, dtype=np.uint8) % 5, (11, 1))
        prediction = target.copy()
        prediction[0, 0] = 4
        target_path = tmp_path / f"{image_id}.npy"
        np.save(target_path, target)
        np.save(run_dir / "oof" / f"{image_id}.npy", prediction)
        samples.append(
            {
                "image_id": image_id,
                "case_id": f"case{index}",
                "image_path": str(source),
                "label_cache": str(target_path),
                "width": 21,
                "height": 11,
            }
        )
        records.append(
            {
                "image_id": image_id,
                "case_id": f"case{index}",
                "fold": index,
                "foreground_macro_dice": score,
            }
        )
    (run_dir / "manifest.json").write_text(json.dumps({"classes": CLASS_NAMES, "samples": samples}))
    (run_dir / "oof_images.json").write_text(json.dumps(records))
    return run_dir


def test_display_keeps_background_distinct_from_non_cancer_and_preserves_labels():
    labels = np.array([[0, 1, 2, 3, 4]], dtype=np.uint8)
    mask = _mask_preview(labels, (5, 1))
    np.testing.assert_array_equal(np.asarray(mask)[0], DISPLAY_COLORS)
    rgb = Image.new("RGB", (5, 1), (100, 80, 60))
    overlay = np.asarray(_overlay(rgb, mask, labels))
    np.testing.assert_array_equal(overlay[0, 0], [100, 80, 60])
    assert not np.array_equal(overlay[0, 1], [100, 80, 60])


def test_error_density_counts_full_resolution_mistakes_before_downsampling():
    target = np.zeros((2, 4), dtype=np.uint8)
    prediction = target.copy()
    prediction[:, :2] = [[1, 0], [0, 0]]  # One mistake out of four pixels in the first bin.
    image = np.asarray(_error_density(target, prediction, (2, 1)))
    np.testing.assert_array_equal(image, [[[255, 191, 191], [255, 255, 255]]])
    prediction[:] = 4
    np.testing.assert_array_equal(
        np.asarray(_error_density(target, prediction, (1, 1))), [[[255, 0, 0]]]
    )
    prediction[0, 0] = 5
    with pytest.raises(ValueError, match="0..4"):
        _error_density(target, prediction, (1, 1))


def test_export_sorts_paginates_and_leaves_original_artifacts_unchanged(tmp_path):
    run_dir = make_run(tmp_path)
    paths = list(run_dir.rglob("*")) + list(tmp_path.glob("*.jpg"))
    original = {path: path.read_bytes() for path in paths if path.is_file()}
    result = generate_thumbnails(run_dir, max_side=128, per_page=2)
    assert result["images_generated"] == 3
    assert result["contact_sheets"] == ["sheet_001.jpg", "sheet_002.jpg"]
    output = run_dir / "thumbnails"
    index = json.loads((output / "index.json").read_text())
    assert [item["image_id"] for item in index["images"]] == ["case1_1", "case0_1", "case2_1"]
    assert index["format"] == "JPEG"
    assert index["jpeg_quality"] == 95
    assert index["chroma_subsampling"] == "4:4:4"
    for item in index["images"]:
        with Image.open(output / item["file"]) as image:
            assert image.format == "JPEG"
            assert image.mode == "RGB"
            assert image.size == (440, 204)
            assert get_sampling(image) == 0
    for filename in index["contact_sheets"]:
        with Image.open(output / filename) as image:
            assert image.format == "JPEG"
            assert get_sampling(image) == 0
    for path, content in original.items():
        assert path.read_bytes() == content


def test_cli_filter_limit_and_custom_output(tmp_path, capsys):
    run_dir = make_run(tmp_path)
    destination = tmp_path / "custom"
    main(
        [
            "thumbnails",
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(destination),
            "--image-id",
            "case0_1",
            "--image-id",
            "case1_1",
            "--sort",
            "best",
            "--limit",
            "1",
            "--max-side",
            "256",
            "--contact-side",
            "160",
            "--jpeg-quality",
            "90",
        ]
    )
    assert json.loads(capsys.readouterr().out)["images_generated"] == 1
    index = json.loads((destination / "index.json").read_text())
    assert index["images"][0]["image_id"] == "case0_1"
    assert index["contact_side"] == 160
    assert index["jpeg_quality"] == 90


def test_missing_predictions_and_mismatched_shapes_fail_clearly(tmp_path):
    run_dir = make_run(tmp_path)
    with pytest.raises(ValueError, match="No saved OOF"):
        generate_thumbnails(run_dir, image_ids=["unknown"])
    prediction = run_dir / "oof" / "case0_1.npy"
    np.save(prediction, np.zeros((3, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="geometry mismatch"):
        generate_thumbnails(run_dir, image_ids=["case0_1"])
    prediction.unlink()
    with pytest.raises(FileNotFoundError, match="Missing OOF"):
        generate_thumbnails(run_dir, image_ids=["case0_1"])


@pytest.mark.parametrize(
    "settings",
    [
        {"max_side": 0},
        {"per_page": 0},
        {"limit": 0},
        {"sort": "bad"},
        {"contact_side": 0},
        {"jpeg_quality": 0},
        {"jpeg_quality": 101},
    ],
)
def test_invalid_thumbnail_options_fail_early(tmp_path, settings):
    with pytest.raises(ValueError):
        generate_thumbnails(tmp_path, **settings)
