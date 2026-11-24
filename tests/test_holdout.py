import csv
import json
from dataclasses import replace

import h5py
import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from prostate.__main__ import main
from prostate.config import Config
from prostate.data import grid_positions
from prostate.folds import assign_holdout
from prostate.inference import predict_image
from prostate.pipeline import evaluate_run, prepare_run, run_cv, run_holdout
from prostate.training_evaluation import evaluate_training_run


@pytest.fixture
def holdout_config(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    for image_id, color in (
        ("trainA_1", "blue"),
        ("trainA_2", "blue"),
        ("trainB_1", "black"),
        ("15-1428_1_validation", "red"),
        ("15-1428_2", "red"),
    ):
        Image.new("RGB", (21, 19), (180, 80, 160)).save(source / f"{image_id}.jpg")
        Image.new("RGB", (21, 19), color).save(source / f"{image_id}.png")
    return Config(
        data_dir=source,
        cache_dir=tmp_path / "cache",
        token_dir=tmp_path / "tokens",
        run_dir=tmp_path / "holdout",
        holdout_case_id="15-1428",
        epochs=1,
        patch_size=32,
        halo=4,
        base_channels=2,
        class_weighting="inverse-sqrt",
        device="cpu",
        cpu_threads=1,
    )


def test_holdout_groups_all_images_of_the_case_and_rejects_empty_splits(holdout_config):
    samples, assignments = prepare_run(holdout_config)
    assert {s.image_id for s in samples if assignments[s.image_id] == 0} == {
        "15-1428_1_validation",
        "15-1428_2",
    }
    assert set(assignments.values()) == {-1, 0}
    assert assign_holdout(list(reversed(samples)), "15-1428") == assignments
    with pytest.raises(ValueError, match="not found"):
        assign_holdout(samples, "absent")
    with pytest.raises(ValueError, match="training case"):
        assign_holdout([s for s in samples if s.case_id == "15-1428"], "15-1428")
    with (holdout_config.run_dir / "split.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["split"] for row in rows} == {"train", "holdout"}
    assert not (holdout_config.run_dir / "folds.csv").exists()
    settings = json.loads((holdout_config.run_dir / "config.json").read_text())
    assert settings["holdout_case_id"] == "15-1428"
    assert "n_splits" not in settings
    assert "holdout_case_id" not in Config().to_dict()


def test_cli_prepares_default_validation_case_without_initializing_training(
    holdout_config, monkeypatch, capsys
):
    def fail(*args, **kwargs):
        pytest.fail("Preparing the split must not run training or require a GPU")

    monkeypatch.setattr("prostate.__main__.run_holdout", fail)
    main(
        [
            "holdout",
            "--prepare-only",
            "--device",
            "cuda",
            "--data-dir",
            str(holdout_config.data_dir),
            "--cache-dir",
            str(holdout_config.cache_dir),
            "--run-dir",
            str(holdout_config.run_dir),
        ]
    )
    output = capsys.readouterr().out
    result = json.loads(output[output.index("{") :])
    assert result["holdout_case_id"] == "15-1428"
    assert len(result["train_images"]) == 3
    assert result["holdout_images"] == ["15-1428_1_validation", "15-1428_2"]
    assert not list(holdout_config.run_dir.rglob("checkpoint.pt"))


class FakeFrozenEncoder(nn.Module):
    def forward_features(self, x):
        assert not torch.is_grad_enabled()
        return x.new_zeros((len(x), 1 + (x.shape[-1] // 16) ** 2, 384))


@pytest.mark.parametrize("encoder", ["none", "gigapath-flash"])
def test_holdout_training_predictions_weights_resume_and_diagnostics(
    holdout_config, monkeypatch, encoder
):
    config = replace(holdout_config, encoder=encoder)
    samples, assignments = prepare_run(config)
    training = [sample for sample in samples if assignments[sample.image_id] == -1]
    expected = {"15-1428_1_validation", "15-1428_2"}
    if encoder == "gigapath-flash":
        config.token_dir.mkdir()
        for sample in training:
            positions = list(grid_positions(sample.height, sample.width, config.patch_size))
            with h5py.File(config.token_dir / f"{sample.image_id}.h5", "w") as cache:
                cache.attrs["patch_size"] = config.patch_size
                cache.create_dataset("positions", data=np.array(positions, dtype=np.int32))
                cache.create_dataset(
                    "tokens", data=np.zeros((len(positions), 2, 2, 384), np.float16)
                )
        monkeypatch.setattr(
            "prostate.extractor.load_gigapath_flash",
            lambda device: (FakeFrozenEncoder(), [0, 0, 0], [1, 1, 1]),
        )

    result = run_holdout(config)
    assert result["complete"] and result["split"] == "holdout"
    assert result["images_evaluated"] == 2 and result["cases_evaluated"] == 1
    assert len(result["primary_case_mean"]["iou"]) == 5
    assert "fold_case_means" not in result
    assert {p.stem for p in (config.run_dir / "holdout").glob("*.png")} == expected
    assert not (config.run_dir / "oof").exists()
    assert not (config.run_dir / "oof_metrics.json").exists()

    checkpoint_path = config.run_dir / "fold_0/checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert set(checkpoint["train_image_ids"]) == {s.image_id for s in training}
    assert checkpoint["train_case_ids"] == ["trainA", "trainB"]
    assert checkpoint["config"]["holdout_case_id"] == "15-1428"
    loss = json.loads((checkpoint_path.parent / "loss.json").read_text())
    assert loss["class_counts"]["GP5"] == 0 and loss["class_weights"]["GP5"] == 0
    assert sum(loss["class_counts"].values()) == len(training) * 21 * 19
    assert checkpoint["loss"] == loss
    records = json.loads((config.run_dir / "holdout_images.json").read_text())
    assert {r["image_id"] for r in records} == expected
    report = json.loads((config.run_dir / "train_metrics.json").read_text())
    assert report["complete"] and report["folds_evaluated"] == [0]
    assert report["evaluation_split"] == "holdout"
    assert report["folds"]["0"]["train_images_evaluated"] == 3
    assert "holdout_primary_case_mean" in report["folds"]["0"]
    assert "oof_primary_case_mean" not in report["folds"]["0"]
    assert evaluate_run(config.run_dir) == result

    output = config.run_dir / "single.png"
    predict_image(checkpoint_path, config.data_dir / "15-1428_2.jpg", output, torch.device("cpu"))
    with Image.open(output) as image:
        assert image.size == (21, 19)

    def fail(*args, **kwargs):
        pytest.fail("A completed holdout must not be retrained")

    monkeypatch.setattr("prostate.pipeline.train_fold", fail)
    monkeypatch.setattr("prostate.pipeline.train_token_fold", fail)
    with pytest.raises(ValueError, match="already exists"):
        run_holdout(config)
    (checkpoint_path.parent / "train_metrics.json").unlink()
    assert run_holdout(config, resume=True) == result
    (checkpoint_path.parent / "train_metrics.json").unlink()
    assert evaluate_training_run(config.run_dir, "cpu")["complete"]
    with pytest.raises(ValueError, match="valid fold IDs"):
        evaluate_training_run(config.run_dir, "cpu", selected_folds=[1])
    checkpoint["train_case_ids"].append("15-1428")
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="Checkpoint training split"):
        evaluate_run(config.run_dir)


def test_changed_holdout_case_and_cv_directory_reuse_are_rejected(holdout_config):
    prepare_run(holdout_config)
    with pytest.raises(ValueError, match="Experiment differs"):
        prepare_run(replace(holdout_config, holdout_case_id="trainA"))
    with pytest.raises(ValueError, match="split differs"):
        prepare_run(replace(holdout_config, holdout_case_id=None, n_splits=2))
    with pytest.raises(ValueError, match="run_holdout"):
        run_cv(holdout_config)
    with pytest.raises(ValueError, match="case ID is required"):
        run_holdout(replace(holdout_config, holdout_case_id=None))


def test_tampered_holdout_split_is_rejected_before_training_diagnostics(holdout_config):
    config = holdout_config
    prepare_run(config)
    split = config.run_dir / "split.csv"
    split.write_text(
        split.read_text().replace("15-1428_2,15-1428,holdout", "15-1428_2,15-1428,train")
    )
    with pytest.raises(ValueError, match="configured case"):
        evaluate_training_run(config.run_dir, "cpu")


@pytest.mark.parametrize("case_id", ["", " ", "15-1428_5_2_validation"])
def test_holdout_config_requires_a_case_id(case_id):
    with pytest.raises(ValueError, match="case ID"):
        Config(holdout_case_id=case_id)
