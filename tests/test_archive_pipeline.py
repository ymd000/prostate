import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from prostate.__main__ import main
from prostate.config import ArchiveConfig
from prostate.inference import predict_image
from prostate.pipeline import evaluate_run, prepare_run, run_holdout
from prostate.training_evaluation import ARCHIVE_EVALUATION_PROTOCOL, evaluate_training_run


@pytest.fixture
def archive_config(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    for name, color in (("A_1", "blue"), ("B_1", "black"), ("15-1428_1_validation", "red")):
        Image.new("RGB", (69, 67), (180, 80, 160)).save(source / f"{name}.jpg")
        Image.new("RGBA", (69, 67), color).save(source / f"{name}.png")
    return ArchiveConfig(
        data_dir=source,
        cache_dir=tmp_path / "cache",
        run_dir=tmp_path / "run",
        patch_size=64,
        batch_size=2,
        epochs=2,
        samples_multiplier=1,
        inference_tile_size=50,
        num_workers=0,
        cpu_threads=1,
    )


@pytest.mark.parametrize("side,batch", [(256, 32), (512, 8), (768, 4)])
def test_cli_archive_recipe_prepares_without_loading_pretrained_weights(
    archive_config, monkeypatch, capsys, side, batch
):
    monkeypatch.setattr(
        "prostate.__main__.run_holdout", lambda *args: pytest.fail("must not train")
    )
    main(
        [
            "archive-holdout",
            "--prepare-only",
            "--patch-size",
            str(side),
            "--device",
            "cuda",
            "--data-dir",
            str(archive_config.data_dir),
            "--cache-dir",
            str(archive_config.cache_dir),
            "--run-dir",
            str(archive_config.run_dir),
        ]
    )
    settings = json.loads((archive_config.run_dir / "config.json").read_text())
    assert settings["model_name"] == "unet11" and settings["pretrained"]
    assert settings["batch_size"] == batch and settings["epochs"] == 30
    assert settings["optimizer"] == "sgd" and settings["momentum"] == 0.9
    assert settings["learning_rate"] == 0.01 and settings["lr_decay"] == 0.95
    assert settings["weight_decay"] == 0 and settings["gradient_clip"] is None
    assert settings["samples_multiplier"] == 8 and not settings["amp"]
    assert settings["input_color"] == "BGR" and settings["normalization"] == "batch"
    assert settings["patch_sampling"] == "archive_random_v1"
    assert not list(archive_config.run_dir.rglob("checkpoint.pt"))
    stdout = capsys.readouterr().out
    split = json.loads(stdout[stdout.index("{") :])
    assert split["holdout_images"] == ["15-1428_1_validation"]


class TinyArchiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.head = nn.Conv2d(3, 5, 1)

    def forward(self, x):
        return self.head(self.bn(x))


def test_archive_sgd_training_holdout_metrics_resume_and_prediction(archive_config, monkeypatch):
    import prostate.archive_model as architecture
    import prostate.archive_train as training

    initialized = []

    def create(model_name="unet11", num_classes=5, pretrained=True, **kwargs):
        initialized.append(pretrained)
        assert num_classes == 5 and model_name == "unet11"
        return TinyArchiveModel()

    monkeypatch.setattr(architecture, "build_archive_model", create)
    monkeypatch.setattr(training, "build_archive_model", create)
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: pytest.fail("SGD is required"))
    monkeypatch.setattr(
        nn.utils,
        "clip_grad_norm_",
        lambda *a, **k: pytest.fail("No gradient clipping in the archive recipe"),
    )
    result = run_holdout(archive_config)
    assert initialized == [True]
    assert result["complete"] and result["cases_evaluated"] == 1
    assert result["images_evaluated"] == 1 and len(result["primary_case_mean"]["iou"]) == 5
    assert set(result["archive_metrics"]["image_mean"]) == {"dice", "jac", "pdice", "pjac"}
    path = archive_config.run_dir / "fold_0/checkpoint.pt"
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["architecture"] == "archive_unet_v1"
    assert checkpoint["train_case_ids"] == ["A", "B"]
    assert checkpoint["train_image_ids"] == ["A_1", "B_1"]
    assert checkpoint["precision"] == "torch.float32"
    assert checkpoint["optimizer"] == {"name": "SGD", "momentum": 0.9, "weight_decay": 0.0}
    assert checkpoint["patches_per_epoch"] == 2 and checkpoint["batches_per_epoch"] == 1
    assert checkpoint["loss"]["class_counts"]["GP5"] == 0
    assert checkpoint["loss"]["class_weights"] is None
    assert checkpoint["loss"]["implementation"] == "softmax_add_1e-24_log_nll"
    with (path.parent / "archive_history.csv").open() as stream:
        history = list(csv.DictReader(stream))
    assert [float(row["learning_rate"]) for row in history] == pytest.approx([0.01, 0.0095])
    assert [row["epoch"] for row in history] == ["1", "2"]
    assert all(0 <= float(row["pjac"]) <= 1 for row in history)
    modern_history = (path.parent / "history.csv").read_text()
    assert "online_iou_GP5" in modern_history
    records = json.loads((archive_config.run_dir / "holdout_images.json").read_text())
    assert records[0]["archive_metrics"]["units"] == 4
    report = json.loads((archive_config.run_dir / "train_metrics.json").read_text())
    assert report["evaluation_protocol"] == ARCHIVE_EVALUATION_PROTOCOL
    assert report["folds"]["0"]["train_archive_metrics"]["images_evaluated"] == 2
    assert report["folds"]["0"]["holdout_archive_metrics"]["images_evaluated"] == 1
    assert evaluate_run(archive_config.run_dir) == result

    output = archive_config.run_dir / "single.png"
    predict_image(
        path, archive_config.data_dir / "15-1428_1_validation.jpg", output, torch.device("cpu")
    )
    assert initialized[-1] is False  # Loading a checkpoint never downloads pretrained weights.
    with Image.open(output) as image:
        assert image.size == (69, 67) and np.asarray(image).max() < 5
    with pytest.raises(ValueError, match="saved inference_tile_size"):
        predict_image(
            path, archive_config.data_dir / "A_1.jpg", output, torch.device("cpu"), patch_size=256
        )
    with pytest.raises(ValueError, match="already exists"):
        run_holdout(archive_config)
    monkeypatch.setattr(training, "train_archive_fold", lambda *a: pytest.fail("must not retrain"))
    (path.parent / "train_metrics.json").unlink()
    assert run_holdout(archive_config, resume=True) == result
    assert evaluate_training_run(archive_config.run_dir, "cpu")["complete"]
    with pytest.raises(ValueError, match="Experiment differs"):
        prepare_run(replace(archive_config, validation_mode="archive-train"))


@pytest.mark.parametrize(
    "settings",
    [
        {"model_name": "invalid"},
        {"patch_size": 32},
        {"epochs": 0},
        {"batch_size": 0},
        {"momentum": 1},
        {"lr_decay": 0},
        {"samples_multiplier": 0},
        {"max_sampling_attempts": 0},
        {"inference_tile_size": 0},
        {"validation_mode": "invalid"},
    ],
)
def test_invalid_archive_settings_fail_early(settings):
    with pytest.raises(ValueError):
        ArchiveConfig(**settings)
