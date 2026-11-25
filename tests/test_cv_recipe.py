import csv
import json
from dataclasses import replace

import pytest
import torch
from PIL import Image
from torch import nn

from prostate.__main__ import main
from prostate.config import ArchiveConfig, Config
from prostate.folds import read_assignments
from prostate.pipeline import prepare_run, run_cv


@pytest.fixture
def cv_config(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    names = ["A_1", "A_2", "B_1", "C_1", "D_1", "15-1428_1_validation"]
    for index, name in enumerate(names):
        Image.new("RGB", (69, 67), (180, 80, 160)).save(source / f"{name}.jpg")
        color = ["white", "black", "blue", "green", "red"][index % 5]
        Image.new("RGBA", (69, 67), color).save(source / f"{name}.png")
    return ArchiveConfig(
        data_dir=source,
        cache_dir=tmp_path / "cache",
        run_dir=tmp_path / "run",
        holdout_case_id=None,
        patch_size=64,
        batch_size=2,
        epochs=2,
        samples_multiplier=1,
        inference_tile_size=50,
        device="cpu",
        num_workers=0,
        cpu_threads=1,
    )


def command(config):
    args = ["cv", "--recipe", "archive"]
    for name in (
        "data_dir",
        "cache_dir",
        "run_dir",
        "patch_size",
        "batch_size",
        "epochs",
        "samples_multiplier",
        "inference_tile_size",
        "device",
        "num_workers",
        "cpu_threads",
    ):
        args.extend([f"--{name.replace('_', '-')}", str(getattr(config, name))])
    return args


@pytest.mark.parametrize("side,batch", [(256, 32), (512, 8), (768, 4)])
def test_cv_recipe_defaults_match_holdout_conditions(monkeypatch, capsys, side, batch):
    captured = []

    def capture(config, folds, resume):
        captured.append(config)
        assert folds == [2] and resume
        return {
            "complete": False,
            "images_evaluated": 0,
            "cases_evaluated": 0,
            "primary_case_mean": {},
        }

    monkeypatch.setattr("prostate.__main__.run_cv", capture)
    main(["cv", "--patch-size", str(side), "--fold", "2", "--resume", "--recipe", "archive"])
    assert not json.loads(capsys.readouterr().out)["complete"]
    config = captured[0]
    assert config.holdout_case_id is None and config.n_splits == 5
    assert config.batch_size == batch
    settings = config.to_dict()
    settings.pop("n_splits")
    expected = ArchiveConfig(patch_size=side, run_dir=config.run_dir).to_dict()
    expected.pop("holdout_case_id")
    assert settings == expected


def test_cv_recipe_preserves_explicit_overrides(monkeypatch, capsys):
    captured = []
    monkeypatch.setattr(
        "prostate.__main__.run_cv",
        lambda config, *a: (
            captured.append(config)
            or {
                "complete": False,
                "images_evaluated": 0,
                "cases_evaluated": 0,
                "primary_case_mean": {},
            }
        ),
    )
    main(
        [
            "cv",
            "--recipe",
            "archive",
            "--model",
            "unet16",
            "--batch-size",
            "8",
            "--learning-rate",
            "0.02",
            "--momentum",
            "0.8",
            "--lr-decay",
            "0.9",
            "--epochs",
            "7",
            "--n-splits",
            "3",
            "--no-pretrained",
        ]
    )
    capsys.readouterr()
    config = captured[0]
    assert config.model_name == "unet16" and not config.pretrained
    assert config.batch_size == 8 and config.epochs == 7 and config.n_splits == 3
    assert (config.learning_rate, config.momentum, config.lr_decay) == (0.02, 0.8, 0.9)


def test_cv_recipe_prepare_uses_the_same_case_folds_as_current_cv(cv_config, monkeypatch):
    monkeypatch.setattr("prostate.__main__.run_cv", lambda *a: pytest.fail("must not train"))
    main(command(cv_config) + ["--prepare-only", "--device", "cuda"])
    settings = json.loads((cv_config.run_dir / "config.json").read_text())
    assert "holdout_case_id" not in settings and settings["n_splits"] == 5
    assert not (cv_config.run_dir / "split.csv").exists()
    current = Config(
        data_dir=cv_config.data_dir,
        cache_dir=cv_config.cache_dir,
        run_dir=cv_config.run_dir.parent / "current",
        seed=cv_config.seed,
    )
    samples, expected = prepare_run(current)
    assert read_assignments(cv_config.run_dir, settings, samples) == expected
    assert (current.run_dir / "folds.csv").read_bytes() == (
        cv_config.run_dir / "folds.csv"
    ).read_bytes()
    assert "15-1428_1_validation" in expected
    with pytest.raises(ValueError, match="Experiment differs"):
        prepare_run(replace(cv_config, batch_size=8, device="cuda"))


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.head = nn.Conv2d(3, 5, 1)

    def forward(self, inputs):
        return self.head(self.bn(inputs))


def test_cv_recipe_trains_independent_folds_and_saves_oof_and_training_metrics(
    cv_config,
    monkeypatch,
    capsys,
):
    initialized = []

    def create(model_name="unet11", num_classes=5, pretrained=True, **kwargs):
        assert model_name == "unet11" and num_classes == 5
        model = TinyUNet()
        initialized.append((pretrained, model.head.weight.detach().clone()))
        return model

    monkeypatch.setattr("prostate.archive_train.build_archive_model", create)
    monkeypatch.setattr("prostate.archive_model.build_archive_model", create)
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: pytest.fail("SGD is required"))
    main(command(cv_config) + ["--fold", "0"])
    partial = json.loads((cv_config.run_dir / "oof_metrics.json").read_text())
    assert not partial["complete"] and not partial["archive_metrics"]["complete"]
    main(command(cv_config) + ["--resume"])
    capsys.readouterr()
    summary = json.loads((cv_config.run_dir / "oof_metrics.json").read_text())
    assert summary["complete"] and summary["cases_evaluated"] == 5
    assert summary["images_evaluated"] == 6 and summary["archive_metrics"]["complete"]
    assert set(summary["fold_case_means"]) == {"0", "1", "2", "3", "4"}
    assert set(summary["archive_metrics"]["fold_image_means"]) == set(summary["fold_case_means"])
    assert [pretrained for pretrained, _ in initialized] == [True] * 5
    assert all(not torch.equal(initialized[0][1], weights) for _, weights in initialized[1:])
    records = json.loads((cv_config.run_dir / "oof_images.json").read_text())
    for fold in range(5):
        folder = cv_config.run_dir / f"fold_{fold}"
        checkpoint = torch.load(folder / "checkpoint.pt", weights_only=True)
        held_out = [r for r in records if r["fold"] == fold]
        assert not set(checkpoint["train_case_ids"]) & {r["case_id"] for r in held_out}
        assert checkpoint["config"] == cv_config.to_dict()
        assert checkpoint["precision"] == "torch.float32"
        assert checkpoint["optimizer"] == {"name": "SGD", "momentum": 0.9, "weight_decay": 0.0}
        assert checkpoint["loss"]["class_weights"] is None
        assert checkpoint["patches_per_epoch"] == 6 - len(held_out)
        with (folder / "archive_history.csv").open() as stream:
            history = list(csv.DictReader(stream))
        assert [float(r["learning_rate"]) for r in history] == pytest.approx([0.01, 0.0095])
        training = json.loads((folder / "train_metrics.json").read_text())
        assert training["complete"] and training["archive_metrics"]["images_evaluated"] == (
            6 - len(held_out)
        )
        assert training["inference_tile_size"] == cv_config.inference_tile_size
        for record in held_out:
            assert record["archive_metrics"]["inference_tile_size"] == cv_config.inference_tile_size
            with Image.open(cv_config.run_dir / "oof" / f"{record['image_id']}.png") as image:
                assert image.size == (69, 67)
    comparison = json.loads((cv_config.run_dir / "train_metrics.json").read_text())
    assert comparison["complete"] and comparison["evaluation_split"] == "oof"
    assert all(fold["oof_archive_metrics"]["complete"] for fold in comparison["folds"].values())
    monkeypatch.setattr(
        "prostate.archive_train.train_archive_fold", lambda *a: pytest.fail("must not retrain")
    )
    assert run_cv(cv_config, resume=True) == summary
    assert len(initialized) == 5
    path = cv_config.run_dir / "oof" / f"{records[0]['image_id']}.archive_metrics.json"
    path.unlink()
    main(["evaluate", "--run-dir", str(cv_config.run_dir), "--device", "cpu"])
    assert initialized[-1][0] is False and len(initialized) == 6
    assert path.exists()
    assert json.loads((cv_config.run_dir / "oof_metrics.json").read_text()) == summary


@pytest.mark.parametrize(
    "args",
    [
        ["--encoder", "gigapath-flash"],
        ["--class-weighting", "inverse-sqrt"],
        ["--normalization", "group"],
        ["--n-splits", "1"],
        ["--holdout-case-id", "15-1428"],
    ],
)
def test_cv_recipe_rejects_incompatible_options(args, capsys):
    with pytest.raises(SystemExit) as error:
        main(["cv", "--recipe", "archive", *args])
    assert error.value.code == 2
    capsys.readouterr()
