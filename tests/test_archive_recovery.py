import json
from dataclasses import replace

import pytest
import torch
from PIL import Image
from torch import nn

from prostate.__main__ import main
from prostate.config import ArchiveConfig, ArchiveEvaluationConfig
from prostate.pipeline import evaluate_archive_checkpoint, prepare_run, run_holdout


@pytest.fixture
def failed_archive_run(tmp_path, monkeypatch):
    import prostate.archive_model as architecture
    import prostate.archive_train as training

    source = tmp_path / "data"
    source.mkdir()
    for image_id, color in (("A_1", "blue"), ("B_1", "black"), ("15-1428_1_validation", "red")):
        Image.new("RGB", (69, 67), (180, 80, 160)).save(source / f"{image_id}.jpg")
        Image.new("RGBA", (69, 67), color).save(source / f"{image_id}.png")
    config = ArchiveConfig(
        data_dir=source,
        cache_dir=tmp_path / "cache",
        run_dir=tmp_path / "run",
        patch_size=64,
        batch_size=2,
        epochs=1,
        samples_multiplier=1,
        inference_tile_size=3000,
        num_workers=0,
        cpu_threads=1,
        pretrained=False,
        device="cpu",
    )

    class LimitedMemoryModel(nn.Module):
        fail_large = True

        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(3)
            self.head = nn.Conv2d(3, 5, 1)

        def forward(self, x):
            if self.fail_large and x.shape[-1] > 64:
                raise torch.OutOfMemoryError("Simulated large inference tile OOM")
            return self.head(self.bn(x))

    def create(*args, pretrained=True, **kwargs):
        assert not pretrained
        return LimitedMemoryModel()

    monkeypatch.setattr(architecture, "build_archive_model", create)
    monkeypatch.setattr(training, "build_archive_model", create)
    with pytest.raises(torch.OutOfMemoryError, match="large inference tile"):
        run_holdout(config)
    assert (config.run_dir / "fold_0/checkpoint.pt").is_file()
    assert not (config.run_dir / "fold_0/evaluation.json").exists()

    def fail(*args, **kwargs):
        pytest.fail("Training must never run after a final checkpoint has been saved")

    monkeypatch.setattr(training, "train_archive_fold", fail)
    monkeypatch.setattr(torch.optim, "SGD", fail)
    originals = {
        path: path.read_bytes()
        for path in config.run_dir.rglob("*")
        if path.is_file() and ".partial." not in path.name
    }
    return config, originals, LimitedMemoryModel


def test_smaller_tile_recovery_preserves_training_and_uses_one_protocol_for_both_splits(
    failed_archive_run, tmp_path
):
    config, originals, _ = failed_archive_run
    output = tmp_path / "evaluation"
    options = ArchiveEvaluationConfig(
        run_dir=config.run_dir, device="cpu", inference_tile_size=32, output_dir=output
    )
    result = evaluate_archive_checkpoint(options)
    assert result["complete"] and result["images_evaluated"] == 1
    assert result["cases_evaluated"] == 1 and result["output_dir"] == str(output)
    assert result["inference_settings"]["inference_tile_size"] == 32
    assert set(result["primary_case_mean"]["iou"]) == {
        "non_gland",
        "non_cancer",
        "GP3",
        "GP4",
        "GP5",
    }
    assert "pjac" in result["archive_metrics"]["image_mean"]
    metadata = json.loads((output / "evaluation_config.json").read_text())
    assert metadata["training_config"]["inference_tile_size"] == 3000
    assert metadata["training_config"]["batch_size"] == 2
    assert metadata["inference_settings"]["inference_tile_size"] == 32
    training = json.loads((output / "fold_0/train_metrics.json").read_text())
    assert training["inference_tile_size"] == 32 and training["images_evaluated"] == 2
    assert training["validation_mode"] == result["inference_settings"]["validation_mode"]
    comparison = json.loads((output / "train_metrics.json").read_text())
    assert comparison["complete"] and comparison["folds_evaluated"] == [0]
    assert comparison["folds"]["0"]["holdout_primary_case_mean"] == result["primary_case_mean"]
    for path, before in originals.items():
        assert path.read_bytes() == before
    with pytest.raises(ValueError, match="settings/checkpoint differ"):
        evaluate_archive_checkpoint(replace(options, inference_tile_size=16))
    with pytest.raises(ValueError, match="separate from the source"):
        evaluate_archive_checkpoint(replace(options, output_dir=config.run_dir))
    protected = tmp_path / "existing-files"
    protected.mkdir()
    (protected / "keep.txt").write_text("keep this file")
    with pytest.raises(ValueError, match="not empty"):
        evaluate_archive_checkpoint(replace(options, output_dir=protected))
    assert (protected / "keep.txt").read_text() == "keep this file"


def test_cli_recovery_uses_saved_settings_and_default_separate_output(failed_archive_run, capsys):
    config, originals, _ = failed_archive_run
    main(
        [
            "archive-evaluate",
            "--run-dir",
            str(config.run_dir),
            "--device",
            "cpu",
            "--inference-tile-size",
            "32",
        ]
    )
    stdout = capsys.readouterr().out
    result = json.loads(stdout[stdout.index("{") :])
    assert result["complete"] and result["images_evaluated"] == 1
    destination = config.run_dir / "evaluation-tile32-eval-cpu"
    assert result["output_dir"] == str(destination)
    assert (destination / "holdout/15-1428_1_validation.png").is_file()
    assert (destination / "train_metrics.json").is_file()
    for path, before in originals.items():
        assert path.read_bytes() == before


def test_resume_after_inference_oom_loads_final_checkpoint_without_training(failed_archive_run):
    config, originals, model_type = failed_archive_run
    with pytest.raises(ValueError, match="Final checkpoint already exists"):
        run_holdout(config)
    # Simulate retrying on a device with enough memory for the saved tile size.
    model_type.fail_large = False
    assert run_holdout(config, resume=True)["complete"]
    for path, before in originals.items():
        assert path.read_bytes() == before


@pytest.mark.parametrize("problem", ["train_case_ids", "config"])
def test_recovery_rejects_wrong_training_metadata_before_loading_the_model(
    failed_archive_run, monkeypatch, tmp_path, problem
):
    config, _, _ = failed_archive_run
    path = config.run_dir / "fold_0/checkpoint.pt"
    checkpoint = torch.load(path, weights_only=True)
    if problem == "train_case_ids":
        checkpoint["train_case_ids"].append("15-1428")
    else:
        checkpoint["config"]["batch_size"] = 999
    torch.save(checkpoint, path)
    monkeypatch.setattr(
        "prostate.archive_evaluation.load_checkpoint_model",
        lambda *a: pytest.fail("must validate first"),
    )
    with pytest.raises(ValueError, match="Checkpoint"):
        evaluate_archive_checkpoint(
            ArchiveEvaluationConfig(
                run_dir=config.run_dir, inference_tile_size=32, output_dir=tmp_path / "evaluation"
            )
        )
    assert not (tmp_path / "evaluation").exists()


def test_missing_final_checkpoint_cannot_trigger_training(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    for name in ("A_1", "15-1428_1_validation"):
        Image.new("RGB", (69, 67), "white").save(data / f"{name}.jpg")
        Image.new("RGBA", (69, 67), "blue").save(data / f"{name}.png")
    config = ArchiveConfig(data_dir=data, cache_dir=tmp_path / "cache", run_dir=tmp_path / "run")
    prepare_run(config)
    monkeypatch.setattr(
        "prostate.archive_evaluation.load_checkpoint_model", lambda *a: pytest.fail("no checkpoint")
    )
    with pytest.raises(FileNotFoundError, match="No final checkpoint"):
        evaluate_archive_checkpoint(ArchiveEvaluationConfig(run_dir=config.run_dir))


@pytest.mark.parametrize(
    "settings", [{"inference_tile_size": 0}, {"device": "invalid"}, {"validation_mode": "invalid"}]
)
def test_invalid_recovery_settings_fail_early(tmp_path, settings):
    with pytest.raises(ValueError):
        ArchiveEvaluationConfig(run_dir=tmp_path, **settings)
