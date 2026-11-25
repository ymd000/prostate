import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from prostate.__main__ import main
from prostate.config import Config
from prostate.inference import save_label_png
from prostate.metrics import confusion_matrix
from prostate.model import TokenDecoder, UNet
from prostate.pipeline import evaluate_run, prepare_run, run_cv, write_json
from prostate.training_evaluation import (
    EVALUATION_PROTOCOL,
    collect_training_reports,
    evaluate_training_run,
    training_report_current,
)


def save_constant_model(samples, config, fold, architecture="unet_groupnorm_v2", settings=None):
    """A fixed predictor, constructed without an optimizer or any training step."""
    normalization = "none" if architecture == "unet_v1" else "group"
    model_type = TokenDecoder if architecture == "token_decoder_v1" else UNet
    model = model_type(base_channels=2, normalization=normalization)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head.bias[4] = 1
    model_settings = {"num_classes": 5, "base_channels": 2}
    if architecture != "unet_v1":
        model_settings["normalization"] = normalization
    checkpoint = {
        "architecture": architecture,
        "model_state": model.state_dict(),
        "model": model_settings,
        "config": settings or config.to_dict(),
        "fold": fold,
        "train_image_ids": [sample.image_id for sample in samples],
        "train_case_ids": sorted({sample.case_id for sample in samples}),
    }
    path = config.run_dir / f"fold_{fold}/checkpoint.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return model


def make_run(tmp_path, architecture="unet_groupnorm_v2", legacy_config=True):
    torch.set_num_threads(1)
    source = tmp_path / "data"
    source.mkdir()
    for image_id, color in (("caseA_1", "red"), ("caseA_2", "black"), ("caseB_1", "blue")):
        Image.new("RGB", (21, 19), (180, 80, 160)).save(source / f"{image_id}.jpg")
        Image.new("RGB", (21, 19), color).save(source / f"{image_id}.png")
    config = Config(
        data_dir=source,
        cache_dir=tmp_path / "cache",
        run_dir=tmp_path / "run",
        n_splits=2,
        epochs=2,
        patch_size=32,
        halo=4,
        base_channels=2,
        device="cpu",
        cpu_threads=1,
    )
    samples, assignments = prepare_run(config)
    settings = config.to_dict()
    if legacy_config:
        del settings["patch_sampling"]
        settings["samples_per_epoch"] = 256
    if architecture == "token_decoder_v1":
        settings["encoder"] = "gigapath-flash"
    write_json(config.run_dir / "config.json", settings)
    for fold in range(2):
        training = [sample for sample in samples if assignments[sample.image_id] != fold]
        save_constant_model(training, config, fold, architecture, settings)
        records = []
        for sample in samples:
            if assignments[sample.image_id] == fold:
                truth = np.load(sample.label_cache)
                prediction = np.full_like(truth, 4)
                output = config.run_dir / "oof" / f"{sample.image_id}.npy"
                output.parent.mkdir(exist_ok=True)
                np.save(output, prediction)
                save_label_png(prediction, output.with_suffix(".png"))
                records.append(
                    {
                        "image_id": sample.image_id,
                        "case_id": sample.case_id,
                        "fold": fold,
                        "confusion_matrix": confusion_matrix(truth, prediction).tolist(),
                    }
                )
        write_json(config.run_dir / f"fold_{fold}/evaluation.json", records)
        (config.run_dir / f"fold_{fold}/history.csv").write_text("epoch,train_loss\n2,0.25\n")
    write_json(config.run_dir / "oof_metrics.json", {"old_report": True})
    return config, samples, assignments


class FakeFrozenEncoder(nn.Module):
    def forward_features(self, x):
        assert not torch.is_grad_enabled()
        return x.new_zeros((len(x), 1 + (x.shape[-1] // 16) ** 2, 384))


@pytest.mark.parametrize("architecture", ["unet_v1", "unet_groupnorm_v2", "token_decoder_v1"])
def test_backfill_uses_exact_training_split_and_leaves_existing_artifacts_unchanged(
    tmp_path, monkeypatch, architecture
):
    config, samples, assignments = make_run(tmp_path, architecture)
    encoder_loads = []

    def load_encoder(device):
        encoder_loads.append(device)
        return FakeFrozenEncoder(), [0, 0, 0], [1, 1, 1]

    monkeypatch.setattr("prostate.extractor.load_gigapath_flash", load_encoder)
    original = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    destination = tmp_path / "diagnostics"
    summary = evaluate_training_run(config.run_dir, "cpu", output_dir=destination)
    assert summary["complete"]
    assert summary["folds_evaluated"] == [0, 1]
    for fold in range(2):
        training = [sample for sample in samples if assignments[sample.image_id] != fold]
        folder = destination / f"fold_{fold}"
        records = json.loads((folder / "train_images.json").read_text())
        report = json.loads((folder / "train_metrics.json").read_text())
        assert {record["image_id"] for record in records} == {
            sample.image_id for sample in training
        }
        assert report["images_evaluated"] == len(training)
        assert report["cases_evaluated"] == len({sample.case_id for sample in training})
        assert report["evaluation_protocol"] == EVALUATION_PROTOCOL
        assert report["split"] == "train"
        assert report["epoch"] == 2 and report["halo"] == 4
        assert not report["predictions_saved"]
        assert sum(map(sum, report["pooled_confusion_matrix"])) == sum(
            sample.height * sample.width for sample in training
        )
        assert "precision" in report["primary_case_mean"]
        assert "recall" in summary["folds"][str(fold)]["oof_primary_case_mean"]
        assert training_report_current(config.run_dir / f"fold_{fold}/checkpoint.pt", folder)
        assert report["archive_metrics"]["complete"]
        assert report["archive_metrics"]["images_evaluated"] == len(training)
        # Existing hard-only OOF cannot reconstruct the probabilities.
        assert not summary["folds"][str(fold)]["oof_archive_metrics"]["complete"]
    assert len(encoder_loads) == (1 if architecture == "token_decoder_v1" else 0)
    assert not list(destination.rglob("*.npy"))
    assert not list(destination.rglob("train-eval-*"))
    for path, data in original.items():
        assert path.read_bytes() == data


def test_cli_can_evaluate_one_existing_fold_and_marks_other_folds_missing(tmp_path, capsys):
    config, _, _ = make_run(tmp_path)
    output = tmp_path / "diagnostics"
    main(
        [
            "evaluate-train",
            "--run-dir",
            str(config.run_dir),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--fold",
            "0",
        ]
    )
    stdout = capsys.readouterr().out
    result = json.loads(stdout[stdout.index("{") :])
    assert not result["complete"]
    assert result["folds_evaluated"] == [0] and result["missing_folds"] == [1]
    assert not (output / "fold_1").exists()


@pytest.mark.parametrize("problem", ["fold", "train_image_ids", "train_case_ids"])
def test_bad_checkpoint_metadata_is_rejected_before_inference(tmp_path, monkeypatch, problem):
    config, _, assignments = make_run(tmp_path)
    path = config.run_dir / "fold_0/checkpoint.pt"
    checkpoint = torch.load(path, weights_only=True)
    checkpoint[problem] = (
        1 if problem == "fold" else [next(key for key, value in assignments.items() if value == 0)]
    )
    torch.save(checkpoint, path)

    def fail(*args, **kwargs):
        pytest.fail("Inference must not run with incorrect training metadata")

    monkeypatch.setattr("prostate.training_evaluation.predict_sample", fail)
    with pytest.raises(ValueError, match="Checkpoint"):
        evaluate_training_run(config.run_dir, "cpu", [0], tmp_path / "diagnostics")


def test_reports_become_stale_when_checkpoint_changes(tmp_path):
    config, _, _ = make_run(tmp_path)
    output = tmp_path / "diagnostics"
    evaluate_training_run(config.run_dir, "cpu", [0], output)
    path = config.run_dir / "fold_0/checkpoint.pt"
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["model_state"]["head.bias"][4] = 2
    torch.save(checkpoint, path)
    assert not training_report_current(path, output / "fold_0")
    assert collect_training_reports(config.run_dir, output)["folds_evaluated"] == []


def test_cv_automatically_evaluates_training_split_and_resume_backfills_without_training(
    tmp_path, monkeypatch
):
    config, _, _ = make_run(tmp_path, legacy_config=False)
    (config.run_dir / "fold_1/evaluation.json").unlink()
    (config.run_dir / "fold_1/checkpoint.pt").unlink()
    trained = []

    def fixed_model(samples, config, fold, device):
        trained.append(fold)
        return save_constant_model(samples, config, fold)

    monkeypatch.setattr("prostate.pipeline.train_fold", fixed_model)
    result = run_cv(config, resume=True)
    assert result["complete"]
    assert trained == [1]
    summary = json.loads((config.run_dir / "train_metrics.json").read_text())
    assert summary["complete"] and summary["folds_evaluated"] == [0, 1]

    def fail(*args, **kwargs):
        pytest.fail("A completed fold must not be retrained")

    monkeypatch.setattr("prostate.pipeline.train_fold", fail)
    (config.run_dir / "fold_0/train_metrics.json").unlink()
    assert run_cv(config, resume=True) == result
    assert (config.run_dir / "fold_0/train_metrics.json").is_file()

    preserved = {
        path: path.read_bytes()
        for pattern in ("checkpoint.pt", "history.csv", "config.json")
        for path in config.run_dir.rglob(pattern)
    }
    for path in (config.run_dir / "oof").glob("*.archive_metrics.json"):
        path.unlink()
    incomplete = evaluate_run(config.run_dir)
    assert incomplete["primary_case_mean"] == result["primary_case_mean"]
    assert not incomplete["archive_metrics"]["complete"]
    assert incomplete["archive_metrics"]["image_mean"]["jac"] is None
    monkeypatch.setattr("prostate.pipeline.evaluate_training_fold", fail)
    assert run_cv(config, resume=True) == result
    comparison = json.loads((config.run_dir / "train_metrics.json").read_text())
    assert all(fold["oof_archive_metrics"]["complete"] for fold in comparison["folds"].values())
    for path, data in preserved.items():
        assert path.read_bytes() == data
    (config.run_dir / "fold_0/checkpoint.pt").unlink()
    with pytest.raises(FileNotFoundError, match="Cannot resume completed fold"):
        run_cv(config, resume=True)


@pytest.mark.parametrize("architecture", ["unet_v1", "unet_groupnorm_v2", "token_decoder_v1"])
def test_evaluate_cli_infers_missing_soft_scores_without_training_or_repeating_complete_images(
    tmp_path, monkeypatch, capsys, architecture
):
    from prostate import evaluation

    config, samples, _ = make_run(tmp_path, architecture)
    previous = evaluate_run(config.run_dir)
    originals = {
        path: path.read_bytes()
        for pattern in ("config.json", "manifest.json", "folds.csv", "checkpoint.pt", "history.csv")
        for path in config.run_dir.rglob(pattern)
    }
    calls, encoder_loads = [], []
    predict, load_model = evaluation.predict_sample, evaluation.load_checkpoint_model

    def capture(model, sample, *args):
        calls.append(sample.image_id)
        return predict(model, sample, *args)

    def encoder(device):
        encoder_loads.append(device)
        return FakeFrozenEncoder(), [0, 0, 0], [1, 1, 1]

    def fail(*args, **kwargs):
        pytest.fail("Evaluation must not train or repeat completed inference")

    monkeypatch.setattr(evaluation, "predict_sample", capture)
    monkeypatch.setattr("prostate.extractor.load_gigapath_flash", encoder)
    monkeypatch.setattr("prostate.pipeline.train_fold", fail)
    monkeypatch.setattr("prostate.pipeline.train_token_fold", fail)
    monkeypatch.setattr(torch.optim, "AdamW", fail)
    command = ["evaluate", "--run-dir", str(config.run_dir), "--device", "cpu"]
    capsys.readouterr()
    main(command)
    stdout = capsys.readouterr().out
    result = json.loads(stdout[stdout.index("{") :])
    old = result["archive_metrics"]
    assert result["complete"] and old["complete"]
    assert result["primary_case_mean"] == previous["primary_case_mean"]
    assert old["images_evaluated"] == len(samples) == len(calls)
    assert old["missing_image_ids"] == [] and set(old["fold_image_means"]) == {"0", "1"}
    assert len(encoder_loads) == (1 if architecture == "token_decoder_v1" else 0)
    # Fixed logits [0,0,0,0,1]; one red mask and two masks of other classes.
    probability = np.exp(1) / (np.exp(1) + 4)
    soft = [399 * probability, 399 / (np.exp(1) + 4), 399 / (np.exp(1) + 4)]
    assert old["image_mean"]["jac"] == pytest.approx(
        np.mean([(s + 1) / (798 - s + 1) for s in soft])
    )
    for path, before in originals.items():
        assert path.read_bytes() == before

    monkeypatch.setattr(evaluation, "predict_sample", fail)
    monkeypatch.setattr(evaluation, "load_checkpoint_model", fail)
    main(command)
    stdout = capsys.readouterr().out
    assert json.loads(stdout[stdout.index("{") :]) == result
    missing = samples[0]
    (config.run_dir / "oof" / f"{missing.image_id}.archive_metrics.json").unlink()
    calls.clear()
    monkeypatch.setattr(evaluation, "predict_sample", capture)
    monkeypatch.setattr(evaluation, "load_checkpoint_model", load_model)
    main(command)
    assert calls == [missing.image_id]


def test_evaluate_saved_only_keeps_missing_metrics_without_loading_models(
    tmp_path, monkeypatch, capsys
):
    config, samples, _ = make_run(tmp_path)
    monkeypatch.setattr(
        "prostate.evaluation.load_checkpoint_model", lambda *a: pytest.fail("No inference")
    )
    capsys.readouterr()
    main(["evaluate", "--run-dir", str(config.run_dir), "--saved-only"])
    result = json.loads(capsys.readouterr().out)
    assert result["complete"] and not result["archive_metrics"]["complete"]
    assert result["archive_metrics"]["images_evaluated"] == 0
    assert len(result["archive_metrics"]["missing_image_ids"]) == len(samples)


def test_evaluate_recovers_after_incomplete_inference_and_keeps_untrained_folds_missing(tmp_path):
    config, samples, assignments = make_run(tmp_path)
    (config.run_dir / "fold_0/evaluation.json").unlink()
    (config.run_dir / "fold_1/evaluation.json").unlink()
    (config.run_dir / "fold_1/checkpoint.pt").unlink()
    for path in (config.run_dir / "oof").iterdir():
        path.unlink()
    result = evaluate_run(config.run_dir, infer_missing=True, requested_device="cpu")
    missing = sorted(sample.image_id for sample in samples if assignments[sample.image_id] == 1)
    assert not result["complete"] and not result["archive_metrics"]["complete"]
    assert result["archive_metrics"]["missing_image_ids"] == missing
    assert (config.run_dir / "fold_0/evaluation.json").is_file()
    assert not (config.run_dir / "fold_1/checkpoint.pt").exists()


@pytest.mark.parametrize("problem", ["train_case_ids", "config"])
def test_evaluate_rejects_inconsistent_checkpoint_before_inference(tmp_path, monkeypatch, problem):
    config, _, _ = make_run(tmp_path)
    path = config.run_dir / "fold_0/checkpoint.pt"
    checkpoint = torch.load(path, weights_only=True)
    if problem == "config":
        checkpoint["config"]["halo"] = 0
    else:
        checkpoint[problem].append("heldout")
    torch.save(checkpoint, path)
    monkeypatch.setattr(
        "prostate.evaluation.load_checkpoint_model", lambda *a: pytest.fail("Must validate first")
    )
    with pytest.raises(ValueError, match="Checkpoint"):
        evaluate_run(config.run_dir, infer_missing=True, requested_device="cpu")
