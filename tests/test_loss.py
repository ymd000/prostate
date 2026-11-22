import json

import numpy as np
import pytest
import torch
from torch import nn

from prostate.__main__ import main
from prostate.config import Config
from prostate.data import CLASS_NAMES, IGNORE_INDEX, Sample
from prostate.loss import class_weights, prepare_cross_entropy
from prostate.pipeline import run_cv


def sample(image_id, counts):
    return Sample(
        image_id, image_id.split("_")[0], "", "", "", "", sum(counts), 1, tuple(counts), {}
    )


def test_inverse_sqrt_weights_are_scale_invariant_and_finite_for_absent_classes():
    counts = np.array([1600, 400, 100, 25, 0])
    weights = class_weights(counts, "inverse-sqrt")
    assert weights.dtype == np.float32
    np.testing.assert_allclose(weights[:4] / weights[0], [1, 2, 4, 8])
    np.testing.assert_allclose(weights, class_weights(counts * 100, "inverse-sqrt"))
    assert weights[:4].mean() == pytest.approx(1)
    assert weights[4] == 0
    assert np.isfinite(weights).all()
    np.testing.assert_array_equal(
        class_weights(np.array([0, 0, 0, 0, 7]), "inverse-sqrt"), [0, 0, 0, 0, 1]
    )


@pytest.mark.parametrize(
    "counts",
    [
        [0, 0, 0, 0, 0],
        [1, 2, 3, 4],
        [-1, 2, 3, 4, 5],
        [1.5, 2, 3, 4, 5],
        [float("nan"), 2, 3, 4, 5],
    ],
)
def test_invalid_class_counts_are_rejected(counts):
    with pytest.raises(ValueError, match="class counts"):
        class_weights(np.array(counts), "inverse-sqrt")


def test_weighted_loss_emphasizes_rare_pixel_errors_and_ignores_padding(tmp_path):
    training = [sample("train_1", [1000, 100, 100, 100, 1])]
    config = Config(run_dir=tmp_path, class_weighting="inverse-sqrt")
    criterion, report = prepare_cross_entropy(training, config, 0, torch.device("cpu"))
    assert criterion.weight.dtype == torch.float32
    logits = torch.zeros(1, 5, 1, 3)
    logits[:, 0] = 3
    targets = torch.tensor([[[0, 4, IGNORE_INDEX]]])
    with torch.inference_mode():
        weighted = criterion(logits, targets)
        unweighted = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)(logits, targets)
        per_pixel = -logits.log_softmax(dim=1)[0, [0, 4], 0, [0, 1]]
        chosen_weights = criterion.weight[[0, 4]]
        expected = (per_pixel * chosen_weights).sum() / chosen_weights.sum()
        torch.testing.assert_close(weighted, expected)
        torch.testing.assert_close(weighted, criterion(logits[..., :2], targets[..., :2]))
    assert weighted > unweighted
    assert report["class_counts"]["GP5"] == 1
    saved = json.loads((tmp_path / "fold_0/loss.json").read_text())
    assert saved == report
    assert saved["train_image_ids"] == ["train_1"]
    assert saved["ignore_index"] == 255 and saved["loss_dtype"] == "float32"
    np.testing.assert_array_equal(
        [saved["class_weights"][name] for name in CLASS_NAMES], criterion.weight.numpy()
    )


def test_unweighted_default_matches_previous_ce(tmp_path):
    config = Config(run_dir=tmp_path)
    criterion, report = prepare_cross_entropy(
        [sample("A_1", [1, 1, 1, 1, 1])], config, 0, torch.device("cpu")
    )
    assert criterion.weight is None
    assert report["class_weighting"] == "none" and report["class_weights"] is None
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]).reshape(1, 5, 1, 1)
    target = torch.tensor([[[4]]])
    torch.testing.assert_close(
        criterion(logits, target), nn.CrossEntropyLoss(ignore_index=255)(logits, target)
    )


@pytest.mark.parametrize("encoder", ["none", "gigapath-flash"])
def test_cv_weight_calculation_uses_training_split_without_optimization(
    tmp_path, monkeypatch, encoder
):
    training = sample("train_1", [10000, 2000, 1000, 500, 100])
    held_out = sample("heldout_1", [1, 1, 1, 1, 500000])
    config = Config(
        run_dir=tmp_path,
        token_dir=tmp_path / "tokens",
        encoder=encoder,
        class_weighting="inverse-sqrt",
        n_splits=2,
        device="cpu",
    )
    monkeypatch.setattr(
        "prostate.pipeline.prepare_run",
        lambda config: ([training, held_out], {training.image_id: 1, held_out.image_id: 0}),
    )
    if encoder == "gigapath-flash":
        config.token_dir.mkdir()
        for item in (training, held_out):
            (config.token_dir / f"{item.image_id}.h5").touch()
        monkeypatch.setattr("prostate.extractor.load_gigapath_flash", lambda device: (None, [], []))

    class LossPrepared(Exception):
        pass

    def prepare_only(samples, config, fold, device):
        _, report = prepare_cross_entropy(samples, config, fold, device)
        assert report["train_image_ids"] == [training.image_id]
        assert report["class_counts"]["GP5"] == 100
        assert report["class_counts"]["non_gland"] == 10000
        raise LossPrepared

    monkeypatch.setattr("prostate.pipeline.train_fold", prepare_only)
    monkeypatch.setattr("prostate.pipeline.train_token_fold", prepare_only)
    with pytest.raises(LossPrepared):
        run_cv(config, selected_folds=[0])
    saved = json.loads((tmp_path / "fold_0/loss.json").read_text())
    assert saved["class_weights"]["GP5"] / saved["class_weights"]["non_gland"] == pytest.approx(10)


def test_cli_saves_weighting_mode_without_training(tmp_path, monkeypatch, capsys):
    captured = []

    def capture(config, folds, resume):
        captured.append(config.to_dict())
        return {
            "complete": False,
            "images_evaluated": 0,
            "cases_evaluated": 0,
            "primary_case_mean": {},
        }

    monkeypatch.setattr("prostate.__main__.run_cv", capture)
    main(["cv", "--class-weighting", "inverse-sqrt", "--run-dir", str(tmp_path)])
    assert captured[0]["class_weighting"] == "inverse-sqrt"
    assert json.loads(capsys.readouterr().out)["complete"] is False
    with pytest.raises(ValueError, match="class_weighting"):
        Config(class_weighting="invalid")


@pytest.mark.parametrize("weighting", ["none", "inverse-sqrt"])
def test_legacy_unweighted_config_can_only_resume_with_the_same_objective(
    tmp_path, monkeypatch, weighting
):
    config = Config(run_dir=tmp_path, device="cpu", class_weighting=weighting)
    old = config.to_dict()
    del old["class_weighting"]
    content = json.dumps(old, indent=2) + "\n"
    (tmp_path / "config.json").write_text(content)
    monkeypatch.setattr("prostate.pipeline.prepare_run", lambda config: ([], {}))

    class BeforeTraining(Exception):
        pass

    def fail_before_training(*args, **kwargs):
        raise BeforeTraining

    monkeypatch.setattr("prostate.pipeline.train_fold", fail_before_training)
    if weighting == "none":
        with pytest.raises(BeforeTraining):
            run_cv(config, selected_folds=[0], resume=True)
    else:
        with pytest.raises(ValueError, match="Experiment differs"):
            run_cv(config, selected_folds=[0], resume=True)
    assert (tmp_path / "config.json").read_text() == content
