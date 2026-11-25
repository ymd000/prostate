import json

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from prostate.__main__ import main
from prostate.config import Config
from prostate.data import grid_positions
from prostate.inference import predict_image
from prostate.pipeline import evaluate_run, prepare_run, run_cv


def test_cli_full_grid_config_reaches_cv_without_training(monkeypatch, capsys, tmp_path):
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
    main(["cv", "--encoder", "gigapath-flash", "--run-dir", str(tmp_path)])
    assert json.loads(capsys.readouterr().out)["complete"] is False
    assert captured[0]["patch_sampling"] == "full_grid_v1"
    assert "samples_per_epoch" not in captured[0]


def test_old_random_run_cannot_be_resumed_as_full_grid(monkeypatch, tmp_path):
    config = Config(run_dir=tmp_path, device="cpu")
    old = config.to_dict()
    del old["patch_sampling"]
    old["samples_per_epoch"] = 256
    content = json.dumps(old, indent=2) + "\n"
    (tmp_path / "config.json").write_text(content)
    monkeypatch.setattr("prostate.pipeline.prepare_run", lambda config: ([], {}))
    with pytest.raises(ValueError, match="Experiment differs"):
        run_cv(config, resume=True)
    assert (tmp_path / "config.json").read_text() == content


@pytest.mark.parametrize("encoder", ["none", "gigapath-flash"])
def test_five_fold_training_oof_resume_and_standalone_prediction(tmp_path, monkeypatch, encoder):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for case in range(5):
        for image_number in range(2 if case == 0 else 1):
            name = f"case{case}_{image_number}"
            Image.new("RGB", (21, 19), (180, 80, 160)).save(data_dir / f"{name}.jpg")
            Image.new("RGBA", (21, 19), (0, 0, 255, 255)).save(data_dir / f"{name}.png")
    config = Config(
        data_dir=data_dir,
        cache_dir=tmp_path / "cache",
        token_dir=tmp_path / "tokens",
        run_dir=tmp_path / "run",
        epochs=1,
        patch_size=32,
        halo=4,
        base_channels=2,
        class_weighting="inverse-sqrt",
        device="cpu",
        cpu_threads=1,
        encoder=encoder,
    )
    if encoder == "gigapath-flash":
        samples, _ = prepare_run(config)
        config.token_dir.mkdir()
        for sample in samples:
            positions = list(grid_positions(sample.height, sample.width, config.patch_size))
            with h5py.File(config.token_dir / f"{sample.image_id}.h5", "w") as cache:
                cache.attrs["patch_size"] = config.patch_size
                cache.create_dataset("positions", data=np.array(positions, dtype=np.int32))
                cache.create_dataset(
                    "tokens", data=np.zeros((len(positions), 2, 2, 384), np.float16)
                )

        class FakeFrozenEncoder(torch.nn.Module):
            def forward_features(self, inputs):
                assert not torch.is_grad_enabled()
                return inputs.new_zeros((len(inputs), 5, 384))

        monkeypatch.setattr(
            "prostate.extractor.load_gigapath_flash",
            lambda device: (FakeFrozenEncoder(), [0, 0, 0], [1, 1, 1]),
        )
    partial = run_cv(config, selected_folds=[0])
    assert not partial["complete"]
    assert not partial["archive_metrics"]["complete"]
    assert partial["archive_metrics"]["missing_image_ids"]
    result = run_cv(config, resume=True)
    assert result["complete"]
    assert result["images_evaluated"] == 6
    assert result["cases_evaluated"] == 5
    assert result["archive_metrics"]["complete"]
    assert result["archive_metrics"]["images_evaluated"] == 6
    assert set(result["archive_metrics"]["fold_image_means"]) == {"0", "1", "2", "3", "4"}
    assert len(list((config.run_dir / "oof").glob("*.png"))) == 6
    records = json.loads((config.run_dir / "oof_images.json").read_text())
    for record in records:
        checkpoint_path = config.run_dir / f"fold_{record['fold']}" / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        assert record["case_id"] not in checkpoint["train_case_ids"]
        loss_settings = json.loads((checkpoint_path.parent / "loss.json").read_text())
        assert checkpoint["loss"] == loss_settings
        assert loss_settings["class_weighting"] == "inverse-sqrt"
        assert loss_settings["class_weights"]["GP3"] == 1
        assert loss_settings["class_counts"]["GP3"] == len(checkpoint["train_image_ids"]) * 21 * 19
        with Image.open(config.run_dir / "oof" / f"{record['image_id']}.png") as image:
            assert image.size == (21, 19)
            assert np.asarray(image).max() < 5
    assert evaluate_run(config.run_dir) == result
    assert run_cv(config, resume=True) == result
    with pytest.raises(ValueError, match="already exists"):
        run_cv(config)
    output = tmp_path / "single.png"
    predict_image(
        config.run_dir / "fold_0" / "checkpoint.pt",
        data_dir / "case0_0.jpg",
        output,
        torch.device("cpu"),
    )
    assert output.exists()


@pytest.mark.parametrize(
    "settings",
    [
        {"patch_size": 17},
        {"halo": 128},
        {"epochs": 0},
        {"batch_size": 0},
        {"num_workers": -1},
        {"n_splits": 1},
        {"normalization": "batch"},
        {"amp_dtype": "invalid"},
    ],
)
def test_invalid_training_config_fails_early(settings):
    with pytest.raises(ValueError):
        Config(**settings)
