import json

import numpy as np
import pytest
import torch
from PIL import Image

from prostate.config import Config
from prostate.inference import predict_image
from prostate.pipeline import evaluate_run, run_cv


def test_five_fold_training_oof_resume_and_standalone_prediction(tmp_path):
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
        run_dir=tmp_path / "run",
        epochs=1,
        samples_per_epoch=1,
        patch_size=32,
        halo=4,
        base_channels=2,
        device="cpu",
        cpu_threads=1,
    )
    partial = run_cv(config, selected_folds=[0])
    assert not partial["complete"]
    result = run_cv(config, resume=True)
    assert result["complete"]
    assert result["images_evaluated"] == 6
    assert result["cases_evaluated"] == 5
    assert len(list((config.run_dir / "oof").glob("*.png"))) == 6
    records = json.loads((config.run_dir / "oof_images.json").read_text())
    for record in records:
        checkpoint_path = config.run_dir / f"fold_{record['fold']}" / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        assert record["case_id"] not in checkpoint["train_case_ids"]
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
