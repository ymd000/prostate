import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from prostate.__main__ import main
from prostate.inference import predict_image
from prostate.model import DoubleConv, UNet
from prostate.precision import resolve_precision


@pytest.mark.parametrize(
    ("supported", "requested", "expected"),
    [
        (True, "auto", torch.bfloat16),
        (False, "auto", torch.float16),
        (True, "float16", torch.float16),
        (True, "bfloat16", torch.bfloat16),
    ],
)
def test_cuda_precision_selection_without_using_gpu(monkeypatch, supported, requested, expected):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: supported)
    assert resolve_precision(torch.device("cuda"), True, requested) == (True, expected)


def test_disabled_amp_and_cpu_do_not_query_cuda(monkeypatch):
    def unexpected_query():
        raise AssertionError("CUDA queried for a disabled precision context")

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", unexpected_query)
    assert resolve_precision(torch.device("cpu"), True) == (False, torch.float32)
    assert resolve_precision(torch.device("cuda"), False) == (False, torch.float32)


def test_explicit_unsupported_bfloat16_is_an_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(ValueError, match="does not support BF16"):
        resolve_precision(torch.device("cuda"), True, "bfloat16")


def test_groupnorm_bounds_fp16_activations_without_optimization():
    torch.set_num_threads(1)
    normalized = DoubleConv(3, 8, "group").half().eval()
    original = DoubleConv(3, 8, "none").half().eval()
    for block in (normalized, original):
        for layer in block.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
    inputs = torch.full((1, 3, 32, 32), 100, dtype=torch.float16)
    with torch.inference_mode():
        assert not torch.isfinite(original(inputs)).all()
        assert torch.isfinite(normalized(inputs)).all()


@pytest.mark.parametrize("normalization", ["none", "group"])
def test_checkpoint_loading_preserves_old_and_new_architectures(tmp_path, normalization):
    torch.set_num_threads(1)
    model = UNet(base_channels=2, normalization=normalization)
    architecture = "unet_v1" if normalization == "none" else "unet_groupnorm_v2"
    model_settings = {"num_classes": 5, "base_channels": 2}
    if normalization == "group":
        model_settings["normalization"] = normalization
    checkpoint = {
        "architecture": architecture,
        "model": model_settings,
        "model_state": model.state_dict(),
        "config": {"patch_size": 32, "halo": 4, "amp": False},
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    Image.new("RGB", (21, 19), (100, 40, 80)).save(tmp_path / "input.jpg")
    predict_image(
        checkpoint_path, tmp_path / "input.jpg", tmp_path / "out.png", torch.device("cpu")
    )
    with Image.open(tmp_path / "out.png") as image:
        assert image.size == (21, 19)
        assert np.asarray(image).max() < 5


def test_cli_propagates_new_precision_and_normalization_settings(monkeypatch):
    def fake_cv(config, selected_folds, resume):
        assert config.normalization == "group"
        assert config.amp_dtype == "auto"
        assert config.learning_rate == 1e-4
        assert selected_folds == [0]
        return {
            "complete": False,
            "images_evaluated": 0,
            "cases_evaluated": 0,
            "primary_case_mean": {},
        }

    monkeypatch.setattr("prostate.__main__.run_cv", fake_cv)
    main(["cv", "--device", "cpu", "--fold", "0"])
