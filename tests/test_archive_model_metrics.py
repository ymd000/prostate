import importlib.util
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from torchvision import models
from torchvision.models.vgg import cfgs, make_layers

from prostate.archive_metrics import ArchiveMetrics
from prostate.archive_model import build_archive_model
from prostate.archive_train import archive_loss
from prostate.config import ARCHIVE_MODELS
from prostate.metrics import confusion_matrix, scores

ARCHIVE = Path(__file__).resolve().parents[1] / "__archive"


@pytest.mark.parametrize("name", ARCHIVE_MODELS)
def test_port_matches_the_archived_architecture_and_output(name, monkeypatch):
    torch.set_num_threads(1)
    # Keep the original constructors offline and avoid their unused VGG classifier.
    original_resnet = models.resnet34
    monkeypatch.setattr(
        models,
        "vgg11_bn",
        lambda **kwargs: SimpleNamespace(features=make_layers(cfgs["A"], batch_norm=True)),
    )
    monkeypatch.setattr(
        models,
        "vgg16_bn",
        lambda **kwargs: SimpleNamespace(features=make_layers(cfgs["D"], batch_norm=True)),
    )
    monkeypatch.setattr(models, "resnet34", lambda **kwargs: original_resnet(weights=None))
    monkeypatch.setitem(
        sys.modules,
        "utils",
        SimpleNamespace(curry=partial, pack=lambda values: [v for v in values if v]),
    )
    spec = importlib.util.spec_from_file_location("archive_oracle_models", ARCHIVE / "models.py")
    archived = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archived)
    port = build_archive_model(name, pretrained=False).eval()
    original = archived.get_model(name)(num_classes=5).eval()
    original.load_state_dict(port.state_dict(), strict=True)
    image = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        expected = original(image)
        logits = port(image)
    assert logits.shape == (1, 5, 64, 64)
    torch.testing.assert_close(logits.softmax(1), expected, rtol=1e-5, atol=1e-6)


def test_archive_metrics_match_the_original_functions_and_keep_background_explicit(monkeypatch):
    monkeypatch.setitem(sys.modules, "formula", SimpleNamespace(NUM_CLASSES=5))
    spec = importlib.util.spec_from_file_location("archive_oracle_utils", ARCHIVE / "utils.py")
    archived = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archived)
    logits = torch.randn(2, 5, 7, 9)
    target = torch.randint(5, (2, 7, 9))
    one_hot = F.one_hot(target, 5).permute(0, 3, 1, 2).float()
    probabilities = logits.softmax(1)
    dice, jac = archived.similarity_index(probabilities, one_hot)
    pdice, pjac = archived.pixel_similarity_index(
        probabilities.argmax(1).reshape(-1), target.reshape(-1)
    )
    metrics = ArchiveMetrics()
    metrics.update(logits, target)
    assert metrics.report("batch")["mean"] == pytest.approx(
        {"dice": dice, "jac": jac, "pdice": pdice, "pjac": pjac}
    )

    # Background-only predictions can score >0.8 in the old pjac while missing all foreground.
    truth = np.array([[0] * 8 + [4]], dtype=np.uint8)
    background = torch.zeros(1, 5, 1, 9)
    background[:, 0] = 30
    metrics = ArchiveMetrics()
    metrics.update(background, torch.from_numpy(truth[None]).long())
    assert metrics.report("tile")["mean"]["pjac"] > 0.8
    assert scores(confusion_matrix(truth, np.zeros_like(truth)))["foreground_macro_iou"] == 0


@pytest.mark.parametrize("ignore_pixel", [False, True])
def test_archived_loss_has_the_same_values_and_finite_gradients_as_ce(ignore_pixel):
    logits = torch.randn(2, 5, 4, 6, requires_grad=True)
    target = torch.randint(5, (2, 4, 6))
    if ignore_pixel:
        target[0, 0, 0] = 255
    loss = archive_loss(logits, target)
    torch.testing.assert_close(loss, F.cross_entropy(logits, target, ignore_index=255))
    loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("name,depth", [("unet11", 11), ("unet16", 16), ("albunet", None)])
def test_pretrained_features_are_loaded_from_the_output_cache(name, depth, monkeypatch, tmp_path):
    torch.set_num_threads(1)
    if depth is None:
        reference = models.resnet34(weights=None)
        state = reference.state_dict()
        reference.conv1.weight.data.fill_(0.125)
        enum_name = "ResNet34_Weights"
    else:
        reference = make_layers(cfgs["A" if depth == 11 else "D"], batch_norm=True)
        reference[0].weight.data.fill_(0.125)
        state = {f"features.{key}": value for key, value in reference.state_dict().items()}
        state["classifier.0.weight"] = torch.zeros(1)
        enum_name = f"VGG{depth}_BN_Weights"
    calls = []

    def load(**kwargs):
        calls.append(kwargs)
        return state

    monkeypatch.setattr(
        models, enum_name, SimpleNamespace(DEFAULT=SimpleNamespace(get_state_dict=load))
    )
    model = build_archive_model(name, pretrained=True, cache_dir=tmp_path)
    first = model.first[0] if depth is None else model.conv1[0]
    torch.testing.assert_close(first.weight, torch.full_like(first.weight, 0.125))
    assert calls == [{"progress": True, "check_hash": True, "model_dir": str(tmp_path)}]


def test_ported_unet_has_finite_encoder_and_decoder_gradients():
    torch.set_num_threads(1)
    model = build_archive_model("unet11", pretrained=False).train()
    loss = archive_loss(model(torch.randn(2, 3, 64, 64)), torch.randint(5, (2, 64, 64)))
    loss.backward()
    assert torch.isfinite(model.conv1[0].weight.grad).all()
    assert torch.isfinite(model.final.weight.grad).all()
