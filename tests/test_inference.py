import numpy as np
import torch
from PIL import Image
from torch import nn

from prostate.inference import predict_tiled, save_label_png, tile_positions
from prostate.model import UNet


class PointwiseModel(nn.Module):
    def forward(self, x):
        values = x[:, :1] * 255
        return torch.cat([-torch.square(values - value) for value in range(5)], dim=1)


def test_halo_tiles_cover_every_pixel_once_and_recover_non_square_image(tmp_path):
    truth = (np.indices((43, 57)).sum(axis=0) % 5).astype(np.uint8)
    rgb = np.repeat(truth[..., None], 3, axis=2)
    coverage = np.zeros(truth.shape, dtype=np.uint8)
    for y, x, h, w in tile_positions(43, 57, 32, 5):
        coverage[y : y + h, x : x + w] += 1
    assert np.all(coverage == 1)
    model = PointwiseModel()
    model.train()
    prediction = predict_tiled(model, rgb, tmp_path / "prediction.npy", 32, 5, torch.device("cpu"))
    assert model.training
    np.testing.assert_array_equal(prediction, truth)
    save_label_png(prediction, tmp_path / "prediction.png")
    with Image.open(tmp_path / "prediction.png") as image:
        assert image.mode == "P"
        np.testing.assert_array_equal(np.asarray(image), truth)


def test_single_pixel_image_is_supported(tmp_path):
    rgb = np.full((1, 1, 3), 2, dtype=np.uint8)
    prediction = predict_tiled(
        PointwiseModel(), rgb, tmp_path / "tiny.npy", 32, 4, torch.device("cpu")
    )
    np.testing.assert_array_equal(prediction, [[2]])


def test_unet_outputs_full_resolution_logits_for_odd_sizes():
    torch.set_num_threads(1)
    model = UNet(base_channels=2)
    inputs = torch.randn(1, 3, 33, 49)
    output = model(inputs)
    assert output.shape == (1, 5, 33, 49)
    output.square().mean().backward()
    assert model.head.weight.grad is not None
