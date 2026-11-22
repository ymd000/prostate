"""Extract frozen gigapath-flash ViT tokens for cached images and save to h5."""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from prostate.data import Sample, cached_arrays, grid_positions, read_tile

ENCODER_ID = "gigapath-flash"
TOKEN_DIM = 384
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_gigapath_flash(device: torch.device) -> tuple[nn.Module, list[float], list[float]]:
    from wsi_toolbox.presets.tile.gigapath_flash import create_gigapath_flash_model

    model = create_gigapath_flash_model()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model.to(device), IMAGENET_MEAN, IMAGENET_STD


def _encode_batch(
    model: nn.Module,
    rgb_batch: np.ndarray,
    mean: list[float],
    std: list[float],
    device: torch.device,
) -> np.ndarray:
    """Run encoder on (B, H, W, 3) uint8 → (B, grid, grid, TOKEN_DIM) float16."""
    x = torch.from_numpy(np.ascontiguousarray(rgb_batch.transpose(0, 3, 1, 2))).float().div_(255)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    x = (x - mean_t) / std_t
    x = x.to(device)
    with torch.inference_mode():
        all_tokens = model.forward_features(x)  # (B, 1 + N, TOKEN_DIM)
    patch_tokens = all_tokens[:, 1:, :]  # drop CLS → (B, N, TOKEN_DIM)
    n = patch_tokens.shape[1]
    grid = int(n**0.5)
    tokens_2d = patch_tokens.reshape(patch_tokens.shape[0], grid, grid, TOKEN_DIM)
    return tokens_2d.half().cpu().numpy()


def extract_sample(
    sample: Sample,
    model: nn.Module,
    mean: list[float],
    std: list[float],
    token_dir: Path,
    patch_size: int,
    batch_size: int,
    device: torch.device,
) -> Path:
    h5_path = token_dir / f"{sample.image_id}.h5"
    if h5_path.exists():
        return h5_path

    rgb, _ = cached_arrays(sample.rgb_cache, sample.label_cache)
    height, width = rgb.shape[:2]
    positions = list(grid_positions(height, width, patch_size))

    all_tokens: list[np.ndarray] = []
    buf: list[np.ndarray] = []

    def _flush():
        all_tokens.append(_encode_batch(model, np.stack(buf), mean, std, device))
        buf.clear()

    for top, left in positions:
        buf.append(read_tile(rgb, top, left, patch_size))
        if len(buf) == batch_size:
            _flush()
    if buf:
        _flush()

    tokens_array = np.concatenate(all_tokens, axis=0)  # (N, grid, grid, TOKEN_DIM)
    pos_array = np.array(positions, dtype=np.int32)  # (N, 2)

    temp = h5_path.with_suffix(".tmp.h5")
    token_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(temp, "w") as f:
        f.attrs["image_id"] = sample.image_id
        f.attrs["width"] = width
        f.attrs["height"] = height
        f.attrs["patch_size"] = patch_size
        f.attrs["encoder"] = ENCODER_ID
        f.create_dataset("tokens", data=tokens_array, compression="lzf")
        f.create_dataset("positions", data=pos_array)
    temp.replace(h5_path)
    return h5_path


def extract_dataset(
    samples: list[Sample],
    token_dir: Path,
    patch_size: int,
    batch_size: int,
    device: torch.device,
):
    model, mean, std = load_gigapath_flash(device)
    for sample in tqdm(samples, desc="Extracting tokens"):
        extract_sample(sample, model, mean, std, token_dir, patch_size, batch_size, device)
