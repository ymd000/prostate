"""Bounded-memory tiled inference and class-ID PNG export."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from prostate.data import NUM_CLASSES, Sample, cached_arrays, image_tensor, open_image, read_tile
from prostate.model import TokenDecoder, UNet
from prostate.precision import resolve_precision

PALETTE = [0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 255, 0, 255, 0, 0] + [0] * (768 - 15)


def tile_positions(height: int, width: int, patch_size: int, halo: int):
    if patch_size < 16 or not 0 <= halo < patch_size / 2:
        raise ValueError("Invalid patch_size or halo")
    core = patch_size - 2 * halo
    for top in range(0, height, core):
        for left in range(0, width, core):
            yield top, left, min(core, height - top), min(core, width - left)


def predict_tiled(
    model: torch.nn.Module,
    rgb: np.ndarray,
    output_path: Path,
    patch_size: int,
    halo: int,
    device: torch.device,
    amp: bool = True,
    amp_dtype: str = "auto",
) -> np.ndarray:
    """Predict overlapping context tiles, keeping each tile's central core exactly once."""
    height, width = rgb.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".partial.npy")
    predictions = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.uint8, shape=(height, width)
    )
    was_training = model.training
    amp_enabled, autocast_dtype = resolve_precision(device, amp, amp_dtype)
    model.eval()
    try:
        with torch.inference_mode():
            for top, left, core_height, core_width in tile_positions(
                height, width, patch_size, halo
            ):
                tile = read_tile(rgb, top - halo, left - halo, patch_size)
                inputs = image_tensor(tile).unsqueeze(0).to(device)
                with torch.autocast(device.type, dtype=autocast_dtype, enabled=amp_enabled):
                    logits = model(inputs)
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Non-finite logits during tiled inference")
                labels = logits[0].argmax(dim=0).to(torch.uint8).cpu().numpy()
                predictions[top : top + core_height, left : left + core_width] = labels[
                    halo : halo + core_height, halo : halo + core_width
                ]
        predictions.flush()
    finally:
        model.train(was_training)
        del predictions
    temporary.replace(output_path)
    return np.load(output_path, mmap_mode="r")


def predict_tiled_with_encoder(
    encoder: torch.nn.Module,
    encoder_mean: list[float],
    encoder_std: list[float],
    decoder: torch.nn.Module,
    rgb: np.ndarray,
    output_path: Path,
    patch_size: int,
    halo: int,
    device: torch.device,
    amp: bool = True,
    amp_dtype: str = "auto",
) -> np.ndarray:
    """Encode each tile with gigapath-flash then decode to class labels."""
    height, width = rgb.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".partial.npy")
    predictions = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.uint8, shape=(height, width)
    )
    amp_enabled, autocast_dtype = resolve_precision(device, amp, amp_dtype)
    mean_t = torch.tensor(encoder_mean, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    std_t = torch.tensor(encoder_std, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    was_training = decoder.training
    encoder.eval()
    decoder.eval()
    try:
        with torch.inference_mode():
            for top, left, core_height, core_width in tile_positions(
                height, width, patch_size, halo
            ):
                tile = read_tile(rgb, top - halo, left - halo, patch_size)
                x = image_tensor(tile).unsqueeze(0).to(device)
                x = (x - mean_t) / std_t
                with torch.autocast(device.type, dtype=autocast_dtype, enabled=amp_enabled):
                    all_tokens = encoder.forward_features(x)  # (1, 1+N, D)
                    patch_tokens = all_tokens[:, 1:, :].float()  # (1, N, D)
                    n = patch_tokens.shape[1]
                    grid = int(n**0.5)
                    token_map = patch_tokens.reshape(1, grid, grid, -1).permute(0, 3, 1, 2)
                    logits = decoder(token_map)
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Non-finite logits during tiled inference")
                labels = logits[0].argmax(dim=0).to(torch.uint8).cpu().numpy()
                predictions[top : top + core_height, left : left + core_width] = labels[
                    halo : halo + core_height, halo : halo + core_width
                ]
        predictions.flush()
    finally:
        decoder.train(was_training)
        del predictions
    temporary.replace(output_path)
    return np.load(output_path, mmap_mode="r")


def save_label_png(labels: np.ndarray, path: Path):
    """Palette colors are for display; stored pixel values remain class IDs 0..4."""
    image = Image.fromarray(np.asarray(labels, dtype=np.uint8))
    image.putpalette(PALETTE)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.png")
    image.save(temporary, format="PNG", transparency=0)
    temporary.replace(path)


def predict_sample(
    model: torch.nn.Module,
    sample: Sample,
    output_path: Path,
    settings: dict,
    device: torch.device,
    encoder_bundle: tuple | None = None,
) -> np.ndarray:
    """Use the same full-image halo tiling for training diagnostics and OOF."""
    rgb, _ = cached_arrays(sample.rgb_cache, sample.label_cache)
    if settings.get("training_recipe") == "archive_random_v1":
        from prostate.archive_inference import predict_archive_tiled

        target = np.load(sample.label_cache, mmap_mode="r")
        return predict_archive_tiled(model, rgb, output_path, settings, device, target)
    options = {"amp": settings["amp"], "amp_dtype": settings.get("amp_dtype", "float16")}
    if encoder_bundle is None:
        return predict_tiled(
            model, rgb, output_path, settings["patch_size"], settings["halo"], device, **options
        )
    encoder, mean, std = encoder_bundle
    return predict_tiled_with_encoder(
        encoder,
        mean,
        std,
        model,
        rgb,
        output_path,
        settings["patch_size"],
        settings["halo"],
        device,
        **options,
    )


def load_checkpoint_model(checkpoint: dict, device: torch.device) -> torch.nn.Module:
    """Load the segmentation model; a frozen encoder can be shared across fold models."""
    architecture = checkpoint.get("architecture")
    if checkpoint["model"]["num_classes"] != NUM_CLASSES:
        raise ValueError("Checkpoint class count does not match the baseline")
    model_settings = dict(checkpoint["model"])
    if architecture == "archive_unet_v1":
        from prostate.archive_model import build_archive_model

        model = build_archive_model(**model_settings, pretrained=False).to(device)
    elif architecture in ("unet_v1", "unet_groupnorm_v2"):
        model_settings.setdefault("normalization", "none" if architecture == "unet_v1" else "group")
        model = UNet(**model_settings).to(device)
    elif architecture == "token_decoder_v1":
        model = TokenDecoder(**model_settings).to(device)
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    model.load_state_dict(checkpoint["model_state"])
    return model


def predict_image(
    checkpoint_path: Path,
    image_path: Path,
    output_path: Path,
    device: torch.device,
    patch_size: int | None = None,
    halo: int | None = None,
) -> Path:
    if output_path.suffix.lower() != ".png":
        raise ValueError("Prediction output must be a .png file")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture")
    model = load_checkpoint_model(checkpoint, device)
    settings = checkpoint["config"]
    size = patch_size if patch_size is not None else settings["patch_size"]
    context = halo if halo is not None else settings["halo"]
    with open_image(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    npy_path = output_path.with_suffix(".npy")
    amp_kwargs = {"amp": settings["amp"], "amp_dtype": settings.get("amp_dtype", "float16")}

    if architecture == "archive_unet_v1":
        from prostate.archive_inference import predict_archive_tiled

        if patch_size is not None or halo is not None:
            raise ValueError(
                "Archive prediction uses its saved inference_tile_size, not patch-size/halo"
            )
        labels = predict_archive_tiled(model, rgb, npy_path, settings, device)
    elif architecture in ("unet_v1", "unet_groupnorm_v2"):
        labels = predict_tiled(model, rgb, npy_path, size, context, device, **amp_kwargs)
    elif architecture == "token_decoder_v1":
        from prostate.extractor import load_gigapath_flash

        encoder, mean, std = load_gigapath_flash(device)
        labels = predict_tiled_with_encoder(
            encoder, mean, std, model, rgb, npy_path, size, context, device, **amp_kwargs
        )
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")

    save_label_png(labels, output_path)
    return output_path
