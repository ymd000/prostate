"""Archived balanced tiles, centered zero padding and optional train-mode BatchNorm."""

from pathlib import Path

import numpy as np
import torch

from prostate.archive_data import archive_image_tensor
from prostate.archive_metrics import ArchiveMetrics, save_archive_prediction_metrics


def archive_tile_positions(height: int, width: int, maximum: int):
    if min(height, width, maximum) < 1:
        raise ValueError("Archive image and tile dimensions must be positive")
    rows, columns = -(-height // maximum), -(-width // maximum)
    heights = [(height + i) // rows for i in range(rows)]
    widths = [(width + i) // columns for i in range(columns)]
    top = 0
    for h in heights:
        left = 0
        for w in widths:
            yield top, left, h, w
            left += w
        top += h


def predict_archive_tiled(
    model: torch.nn.Module,
    rgb: np.ndarray,
    output_path: Path,
    settings: dict,
    device: torch.device,
    target: np.ndarray | None = None,
) -> np.ndarray:
    height, width = rgb.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".partial.npy")
    predictions = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.uint8, shape=(height, width)
    )
    archive_train = settings["validation_mode"] == "archive-train"
    was_training = model.training
    # The historical validation kept BatchNorm in training mode. Restore buffers
    # so this optional comparison never changes the checkpoint or later diagnostics.
    buffers = (
        {name: value.clone() for name, value in model.named_buffers()} if archive_train else {}
    )
    model.train(archive_train)
    metrics = ArchiveMetrics() if target is not None else None
    try:
        with torch.inference_mode():
            for top, left, h, w in archive_tile_positions(
                height, width, settings["inference_tile_size"]
            ):
                tile = np.asarray(rgb[top : top + h, left : left + w])
                padded_h, padded_w = -(-h // 64) * 64, -(-w // 64) * 64
                y, x = (padded_h - h) // 2, (padded_w - w) // 2
                padded = np.pad(tile, ((y, padded_h - h - y), (x, padded_w - w - x), (0, 0)))
                image = archive_image_tensor(padded).unsqueeze(0).to(device)
                logits = model(image)[:, :, y : y + h, x : x + w]
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Non-finite logits in archive inference")
                predictions[top : top + h, left : left + w] = logits.argmax(1)[0].cpu().numpy()
                if metrics is not None:
                    labels = torch.from_numpy(
                        np.array(target[top : top + h, left : left + w])
                    ).long()
                    metrics.update(logits, labels.unsqueeze(0).to(device))
        predictions.flush()
    finally:
        if buffers:
            with torch.no_grad():
                for name, value in model.named_buffers():
                    value.copy_(buffers[name])
        model.train(was_training)
        del predictions
    temporary.replace(output_path)
    if metrics is not None:
        save_archive_prediction_metrics(
            output_path,
            metrics,
            tile_region="unpadded_tile",
            image_shape=[height, width],
            validation_mode=settings["validation_mode"],
            inference_tile_size=settings["inference_tile_size"],
        )
    return np.load(output_path, mmap_mode="r")
