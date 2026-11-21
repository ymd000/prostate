"""Fixed-epoch training of one fold; held-out labels are never used for selection."""

import csv
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from prostate.config import Config
from prostate.data import IGNORE_INDEX, NUM_CLASSES, PatchDataset, Sample
from prostate.model import UNet
from prostate.precision import resolve_precision


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Install a CUDA-enabled PyTorch or use --device cpu.")
    return torch.device(requested)


def seed_training(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_fold(samples: list[Sample], config: Config, fold: int, device: torch.device) -> UNet:
    seed = config.seed + fold
    seed_training(seed)
    torch.set_num_threads(config.cpu_threads)
    model = UNet(NUM_CLASSES, config.base_channels, config.normalization).to(device)
    dataset = PatchDataset(samples, config.patch_size, config.samples_per_epoch, seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    amp_enabled, amp_dtype = resolve_precision(device, config.amp, config.amp_dtype)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled and amp_dtype == torch.float16)
    fold_dir = config.run_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    with (fold_dir / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "seconds"])
        writer.writeheader()
        for epoch in range(config.epochs):
            dataset.epoch = epoch
            model.train()
            loss_sum, count = 0.0, 0
            started = time.monotonic()
            progress = tqdm(loader, desc=f"Fold {fold} epoch {epoch + 1}/{config.epochs}")
            for batch_index, (images, targets) in enumerate(progress, start=1):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                    logits = model(images)
                # Keep the loss in FP32 for both BF16 and FP16 autocast.
                loss = criterion(logits.float(), targets)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite training loss in fold {fold}, epoch {epoch + 1}, "
                        f"batch {batch_index} (precision={amp_dtype}, "
                        f"finite_logits={bool(torch.isfinite(logits).all())}). "
                        "Retry with --no-amp and a new --run-dir to isolate AMP overflow."
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=not scaler.is_enabled()
                )
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item() * len(images)
                count += len(images)
                progress.set_postfix(loss=f"{loss_sum / count:.4f}")
            writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": loss_sum / count,
                    "seconds": time.monotonic() - started,
                }
            )
            stream.flush()
    checkpoint = {
        "architecture": "unet_groupnorm_v2" if config.normalization == "group" else "unet_v1",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model": {
            "num_classes": NUM_CLASSES,
            "base_channels": config.base_channels,
            "normalization": config.normalization,
        },
        "precision": str(amp_dtype),
        "config": config.to_dict(),
        "fold": fold,
        "train_image_ids": [sample.image_id for sample in samples],
        "train_case_ids": sorted({sample.case_id for sample in samples}),
    }
    path = fold_dir / "checkpoint.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    return model
