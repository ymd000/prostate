"""Fixed-epoch training of one fold; held-out labels are never used for selection."""

import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from prostate.config import ArchiveConfig, Config
from prostate.data import NUM_CLASSES, PatchDataset, Sample
from prostate.loss import prepare_cross_entropy
from prostate.model import TokenDecoder, UNet
from prostate.precision import resolve_precision
from prostate.token_data import TokenPatchDataset
from prostate.training_history import ONLINE_PROTOCOL, EpochMetrics, TrainingHistory


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


def training_loader(
    dataset: Dataset, config: Config | ArchiveConfig, fold: int, device: torch.device
) -> DataLoader:
    """Shuffle without replacement; retain the final, possibly smaller batch."""
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(config.seed + fold),
    )
    sampling = (
        "archive random crops" if config.patch_sampling == "archive_random_v1" else "full grid"
    )
    print(f"Fold {fold}: {sampling}, {len(dataset)} patches/epoch, {len(loader)} batches/epoch")
    return loader


def train_fold(samples: list[Sample], config: Config, fold: int, device: torch.device) -> UNet:
    seed = config.seed + fold
    seed_training(seed)
    torch.set_num_threads(config.cpu_threads)
    model = UNet(NUM_CLASSES, config.base_channels, config.normalization).to(device)
    dataset = PatchDataset(samples, config.patch_size, seed)
    loader = training_loader(dataset, config, fold, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion, loss_settings = prepare_cross_entropy(samples, config, fold, device)
    amp_enabled, amp_dtype = resolve_precision(device, config.amp, config.amp_dtype)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled and amp_dtype == torch.float16)
    fold_dir = config.run_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    with TrainingHistory(fold_dir) as history:
        for epoch in range(config.epochs):
            dataset.epoch = epoch
            model.train()
            loss_sum, count = 0.0, 0
            metrics = EpochMetrics(device)
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
                metrics.update(logits, targets)
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
            history.log(
                epoch + 1, loss_sum / count, time.monotonic() - started, count, len(loader), metrics
            )
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
        "patches_per_epoch": len(dataset),
        "batches_per_epoch": len(loader),
        "history_metrics_protocol": ONLINE_PROTOCOL,
        "loss": loss_settings,
    }
    path = fold_dir / "checkpoint.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    return model


def train_token_fold(
    samples: list[Sample], config: Config, fold: int, device: torch.device
) -> TokenDecoder:
    seed = config.seed + fold
    seed_training(seed)
    torch.set_num_threads(config.cpu_threads)
    model = TokenDecoder(NUM_CLASSES, config.base_channels, config.normalization).to(device)
    dataset = TokenPatchDataset(samples, config.token_dir, config.patch_size, seed)
    loader = training_loader(dataset, config, fold, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion, loss_settings = prepare_cross_entropy(samples, config, fold, device)
    amp_enabled, amp_dtype = resolve_precision(device, config.amp, config.amp_dtype)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled and amp_dtype == torch.float16)
    fold_dir = config.run_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    with TrainingHistory(fold_dir) as history:
        for epoch in range(config.epochs):
            dataset.epoch = epoch
            model.train()
            loss_sum, count = 0.0, 0
            metrics = EpochMetrics(device)
            started = time.monotonic()
            progress = tqdm(loader, desc=f"Fold {fold} epoch {epoch + 1}/{config.epochs}")
            for batch_index, (tokens, targets) in enumerate(progress, start=1):
                tokens = tokens.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                    logits = model(tokens)
                loss = criterion(logits.float(), targets)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite training loss in fold {fold}, epoch {epoch + 1}, "
                        f"batch {batch_index}"
                    )
                metrics.update(logits, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=not scaler.is_enabled()
                )
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item() * len(tokens)
                count += len(tokens)
                progress.set_postfix(loss=f"{loss_sum / count:.4f}")
            history.log(
                epoch + 1, loss_sum / count, time.monotonic() - started, count, len(loader), metrics
            )
    checkpoint = {
        "architecture": "token_decoder_v1",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model": {
            "num_classes": NUM_CLASSES,
            "base_channels": config.base_channels,
            "normalization": config.normalization,
        },
        "encoder": config.encoder,
        "precision": str(amp_dtype),
        "config": config.to_dict(),
        "fold": fold,
        "train_image_ids": [sample.image_id for sample in samples],
        "train_case_ids": sorted({sample.case_id for sample in samples}),
        "patches_per_epoch": len(dataset),
        "batches_per_epoch": len(loader),
        "history_metrics_protocol": ONLINE_PROTOCOL,
        "loss": loss_settings,
    }
    path = fold_dir / "checkpoint.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    return model
