"""Historical random crops, full-model SGD and fixed-epoch training."""

import csv
import json
import time

import torch
from torch.nn import functional as F
from tqdm import tqdm

from prostate.archive_data import ArchivePatchDataset
from prostate.archive_metrics import ARCHIVE_METRICS
from prostate.archive_model import build_archive_model
from prostate.config import ArchiveConfig
from prostate.data import IGNORE_INDEX, NUM_CLASSES, Sample
from prostate.loss import prepare_cross_entropy
from prostate.train import seed_training, training_loader
from prostate.training_history import ONLINE_PROTOCOL, EpochMetrics, TrainingHistory


def archive_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.nll_loss((logits.softmax(dim=1) + 1e-24).log(), targets, ignore_index=IGNORE_INDEX)


def train_archive_fold(
    samples: list[Sample], config: ArchiveConfig, fold: int, device: torch.device
) -> torch.nn.Module:
    seed_training(config.seed + fold)
    torch.set_num_threads(config.cpu_threads)
    dataset = ArchivePatchDataset(samples, config)
    loader = training_loader(dataset, config, fold, device)
    model = build_archive_model(
        config.model_name, pretrained=config.pretrained, cache_dir=config.cache_dir / "pretrained"
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=0
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: config.lr_decay**epoch)
    _, loss_settings = prepare_cross_entropy(samples, config, fold, device)
    loss_settings["implementation"] = "softmax_add_1e-24_log_nll"
    folder = config.run_dir / f"fold_{fold}"
    (folder / "loss.json").write_text(json.dumps(loss_settings, indent=2, allow_nan=False) + "\n")
    with (
        TrainingHistory(folder) as history,
        (folder / "archive_history.csv").open("w", newline="") as stream,
        (folder / "archive_history_metrics.jsonl").open("w") as json_stream,
    ):
        writer = csv.DictWriter(stream, fieldnames=["epoch", "learning_rate", *ARCHIVE_METRICS])
        writer.writeheader()
        for epoch in range(config.epochs):
            dataset.epoch = epoch
            model.train()
            metrics = EpochMetrics(device)
            loss_sum, count, batch_losses = 0.0, 0, []
            started = time.monotonic()
            lr = optimizer.param_groups[0]["lr"]
            for images, targets in tqdm(loader, desc=f"Archive epoch {epoch + 1}/{config.epochs}"):
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = archive_loss(logits, targets)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite archive loss at epoch {epoch + 1}")
                metrics.update(logits, targets)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(images)
                count += len(images)
                batch_losses.append(loss.item())
            history.log(
                epoch + 1, loss_sum / count, time.monotonic() - started, count, len(loader), metrics
            )
            report = metrics.archive_metrics.report("batch")
            writer.writerow({"epoch": epoch + 1, "learning_rate": lr, **report["mean"]})
            json_stream.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "learning_rate": lr,
                        "mean_batch_loss": sum(batch_losses) / len(batch_losses),
                        **report,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            stream.flush()
            json_stream.flush()
            scheduler.step()
    checkpoint = {
        "architecture": "archive_unet_v1",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model": {"num_classes": NUM_CLASSES, "model_name": config.model_name},
        "config": config.to_dict(),
        "fold": fold,
        "train_image_ids": [sample.image_id for sample in samples],
        "train_case_ids": sorted({sample.case_id for sample in samples}),
        "precision": "torch.float32",
        "patches_per_epoch": len(dataset),
        "batches_per_epoch": len(loader),
        "history_metrics_protocol": ONLINE_PROTOCOL,
        "optimizer": {"name": "SGD", "momentum": config.momentum, "weight_decay": 0.0},
        "scheduler": {"name": "LambdaLR", "factor_per_epoch": config.lr_decay},
        "loss": loss_settings,
    }
    temporary = folder / "checkpoint.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(folder / "checkpoint.pt")
    return model
