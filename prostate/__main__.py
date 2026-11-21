"""CLI entry point: python -m prostate {prepare,cv,evaluate,predict}."""

import argparse
import json
from pathlib import Path

import torch

from prostate.config import Config
from prostate.inference import predict_image
from prostate.pipeline import evaluate_run, prepare_run, run_cv
from prostate.train import resolve_device


def _training_arguments(parser: argparse.ArgumentParser):
    defaults = Config()
    for name in ("data_dir", "cache_dir", "run_dir"):
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=Path, default=getattr(defaults, name)
        )
    for name in (
        "n_splits",
        "seed",
        "epochs",
        "samples_per_epoch",
        "patch_size",
        "halo",
        "batch_size",
        "base_channels",
        "num_workers",
        "cpu_threads",
    ):
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name)
        )
    for name in ("learning_rate", "weight_decay"):
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name)
        )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-amp", action="store_false", dest="amp", default=True)
    parser.add_argument(
        "--normalization", choices=("group", "none"), default=defaults.normalization
    )
    parser.add_argument(
        "--amp-dtype", choices=("auto", "bfloat16", "float16"), default=defaults.amp_dtype
    )


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="U-Net grouped CV and OOF segmentation baseline")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Cache RGB/labels and save case-level folds")
    _training_arguments(prepare)
    cv = commands.add_parser("cv", help="Train fixed-epoch folds and save held-out predictions")
    _training_arguments(cv)
    cv.add_argument("--fold", type=int, action="append", help="Only run this fold (repeatable)")
    cv.add_argument(
        "--resume", action="store_true", help="Skip completed folds; restart incomplete ones"
    )
    evaluate = commands.add_parser("evaluate", help="Recompute OOF metrics from completed folds")
    evaluate.add_argument("--run-dir", type=Path, default=Config().run_dir)
    predict = commands.add_parser("predict", help="Predict a JPEG using a fold checkpoint")
    predict.add_argument("image", type=Path)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    predict.add_argument("--patch-size", type=int)
    predict.add_argument("--halo", type=int)
    predict.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        if args.command in ("prepare", "cv"):
            config = Config(**{name: getattr(args, name) for name in Config.__dataclass_fields__})
            if args.command == "prepare":
                samples, _ = prepare_run(config)
                print(f"Prepared {len(samples)} images / {len({s.case_id for s in samples})} cases")
                return
            summary = run_cv(config, args.fold, args.resume)
        elif args.command == "evaluate":
            summary = evaluate_run(args.run_dir)
        else:
            if args.cpu_threads < 1:
                raise ValueError("cpu_threads must be positive")
            torch.set_num_threads(args.cpu_threads)
            path = predict_image(
                args.checkpoint,
                args.image,
                args.output,
                resolve_device(args.device),
                args.patch_size,
                args.halo,
            )
            print(f"Saved: {path}")
            return
        print(
            json.dumps(
                {
                    "complete": summary["complete"],
                    "images_evaluated": summary["images_evaluated"],
                    "cases_evaluated": summary["cases_evaluated"],
                    "primary_case_mean": summary["primary_case_mean"],
                },
                indent=2,
            )
        )
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
