"""Command-line entry points for CV, fixed holdout and frozen feature extraction."""

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch

from prostate.config import (
    ARCHIVE_MODELS,
    DEFAULT_HOLDOUT_CASE_ID,
    DEFAULT_HOLDOUT_RUN_DIR,
    ArchiveConfig,
    ArchiveEvaluationConfig,
    Config,
)
from prostate.inference import predict_image
from prostate.pipeline import (
    evaluate_archive_checkpoint,
    evaluate_run,
    extract_run,
    prepare_run,
    run_cv,
    run_holdout,
)
from prostate.thumbnails import generate_thumbnails
from prostate.train import resolve_device
from prostate.training_evaluation import evaluate_training_run


def _training_arguments(parser: argparse.ArgumentParser, include_folds: bool = True):
    defaults = Config()
    parser.set_defaults(holdout_case_id=None, n_splits=defaults.n_splits)
    if include_folds:
        parser.add_argument("--n-splits", type=int, default=defaults.n_splits)
    for name in ("data_dir", "cache_dir", "run_dir"):
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=Path, default=getattr(defaults, name)
        )
    for name in (
        "seed",
        "epochs",
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
    parser.add_argument("--token-dir", type=Path, default=defaults.token_dir)
    parser.add_argument("--encoder", choices=("none", "gigapath-flash"), default=defaults.encoder)
    parser.add_argument(
        "--class-weighting",
        choices=("none", "inverse-sqrt"),
        default=defaults.class_weighting,
        help="Cross-entropy class weights computed from training-fold pixels only",
    )


def _archive_arguments(parser: argparse.ArgumentParser, include_holdout: bool = True):
    defaults = ArchiveConfig()
    for name in ("data_dir", "cache_dir", "run_dir"):
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=Path, default=getattr(defaults, name)
        )
    for name in (
        "patch_size",
        "epochs",
        "samples_multiplier",
        "max_sampling_attempts",
        "inference_tile_size",
        "seed",
        "num_workers",
        "cpu_threads",
    ):
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name)
        )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Default: 256px=32, 512px=8, 768px=4"
    )
    parser.add_argument(
        "--crop-start-step",
        type=int,
        default=defaults.crop_start_step,
        help="Random crop start-coordinate spacing in pixels (default: 1; use 16 for token alignment)",
    )
    for name in ("learning_rate", "momentum", "lr_decay"):
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name)
        )
    parser.add_argument(
        "--model", dest="model_name", choices=ARCHIVE_MODELS, default=defaults.model_name
    )
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=True)
    if include_holdout:
        parser.set_defaults(n_splits=defaults.n_splits)
        parser.add_argument("--holdout-case-id", default=defaults.holdout_case_id)
    else:
        parser.set_defaults(holdout_case_id=None, run_dir=Path("outputs/unet11-archive-cv"))
        parser.add_argument("--n-splits", type=int, default=defaults.n_splits)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=defaults.device)
    parser.add_argument("--validation-mode", choices=("eval", "archive-train"), default="eval")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")


def main(argv: list[str] | None = None):
    # Select the CV argument set first so recipe defaults are applied before explicit
    # overrides, and options belonging to the other recipe are rejected by argparse.
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--recipe", default="full-grid")
    recipe = selector.parse_known_args(argv)[0].recipe
    parser = argparse.ArgumentParser(
        description="Case-grouped CV or holdout, frozen encoders and segmentation"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Cache RGB/labels and save case-level folds")
    _training_arguments(prepare)
    extract = commands.add_parser(
        "extract", help="Run gigapath-flash on cached images to build token h5"
    )
    _training_arguments(extract)
    extract.add_argument(
        "--extract-batch-size", type=int, default=8, help="Encoder forward-pass batch size"
    )
    cv = commands.add_parser(
        "cv", help="Train fixed-epoch folds and save held-out predictions", allow_abbrev=False
    )
    cv.add_argument(
        "--recipe",
        choices=("full-grid", "archive"),
        default="full-grid",
        help="Training conditions; use cv --recipe archive --help for historical options",
    )
    if recipe == "archive":
        _archive_arguments(cv, include_holdout=False)
    else:
        _training_arguments(cv)
        cv.add_argument(
            "--resume", action="store_true", help="Skip completed folds; restart incomplete ones"
        )
    cv.add_argument("--fold", type=int, action="append", help="Only run this fold (repeatable)")
    holdout = commands.add_parser("holdout", help="Train once with a fixed case held out")
    _training_arguments(holdout, include_folds=False)
    holdout.set_defaults(run_dir=DEFAULT_HOLDOUT_RUN_DIR)
    holdout.add_argument("--holdout-case-id", default=DEFAULT_HOLDOUT_CASE_ID)
    holdout.add_argument(
        "--prepare-only", action="store_true", help="Save the split without training"
    )
    holdout.add_argument("--resume", action="store_true", help="Reuse a completed holdout model")
    archive = commands.add_parser(
        "archive-holdout", help="Verify archived U-Net training conditions"
    )
    _archive_arguments(archive)
    archive_evaluation = commands.add_parser(
        "archive-evaluate", help="Evaluate a final archive checkpoint without training"
    )
    archive_evaluation.add_argument("--run-dir", type=Path, required=True)
    archive_evaluation.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    archive_evaluation.add_argument("--inference-tile-size", type=int)
    archive_evaluation.add_argument("--validation-mode", choices=("eval", "archive-train"))
    archive_evaluation.add_argument("--output-dir", type=Path)
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate saved fold models; infer missing archive metrics"
    )
    evaluate.add_argument("--run-dir", type=Path, default=Config().run_dir)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    evaluate.add_argument(
        "--saved-only",
        action="store_true",
        help="Only aggregate saved predictions and metrics; never run inference",
    )
    training_evaluation = commands.add_parser(
        "evaluate-train",
        help="Evaluate saved fold models on their training images without training",
    )
    training_evaluation.add_argument("--run-dir", type=Path, default=Config().run_dir)
    training_evaluation.add_argument("--output-dir", type=Path)
    training_evaluation.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    training_evaluation.add_argument("--fold", type=int, action="append")
    thumbnails = commands.add_parser("thumbnails", help="Render saved OOF comparisons on CPU")
    thumbnails.add_argument("--run-dir", type=Path, default=Config().run_dir)
    thumbnails.add_argument("--output-dir", type=Path)
    thumbnails.add_argument("--max-side", type=int, default=768)
    thumbnails.add_argument("--contact-side", type=int, default=320)
    thumbnails.add_argument("--jpeg-quality", type=int, default=95)
    thumbnails.add_argument("--per-page", type=int, default=8)
    thumbnails.add_argument("--sort", choices=("worst", "best", "id"), default="worst")
    thumbnails.add_argument("--limit", type=int)
    thumbnails.add_argument(
        "--image-id", action="append", help="Only render this image (repeatable)"
    )
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
        if args.command == "archive-evaluate":
            config = ArchiveEvaluationConfig(
                **{item.name: getattr(args, item.name) for item in fields(ArchiveEvaluationConfig)}
            )
            result = evaluate_archive_checkpoint(config)
            print(
                json.dumps(
                    {
                        key: result[key]
                        for key in (
                            "complete",
                            "output_dir",
                            "images_evaluated",
                            "cases_evaluated",
                            "primary_case_mean",
                            "archive_metrics",
                        )
                    },
                    indent=2,
                )
            )
            return
        if args.command == "evaluate-train":
            summary = evaluate_training_run(args.run_dir, args.device, args.fold, args.output_dir)
            print(
                json.dumps(
                    {
                        key: summary[key]
                        for key in ("complete", "output_dir", "folds_evaluated", "missing_folds")
                    },
                    indent=2,
                )
            )
            return
        if args.command == "thumbnails":
            result = generate_thumbnails(
                args.run_dir,
                output_dir=args.output_dir,
                max_side=args.max_side,
                per_page=args.per_page,
                sort=args.sort,
                limit=args.limit,
                image_ids=args.image_id,
                contact_side=args.contact_side,
                jpeg_quality=args.jpeg_quality,
            )
            print(json.dumps(result, indent=2))
            return
        if args.command in ("prepare", "extract", "cv", "holdout", "archive-holdout"):
            archive_recipe = args.command == "archive-holdout" or (
                args.command == "cv" and args.recipe == "archive"
            )
            config_type = ArchiveConfig if archive_recipe else Config
            config = config_type(
                **{item.name: getattr(args, item.name) for item in fields(config_type) if item.init}
            )
            if args.command == "prepare" or (
                args.command == "cv" and archive_recipe and args.prepare_only
            ):
                samples, _ = prepare_run(config)
                print(f"Prepared {len(samples)} images / {len({s.case_id for s in samples})} cases")
                return
            if args.command == "extract":
                extract_run(config, args.extract_batch_size)
                return
            if args.command in ("holdout", "archive-holdout"):
                if args.prepare_only:
                    samples, assignment = prepare_run(config)
                    print(
                        json.dumps(
                            {
                                "holdout_case_id": config.holdout_case_id,
                                "train_images": [
                                    s.image_id for s in samples if assignment[s.image_id] == -1
                                ],
                                "holdout_images": [
                                    s.image_id for s in samples if assignment[s.image_id] == 0
                                ],
                            },
                            indent=2,
                        )
                    )
                    return
                summary = run_holdout(config, args.resume)
            else:
                summary = run_cv(config, args.fold, args.resume)
        elif args.command == "evaluate":
            summary = evaluate_run(
                args.run_dir, infer_missing=not args.saved_only, requested_device=args.device
            )
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
                }
                | (
                    {"archive_metrics": summary["archive_metrics"]}
                    if "archive_metrics" in summary
                    else {}
                ),
                indent=2,
            )
        )
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
