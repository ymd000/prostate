"""Case-grouped CV or fixed holdout splits and saved assignment validation."""

import csv
from pathlib import Path

import numpy as np

from prostate.data import Sample


def assign_folds(samples: list[Sample], n_splits: int = 5, seed: int = 42) -> dict[str, int]:
    groups = {}
    for sample in samples:
        groups.setdefault(sample.case_id, []).append(sample.image_id)
    if not 2 <= n_splits <= len(groups):
        raise ValueError(f"Need at least {n_splits} cases; found {len(groups)}")
    case_ids = sorted(groups)
    np.random.default_rng(seed).shuffle(case_ids)
    # Randomize ties, then place larger cases first. Equal case counts take
    # precedence over image counts so one six-image case cannot occupy a fold alone.
    case_ids.sort(key=lambda case: -len(groups[case]))
    case_counts = [0] * n_splits
    image_counts = [0] * n_splits
    assignment = {}
    for case in case_ids:
        fold = min(range(n_splits), key=lambda f: (case_counts[f], image_counts[f], f))
        case_counts[fold] += 1
        image_counts[fold] += len(groups[case])
        assignment.update({image_id: fold for image_id in groups[case]})
    return assignment


def assign_holdout(samples: list[Sample], case_id: str) -> dict[str, int]:
    """Use model 0 for holdout predictions and -1 for training-only images."""
    cases = {sample.case_id for sample in samples}
    if case_id not in cases:
        raise ValueError(f"Holdout case not found: {case_id}")
    if cases == {case_id}:
        raise ValueError("Holdout requires at least one training case")
    return {sample.image_id: 0 if sample.case_id == case_id else -1 for sample in samples}


def evaluation_folds(settings: dict) -> range:
    return range(1 if settings.get("holdout_case_id") else settings["n_splits"])


def prediction_split(settings: dict) -> str:
    return "holdout" if settings.get("holdout_case_id") else "oof"


def read_assignments(run_dir: Path, settings: dict, samples: list[Sample]) -> dict[str, int]:
    """Check saved image assignments and preserve case separation before evaluating."""
    holdout = settings.get("holdout_case_id")
    with (run_dir / ("split.csv" if holdout else "folds.csv")).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if holdout:
        if any(row["split"] not in ("train", "holdout") for row in rows):
            raise ValueError("Invalid saved holdout split")
        assignments = {row["image_id"]: 0 if row["split"] == "holdout" else -1 for row in rows}
    else:
        assignments = {row["image_id"]: int(row["fold"]) for row in rows}
    ids = {sample.image_id for sample in samples}
    if len(ids) != len(samples) or len(rows) != len(ids) or set(assignments) != ids:
        raise ValueError("Saved split assignments do not match the manifest")
    case_ids = {sample.image_id: sample.case_id for sample in samples}
    if any(row["case_id"] != case_ids[row["image_id"]] for row in rows):
        raise ValueError("Saved case IDs do not match the manifest")
    if holdout:
        if assignments != assign_holdout(samples, holdout):
            raise ValueError("Saved holdout split does not match the configured case")
    else:
        case_folds = {}
        for sample in samples:
            fold = assignments[sample.image_id]
            if fold not in evaluation_folds(settings):
                raise ValueError(f"Invalid saved fold: {fold}")
            if case_folds.setdefault(sample.case_id, fold) != fold:
                raise ValueError(f"Case is split across folds: {sample.case_id}")
    return assignments


def validate_checkpoint_split(checkpoint: dict, training: list[Sample], fold: int):
    if checkpoint["fold"] != fold:
        raise ValueError("Checkpoint fold mismatch")
    if (
        set(checkpoint["train_image_ids"]) != {sample.image_id for sample in training}
        or len(checkpoint["train_image_ids"]) != len(training)
        or set(checkpoint["train_case_ids"]) != {sample.case_id for sample in training}
    ):
        raise ValueError("Checkpoint training split does not match the saved assignments")
