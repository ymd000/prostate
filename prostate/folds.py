"""Reproducible grouped folds, balancing the number of cases first."""

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
