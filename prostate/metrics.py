"""Confusion-based segmentation metrics with explicit aggregation units."""

from collections import defaultdict

import numpy as np

from prostate.data import CLASS_NAMES, NUM_CLASSES


def confusion_matrix(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("Target and prediction must be equally sized 2-D class-ID arrays")
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for top in range(0, target.shape[0], 256):
        truth = np.asarray(target[top : top + 256])
        predicted = np.asarray(prediction[top : top + 256])
        if truth.min() < 0 or predicted.min() < 0:
            raise ValueError("Class IDs must be nonnegative")
        if truth.max() >= NUM_CLASSES or predicted.max() >= NUM_CLASSES:
            raise ValueError("Class IDs must be in 0..4")
        codes = truth.astype(np.int64) * NUM_CLASSES + predicted
        matrix += np.bincount(codes.ravel(), minlength=NUM_CLASSES**2).reshape(
            NUM_CLASSES, NUM_CLASSES
        )
    return matrix


def mean_defined(values) -> float | None:
    valid = [value for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def scores(matrix: np.ndarray) -> dict:
    matrix = np.asarray(matrix, dtype=np.int64)
    true_positive = np.diag(matrix)
    actual = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    dice, iou, precision, recall = {}, {}, {}, {}
    for index, name in enumerate(CLASS_NAMES):
        total = int(actual[index] + predicted[index])
        union = int(total - true_positive[index])
        dice[name] = 2 * int(true_positive[index]) / total if total else None
        iou[name] = int(true_positive[index]) / union if union else None
        precision[name] = (
            int(true_positive[index]) / int(predicted[index]) if predicted[index] else None
        )
        recall[name] = int(true_positive[index]) / int(actual[index]) if actual[index] else None
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "foreground_macro_dice": mean_defined(dice[name] for name in CLASS_NAMES[1:]),
        "foreground_macro_iou": mean_defined(iou[name] for name in CLASS_NAMES[1:]),
        "pixel_accuracy": float(true_positive.sum() / matrix.sum()) if matrix.sum() else None,
        "target_pixels": {name: int(actual[i]) for i, name in enumerate(CLASS_NAMES)},
        "predicted_pixels": {name: int(predicted[i]) for i, name in enumerate(CLASS_NAMES)},
        "true_positive_pixels": {name: int(true_positive[i]) for i, name in enumerate(CLASS_NAMES)},
    }


def mean_scores(reports: list[dict]) -> dict:
    return {
        metric: {
            name: mean_defined(report[metric][name] for report in reports) for name in CLASS_NAMES
        }
        for metric in ("dice", "iou", "precision", "recall")
    } | {
        metric: mean_defined(report[metric] for report in reports)
        for metric in ("foreground_macro_dice", "foreground_macro_iou", "pixel_accuracy")
    }


def summarize_predictions(records: list[dict], expected_ids: list[str]) -> dict:
    """Aggregate one prediction per image, weighting cases equally in the primary report."""
    image_ids = [record["image_id"] for record in records]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Duplicate image predictions")
    if set(image_ids) - set(expected_ids):
        raise ValueError("Predictions contain unknown images")
    cases = defaultdict(list)
    total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    image_reports = []
    for record in records:
        matrix = np.asarray(record["confusion_matrix"], dtype=np.int64)
        total += matrix
        cases[(record["fold"], record["case_id"])].append(matrix)
        image_reports.append(scores(matrix))
    case_reports = []
    for (fold, case_id), matrices in sorted(cases.items()):
        report = scores(np.sum(matrices, axis=0))
        case_reports.append({"case_id": case_id, "fold": fold, **report})
    missing = sorted(set(expected_ids) - set(image_ids))
    return {
        "complete": not missing,
        "images_evaluated": len(records),
        "cases_evaluated": len(case_reports),
        "missing_image_ids": missing,
        "primary_case_mean": mean_scores(case_reports),
        "image_mean": mean_scores(image_reports),
        "pooled_pixels": scores(total),
        "pooled_confusion_matrix": total.tolist(),
        "cases": case_reports,
        "undefined_class_policy": "Dice/IoU: null if absent in both truth and prediction; "
        "precision: null if no predictions; recall: null if no target pixels; null excluded from mean",
    }


def summarize_oof(records: list[dict], expected_ids: list[str], n_splits: int) -> dict:
    summary = summarize_predictions(records, expected_ids)
    folds = defaultdict(list)
    for report in summary["cases"]:
        folds[report["fold"]].append(report)
    fold_reports = {str(fold): mean_scores(reports) for fold, reports in sorted(folds.items())}
    dispersion = {}
    for metric in ("foreground_macro_dice", "foreground_macro_iou"):
        values = [report[metric] for report in fold_reports.values() if report[metric] is not None]
        dispersion[metric] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
        }
    return summary | {
        "complete": summary["complete"] and len(fold_reports) == n_splits,
        "fold_case_means": fold_reports,
        "fold_mean_std": dispersion,
    }
