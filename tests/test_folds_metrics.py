import numpy as np
import pytest

from prostate.data import Sample
from prostate.folds import assign_folds
from prostate.metrics import confusion_matrix, scores, summarize_oof


def sample(image_id, case_id):
    return Sample(image_id, case_id, "", "", "", "", 1, 1, (1, 0, 0, 0, 0), {})


def test_folds_are_reproducible_and_never_split_a_case():
    samples = [
        sample(f"case{i}_{j}", f"case{i}") for i in range(11) for j in range(6 if i == 0 else 1)
    ]
    assignment = assign_folds(samples, 5, seed=42)
    assert assignment == assign_folds(list(reversed(samples)), 5, seed=42)
    assert set(assignment.values()) == set(range(5))
    for case_id in {s.case_id for s in samples}:
        assert len({assignment[s.image_id] for s in samples if s.case_id == case_id}) == 1
    counts = [len({s.case_id for s in samples if assignment[s.image_id] == f}) for f in range(5)]
    assert max(counts) - min(counts) <= 1
    with pytest.raises(ValueError, match="cases"):
        assign_folds(samples, 12)


def test_metrics_handle_false_positives_and_absent_classes():
    truth = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.uint8)
    prediction = np.array([[0, 1, 0], [2, 3, 0]], dtype=np.uint8)
    matrix = confusion_matrix(truth, prediction)
    result = scores(matrix)
    assert matrix.sum() == 6
    assert result["dice"]["non_cancer"] == pytest.approx(2 / 3)
    assert result["iou"]["GP3"] == pytest.approx(1 / 2)
    assert result["dice"]["GP4"] == 0
    assert result["dice"]["GP5"] is None
    assert result["foreground_macro_dice"] == pytest.approx(4 / 9)
    assert result["precision"]["non_cancer"] == 1
    assert result["recall"]["non_cancer"] == 0.5
    assert result["precision"]["GP4"] == 0
    assert result["recall"]["GP4"] is None
    assert result["precision"]["GP5"] is None
    assert result["recall"]["GP5"] is None
    assert result["predicted_pixels"]["GP4"] == 1
    assert result["true_positive_pixels"]["GP4"] == 0
    with pytest.raises(ValueError, match="0..4"):
        confusion_matrix(truth, np.full_like(truth, 5))


def test_case_mean_is_not_dominated_by_large_images_and_partial_is_explicit():
    perfect = np.zeros((5, 5), dtype=np.int64)
    perfect[2, 2] = 10000
    failed = np.zeros((5, 5), dtype=np.int64)
    failed[2, 0] = 1
    records = [
        {"image_id": "A_1", "case_id": "A", "fold": 0, "confusion_matrix": perfect.tolist()},
        {"image_id": "B_1", "case_id": "B", "fold": 1, "confusion_matrix": failed.tolist()},
    ]
    summary = summarize_oof(records, ["A_1", "B_1", "C_1"], 3)
    assert not summary["complete"]
    assert summary["missing_image_ids"] == ["C_1"]
    assert summary["primary_case_mean"]["dice"]["GP3"] == 0.5
    assert summary["pooled_pixels"]["dice"]["GP3"] > 0.99
    assert summary["primary_case_mean"]["recall"]["GP3"] == 0.5
    assert summary["pooled_pixels"]["recall"]["GP3"] > 0.99
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_oof(records + records, ["A_1", "B_1"], 2)
