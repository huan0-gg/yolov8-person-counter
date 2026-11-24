"""评测指标的单元测试：确保"准确率/误检/漏检"的算法本身是对的。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import compute_ap, iou, match_predictions, prf  # noqa: E402


def _pred(bbox, conf=0.9):
    return {"bbox": bbox, "conf": conf}


def test_iou_basics():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_perfect_detection():
    gt = [(0, 0, 10, 10), (20, 20, 30, 30)]
    preds = [_pred((0, 0, 10, 10)), _pred((20, 20, 30, 30))]
    flags, fn = match_predictions(preds, gt, 0.5)
    assert fn == 0
    assert all(f[1] for f in flags)
    m = prf(2, 0, 0)
    assert m == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "fp": 0, "fn": 0, "tp": 2}


def test_false_positive_counted():
    gt = [(0, 0, 10, 10)]
    preds = [_pred((0, 0, 10, 10)), _pred((50, 50, 60, 60))]
    flags, fn = match_predictions(preds, gt, 0.5)
    assert fn == 0
    assert flags.count((0.9, False)) == 1
    m = prf(1, 1, 0)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(1.0)


def test_missed_detection_counted():
    gt = [(0, 0, 10, 10), (20, 20, 30, 30)]
    preds = [_pred((0, 0, 10, 10))]
    flags, fn = match_predictions(preds, gt, 0.5)
    assert fn == 1
    assert len(flags) == 1
    m = prf(1, 0, 1)
    assert m["recall"] == pytest.approx(0.5)


def test_one_to_one_matching_prevents_double_count():
    gt = [(0, 0, 10, 10)]
    preds = [_pred((0, 0, 10, 10), 0.9), _pred((0, 0, 10, 10), 0.8)]
    flags, fn = match_predictions(preds, gt, 0.5)
    assert sum(1 for _, is_tp in flags if is_tp) == 1
    assert sum(1 for _, is_tp in flags if not is_tp) == 1
    assert fn == 0


def test_match_respects_confidence_order():
    gt = [(0, 0, 10, 10)]
    preds = [_pred((0, 0, 10, 10), 0.3), _pred((0, 0, 10, 10), 0.95)]
    flags, _ = match_predictions(preds, gt, 0.5)
    assert flags[0] == (0.95, True)


def test_ap_perfect_is_one():
    flags = [(0.9, True), (0.8, True)]
    assert compute_ap(flags, n_gt=2) == pytest.approx(1.0)


def test_ap_penalises_false_positives():
    flags = [(0.9, False), (0.8, True)]
    ap = compute_ap(flags, n_gt=1)
    assert 0.0 < ap < 0.6


def test_ap_with_no_gt_is_zero():
    assert compute_ap([(0.9, False)], n_gt=0) == 0.0
