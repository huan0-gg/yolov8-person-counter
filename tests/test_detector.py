"""检测后处理（NMS / Soft-NMS / WBF / 切片窗口）的单元测试。

这些函数不依赖模型权重，可以直接构造框来验证行为边界。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detector import (  # noqa: E402
    ioS,
    nms_detections,
    soft_nms_detections,
    suppress_contained,
    tile_windows,
    wbf_detections,
)


def _det(bbox, conf):
    return {"bbox": bbox, "conf": conf, "class": 0, "class_name": "person"}


def test_nms_removes_duplicate():
    dets = [_det((0, 0, 100, 200), 0.9), _det((2, 2, 102, 202), 0.8), _det((300, 0, 400, 200), 0.7)]
    kept = nms_detections(dets, 0.5)
    assert len(kept) == 2
    assert kept[0]["conf"] == 0.9


def test_nms_keeps_separated_boxes():
    dets = [_det((0, 0, 100, 200), 0.9), _det((120, 0, 220, 200), 0.85)]
    assert len(nms_detections(dets, 0.5)) == 2


def test_soft_nms_keeps_duplicate_but_decays_score():
    dets = [_det((0, 0, 100, 200), 0.9), _det((2, 2, 102, 202), 0.8)]
    kept = soft_nms_detections(dets, iou_threshold=0.5, sigma=0.5, score_thr=0.001)
    assert len(kept) == 2
    assert kept[1]["conf"] < 0.8


def test_wbf_merges_duplicates_to_weighted_midpoint():
    dets = [_det((0, 0, 100, 200), 0.9), _det((10, 0, 110, 200), 0.3)]
    fused = wbf_detections(dets, 0.55)
    assert len(fused) == 1
    x1 = fused[0]["bbox"][0]
    # 0.9/1.2 * 0 + 0.3/1.2 * 10 = 2.5
    assert x1 == pytest.approx(2.5, abs=0.01)
    assert fused[0]["conf"] == pytest.approx(0.9)
    assert fused[0]["merged"] == 2


def test_wbf_keeps_distinct_people():
    dets = [_det((0, 0, 100, 200), 0.9), _det((150, 0, 250, 200), 0.8)]
    assert len(wbf_detections(dets, 0.55)) == 2


def test_wbf_center_updated_after_fusion():
    dets = [_det((0, 0, 100, 200), 0.8), _det((20, 20, 120, 220), 0.6)]
    fused = wbf_detections(dets, 0.5)[0]
    x1, y1, x2, y2 = fused["bbox"]
    assert fused["center"] == pytest.approx(((x1 + x2) / 2, (y1 + y2) / 2))


def test_tile_windows_cover_image_and_stay_inside():
    w, h = 1536, 1024
    wins = tile_windows(w, h, 2, 2, 0.2)
    assert len(wins) == 4
    for x0, y0, x1, y1 in wins:
        assert 0 <= x0 < x1 <= w
        assert 0 <= y0 < y1 <= h
    assert min(x0 for x0, _, _, _ in wins) == 0
    assert min(y0 for _, y0, _, _ in wins) == 0
    assert max(x1 for _, _, x1, _ in wins) == w
    assert max(y1 for _, _, _, y1 in wins) == h


def test_tile_windows_have_overlap():
    wins = tile_windows(1000, 1000, 2, 2, 0.25)
    right_top = [w for w in wins if w[0] > 0 and w[1] == 0][0]
    left_top = [w for w in wins if w[0] == 0 and w[1] == 0][0]
    assert left_top[2] > right_top[0]  # 横向有重叠
    assert right_top[2] == 1000


def test_tile_windows_single_tile():
    assert tile_windows(640, 480, 1, 1, 0.2) == [(0, 0, 640, 480)]


def test_ios_detects_nested_boxes():
    big = (0, 0, 100, 200)
    head = (30, 0, 70, 40)  # 完全落在 big 内
    assert ioS(head, big) == pytest.approx(1.0)
    assert ioS((0, 0, 100, 200), (200, 200, 300, 300)) == 0.0


def test_suppress_contained_removes_fragment_keeps_full_body():
    full = _det((0, 0, 100, 200), 0.8)
    head = _det((30, 0, 70, 40), 0.6)
    other = _det((300, 0, 400, 200), 0.7)
    kept = suppress_contained([full, head, other], 0.9)
    assert len(kept) == 2
    assert full in kept and other in kept


def test_suppress_contained_disabled():
    full = _det((0, 0, 100, 200), 0.8)
    head = _det((30, 0, 70, 40), 0.6)
    assert len(suppress_contained([full, head], 0.0)) == 2
