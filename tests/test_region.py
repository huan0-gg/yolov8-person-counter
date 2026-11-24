"""指定区域（ROI）几何与统计的单元测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.region import (  # noqa: E402
    Region,
    annotate_detections,
    filter_outside,
    inside_ratio,
    parse_regions,
    region_summary,
)

SHAPE = (100, 100, 3)


def _det(bbox, conf=0.9):
    return {
        "bbox": bbox,
        "conf": conf,
        "center": ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
    }


def test_parse_polygon_and_rect():
    cfg = {
        "enabled": True,
        "items": [
            {"name": "a", "label": "区域A", "points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]]},
            {"name": "b", "label": "区域B", "rect": [0.0, 0.0, 0.5, 0.5]},
        ],
    }
    regions = parse_regions(cfg)
    assert [r.name for r in regions] == ["a", "b"]
    assert regions[0].polygon[0] == (0.1, 0.1)
    assert len(regions[1].polygon) == 4


def test_parse_disabled_returns_empty():
    assert parse_regions({"enabled": False, "items": [{"name": "a", "rect": [0, 0, 1, 1]}]}) == []
    assert parse_regions(None) == []


def test_parse_rejects_bad_geometry():
    with pytest.raises(ValueError):
        parse_regions({"items": [{"name": "x", "points": [[0.1, 0.1], [0.2, 0.2]]}]})
    with pytest.raises(ValueError):
        parse_regions({"items": [{"name": "y", "points": [[0.1, 0.1], [1.5, 0.2], [0.5, 0.9]]}]})
    with pytest.raises(ValueError):
        parse_regions({"items": [{"name": "z"}]})


def test_region_geometry():
    r = Region(name="mid", label="中间", polygon=((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)))
    mask = r.mask(SHAPE)
    assert mask.shape == SHAPE[:2]
    assert mask[50, 50] == 255
    assert mask[5, 5] == 0
    assert r.contains_point((50, 50), SHAPE) is True
    assert r.contains_point((5, 50), SHAPE) is False
    x1, y1, x2, y2 = r.bbox(SHAPE)
    assert (x1, y1) == (20, 20) and (x2, y2) == (80, 80)


def test_inside_ratio_full_half_outside():
    r = Region(name="mid", label="中间", polygon=((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)))
    mask = r.mask(SHAPE)
    assert inside_ratio((10, 10, 40, 40), mask) == pytest.approx(1.0)
    assert inside_ratio((60, 10, 90, 40), mask) == pytest.approx(0.0)
    # 边线像素会被 fillPoly 计入，允许 1 像素的栅格化误差
    assert inside_ratio((30, 10, 70, 40), mask) == pytest.approx(0.5, abs=0.03)


def test_annotate_and_filter_outside():
    r = Region(name="mid", label="中间", polygon=((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)))
    dets = [_det((10, 10, 40, 40)), _det((60, 10, 90, 40))]
    annotate_detections(dets, [r], SHAPE, min_inside_ratio=0.6)
    assert dets[0]["region"] == "mid"
    assert dets[1]["region"] is None
    assert dets[0]["inside_ratio"] == pytest.approx(1.0)
    kept = filter_outside(dets)
    assert len(kept) == 1 and kept[0]["region"] == "mid"


def test_min_inside_ratio_controls_boundary():
    r = Region(name="mid", label="中间", polygon=((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)))
    dets = [_det((30, 10, 70, 40))]  # 一半在内
    annotate_detections(dets, [r], SHAPE, min_inside_ratio=0.6)
    assert dets[0]["region"] is None
    dets = [_det((30, 10, 70, 40))]
    annotate_detections(dets, [r], SHAPE, min_inside_ratio=0.4)
    assert dets[0]["region"] == "mid"


def test_no_regions_means_all_inside():
    dets = [_det((10, 10, 40, 40))]
    annotate_detections(dets, [], SHAPE)
    assert dets[0]["region"] is None and dets[0]["inside_ratio"] == 1.0


def test_region_summary_counts():
    r1 = Region(name="a", label="A", polygon=((0, 0), (0.5, 0), (0.5, 1), (0, 1)))
    r2 = Region(name="b", label="B", polygon=((0.5, 0), (1, 0), (1, 1), (0.5, 1)))
    dets = [_det((5, 5, 40, 40)), _det((60, 5, 90, 40)), _det((5, 60, 40, 90))]
    annotate_detections(dets, [r1, r2], SHAPE, min_inside_ratio=0.6)
    summary, outside = region_summary(dets, [r1, r2])
    assert summary[0]["count"] == 2
    assert summary[1]["count"] == 1
    assert outside == 0


def test_mask_cache_is_not_shared_between_shapes():
    r = Region(name="a", label="A", polygon=((0, 0), (0.5, 0), (0.5, 1), (0, 1)))
    small = r.mask((50, 50, 3))
    big = r.mask((200, 200, 3))
    assert small.shape == (50, 50)
    assert big.shape == (200, 200)
    assert small.dtype == np.uint8
