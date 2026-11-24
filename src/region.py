"""指定区域（Region of Interest）标注与统计。

用途：
    场景里通常只有一部分画面是"真正要管的地方"（如教室座位区、会议室桌前、
    考场通道）。v2.2 支持在配置里用**归一化多边形**（或矩形）显式指定这些区域，
    检测结果图上会：

    1. 用高对比描边 + 半透明填充把区域"框出来"，标注区域名与区域内人数；
    2. 区域内的人员框正常着色，区域外的人员框降级为灰色，一眼看出"框在哪"；
    3. 可选按区域过滤（filter_outside），把墙上海报、走廊路人等区域外误检直接丢弃。

坐标约定：
    全部用 0~1 的归一化坐标，与画面分辨率无关；换成任意一张图都不用改代码，
    只改配置即可。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

Box = Sequence[float]

# 默认配色（BGR）：青绿、琥珀、天蓝、紫、珊瑚
PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (170, 212, 45),
    (60, 190, 250),
    (235, 170, 70),
    (215, 120, 200),
    (110, 110, 245),
)


@dataclass
class Region:
    """一个监测区域。

    polygon: 归一化顶点序列，至少 3 个点，顺序为多边形环（自动闭合）。
    """

    name: str
    label: str
    polygon: Tuple[Tuple[float, float], ...]
    color: Tuple[int, int, int] = PALETTE[0]
    count: bool = True
    enabled: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---------- 几何 ----------
    def pixel_points(self, shape: Tuple[int, ...]) -> np.ndarray:
        h, w = shape[:2]
        pts = np.array(
            [[int(round(x * w)), int(round(y * h))] for x, y in self.polygon],
            dtype=np.int32,
        )
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        return pts

    def mask(self, shape: Tuple[int, ...]) -> np.ndarray:
        h, w = shape[:2]
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, [self.pixel_points(shape)], 255)
        return m

    def contains_point(self, point: Tuple[float, float], shape: Tuple[int, ...]) -> bool:
        return cv2.pointPolygonTest(
            self.pixel_points(shape).reshape(-1, 1, 2), (float(point[0]), float(point[1])), False
        ) >= 0

    def bbox(self, shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        pts = self.pixel_points(shape)
        return (
            int(pts[:, 0].min()),
            int(pts[:, 1].min()),
            int(pts[:, 0].max()),
            int(pts[:, 1].max()),
        )


def _as_color(raw: Any, idx: int) -> Tuple[int, int, int]:
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return tuple(int(max(0, min(255, v))) for v in raw)  # type: ignore[return-value]
    return PALETTE[idx % len(PALETTE)]


def _points_from_item(item: Dict[str, Any]) -> List[Tuple[float, float]]:
    """支持两种写法：points=[[x,y],...] 多边形；rect=[x1,y1,x2,y2] 矩形。"""
    if "points" in item and item["points"]:
        pts = [(float(p[0]), float(p[1])) for p in item["points"]]
    elif "rect" in item and item["rect"]:
        x1, y1, x2, y2 = [float(v) for v in item["rect"][:4]]
        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    else:
        raise ValueError("区域 %r 缺少 points 或 rect" % item.get("name"))
    return pts


def parse_regions(cfg: Dict[str, Any] | None) -> List[Region]:
    """从配置解析区域列表。

    配置形如：
        regions:
          enabled: true
          items:
            - name: seat_zone
              label: 监测区域
              points: [[0.05, 0.32], [0.96, 0.32], [0.99, 0.99], [0.02, 0.99]]
              color: [170, 212, 45]
              count: true
    """
    if not cfg or not cfg.get("enabled", True):
        return []
    items: Iterable[Dict[str, Any]] = cfg.get("items") or []
    regions: List[Region] = []
    for idx, item in enumerate(items):
        if not item or not item.get("enabled", True):
            continue
        name = str(item.get("name") or f"region_{idx + 1}")
        pts = _points_from_item(item)
        if len(pts) < 3:
            raise ValueError(f"区域 {name} 至少需要 3 个顶点，当前 {len(pts)} 个")
        for x, y in pts:
            if not (-0.01 <= x <= 1.01 and -0.01 <= y <= 1.01):
                raise ValueError(f"区域 {name} 的坐标必须归一化到 0~1，当前点 ({x}, {y})")
        regions.append(
            Region(
                name=name,
                label=str(item.get("label") or name),
                polygon=tuple(pts),
                color=_as_color(item.get("color"), idx),
                count=bool(item.get("count", True)),
                enabled=True,
            )
        )
    return regions


def inside_ratio(bbox: Box, mask: np.ndarray) -> float:
    """人员框面积中被区域覆盖的比例（0~1）。"""
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = mask[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float(np.count_nonzero(patch)) / float(patch.size)


def annotate_detections(
    detections: List[Dict[str, Any]],
    regions: List[Region],
    shape: Tuple[int, ...],
    min_inside_ratio: float = 0.6,
) -> List[Dict[str, Any]]:
    """给每个检测结果打上区域归属：det["region"] / det["inside_ratio"]。

    region 为 None 表示落在所有区域之外。会被原地修改并返回。
    """
    if not regions:
        for det in detections:
            det["region"] = None
            det["inside_ratio"] = 1.0
        return detections

    masks = [(r, r.mask(shape)) for r in regions]
    for det in detections:
        best_name: str | None = None
        best_ratio = 0.0
        for region, mask in masks:
            if not region.enabled:
                continue
            ratio = inside_ratio(det["bbox"], mask)
            if ratio > best_ratio:
                best_ratio, best_name = ratio, region.name
        det["inside_ratio"] = round(best_ratio, 4)
        det["region"] = best_name if best_ratio >= min_inside_ratio else None
    return detections


def filter_outside(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """只保留已归属区域的检测（需先调用 annotate_detections）。"""
    return [d for d in detections if d.get("region")]


def region_summary(detections: List[Dict[str, Any]], regions: List[Region]) -> List[Dict[str, Any]]:
    """按区域统计人数，返回可视化/接口用的结构。"""
    summary: List[Dict[str, Any]] = []
    for region in regions:
        members = [d for d in detections if d.get("region") == region.name]
        summary.append(
            {
                "name": region.name,
                "label": region.label,
                "color": list(region.color),
                "polygon": region.polygon,
                "count": len(members) if region.count else 0,
                "count_enabled": region.count,
            }
        )
    outside = [d for d in detections if not d.get("region")]
    return summary, len(outside)  # type: ignore[return-value]
