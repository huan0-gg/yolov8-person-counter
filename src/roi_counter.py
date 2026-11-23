"""座位占用判定与框关联（改进版）。

改进点：
1. 座位框由 seat_layout 动态生成，贴合画面透视，不再硬编码；
2. 一人只占一个座位（按最优匹配分配），避免同一人命中多个座位造成"重复占用"；
3. 过滤无效人员框（面积过小/超出画面/宽高比异常），减少杂乱；
4. 支持 center / overlap 两种判定模式，兼容原有 A/B 实验。
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .seat_layout import build_seats

Box = Tuple[int, int, int, int]


def _bbox_area(b: Box) -> float:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def filter_valid_detections(
    detections: List[Dict[str, Any]], shape: Tuple[int, ...], min_area_ratio: float = 0.002
) -> List[Dict[str, Any]]:
    """过滤无效人员框：面积过小、宽高比异常、中心在画面外。"""
    h, w = shape[:2]
    min_area = min_area_ratio * w * h
    valid: List[Dict[str, Any]] = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        bw, bh = max(0, x2 - x1), max(0, y2 - y1)
        area = bw * bh
        if area < min_area:
            continue
        if bh < 1 or bw < 1:
            continue
        ratio = bh / bw
        # 人形框通常是竖长条；坐姿会矮一些，放宽到 [0.5, 4.0]
        if ratio < 0.5 or ratio > 4.0:
            continue
        cx, cy = det["center"]
        if not (0 <= cx <= w and 0 <= cy <= h):
            continue
        valid.append(det)
    return valid


def _match_score(
    det: Dict[str, Any], seat: Box, mode: str, threshold: float, img_area: float
) -> float:
    """计算人员与座位的匹配分数；<=0 表示不匹配。"""
    x1, y1, x2, y2 = det["bbox"]
    sx1, sy1, sx2, sy2 = seat
    cx, cy = det["center"]

    inside = sx1 <= cx <= sx2 and sy1 <= cy <= sy2
    if mode == "center":
        if inside:
            # 中心越靠近座位中心，分数越高
            scx, scy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
            dist = ((cx - scx) ** 2 + (cy - scy) ** 2) ** 0.5
            diag = ((sx2 - sx1) ** 2 + (sy2 - sy1) ** 2) ** 0.5
            return 1.0 - min(dist / max(diag, 1.0), 1.0)
        # 中心不在座位内：允许小范围邻近（画面下方的人可能坐在框略上方）
        ix1, iy1 = max(x1, sx1), max(y1, sy1)
        ix2, iy2 = min(x2, sx2), min(y2, sy2)
        if ix2 > ix1 and iy2 > iy1:
            inter = (ix2 - ix1) * (iy2 - iy1)
            if inter / max(img_area, 1.0) > 0.0005:
                return 0.3
        return 0.0

    # overlap 模式：用人员框与座位的重叠比例
    ix1, iy1 = max(x1, sx1), max(y1, sy1)
    ix2, iy2 = min(x2, sx2), min(y2, sy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ratio = inter / max(_bbox_area((x1, y1, x2, y2)), 1.0)
    if ratio >= threshold:
        return ratio
    return 0.0


class SeatROICounter:
    def __init__(
        self,
        cfg: Dict[str, Any],
        mode: str = "center",
        overlap_threshold: float = 0.30,
        min_area_ratio: float = 0.002,
    ) -> None:
        if mode not in ("center", "overlap"):
            raise ValueError("count_mode 只能是 center 或 overlap")
        self.cfg = cfg
        self.mode = mode
        self.overlap_threshold = overlap_threshold
        self.min_area_ratio = min_area_ratio

    def count(
        self, detections: List[Dict[str, Any]], shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        seat_boxes, seat_source = build_seats(self.cfg, shape)
        valid_dets = filter_valid_detections(detections, shape, self.min_area_ratio)
        img_area = float(shape[0] * shape[1])

        occupied = [False] * len(seat_boxes)
        seat_of_person: List[int] = [-1] * len(valid_dets)

        # 生成所有 (人员, 座位) 候选匹配并按分数降序，做一对一分配
        candidates: List[Tuple[float, int, int]] = []
        for pi, det in enumerate(valid_dets):
            for si, seat in enumerate(seat_boxes):
                s = _match_score(det, seat, self.mode, self.overlap_threshold, img_area)
                if s > 0:
                    candidates.append((s, pi, si))
        candidates.sort(key=lambda x: -x[0])

        used_persons: set[int] = set()
        used_seats: set[int] = set()
        for score, pi, si in candidates:
            if pi in used_persons or si in used_seats:
                continue
            used_persons.add(pi)
            used_seats.add(si)
            occupied[si] = True
            seat_of_person[pi] = si

        for pi, det in enumerate(valid_dets):
            det["seat_index"] = seat_of_person[pi]

        return {
            "occupied": occupied,
            "occupied_count": int(sum(occupied)),
            "total_seats": len(seat_boxes),
            "total_persons": len(valid_dets),
            "raw_persons": len(detections),
            "filtered_out": len(detections) - len(valid_dets),
            "seat_boxes": seat_boxes,
            "seat_source": seat_source,
            "count_mode": self.mode,
            "detections": valid_dets,
        }
