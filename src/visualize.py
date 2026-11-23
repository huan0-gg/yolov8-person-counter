"""结果可视化（改进版）：统一展示规则，减少杂乱。

规则：
- 座位框：统一细线，占用=红色、空闲=绿色，标签简洁；
- 人员框：黄色中等线宽，只标序号（不堆置信度），可关联座位号；
- 只画通过过滤的有效框；
- 统计面板显示过滤信息，便于复核。
"""
from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

# 颜色（BGR）
COLOR_SEAT_FREE = (0, 170, 0)
COLOR_SEAT_OCCUPIED = (0, 0, 220)
COLOR_PERSON = (0, 200, 255)
COLOR_TEXT = (25, 25, 25)


def draw_result(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    counter_result: Dict[str, Any],
    show_track: bool = False,
    display_cfg: Dict[str, Any] | None = None,
) -> np.ndarray:
    display_cfg = display_cfg or {}
    show_person = bool(display_cfg.get("show_person_boxes", True))
    show_ids = bool(display_cfg.get("show_seat_ids", True))
    show_panel = bool(display_cfg.get("show_panel", True))

    vis = frame.copy()
    occupied = counter_result["occupied"]
    seat_boxes = counter_result["seat_boxes"]

    # 1) 座位框：统一细线
    for i, (x1, y1, x2, y2) in enumerate(seat_boxes):
        is_occ = i < len(occupied) and occupied[i]
        color = COLOR_SEAT_OCCUPIED if is_occ else COLOR_SEAT_FREE
        thickness = 2 if is_occ else 1
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        if show_ids:
            label = f"S{i + 1}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(vis, (x1, y1), (x1 + tw + 6, y1 + th + 6), (255, 255, 255), -1)
            cv2.putText(
                vis, label, (x1 + 3, y1 + th + 3), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, color, 1, cv2.LINE_AA,
            )

    # 2) 人员框：只画有效框，标注序号与座位号
    if show_person:
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_PERSON, 2, lineType=cv2.LINE_AA)
            parts = [f"P{i + 1}"]
            if show_track and det.get("track_id") is not None:
                parts.append(f"#{det['track_id']}")
            seat_idx = det.get("seat_index", -1)
            if seat_idx is not None and seat_idx >= 0:
                parts.append(f"S{seat_idx + 1}")
            label = " ".join(parts)
            ty = max(y1 - 6, 12)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), (255, 255, 255), -1)
            cv2.putText(
                vis, label, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, COLOR_PERSON, 1, cv2.LINE_AA,
            )

    # 3) 统计面板：紧凑、半透明、置于右上角，尽量少遮挡检测区域
    if show_panel:
        h, w = vis.shape[:2]
        panel_w, panel_h = 250, 104
        px, py = w - panel_w - 12, 12
        overlay = vis.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.78, vis, 0.22, 0, vis)
        lines = [
            (f"Persons: {counter_result['total_persons']}", 0.65, COLOR_TEXT),
            (
                f"Seats occupied: {counter_result['occupied_count']}/{counter_result['total_seats']}",
                0.55, COLOR_SEAT_OCCUPIED,
            ),
            (f"Grid: {counter_result.get('seat_source', 'n/a')}", 0.4, (90, 90, 90)),
            (
                f"Filtered: {counter_result.get('filtered_out', 0)}  Mode: {counter_result.get('count_mode', '')}",
                0.4, (90, 90, 90),
            ),
        ]
        y = py + 22
        for text, scale, color in lines:
            cv2.putText(
                vis, text, (px + 12, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1 if scale < 0.6 else 2, cv2.LINE_AA,
            )
            y += 22
    return vis
