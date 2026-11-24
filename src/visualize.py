"""结果可视化（v2.2）：在检测结果上清晰标注"指定区域"，并统一展示规则。

规则：
- **指定区域**：彩色描边 + 半透明填充 + 左上角标签条（区域名 + 区域内人数）；
  可开启"区域外压暗"，让监测范围一眼可辨；
- 座位框：统一细线，占用=红色、空闲=绿色，标签简洁；
- 人员框：区域内=黄色实线、区域外=灰色细线，只标序号（不堆置信度），可关联座位号；
- 统计面板显示区域人数、切片状态与过滤信息，便于复核。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .region import Region

# 颜色（BGR）
COLOR_SEAT_FREE = (0, 170, 0)
COLOR_SEAT_OCCUPIED = (0, 0, 220)
COLOR_PERSON = (0, 200, 255)
COLOR_OUTSIDE = (150, 150, 150)
COLOR_TEXT = (25, 25, 25)

_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)
_font_cache: Dict[int, Any] = {}


def _cjk_font(size: int):
    if size in _font_cache:
        return _font_cache[size]
    try:
        from PIL import ImageFont

        for path in _FONT_CANDIDATES:
            if Path(path).exists():
                _font_cache[size] = ImageFont.truetype(path, size)
                return _font_cache[size]
    except Exception:  # pragma: no cover - 缺字体时静默降级
        pass
    _font_cache[size] = None
    return None


def _label_chip(
    vis: np.ndarray, xy: Tuple[int, int], text: str, color: Tuple[int, int, int], size: int = 22
) -> np.ndarray:
    """在 (x, y) 画一个带底色的中文标签条（PIL 渲染，避免 cv2 中文变问号）。"""
    font = _cjk_font(size)
    if font is None:
        cv2.putText(
            vis, "ROI", (xy[0] + 8, xy[1] + size), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        return vis

    from PIL import Image, ImageDraw

    h, w = vis.shape[:2]
    pad_x, pad_y = 10, 6
    tmp = Image.new("RGB", (max(w, 420), 80), (0, 0, 0))
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    chip_w, chip_h = tw + pad_x * 2, th + pad_y * 2
    x = int(np.clip(xy[0], 2, max(2, w - chip_w - 2)))
    y = int(np.clip(xy[1], 2, max(2, h - chip_h - 2)))

    overlay = vis.copy()
    cv2.rectangle(overlay, (x, y), (x + chip_w, y + chip_h), color, -1)
    cv2.addWeighted(overlay, 0.92, vis, 0.08, 0, vis)

    rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    ImageDraw.Draw(pil).text(
        (x + pad_x, y + pad_y - bbox[1]), text, font=font, fill=(12, 20, 26)
    )
    vis[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return vis


def _draw_regions(
    vis: np.ndarray,
    regions: List[Dict[str, Any]],
    shape: Tuple[int, ...],
    dim_outside: bool = False,
) -> np.ndarray:
    """绘制区域：填充、描边、标签、可选压暗区域外。"""
    h, w = shape[:2]
    if not regions:
        return vis

    if dim_outside:
        mask = np.zeros((h, w), dtype=np.uint8)
        for region in regions:
            pts = np.array(
                [[int(x * w), int(y * h)] for x, y in region["polygon"]], dtype=np.int32
            )
            cv2.fillPoly(mask, [pts], 255)
        dark = (vis * 0.45).astype(np.uint8)
        vis[:, :] = np.where(mask[..., None] > 0, vis, dark)

    for idx, region in enumerate(regions):
        color = tuple(region["color"])
        pts = np.array(
            [[int(x * w), int(y * h)] for x, y in region["polygon"]], dtype=np.int32
        )
        fill = vis.copy()
        cv2.fillPoly(fill, [pts], color)
        cv2.addWeighted(fill, 0.10, vis, 0.90, 0, vis)
        cv2.polylines(vis, [pts], True, color, 2, lineType=cv2.LINE_AA)

        # 角点小方块，让"这是一个人工指定的范围"更明确
        for px, py in pts:
            cv2.rectangle(vis, (px - 4, py - 4), (px + 4, py + 4), color, -1, cv2.LINE_AA)

        label = region["label"]
        if region.get("count_enabled", True):
            label = f"{label}  {region['count']} 人"
        top = pts[:, 1].min()
        left = pts[:, 0].min()
        vis = _label_chip(vis, (int(left) + 6, int(top) + 6), label, color, size=22)
    return vis


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
    show_regions = bool(display_cfg.get("show_regions", True))
    dim_outside = bool(display_cfg.get("dim_outside", False))

    vis = frame.copy()
    occupied = counter_result["occupied"]
    seat_boxes = counter_result["seat_boxes"]
    regions = counter_result.get("regions") or []

    # 0) 指定区域（画在最底层，避免压住人员框）
    if show_regions and regions:
        vis = _draw_regions(vis, regions, frame.shape, dim_outside=dim_outside)

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

    # 2) 人员框：区域内黄色、区域外灰色
    if show_person:
        has_regions = bool(regions)
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            outside = has_regions and det.get("region") is None
            color = COLOR_OUTSIDE if outside else COLOR_PERSON
            thickness = 1 if outside else 2
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
            parts = [f"P{i + 1}"]
            if show_track and det.get("track_id") is not None:
                parts.append(f"#{det['track_id']}")
            seat_idx = det.get("seat_index", -1)
            if seat_idx is not None and seat_idx >= 0:
                parts.append(f"S{seat_idx + 1}")
            if outside:
                parts.append("OUT")
            label = " ".join(parts)
            ty = max(y1 - 6, 12)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), (255, 255, 255), -1)
            cv2.putText(
                vis, label, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, color, 1, cv2.LINE_AA,
            )

    # 3) 统计面板：紧凑、半透明、置于右上角，尽量少遮挡检测区域
    if show_panel:
        h, w = vis.shape[:2]
        panel_w = 268
        panel_h = 132 if regions else 104
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
        ]
        if regions:
            lines.append(
                (
                    f"Inside ROI: {counter_result.get('inside_persons', 0)}  "
                    f"Outside: {counter_result.get('outside_persons', 0)}",
                    0.46, (20, 90, 70),
                )
            )
        lines += [
            (
                f"Mode: {counter_result.get('count_mode', '')}  "
                f"Tiling: {'ON' if counter_result.get('tiling') else 'OFF'}",
                0.4, (90, 90, 90),
            ),
            (f"Filtered: {counter_result.get('filtered_out', 0)}", 0.4, (90, 90, 90)),
        ]
        y = py + 22
        for text, scale, color in lines:
            cv2.putText(
                vis, text, (px + 12, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1 if scale < 0.6 else 2, cv2.LINE_AA,
            )
            y += 22
    return vis
