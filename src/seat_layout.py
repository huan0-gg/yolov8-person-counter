"""自适应座位网格：根据图像透视生成贴合画面的座位框。

背景：
    旧版把 16 个座位框的归一化坐标硬编码在 config.yaml，任何一张视角不同的
    图片都会出现"框跑到墙壁/黑板/天花板"的错位问题（见用户截图）。

新方案：
    用一组"标定锚点"描述座位区在画面中的梯形范围（左上/右上/左下/右下四个
    归一化角点），再按 row x col 在梯形内做双线性插值，生成透视网格。
    - 顶边窄、底边宽，天然贴合教室近大远小的透视；
    - 锚点可按不同教室调整，一套配置适配任意视角；
    - 生成的框始终落在标定区域内，不会跑到黑板/墙壁；
    - 无人场景同样稳定显示，因为框不依赖检测结果。
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

# 默认标定：座位区梯形四角（归一化 0~1），左上/右上/左下/右下
DEFAULT_QUAD: List[List[float]] = [
    [0.10, 0.44],
    [0.90, 0.44],
    [0.03, 0.97],
    [0.97, 0.97],
]


def _lerp(a: Tuple[float, float], b: Tuple[float, float], t: float) -> Tuple[float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def bilinear(quad: List[List[float]], u: float, v: float) -> Tuple[float, float]:
    """在梯形内做双线性插值。u 横向 0->1，v 纵向 0->1。"""
    tl, tr, br, bl = [tuple(p) for p in quad]
    top = _lerp(tl, tr, u)
    bottom = _lerp(bl, br, u)
    return _lerp(top, bottom, v)


def build_grid(
    quad: List[List[float]] | None,
    rows: int,
    cols: int,
    shape: Tuple[int, ...],
    shrink: float = 0.86,
    inset_x: float = 0.0,
    inset_y: float = 0.0,
) -> List[Tuple[int, int, int, int]]:
    """生成 rows*cols 个透视座位框（像素坐标 xyxy）。

    shrink: 每个格子的收缩比例，避免相邻框紧贴造成"杂乱"观感。
    inset_x/inset_y: 梯形四边向内收缩，避免框压到画面边缘。
    """
    h, w = shape[:2]
    quad = quad or DEFAULT_QUAD
    # 先把梯形按 inset 向内收一点
    q = [
        [quad[0][0] + inset_x, quad[0][1] + inset_y],
        [quad[1][0] - inset_x, quad[1][1] + inset_y],
        [quad[2][0] - inset_x, quad[2][1] - inset_y],
        [quad[3][0] + inset_x, quad[3][1] - inset_y],
    ]
    boxes: List[Tuple[int, int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            u0, u1 = c / cols, (c + 1) / cols
            v0, v1 = r / rows, (r + 1) / rows
            # 格子四角
            corners = [
                bilinear(q, u0, v0),
                bilinear(q, u1, v0),
                bilinear(q, u1, v1),
                bilinear(q, u0, v1),
            ]
            cx = sum(p[0] for p in corners) / 4.0
            cy = sum(p[1] for p in corners) / 4.0
            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
            bw = (max(xs) - min(xs)) * shrink
            bh = (max(ys) - min(ys)) * shrink
            x1 = int(max(0, (cx - bw / 2) * w))
            y1 = int(max(0, (cy - bh / 2) * h))
            x2 = int(min(w, (cx + bw / 2) * w))
            y2 = int(min(h, (cy + bh / 2) * h))
            boxes.append((x1, y1, x2, y2))
    return boxes


def build_seats(cfg: Dict[str, Any], shape: Tuple[int, ...]) -> Tuple[List[Tuple[int, int, int, int]], str]:
    """根据配置生成座位框。

    返回 (座位框列表, 来源说明)。优先用透视网格，未配置则回退旧版固定 seats。
    """
    layout = cfg.get("seat_grid")
    if layout and layout.get("enabled", True):
        quad = layout.get("quad")
        rows = int(layout.get("rows", 4))
        cols = int(layout.get("cols", 4))
        shrink = float(layout.get("shrink", 0.86))
        boxes = build_grid(
            quad, rows, cols, shape,
            shrink=shrink,
            inset_x=float(layout.get("inset_x", 0.01)),
            inset_y=float(layout.get("inset_y", 0.01)),
        )
        return boxes, f"perspective_grid_{rows}x{cols}"

    # 回退：旧版固定归一化坐标
    seats = cfg.get("seats") or []
    h, w = shape[:2]
    boxes = [
        (int(s[0] * w), int(s[1] * h), int(s[2] * w), int(s[3] * h))
        for s in seats
    ]
    return boxes, "fixed_seats"
