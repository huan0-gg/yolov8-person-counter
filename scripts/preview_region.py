#!/usr/bin/env python3
"""只渲染指定区域，用来快速确认多边形标得对不对（不跑检测，秒出图）。

用法：
    python scripts/preview_region.py --image assets/ai/classroom_seat.jpg \
        --poly "0.01,0.30 0.99,0.27 0.99,0.99 0.01,1.00" --out preview.jpg

支持多个区域：--poly "..." --poly "..."；或 --rect "x1,y1,x2,y2"。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2

from src.region import parse_regions
from src.visualize import _draw_regions


def main() -> None:
    ap = argparse.ArgumentParser(description="区域标定预览")
    ap.add_argument("--image", required=True)
    ap.add_argument("--poly", action="append", default=[], help='"x1,y1 x2,y2 ..." 归一化坐标')
    ap.add_argument("--rect", action="append", default=[], help='"x1,y1,x2,y2" 归一化矩形')
    ap.add_argument("--dim", action="store_true", help="区域外压暗")
    ap.add_argument("--out", default=str(ROOT / "docs" / "region" / "_preview.jpg"))
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"读不到图片：{args.image}")

    items = []
    for i, poly in enumerate(args.poly):
        pts = [[float(v) for v in pair.split(",")] for pair in poly.replace(";", " ").split()]
        items.append({"name": f"poly{i + 1}", "label": f"区域 {i + 1}", "points": pts})
    for i, rect in enumerate(args.rect):
        box = [float(v) for v in rect.split(",")]
        items.append({"name": f"rect{i + 1}", "label": f"区域 R{i + 1}", "rect": box})
    if not items:
        raise SystemExit("至少给一个 --poly 或 --rect")

    regions = parse_regions({"enabled": True, "items": items})
    vis = _draw_regions(img, [
        {
            "name": r.name,
            "label": r.label,
            "polygon": r.polygon,
            "color": list(r.color),
            "count": 0,
            "count_enabled": False,
        }
        for r in regions
    ], img.shape, dim_outside=args.dim)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
