#!/usr/bin/env python3
"""生成一张模拟教室场景图，用于无外网时演示（assets/demo.jpg）。

注意：这是示意图，YOLO 对简笔画人形的检测效果有限。
正式实验请替换为真实教室照片或 COCO 图片。
"""
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    random.seed(7)
    w, h = 900, 620
    img = Image.new("RGB", (w, h), (238, 240, 236))
    d = ImageDraw.Draw(img)

    # 黑板与地板
    d.rectangle([0, 0, w, 110], fill=(60, 95, 75))
    d.rectangle([40, 18, w - 40, 92], fill=(35, 60, 45))
    d.text((w // 2 - 90, 45), "CLASSROOM", fill=(220, 230, 220))

    # 4x4 座位，位置与 config.yaml 的归一化区域大致对齐
    seat_w, seat_h = 150, 88
    x0, y0 = 70, 140
    occupied = {(0, 0), (1, 0), (3, 0), (0, 1), (2, 1), (1, 2), (3, 2), (2, 3)}
    for r in range(4):
        for c in range(4):
            sx = x0 + c * 198
            sy = y0 + r * 115
            d.rounded_rectangle(
                [sx, sy, sx + seat_w, sy + seat_h], radius=8,
                outline=(120, 125, 120), width=3,
            )
            if (r, c) in occupied:
                cx = sx + seat_w // 2
                d.ellipse([cx - 20, sy + 8, cx + 20, sy + 48], fill=(90, 110, 160))
                d.rounded_rectangle(
                    [cx - 30, sy + 46, cx + 30, sy + seat_h - 4],
                    radius=10, fill=(70, 90, 140),
                )

    out = ROOT / "assets" / "demo.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=92)
    print(f"已生成演示图: {out} ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
