#!/usr/bin/env python3
"""从 Web 界面截图中裁出左侧教室原图，用作复现/验证素材。

截图里左侧"上传教室/会议室图片"面板大致位于图片左半区域，
裁剪比例按经验值设置，可通过命令行覆盖。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

BASE = Path(r"C:\Users\huan'xin\.workbuddy\clipboard-images")
OUT = Path(__file__).resolve().parent.parent / "assets"


def crop(src: Path, dst: Path, box_ratio: tuple[float, float, float, float]) -> None:
    img = Image.open(src).convert("RGB")
    w, h = img.size
    l, t, r, b = box_ratio
    box = (int(l * w), int(t * h), int(r * w), int(b * h))
    img.crop(box).save(dst, quality=95)
    print(f"{src.name} -> {dst.name}  crop={box}  size={dst.stat().st_size}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", nargs=4, type=float, default=[0.10, 0.40, 0.47, 0.55])
    args = parser.parse_args()
    l, t, r, b = args.ratio
    crop(
        BASE / "clipboard-2026-11-23T03-18-48-400Z-bc271488.jpg",
        OUT / "classroom_crowded.jpg",
        (l, t, r, b),
    )
    # 空教室图在另一张截图中位置不同，单独设定
    crop(
        BASE / "clipboard-2026-11-23T03-18-48-402Z-ba726ebd.jpg",
        OUT / "classroom_empty.jpg",
        (0.10, 0.40, 0.48, 0.86),
    )


if __name__ == "__main__":
    main()
