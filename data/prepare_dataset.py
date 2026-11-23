#!/usr/bin/env python3
"""数据集准备：从本地 COCO 2017 抽取 person 类，生成 YOLO 格式标注。

用法:
  python data/prepare_dataset.py --coco-root D:/coco --out data --max-images 1000

若无 COCO，可用手机自采教室图片 + X-AnyLabeling 标注为 YOLO 格式，
放到 data/images/train 与 data/labels/train 即可，无需运行本脚本。
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def coco_bbox_to_yolo(bbox: list[float], w: int, h: int) -> tuple[float, float, float, float]:
    x, y, bw, bh = bbox
    return ((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h)


def convert_split(
    coco_root: Path, out_dir: Path, split: str, max_images: int, person_cat: int = 1
) -> int:
    ann_path = coco_root / "annotations" / f"instances_{split}2017.json"
    img_dir = coco_root / f"{split}2017"
    if not ann_path.exists() or not img_dir.exists():
        print(f"[跳过] 未找到 {ann_path} 或 {img_dir}")
        return 0

    with ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    person_images: dict[int, list[list[float]]] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] != person_cat or ann.get("iscrowd", 0):
            continue
        person_images.setdefault(ann["image_id"], []).append(ann["bbox"])

    ids = sorted(person_images)
    random.shuffle(ids)
    ids = ids[:max_images]

    out_img_dir = out_dir / "images" / split
    out_lbl_dir = out_dir / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    for image_id in ids:
        img_info = images[image_id]
        src = img_dir / img_info["file_name"]
        if not src.exists():
            continue
        dst = out_img_dir / img_info["file_name"]
        if not dst.exists():
            shutil.copy2(src, dst)
        w, h = img_info["width"], img_info["height"]
        lines = []
        for bbox in person_images[image_id]:
            cx, cy, bw, bh = coco_bbox_to_yolo(bbox, w, h)
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        label_path = out_lbl_dir / (Path(img_info["file_name"]).stem + ".txt")
        label_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[{split}] 已导出 {len(ids)} 张含 person 的图片 -> {out_img_dir}")
    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="COCO person -> YOLO 数据集")
    parser.add_argument("--coco-root", required=True, help="COCO 2017 根目录")
    parser.add_argument("--out", default="data", help="输出目录")
    parser.add_argument("--max-images", type=int, default=1000, help="每个 split 最多图片数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out)
    coco_root = Path(args.coco_root)
    total = convert_split(coco_root, out_dir, "train", args.max_images)
    total += convert_split(coco_root, out_dir, "val", max(200, args.max_images // 5))
    print(f"[完成] 共 {total} 张。请确保 {out_dir / 'dataset.yaml'} 的 path 指向 {out_dir}")


if __name__ == "__main__":
    main()
