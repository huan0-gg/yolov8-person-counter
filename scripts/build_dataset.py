#!/usr/bin/env python3
"""v2.0 数据集构建器：生成可复现的教室人员 YOLO 数据集。

数据来源与标注方式（完整说明见 data/DATA_CARD.md）：
  1) 背景底图：assets/classroom_empty.jpg —— 项目已有的真实空教室截图裁剪；
  2) 人物实例：assets/demo.jpg —— ultralytics 官方 COCO bus.jpg 样例，
     用 COCO 预训练权重检测 person 后裁剪得到；
  3) 合成样本：把人物实例按教室课桌的透视规律粘贴到底图，
     边界框由放置坐标精确计算（不是模型猜的），保证标签可靠；
  4) 真实域对照：并入少量真实图片及其伪标注，避免模型只认识合成图。

固定 seed 可完全复现。本数据集用于工程演示与训练流程验证；
正式科研/生产请使用人工精标的真实课堂数据。

用法:
  python scripts/build_dataset.py --seed 42 --total 100
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def resolve_weights(name: str) -> str:
    """按 给定路径 -> 项目根 -> 上级目录 -> 自动下载 的顺序找权重。"""
    candidates = [
        Path(name),
        ROOT / name,
        ROOT.parent / name,
        ROOT / "weights" / name,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return name  # 交给 ultralytics 自动下载


def extract_person_patches(annotator: YOLO, image_path: Path, conf: float = 0.30):
    """从真实图片中检测并裁剪 person 实例。"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"无法读取 {image_path}")
    results = annotator(img, classes=[0], conf=conf, verbose=False)
    patches = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            patch = img[y1:y2, x1:x2]
            if patch.size == 0 or patch.shape[0] * patch.shape[1] < 1200:
                continue
            # 过滤被严重遮挡的极细条实例（正常人体宽高比约 0.18–0.40）
            ratio = patch.shape[1] / max(patch.shape[0], 1)
            if not (0.15 <= ratio <= 0.60):
                continue
            patches.append(patch)
    return patches


def make_grid_positions(rng: random.Random, n: int, w: int, h: int):
    """按 4 列梯形网格生成就坐位置（上排略窄、下排略宽，贴合透视）。"""
    positions = []
    cols, rows = 4, 4
    for i in range(n):
        row, col = divmod(i, cols)
        spread = 0.19 + 0.025 * row          # 越靠下间距越大
        cx = w * (0.5 + (col - 1.5) * spread)
        cy = h * (0.46 + row * 0.145)
        cx += rng.randint(-int(w * 0.012), int(w * 0.012))
        cy += rng.randint(-int(h * 0.010), int(h * 0.010))
        positions.append((int(cx), int(cy)))
    return positions


def synthesize(bg_bgr, patches, rng: random.Random, n_persons: int):
    """合成一张教室人群图，返回 (图像, YOLO 标签行列表)。"""
    out = bg_bgr.copy()
    H, W = out.shape[:2]
    labels = []
    positions = make_grid_positions(rng, n_persons, W, H)
    for (cx, cy), patch in zip(positions, rng.choices(patches, k=n_persons)):
        target_h = rng.randint(int(H * 0.16), int(H * 0.27))
        scale = target_h / patch.shape[0]
        target_w = max(10, int(patch.shape[1] * scale))
        target_w = min(target_w, int(W * 0.22))
        target_h = max(10, int(target_h))
        person = cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_AREA)
        if rng.random() < 0.5:
            person = cv2.flip(person, 1)
        if rng.random() < 0.4:  # 轻微亮度扰动，增加多样性
            factor = rng.uniform(0.85, 1.15)
            person = np.clip(person.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        x1 = max(0, min(W - target_w, cx - target_w // 2))
        y1 = max(0, min(H - target_h, cy - target_h // 2))
        out[y1:y1 + target_h, x1:x1 + target_w] = person

        # 边界框由放置坐标精确生成 —— 标签可靠性的关键
        ncx = (x1 + target_w / 2) / W
        ncy = (y1 + target_h / 2) / H
        nw = target_w / W
        nh = target_h / H
        labels.append(f"0 {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")
    return out, labels


def detect_labels(annotator: YOLO, image_bgr, conf: float = 0.30):
    """对真实图片做伪标注，返回 YOLO 标签行列表。"""
    results = annotator(image_bgr, classes=[0], conf=conf, verbose=False)
    lines = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes.xywhn.cpu().numpy():
            cx, cy, w, h = box
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def write_split(images, labels, out_dir: Path, split: str):
    img_dir = out_dir / "images" / split
    lbl_dir = out_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i, (img, lines) in enumerate(zip(images, labels)):
        name = f"{split}_{i:04d}.jpg"
        cv2.imwrite(str(img_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (lbl_dir / (Path(name).stem + ".txt")).write_text(
            "\n".join(lines), encoding="utf-8"
        )


def draw_samples(images, labels, out_path: Path, count: int = 6):
    """生成数据集标注样例拼图，供展示与人工抽检。"""
    thumbs = []
    for img, lines in list(zip(images, labels))[:count]:
        vis = img.copy()
        H, W = vis.shape[:2]
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            cx, cy, w, h = map(float, parts[1:])
            x1 = int((cx - w / 2) * W)
            y1 = int((cy - h / 2) * H)
            x2 = int((cx + w / 2) * W)
            y2 = int((cy + h / 2) * H)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 160), 2)
        scale = 360 / W
        vis = cv2.resize(vis, (360, int(H * scale)), interpolation=cv2.INTER_AREA)
        thumbs.append(vis)
    if not thumbs:
        return
    h = max(t.shape[0] for t in thumbs)
    canvas = np.full((h * 2, 360 * 3, 3), 20, dtype=np.uint8)
    for i, t in enumerate(thumbs):
        y = (i // 3) * h
        x = (i % 3) * 360
        canvas[y:y + t.shape[0], x:x + 360] = t
    cv2.imwrite(str(out_path), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description="v2.0 教室人员数据集构建")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total", type=int, default=100, help="合成样本总数")
    parser.add_argument("--annotator", default="yolo26n.pt", help="伪标注/裁剪用权重")
    parser.add_argument("--conf", type=float, default=0.30)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    annotator = YOLO(resolve_weights(args.annotator))

    bg_path = ROOT / "assets" / "classroom_empty.jpg"
    people_path = ROOT / "assets" / "demo.jpg"
    bg = cv2.imread(str(bg_path))
    if bg is None:
        raise FileNotFoundError(f"缺少背景图 {bg_path}")
    # 裁掉截图顶部的界面残留（上传提示条），保留纯教室场景
    bg = bg[int(bg.shape[0] * 0.12):int(bg.shape[0] * 0.94), :]
    patches = extract_person_patches(annotator, people_path, args.conf)
    if not patches:
        raise RuntimeError(f"未能从 {people_path} 提取到 person 实例")
    print(f"[数据] 背景 {bg.shape[1]}x{bg.shape[0]}；提取人物实例 {len(patches)} 个")

    images, labels = [], []
    # 1) 合成教室人群样本
    for i in range(args.total):
        n_persons = rng.randint(4, 12)
        img, lines = synthesize(bg, patches, rng, n_persons)
        images.append(img)
        labels.append(lines)

    # 2) 真实域对照样本（原图 + 水平翻转 + 亮度扰动）
    real_sources = [
        ROOT / "assets" / "demo.jpg",
        ROOT / "assets" / "classroom_crowded.jpg",
    ]
    for src in real_sources:
        real = cv2.imread(str(src))
        if real is None:
            continue
        variants = [real, cv2.flip(real, 1)]
        bright = np.clip(real.astype(np.float32) * 1.12, 0, 255).astype(np.uint8)
        variants.append(bright)
        for var in variants:
            images.append(var)
            labels.append(detect_labels(annotator, var, args.conf))

    # 3) 打乱并按 8:2 划分
    paired = list(zip(images, labels))
    rng.shuffle(paired)
    split_at = int(len(paired) * 0.8)
    train_pairs, val_pairs = paired[:split_at], paired[split_at:]

    out_dir = ROOT / "data"
    write_split([p[0] for p in train_pairs], [p[1] for p in train_pairs], out_dir, "train")
    write_split([p[0] for p in val_pairs], [p[1] for p in val_pairs], out_dir, "val")

    total_boxes = sum(len(l) for p in train_pairs + val_pairs for l in [p[1]])
    stats = {
        "seed": args.seed,
        "annotator": args.annotator,
        "conf": args.conf,
        "background": str(bg_path.relative_to(ROOT)),
        "person_source": str(people_path.relative_to(ROOT)),
        "person_patches": len(patches),
        "train_images": len(train_pairs),
        "val_images": len(val_pairs),
        "total_person_boxes": total_boxes,
        "empty_images": sum(1 for _, l in train_pairs + val_pairs if not l),
    }
    (out_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    results_dir = ROOT / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    draw_samples([p[0] for p in train_pairs], [p[1] for p in train_pairs],
                 results_dir / "dataset_samples.jpg")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[完成] 数据集已写入 {out_dir}；样例图 {results_dir / 'dataset_samples.jpg'}")


if __name__ == "__main__":
    main()
