#!/usr/bin/env python3
"""训练入口：基线 A 与创新点 1（数据增强策略）共用。

示例：
  python scripts/train.py --augment default --name A_baseline --epochs 30
  python scripts/train.py --augment mosaic_strong --name B1_augment --epochs 30
  python scripts/train.py --augment no_mosaic --name ctrl_no_mosaic --epochs 30
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse

from ultralytics import YOLO

from src.augment import get_augment_kwargs
from src.utils import load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 person 检测训练")
    parser.add_argument("--weights", default="yolov8n.pt", help="预训练权重")
    parser.add_argument("--data", default="data/dataset.yaml", help="数据集配置")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="cpu / 0 / cuda")
    parser.add_argument(
        "--augment",
        default="default",
        choices=["default", "mosaic_strong", "no_mosaic"],
        help="创新点1：数据增强策略",
    )
    parser.add_argument("--name", default=None, help="本次实验名称")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    load_config()  # 确保配置文件存在且可读
    name = args.name or f"{args.augment}_e{args.epochs}"
    aug_kwargs = get_augment_kwargs(args.augment)

    model = YOLO(args.weights)
    print(f"[训练] 增强策略={args.augment} 参数={aug_kwargs}")
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=name,
        project="runs/train",
        seed=args.seed,
        exist_ok=True,
        **aug_kwargs,
    )

    metrics = model.val(data=args.data, device=args.device, verbose=False)
    result = {
        "name": name,
        "augment": args.augment,
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
        "weights": f"runs/train/{name}/weights/best.pt",
    }
    results_dir = ROOT / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{name}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
