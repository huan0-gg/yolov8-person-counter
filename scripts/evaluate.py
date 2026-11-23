#!/usr/bin/env python3
"""单方案评估：输出 mAP / FPS / 计数结果 JSON。

示例：
  python scripts/evaluate.py --variant B2_track --weights yolov8n.pt --demo-image assets/demo.jpg
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse

from src.evaluate import evaluate_variant
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="单方案评估")
    parser.add_argument("--variant", default="A_baseline")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--data", default=None, help="验证集 dataset.yaml")
    parser.add_argument("--demo-image", default=None, help="真实场景图片，用于计数测试")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs" / "config.yaml")
    result = evaluate_variant(
        cfg,
        variant=args.variant,
        weights=args.weights,
        data=args.data,
        demo_image=args.demo_image,
        repeats=args.repeats,
    )
    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval_{args.variant}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
