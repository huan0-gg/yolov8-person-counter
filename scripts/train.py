#!/usr/bin/env python3
"""v2.0 训练入口：完整训练流程 + 模型输出 + 结果验证。

相比 1.0 新增：
  1) 训练前保存完整配置快照（可复核）；
  2) 训练中自动记录 loss / mAP 曲线（results.csv 由 ultralytics 产出）；
  3) 训练后导出 best.pt 副本到 experiments/models/，并生成训练报告；
  4) 独立验证集评估，输出 mAP@0.5 / mAP@0.5:0.95 / 每类精度；
  5) 生成 loss 曲线图，便于展示"训练确实发生"。

示例：
  python scripts/train.py --augment default --name A_baseline --epochs 15
  python scripts/train.py --augment mosaic_strong --name B1_augment --epochs 15
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

from src.augment import get_augment_kwargs
from src.utils import load_config, resolve_dataset_yaml, set_seed


def plot_training_curves(csv_path: Path, out_path: Path) -> bool:
    """从 ultralytics results.csv 绘制 loss / mAP 曲线。"""
    if not csv_path.exists():
        return False
    import csv

    rows = []
    with csv_path.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return False

    def col(name, default=0.0):
        try:
            return [float(r.get(name, default) or default) for r in rows]
        except (ValueError, TypeError):
            return [default] * len(rows)

    epochs = list(range(1, len(rows) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, col("train/box_loss"), label="train box_loss", color="#2DD4A7")
    axes[0].plot(epochs, col("train/cls_loss"), label="train cls_loss", color="#5AB4E8")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, col("metrics/mAP50(B)"), label="mAP@0.5", color="#2DD4A7")
    axes[1].plot(epochs, col("metrics/mAP50-95(B)"), label="mAP@0.5:0.95", color="#F5B942")
    axes[1].set_title("Validation mAP")
    axes[1].set_xlabel("epoch")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="v2.0 YOLOv8 person 检测训练")
    parser.add_argument("--weights", default="yolov8n.pt", help="预训练权重")
    parser.add_argument("--data", default="data/dataset.yaml", help="数据集配置")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default="cpu", help="cpu / 0 / cuda")
    parser.add_argument(
        "--augment", default="default", choices=["default", "mosaic_strong", "no_mosaic"],
        help="创新点1：数据增强策略",
    )
    parser.add_argument("--name", default=None, help="本次实验名称")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=0, help="早停耐心，0=不启用")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config()
    args.data = str(resolve_dataset_yaml(args.data))
    name = args.name or f"{args.augment}_e{args.epochs}"
    aug_kwargs = get_augment_kwargs(args.augment)
    # 显式绝对路径，避免 ultralytics 全局 runs_dir 设置导致输出错位
    project_dir = ROOT / "runs" / "train"
    run_dir = project_dir / name
    results_dir = ROOT / "experiments" / "results"
    model_dir = ROOT / "experiments" / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_config = {
        "name": name,
        "weights": args.weights,
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "seed": args.seed,
        "augment": args.augment,
        "augment_kwargs": aug_kwargs,
        "patience": args.patience,
        "config_snapshot": {
            "model": cfg.get("model", {}),
            "roi": cfg.get("roi", {}),
        },
    }
    (results_dir / f"{name}_config.json").write_text(
        json.dumps(train_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[训练] 实验={name} 增强={args.augment} 参数={aug_kwargs}")
    t0 = time.perf_counter()
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=name,
        project=str(project_dir),
        seed=args.seed,
        exist_ok=True,
        patience=args.patience,
        plots=True,
        **aug_kwargs,
    )
    train_seconds = round(time.perf_counter() - t0, 1)

    # 独立验证集评估
    metrics = model.val(data=args.data, device=args.device, verbose=False)
    best_src = run_dir / "weights" / "best.pt"
    exported = model_dir / f"{name}_best.pt"
    if best_src.exists():
        shutil.copy2(best_src, exported)

    csv_path = run_dir / "results.csv"
    curve_path = results_dir / f"{name}_curves.png"
    curve_ok = plot_training_curves(csv_path, curve_path)

    result = {
        "name": name,
        "augment": args.augment,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "train_seconds": train_seconds,
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "weights": str(exported.relative_to(ROOT)) if exported.exists() else str(best_src),
        "loss_curve": str(curve_path.relative_to(ROOT)) if curve_ok else None,
        "results_csv": str(csv_path.relative_to(ROOT)) if csv_path.exists() else None,
        "dataset_note": "指标基于 data/DATA_CARD.md 所述合成+真实对照数据集",
    }
    (results_dir / f"{name}_train_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        f"# 训练报告：{name}",
        "",
        f"- 增强策略：{args.augment}（{aug_kwargs}）",
        f"- 训练轮数：{args.epochs}，batch={args.batch}，imgsz={args.imgsz}，device={args.device}",
        f"- 训练耗时：{train_seconds} 秒",
        f"- 数据集：{args.data}（见 data/DATA_CARD.md）",
        "",
        "## 验证集指标（真实运行）",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| mAP@0.5 | {result['map50']:.4f} |",
        f"| mAP@0.5:0.95 | {result['map50_95']:.4f} |",
        f"| Precision | {result['precision']:.4f} |",
        f"| Recall | {result['recall']:.4f} |",
        "",
        "## 模型输出",
        "",
        f"- 最佳权重：`{result['weights']}`",
        f"- 训练曲线：`{result['loss_curve'] or '未生成'}`",
        f"- 原始 loss 记录：`{result['results_csv'] or '未生成'}`",
        f"- 配置快照：`experiments/results/{name}_config.json`",
        "",
        "> 所有数值来自本次真实训练与验证，未手工填写。",
    ]
    (results_dir / f"{name}_train_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[完成] 训练报告 {results_dir / (name + '_train_report.md')}")


if __name__ == "__main__":
    main()
