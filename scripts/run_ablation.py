#!/usr/bin/env python3
"""A/B 实验：自动跑基线 A 与 3 个创新点，输出可复核的对比证据。

用法:
  python scripts/run_ablation.py --demo-image assets/demo.jpg
  python scripts/run_ablation.py --data data/dataset.yaml \
      --b1-weights runs/train/B1_augment/weights/best.pt

输出:
  experiments/results/ablation_results.json / .csv / ablation_report.md
  （ablation_report.md 内含图片要求的 3 条标准化创新表述）
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse

from src.evaluate import evaluate_variant
from src.utils import load_config

VARIANTS = ["A_baseline", "B1_augment", "B2_track", "B3_center_roi", "B_all"]

VARIANT_DESC = {
    "A_baseline": "默认 YOLOv8n + 逐帧独立计数 + overlap 座位判定",
    "B1_augment": "创新点1：强 Mosaic + Copy-Paste 数据增强训练",
    "B2_track": "创新点2：ByteTrack 跨帧跟踪去重",
    "B3_center_roi": "创新点3：中心点判定替代重叠阈值",
    "B_all": "三个创新点全部启用",
}


def pct(new: float, old: float) -> str:
    if old == 0:
        return "N/A"
    return f"{(new - old) / old * 100:+.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B 实验与可复核创新证据生成")
    parser.add_argument("--data", default=None, help="验证集 dataset.yaml（可选）")
    parser.add_argument("--demo-image", default=None, help="真实场景图片，用于计数测试")
    parser.add_argument("--base-weights", default="yolov8n.pt")
    parser.add_argument("--b1-weights", default="runs/train/B1_augment/weights/best.pt")
    parser.add_argument("--repeats", type=int, default=20, help="FPS 测试重复次数")
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs" / "config.yaml")
    base_w = args.base_weights
    b1_path = ROOT / args.b1_weights
    b1_w = args.b1_weights if b1_path.exists() else base_w

    weights_map = {
        "A_baseline": base_w,
        "B1_augment": b1_w,
        "B2_track": base_w,
        "B3_center_roi": base_w,
        "B_all": b1_w,
    }

    results: dict[str, dict] = {}
    for variant in VARIANTS:
        print(f"[A/B] 正在评估 {variant} ...")
        r = evaluate_variant(
            cfg,
            variant=variant,
            weights=weights_map[variant],
            data=args.data,
            demo_image=args.demo_image,
            repeats=args.repeats,
        )
        if variant in ("B1_augment", "B_all") and b1_w == base_w:
            r["note"] = (
                r.get("note", "")
                + " B1权重未找到，已回退基线权重；请先运行 python scripts/train.py --augment mosaic_strong --name B1_augment"
            ).strip()
        results[variant] = r

    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "ablation_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            ["方案", "改动点", "权重", "mAP@0.5", "mAP@0.5:0.95", "FPS",
             "检测人数", "占用座位", "备注"]
        )
        for v in VARIANTS:
            r = results[v]
            writer.writerow(
                [v, VARIANT_DESC[v], r["weights"], r.get("map50", 0),
                 r.get("map50_95", 0), r.get("fps", 0), r.get("persons", 0),
                 r.get("occupied", 0), r.get("note", "")]
            )

    a = results["A_baseline"]
    b1 = results["B1_augment"]
    b2 = results["B2_track"]
    b3 = results["B3_center_roi"]
    test_desc = args.demo_image or "合成测试帧"

    report = [
        "# A/B 实验报告：教室人员统计与座位占用检测",
        "",
        "## 实验设置",
        f"- 基线 A：{VARIANT_DESC['A_baseline']}",
        f"- 评估图片：{test_desc}",
        f"- FPS 重复次数：{args.repeats}",
        "- 控制变量：每组只改变一个因素，其余参数完全一致",
        "",
        "## 结果对比",
        "",
        "| 方案 | 改动点 | mAP@0.5 | FPS | 检测人数 | 占用座位 |",
        "|---|---|---|---|---|---|",
    ]
    for v in VARIANTS:
        r = results[v]
        report.append(
            f"| {v} | {VARIANT_DESC[v]} | {r.get('map50', 0):.4f} | "
            f"{r.get('fps', 0):.1f} | {r.get('persons', 0)} | {r.get('occupied', 0)} |"
        )

    report += [
        "",
        "## 3 条可复核的创新点表述",
        "",
        f"1. 针对密集场景人员召回不足问题，在【训练模块】作【强 Mosaic + Copy-Paste 数据增强】改动，"
        f"在【验证集】条件下得到【mAP@0.5={b1.get('map50', 0):.4f}】，"
        f"相对基线 A（{a.get('map50', 0):.4f}），提升 {pct(b1.get('map50', 0), a.get('map50', 0))}。",
        "",
        f"2. 针对跨帧重复计数问题，在【推理流水线】作【启用 ByteTrack 跟踪去重】改动，"
        f"在【{test_desc}】条件下得到【检测人数={b2.get('persons', 0)}】，"
        f"相对基线 A（{a.get('persons', 0)}），变化 {pct(b2.get('persons', 0), a.get('persons', 0))}。",
        "",
        f"3. 针对座位擦边误判问题，在【ROI 计数模块】作【中心点判定替代重叠阈值】改动，"
        f"在【{test_desc}】条件下得到【占用座位={b3.get('occupied', 0)}】，"
        f"相对基线 A（{a.get('occupied', 0)}），变化 {pct(b3.get('occupied', 0), a.get('occupied', 0))}。",
        "",
        "## 证据文件",
        "- `ablation_results.json`：各组原始指标",
        "- `ablation_results.csv`：可直接粘贴到 README 的对比表",
        "- 本报告：3 条标准化创新表述（数值由真实运行自动填充）",
    ]
    (out_dir / "ablation_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\n[完成] 结果已保存至 {out_dir}")


if __name__ == "__main__":
    main()
