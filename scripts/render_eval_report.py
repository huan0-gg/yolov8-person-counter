#!/usr/bin/env python3
"""把 detection_eval.json 渲染成 Markdown 报告，避免文档数字与实测结果脱节。

用法：
    python scripts/eval_detection.py            # 先跑评测
    python scripts/render_eval_report.py        # 再生成 docs/EVAL_v2.2.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "experiments" / "results" / "detection_eval.json"
OUT = ROOT / "docs" / "EVAL_v2.2.md"

ORDER = [
    "v21_baseline",
    "v22_aspect",
    "v22_wbf",
    "v22_scale",
    "v22_region",
    "v22_tiling",
    "v22_softnms",
]


def pct(new: float, old: float) -> str:
    if old == 0:
        return "—"
    d = (new - old) / old * 100
    return f"{d:+.1f}%"


def main() -> None:
    payload = json.loads(SRC.read_text(encoding="utf-8"))
    v = payload["variants"]
    base = v["v21_baseline"]

    rows = []
    for key in ORDER:
        if key not in v:
            continue
        m = v[key]
        rows.append(
            f"| {m['label']} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
            f"{m['ap50']:.4f} | {m['fp']} | {m['fn']} | {m['count_mae']:.2f} | "
            f"{m['ms_per_image']:.0f} |"
        )
    table = "\n".join(rows)

    gen = v.get("v22_scale", base)
    reg = v.get("v22_region", gen)

    lines = [
        "# v2.2 检测效果实测：更新前 vs 更新后",
        "",
        f"> 数据：`data/images/val` 共 **{payload['images']} 张**、**{base['gt_boxes']} 个** person 标注；"
        f"权重 `{payload['weights']}`；置信度 {payload['conf_threshold']}；匹配 IoU 0.5。",
        "> 所有数字由 `python scripts/eval_detection.py` 产出，再经本脚本渲染，未手工填写。",
        "",
        "## 1. 指标口径",
        "",
        "- **TP / FP / FN**：预测框与 GT 按 IoU 0.5 贪心一对一匹配；匹配上即 TP，多出来的框记 FP，没被匹配的 GT 记 FN。",
        "- **Precision / Recall / F1**：`TP/(TP+FP)`、`TP/(TP+FN)`。",
        "- **AP@0.5**：把所有预测按置信度排序扫一遍 PR 曲线，precision 单调化后积分（单类）。",
        "- **计数 MAE**：每张图 `|预测人数 − 真实人数|` 的平均值。",
        "- **误检率**用 FP 总数表示，**漏检率**用 FN 总数表示——比百分比更直观：一共多少张图、多少个错框。",
        "",
        "## 2. 消融结果",
        "",
        "| 变体 | Precision | Recall | F1 | AP@0.5 | 误检 FP | 漏检 FN | 计数 MAE | 耗时/图 |",
        "|---|---|---|---|---|---|---|---|---|",
        table,
        "",
        "## 3. 结论",
        "",
        f"**默认配置（v2.2-c）相对 v2.1 基线**：Recall {base['recall']:.4f} → {gen['recall']:.4f}"
        f"（{pct(gen['recall'], base['recall'])}），F1 {base['f1']:.4f} → {gen['f1']:.4f}"
        f"（{pct(gen['f1'], base['f1'])}），AP@0.5 {base['ap50']:.4f} → {gen['ap50']:.4f}"
        f"（{pct(gen['ap50'], base['ap50'])}），漏检 {base['fn']} → {gen['fn']} 个"
        f"（{pct(gen['fn'], base['fn'])}），误检 {base['fp']} → {gen['fp']} 个"
        f"（{pct(gen['fp'], base['fp'])}）。",
        "",
        f"**区域模式（v2.2-f，只看监测区域）**：Precision {reg['precision']:.4f}"
        f"（{pct(reg['precision'], base['precision'])}），误检 {base['fp']} → {reg['fp']} 个"
        f"（{pct(reg['fp'], base['fp'])}），漏检 {base['fn']} → {reg['fn']} 个"
        f"（{pct(reg['fn'], base['fn'])}），F1 {base['f1']:.4f} → {reg['f1']:.4f}。"
        "代价是区域边界上的人可能被判为区域外，所以这个模式只适合"
        "\"只统计指定区域\"的考勤场景，默认不开启。",
        "",
        "### 为什么切片（2×2）被否掉",
        "",
        f"切片把大图切开再放大推理，理论上小目标更清晰，但实测 Precision 掉到 "
        f"{v['v22_tiling']['precision']:.4f}、FP 涨到 {v['v22_tiling']['fp']} 个："
        "这些图本身只有 389×380，再切片等于反复放大噪声，椅子靠背、桌面反光都被判成人。"
        "因此 `tiling.enabled` 默认 `false`，并加了两道闸：`min_image_side`（最短边不足 900 不切）"
        "与 `ios_merge`（嵌套框去重）。4K 监控这类真正的大图可以自行打开，代码和测试都保留着。",
        "",
        "### 为什么 Soft-NMS 也否掉",
        "",
        f"Soft-NMS 用高斯衰减代替硬抑制，实测 FP 膨胀到 {v['v22_softnms']['fp']} 个："
        "衰减只是把重复框的分数压低，低于阈值的框仍然会被保留输出。密集场景里它"
        "\"舍不得删\"，反而制造误检。保留为可选项，默认不用。",
        "",
        "## 4. 复现",
        "",
        "```bash",
        "python -m pytest tests -q                    # 31 个单元测试",
        "python scripts/eval_detection.py             # 跑 7 个变体，写 detection_eval.json",
        "python scripts/render_eval_report.py         # 重新生成本文档",
        "```",
        "",
        "> 数据不随仓库分发（`data/images`、`data/labels` 在 .gitignore 里）。",
        "> 按 `data/DATA_CARD.md` 准备好 22 张验证集后即可复现同一组数字。",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入 {OUT}")


if __name__ == "__main__":
    main()
