#!/usr/bin/env python3
"""检测级评测：把"更新前 vs 更新后"的准确率、误检、漏检量化出来。

与 scripts/evaluate.py（调 ultralytics 的 mAP）不同，这里按**部署时真实发生的事**
来算指标：

1. 用项目自己的流水线（含框过滤、融合、多尺度）出结果；
2. 与人工标注 GT 按 IoU 0.5 做贪心匹配，得到 TP / FP / FN；
3. 输出 Precision / Recall / F1 / 误检数 / 漏检数 / 计数绝对误差；
4. 再扫一遍置信度得到 PR 曲线，算单类 AP@0.5（自算，保证各变体口径一致）。

用法：
    python scripts/eval_detection.py                  # 跑全部变体
    python scripts/eval_detection.py --variants v21_baseline,v22_scale,v22_region
    python scripts/eval_detection.py --limit 8        # 先小样本试跑
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2

from src.metrics import compute_ap, load_yolo_labels, match_predictions, prf
from src.pipeline import PersonCountingPipeline
from src.utils import load_config, resolve_dataset_yaml

Box = Tuple[float, float, float, float]

# 变体定义：名字 -> 配置覆盖。全部在同一份权重、同一批图片上跑，只改推理侧策略。
VARIANTS: Dict[str, Dict[str, Any]] = {
    "v21_baseline": {
        "label": "v2.1 基线（单程 + NMS + 原宽高比过滤）",
        "tiling": False,
        "aspect_range": [0.50, 4.00],
        "merge": "nms",
        "scales": [],
        "ios_merge": 0.0,
    },
    "v22_aspect": {
        "label": "v2.2-a（放宽宽高比过滤）",
        "tiling": False,
        "aspect_range": [0.30, 5.00],
        "merge": "nms",
        "scales": [],
        "ios_merge": 0.0,
    },
    "v22_wbf": {
        "label": "v2.2-b（+ WBF 融合）",
        "tiling": False,
        "aspect_range": [0.30, 5.00],
        "merge": "wbf",
        "scales": [],
        "ios_merge": 0.0,
    },
    "v22_scale": {
        "label": "v2.2-c（+ 960/1152 多尺度 + 包含式去重）",
        "tiling": False,
        "aspect_range": [0.30, 5.00],
        "merge": "wbf",
        "scales": [960, 1152],
        "ios_merge": 0.90,
    },
    "v22_tiling": {
        "label": "v2.2-d（2×2 切片 + WBF，强制关闭小图降级）",
        "tiling": True,
        "aspect_range": [0.30, 5.00],
        "merge": "wbf",
        "scales": [],
        "ios_merge": 0.90,
    },
    "v22_softnms": {
        "label": "v2.2-e（切片 + Soft-NMS）",
        "tiling": True,
        "aspect_range": [0.30, 5.00],
        "merge": "soft_nms",
        "scales": [],
        "ios_merge": 0.90,
    },
    "v22_region": {
        "label": "v2.2-f（多尺度 + 丢弃区域外检测，推荐）",
        "tiling": False,
        "aspect_range": [0.30, 5.00],
        "merge": "wbf",
        "scales": [960, 1152],
        "ios_merge": 0.90,
        "region_filter": True,
    },
}


def resolve_val_dir(data_yaml: Path) -> Tuple[Path, Path]:
    import yaml

    content = yaml.safe_load(resolve_dataset_yaml(data_yaml).read_text(encoding="utf-8"))
    base = Path(content["path"])
    return base / content["val"], base / str(content["val"]).replace("images", "labels")


def run_variant(
    name: str,
    spec: Dict[str, Any],
    images: List[Path],
    labels_dir: Path,
    base_cfg: Dict[str, Any],
    conf_thr: float,
    imgsz: int = 0,
    warmup: int = 1,
    nms_iou: float = 0.0,
) -> Dict[str, Any]:
    import copy

    cfg = copy.deepcopy(base_cfg)
    cfg["roi"]["aspect_range"] = list(spec["aspect_range"])
    if imgsz:
        cfg["model"]["imgsz"] = int(imgsz)
    if nms_iou:
        cfg["model"]["nms_iou"] = float(nms_iou)
    cfg["model"]["merge"] = spec["merge"]
    cfg["model"]["scales"] = list(spec.get("scales") or [])
    cfg["model"]["scales_image"] = bool(spec.get("scales"))
    cfg["model"]["ios_merge"] = float(spec.get("ios_merge", 0.90))
    cfg["model"].setdefault("tiling", {})
    cfg["model"]["tiling"]["merge"] = spec["merge"]
    cfg["model"]["tiling"]["enabled"] = bool(spec["tiling"])
    cfg["model"]["tiling"]["image"] = bool(spec["tiling"])
    # 切片变体强制关闭"小图自动降级"，才能看出切片本身的真实影响
    cfg["model"]["tiling"]["min_image_side"] = 0
    use_regions = bool(spec.get("region_filter", False))
    cfg["regions"]["enabled"] = use_regions
    cfg["regions"]["filter_outside"] = use_regions

    pipe = PersonCountingPipeline(cfg, variant="B3_center_roi")
    pipe.tiling_image = bool(spec["tiling"])

    all_flags: List[Tuple[float, bool]] = []
    tp = fp = fn = 0
    count_errors: List[float] = []
    total_gt = 0
    times: List[float] = []
    per_image: List[Dict[str, Any]] = []

    # 预热：第一次推理含模型加载/算子编译，不计入速度
    if warmup and images:
        warm_frame = cv2.imread(str(images[0]))
        if warm_frame is not None:
            for _ in range(warmup):
                pipe.process(warm_frame, draw=False, tiling=bool(spec["tiling"]))

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        gt = load_yolo_labels(labels_dir / f"{img_path.stem}.txt", w, h)
        t0 = time.perf_counter()
        _, result = pipe.process(frame, draw=False, tiling=bool(spec["tiling"]))
        times.append(time.perf_counter() - t0)

        preds = result["detections"]
        flags, miss = match_predictions(preds, gt, 0.5)
        all_flags.extend(flags)
        t = sum(1 for _, is_tp in flags if is_tp)
        f = len(flags) - t
        tp += t
        fp += f
        fn += miss
        total_gt += len(gt)
        count_errors.append(abs(len(preds) - len(gt)))
        per_image.append(
            {"image": img_path.name, "gt": len(gt), "pred": len(preds), "tp": t, "fp": f, "fn": miss}
        )

    metrics = prf(tp, fp, fn)
    n = max(len(times), 1)
    metrics.update(
        {
            "variant": name,
            "label": spec["label"],
            "images": len(times),
            "gt_boxes": total_gt,
            "ap50": round(compute_ap(all_flags, total_gt), 4),
            "count_mae": round(sum(count_errors) / max(len(count_errors), 1), 2),
            "ms_per_image": round(1000 * sum(times) / n, 1),
            "fps": round(n / max(sum(times), 1e-6), 2),
            "conf_thr": conf_thr,
            "per_image": per_image,
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="检测级前后对比评测")
    parser.add_argument("--data", default=str(ROOT / "data" / "dataset.yaml"))
    parser.add_argument("--variants", default=",".join(VARIANTS.keys()))
    parser.add_argument("--limit", type=int, default=0, help="只用前 N 张（调试用）")
    parser.add_argument(
        "--conf",
        type=float,
        default=float(load_config(ROOT / "configs" / "config.yaml")["model"]["conf"]),
    )
    parser.add_argument("--imgsz", type=int, default=0, help="覆盖推理分辨率（0=跟随配置）")
    parser.add_argument("--nms-iou", type=float, default=0.0, help="覆盖融合 IoU 阈值（0=跟随配置）")
    parser.add_argument("--warmup", type=int, default=1, help="预热次数，避免首次加载影响耗时")
    parser.add_argument("--out", default=str(ROOT / "experiments" / "results" / "detection_eval.json"))
    args = parser.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(f"找不到数据集配置：{data_yaml}\n请先按 data/DATA_CARD.md 准备验证集。")
    images_dir, labels_dir = resolve_val_dir(data_yaml)
    images = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"验证集为空：{images_dir}")

    base_cfg = load_config(ROOT / "configs" / "config.yaml")
    base_cfg["model"]["conf"] = float(args.conf)

    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    results: Dict[str, Any] = {}
    for name in names:
        if name not in VARIANTS:
            raise SystemExit(f"未知变体 {name}，可选：{', '.join(VARIANTS)}")
        print(f"[eval] {name} ...", flush=True)
        results[name] = run_variant(
            name,
            VARIANTS[name],
            images,
            labels_dir,
            base_cfg,
            args.conf,
            args.imgsz,
            args.warmup,
            args.nms_iou,
        )
        m = results[name]
        print(
            f"  P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
            f"AP50={m['ap50']:.4f} FP={m['fp']} FN={m['fn']} {m['ms_per_image']:.0f}ms/img",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": len(images),
        "conf_threshold": args.conf,
        "iou_threshold": 0.5,
        "weights": base_cfg["model"]["weights"],
        "variants": results,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
