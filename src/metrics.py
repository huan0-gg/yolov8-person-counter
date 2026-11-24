"""检测评测指标：TP/FP/FN 匹配、P/R/F1、单类 AP。

放在 src/ 里而不是脚本里，是为了让单元测试和文档生成共用同一套实现 ——
"指标怎么算的"只有一处代码，不会出现脚本改了口径、报告还是旧数字的情况。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

Box = Tuple[float, float, float, float]


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_yolo_labels(label_path: Path, width: int, height: int, cls: int = 0) -> List[Box]:
    """读 YOLO 标注（cls cx cy w h，归一化），只保留指定类别，返回像素 xyxy。"""
    boxes: List[Box] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        if int(float(parts[0])) != cls:
            continue
        cx, cy, w, h = (float(v) for v in parts[1:5])
        boxes.append(
            (
                (cx - w / 2) * width,
                (cy - h / 2) * height,
                (cx + w / 2) * width,
                (cy + h / 2) * height,
            )
        )
    return boxes


def match_predictions(
    preds: List[Dict[str, Any]], gt: List[Box], iou_thr: float = 0.5
) -> Tuple[List[Tuple[float, bool]], int]:
    """贪心匹配：按置信度降序，每个预测找 IoU 最大的未匹配 GT。

    返回 ([(conf, is_tp), ...], 漏检数)。一对一匹配保证同一 GT 不会重复计 TP。
    """
    order = sorted(range(len(preds)), key=lambda i: -preds[i]["conf"])
    used = [False] * len(gt)
    flags: List[Tuple[float, bool]] = []
    for idx in order:
        pb = preds[idx]["bbox"]
        best_j, best_iou = -1, 0.0
        for j, gb in enumerate(gt):
            if used[j]:
                continue
            ov = iou(pb, gb)
            if ov > best_iou:
                best_j, best_iou = j, ov
        if best_j >= 0 and best_iou >= iou_thr:
            used[best_j] = True
            flags.append((float(preds[idx]["conf"]), True))
        else:
            flags.append((float(preds[idx]["conf"]), False))
    return flags, sum(1 for u in used if not u)


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compute_ap(flags: List[Tuple[float, bool]], n_gt: int) -> float:
    """单类 AP@0.5：分数降序累积 PR 曲线，precision 单调化后积分（COCO 风格）。"""
    if n_gt <= 0 or not flags:
        return 0.0
    flags = sorted(flags, key=lambda x: -x[0])
    tp = fp = 0
    rec: List[float] = []
    prec: List[float] = []
    for _, is_tp in flags:
        if is_tp:
            tp += 1
        else:
            fp += 1
        rec.append(tp / n_gt)
        prec.append(tp / (tp + fp))
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    ap = 0.0
    prev_r = 0.0
    for r, p in zip(rec, prec):
        ap += (r - prev_r) * p
        prev_r = r
    return float(ap)


def count_mae(errors: Sequence[float]) -> float:
    """平均每图计数绝对误差（|预测人数 - 真实人数|）。"""
    return round(sum(errors) / len(errors), 2) if errors else 0.0
