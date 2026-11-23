"""评估工具：精度(mAP)、速度(FPS)、单帧计数结果。

被 scripts/evaluate.py 与 scripts/run_ablation.py 共用，
确保 A/B 各组使用完全相同的评估流程。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from .pipeline import PersonCountingPipeline
from .utils import load_config

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装依赖: pip install -r requirements.txt") from exc


def make_test_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """生成合成座位图，用于无真实数据时的 FPS 基准测试。"""
    frame = np.full((height, width, 3), 240, dtype=np.uint8)
    for r in range(4):
        for c in range(4):
            x1 = int(0.06 * width) + c * int(0.23 * width)
            y1 = int(0.28 * height) + r * int(0.17 * height)
            x2 = min(x1 + int(0.20 * width), width - 1)
            y2 = min(y1 + int(0.14 * height), height - 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (210, 210, 210), 2)
    return frame


def evaluate_variant(
    cfg: Dict[str, Any],
    variant: str,
    weights: str,
    data: str | None = None,
    demo_image: str | None = None,
    warmup: int = 3,
    repeats: int = 20,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"variant": variant, "weights": weights}

    # 1) 精度：需要验证集
    if data and Path(data).exists():
        model = YOLO(weights)
        metrics = model.val(
            data=data, device=cfg["model"].get("device", "cpu"), verbose=False
        )
        result["map50"] = round(float(metrics.box.map50), 4)
        result["map50_95"] = round(float(metrics.box.map), 4)
    else:
        result["map50"] = 0.0
        result["map50_95"] = 0.0
        result["note"] = "未提供验证集，mAP 记为 0；准备数据后重跑即可填入真实值"

    # 2) 速度：优先用真实图片，否则用合成帧
    if demo_image and Path(demo_image).exists():
        frame = cv2.imread(demo_image)
        if frame is None:
            frame = make_test_frame()
    else:
        frame = make_test_frame()

    pipe = PersonCountingPipeline(cfg, variant=variant, weights=weights)
    for _ in range(warmup):
        pipe.process(frame, draw=False)
    t0 = time.perf_counter()
    for _ in range(repeats):
        pipe.process(frame, draw=False)
    result["fps"] = round(repeats / max(time.perf_counter() - t0, 1e-6), 2)

    # 3) 单帧计数结果
    pipe.reset()
    _, count_result = pipe.process(frame, draw=False)
    result["persons"] = count_result["total_persons"]
    result["occupied"] = count_result["occupied_count"]
    result["total_seats"] = count_result["total_seats"]
    return result
