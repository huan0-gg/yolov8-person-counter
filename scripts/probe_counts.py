#!/usr/bin/env python3
"""在密集教室图上复现「改进前 13 人 / 改进后 23 人」，确认口径。

分别用两组权重与两组推理参数交叉测试，输出真实检出数，
用于确认宣传视频中引用的数字可以复现。
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import PersonCountingPipeline  # noqa: E402
from src.utils import load_config  # noqa: E402

cfg0 = load_config(ROOT / "configs" / "config.yaml")
img = cv2.imread(str(ROOT / "assets" / "classroom_crowded.jpg"))

WEIGHTS = ["yolov8n.pt", r"D:\gg\yolo26n.pt"]
PARAMS = [("旧参数", 640, 0.25), ("新参数", 960, 0.12)]

for w in WEIGHTS:
    for tag, imgsz, conf in PARAMS:
        cfg = copy.deepcopy(cfg0)
        cfg["model"]["imgsz"] = imgsz
        cfg["model"]["conf"] = conf
        pipe = PersonCountingPipeline(cfg, variant="B3_center_roi", weights=w)
        _, res = pipe.process_image(img)
        print(f"{Path(w).name:14s} {tag} {imgsz}/{conf}  人数={res['total_persons']:3d} "
              f"占用={res['occupied_count']:2d}/{res['total_seats']}  "
              f"过滤={res['filtered_out']}")
