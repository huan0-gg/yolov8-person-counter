#!/usr/bin/env python3
"""调试：对比不同推理参数下的人员检测数量，用于定位漏检原因。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
from ultralytics import YOLO

WEIGHTS = r"D:\gg\yolo26n.pt"
IMAGES = ["assets/classroom_crowded.jpg", "assets/classroom_empty.jpg"]

model = YOLO(WEIGHTS)
for path in IMAGES:
    img = cv2.imread(str(ROOT / path))
    print(f"\n=== {path}  shape={img.shape} ===")
    for imgsz in (640, 960):
        for conf in (0.25, 0.10):
            r = model.predict(
                source=img, conf=conf, iou=0.45, imgsz=imgsz,
                device="cpu", classes=[0], verbose=False,
            )[0]
            n = 0 if r.boxes is None else len(r.boxes)
            confs = [] if r.boxes is None else [round(float(c), 2) for c in r.boxes.conf[:6]]
            print(f"  imgsz={imgsz} conf={conf}: persons={n} top_conf={confs}")
