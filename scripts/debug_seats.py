#!/usr/bin/env python3
"""调试：测试桌椅类检测，评估自适应座位框可行性。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
from ultralytics import YOLO

WEIGHTS = r"D:\gg\yolo26n.pt"
# COCO: 56=chair, 62=dining table, 57=couch, 60=bed
SEAT_CLASSES = [56, 62, 57, 60]
model = YOLO(WEIGHTS)

for path in ["assets/classroom_empty.jpg", "assets/classroom_crowded.jpg", "assets/demo.jpg"]:
    img = cv2.imread(str(ROOT / path))
    print(f"\n=== {path} {img.shape} ===")
    for conf in (0.25, 0.10):
        r = model.predict(
            source=img, conf=conf, iou=0.45, imgsz=960,
            device="cpu", classes=SEAT_CLASSES, verbose=False,
        )[0]
        if r.boxes is None or len(r.boxes) == 0:
            print(f"  conf={conf}: 0 seat-like boxes")
            continue
        cls_counts: dict[int, int] = {}
        for c in r.boxes.cls:
            cid = int(c)
            cls_counts[cid] = cls_counts.get(cid, 0) + 1
        print(f"  conf={conf}: total={len(r.boxes)} classes={cls_counts}")
        boxes = sorted(
            [(int(c), round(float(cf), 2)) for c, cf in zip(r.boxes.cls, r.boxes.conf)],
            key=lambda x: -x[1],
        )[:6]
        print(f"    top(cls,conf)={boxes}")
