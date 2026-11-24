#!/usr/bin/env python3
"""验证改进效果：在两张教室图上跑流水线并保存标注图 + 指标。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2

from src.pipeline import PersonCountingPipeline
from src.utils import load_config

cfg = load_config(ROOT / "configs" / "config.yaml")
WEIGHTS = None  # 跟随 configs/config.yaml，保证与界面、文档一致
OUT = ROOT / "assets"

report = {}
for name in ("classroom_empty", "classroom_crowded"):
    img = cv2.imread(str(OUT / f"{name}.jpg"))
    pipe = PersonCountingPipeline(cfg, variant="B3_center_roi", weights=WEIGHTS)
    annotated, result = pipe.process_image(img)
    cv2.imwrite(str(OUT / f"{name}_fixed.jpg"), annotated)
    report[name] = {
        "persons": result["total_persons"],
        "raw_persons": result["raw_persons"],
        "filtered_out": result["filtered_out"],
        "occupied": result["occupied_count"],
        "total_seats": result["total_seats"],
        "seat_source": result["seat_source"],
        "fps": result["fps"],
        "frame_ms": result["frame_ms"],
    }
    print(f"{name}: {json.dumps(report[name], ensure_ascii=False)}")

(OUT / "verify_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)
