#!/usr/bin/env python3
"""采集宣传视频所需的真实证据帧。

对同一张教室图分别用「v1.0 旧参数」与「v2.0 当前配置」跑一遍流水线，
把标注结果、检测框坐标、人数/座位数写入 docs/video/evidence/，
供 scripts/make_promo.py 合成视频时直接调用，保证画面里的每个数字都来自真实运行。

用法：
    python scripts/collect_promo_assets.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import PersonCountingPipeline  # noqa: E402
from src.utils import load_config  # noqa: E402

OUT = ROOT / "docs" / "video" / "evidence"
OUT.mkdir(parents=True, exist_ok=True)

CASES = [
    # name, image, variant, weights, overrides
    ("crowded_old", "classroom_crowded.jpg", "B3_center_roi", None,
     {"conf": 0.25, "imgsz": 640}),
    ("crowded_new", "classroom_crowded.jpg", "B3_center_roi", None, {}),
    ("empty_new", "classroom_empty.jpg", "B3_center_roi", None, {}),
]


def main() -> int:
    cfg = load_config(ROOT / "configs" / "config.yaml")
    # 不覆盖权重：与 app.py 用同一份默认配置，保证视频、界面、文档三处数字一致
    weights = None
    print("权重:", cfg["model"]["weights"], "| imgsz/conf:", cfg["model"]["imgsz"], cfg["model"]["conf"])
    summary = {}

    for name, img_name, variant, _w, overrides in CASES:
        img_path = ROOT / "assets" / img_name
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[skip] 读不到 {img_path}")
            continue

        cfg_v = copy.deepcopy(cfg)
        cfg_v["model"]["conf"] = overrides.get("conf", cfg["model"]["conf"])
        cfg_v["model"]["imgsz"] = overrides.get("imgsz", cfg["model"]["imgsz"])

        pipe = PersonCountingPipeline(cfg=cfg_v, variant=variant, weights=weights)
        annotated, result = pipe.process_image(frame)

        cv2.imwrite(str(OUT / f"{name}.png"), annotated)
        boxes = [
            {"bbox": [int(v) for v in d["bbox"]], "conf": round(float(d["conf"]), 3)}
            for d in result.get("detections", [])
        ]
        summary[name] = {
            "image": img_name,
            "variant": variant,
            "conf": cfg_v["model"]["conf"],
            "imgsz": cfg_v["model"]["imgsz"],
            "size": [int(frame.shape[1]), int(frame.shape[0])],
            "persons": int(result["total_persons"]),
            "occupied": int(result["occupied_count"]),
            "seats": int(result.get("total_seats", 0)),
            "seat_boxes": [[int(v) for v in s] for s in result.get("seat_boxes", [])],
            "occupied_flags": [bool(v) for v in result.get("occupied", [])],
            "boxes": boxes,
        }
        print(f"{name:12s} 人数={result['total_persons']:2d} "
              f"占用座位={result['occupied_count']:2d}/{result.get('total_seats', 0)}")

    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("写入", OUT / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
