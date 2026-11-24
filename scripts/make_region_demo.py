#!/usr/bin/env python3
"""生成"更新前 / 更新后"对照图（含指定区域标注），用于文档与 Release 说明。

每个场景单独标定监测区域（真实部署也是这样：一台设备一个机位、一套多边形）。
用法：
    python scripts/make_region_demo.py
输出：
    docs/region/before_*.jpg   更新前（v2.1：单尺度、原宽高比过滤、无区域标注）
    docs/region/after_*.jpg    更新后（v2.2：多尺度 + 区域标注 + 区域内计数）
    docs/region/dim_*.jpg      更新后（区域外压暗，现场投屏更醒目）
    docs/region/compare_*.jpg  左右并排对照
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.pipeline import PersonCountingPipeline
from src.utils import load_config

OUT = ROOT / "docs" / "region"

# 场景 -> (图片, 监测区域多边形)。多边形按画面透视手工标定，实际部署改配置即可。
SCENES = {
    "dense": {
        "image": "assets/ai/classroom_dense.jpg",
        "label": "考勤区 A（左半区）",
        "points": [[0.02, 0.13], [0.60, 0.145], [0.66, 0.86], [0.00, 0.84]],
    },
    "seat": {
        "image": "assets/ai/classroom_seat.jpg",
        "label": "监测区域",
        "points": [[0.01, 0.30], [0.99, 0.27], [0.99, 0.99], [0.01, 1.00]],
    },
    "empty": {
        "image": "assets/ai/classroom_empty.jpg",
        "label": "座位区",
        "points": [[0.06, 0.48], [0.96, 0.46], [0.99, 1.00], [0.02, 1.00]],
    },
    "meeting": {
        "image": "assets/ai/meeting_room.jpg",
        "label": "会议桌区",
        "points": [[0.07, 0.22], [0.95, 0.25], [0.93, 0.95], [0.04, 0.93]],
    },
}


def build_pipeline(mode: str, points, label: str = "监测区域") -> PersonCountingPipeline:
    cfg = copy.deepcopy(load_config(ROOT / "configs" / "config.yaml"))
    cfg["seat_grid"]["enabled"] = False  # 对照图只讲"检测框 + 监测区域"，不叠加座位网格
    cfg["seats"] = []                    # 同时清掉旧版固定坐标回退，避免图上出现两套框
    cfg["regions"]["items"] = [
        {
            "name": "monitor",
            "label": label,
            "points": points,
            "color": [170, 212, 45],
            "count": True,
        }
    ]
    if mode == "before":
        cfg["roi"]["aspect_range"] = [0.50, 4.00]
        cfg["regions"]["enabled"] = False
        cfg["model"]["scales"] = []
        cfg["model"]["scales_image"] = False
        cfg["model"]["ios_merge"] = 0.0
        cfg["model"]["tiling"]["enabled"] = False
        cfg["display"]["show_regions"] = False
    else:
        cfg["display"]["show_regions"] = True
    cfg["display"]["dim_outside"] = mode == "dim"
    return PersonCountingPipeline(cfg, variant="B3_center_roi")


def add_caption(img: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    """左上角加标题条（PIL 渲染中文，避免 cv2 中文变问号）。"""
    from PIL import Image, ImageDraw, ImageFont

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    font = None
    for p in (r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc"):
        if Path(p).exists():
            font = ImageFont.truetype(p, 30)
            break
    if font is None:
        return img
    bbox = dr.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0] + 28, bbox[3] - bbox[1] + 18
    dr.rectangle([16, 14, 16 + w, 14 + h], fill=tuple(reversed(color)))
    dr.text((30, 22 - bbox[1]), text, font=font, fill=(10, 18, 24))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {}

    for name, scene in SCENES.items():
        frame = cv2.imread(str(ROOT / scene["image"]))
        if frame is None:
            print(f"跳过（缺文件）{scene['image']}")
            continue
        before_pipe = build_pipeline("before", scene["points"], scene["label"])
        after_pipe = build_pipeline("after", scene["points"], scene["label"])
        dim_pipe = build_pipeline("dim", scene["points"], scene["label"])

        b_img, b_res = before_pipe.process_image(frame)
        a_img, a_res = after_pipe.process_image(frame)
        d_img, d_res = dim_pipe.process_image(frame)

        b_img = add_caption(b_img, f"更新前 v2.1｜{b_res['total_persons']} 人")
        a_img = add_caption(
            a_img,
            f"更新后 v2.2｜{a_res['total_persons']} 人（区域内 {a_res['inside_persons']}）",
        )
        d_img = add_caption(
            d_img,
            f"区域模式｜区域内 {d_res['inside_persons']} 人，区域外不计",
        )

        cv2.imwrite(str(OUT / f"before_{name}.jpg"), b_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(OUT / f"after_{name}.jpg"), a_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(OUT / f"dim_{name}.jpg"), d_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

        gap = 12
        h = max(b_img.shape[0], a_img.shape[0])
        canvas = np.full((h, b_img.shape[1] + a_img.shape[1] + gap, 3), 18, dtype=np.uint8)
        canvas[: b_img.shape[0], : b_img.shape[1]] = b_img
        canvas[: a_img.shape[0], b_img.shape[1] + gap:] = a_img
        cv2.imwrite(str(OUT / f"compare_{name}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])

        stats[name] = {
            "before_persons": b_res["total_persons"],
            "after_persons": a_res["total_persons"],
            "after_inside": a_res["inside_persons"],
            "after_outside": a_res["outside_persons"],
            "region_only": d_res["inside_persons"],
            "scales": a_res.get("scales"),
            "before_ms": b_res.get("frame_ms"),
            "after_ms": a_res.get("frame_ms"),
            "size": [frame.shape[1], frame.shape[0]],
        }
        print(name, stats[name], flush=True)

    (OUT / "demo_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

