"""创新点 1：数据增强策略（训练阶段）。

用不同增强配置训练同一数据集，形成可对比的 A/B 实验组。
baseline = 官方默认增强；mosaic_strong = 强 Mosaic + Copy-Paste（针对密集/遮挡场景）；
no_mosaic = 关闭 Mosaic（对照组，验证增强是否真的有用）。
"""
from __future__ import annotations

from typing import Any, Dict

AUGMENT_PRESETS: Dict[str, Dict[str, float]] = {
    "default": {
        "mosaic": 1.0,
        "copy_paste": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
    },
    "mosaic_strong": {
        "mosaic": 1.0,
        "copy_paste": 0.3,
        "hsv_h": 0.03,
        "hsv_s": 0.8,
        "hsv_v": 0.5,
        "degrees": 5.0,
        "translate": 0.15,
        "scale": 0.7,
        "fliplr": 0.5,
    },
    "no_mosaic": {
        "mosaic": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
    },
}


def get_augment_kwargs(mode: str = "default") -> Dict[str, Any]:
    """返回可直接传给 YOLO.train() 的增强参数。"""
    if mode not in AUGMENT_PRESETS:
        raise ValueError(f"未知增强模式: {mode}，可选 {list(AUGMENT_PRESETS)}")
    return dict(AUGMENT_PRESETS[mode])
