"""通用工具：配置加载、随机种子、指标格式化。"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def load_config(path: str | Path = "configs/config.yaml") -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class Timer:
    """简单的耗时/FPS 统计器。"""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.count = 0

    def tick(self) -> None:
        self.count += 1

    def fps(self) -> float:
        elapsed = time.perf_counter() - self.start
        return self.count / elapsed if elapsed > 0 else 0.0

    def elapsed(self) -> float:
        return time.perf_counter() - self.start


def format_metric_table(results: Dict[str, Dict[str, float]]) -> str:
    """把 A/B 实验结果渲染成 Markdown 表格。"""
    if not results:
        return "| 方案 | mAP@0.5 | mAP@0.5:0.95 | FPS | 计数准确率 |\n|---|---|---|---|---|"
    headers = ["方案", "mAP@0.5", "mAP@0.5:0.95", "FPS", "计数准确率"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for name, metrics in results.items():
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.1f} | {:.4f} |".format(
                name,
                metrics.get("map50", 0.0),
                metrics.get("map50_95", 0.0),
                metrics.get("fps", 0.0),
                metrics.get("count_acc", 0.0),
            )
        )
    return "\n".join(lines)
