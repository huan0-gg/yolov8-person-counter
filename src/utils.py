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


def resolve_dataset_yaml(data: str | Path = "data/dataset.yaml") -> Path:
    """生成一份 path 为绝对路径的临时 dataset.yaml，供训练/评估可靠读取。

    YOLO 的 dataset.yaml 里 path 是相对项；不同工作目录会导致解析错位。
    本函数保留仓库内的模板不变，只在运行时生成绝对路径副本。
    """
    src = Path(data).resolve()
    if not src.exists():
        raise FileNotFoundError(f"数据集配置不存在: {src}")
    with src.open("r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    data_root = src.parent
    content["path"] = str(data_root).replace("\\", "/")
    out = data_root / "dataset_runtime.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, allow_unicode=True, sort_keys=False)
    return out


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
