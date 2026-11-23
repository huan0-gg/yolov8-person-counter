"""创新点 2：基于 track_id 的跨帧去重计数。

基线逐帧独立计数会把同一个人重复统计；启用 ByteTrack 后，
每个 track_id 在整个视频中只计一次，显著降低重复计数。
"""
from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple


def deduplicate_by_track(
    detections: List[Dict[str, Any]], seen_ids: Set[int]
) -> Tuple[List[Dict[str, Any]], Set[int]]:
    """按 track_id 去重：第一次出现的 id 计入，后续帧跳过。

    参数:
        detections: 当前帧检测结果（需含 track_id 字段）
        seen_ids: 历史已计入的 track_id 集合（原地更新）
    返回:
        (本帧新增人员列表, 更新后的 seen_ids)
    """
    unique: List[Dict[str, Any]] = []
    for det in detections:
        tid = det.get("track_id")
        if tid is None:
            # 没有 track_id（跟踪未生效）时退化为普通计数
            unique.append(det)
            continue
        if tid not in seen_ids:
            seen_ids.add(tid)
            unique.append(det)
    return unique, seen_ids
