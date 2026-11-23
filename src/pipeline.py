"""完整推理流水线（改进版）：检测 -> 过滤去重 -> (跟踪) -> 座位关联 -> 可视化。

改进：
- 接入 seat_layout 透视网格与改进计数器；
- 首帧 FPS 用实际耗时估算，不再显示 0；
- 结果含 filtered_out / seat_source 便于复核。
"""
from __future__ import annotations

import time
from typing import Any, Dict

import cv2
import numpy as np

from .detector import PersonDetector
from .roi_counter import SeatROICounter
from .tracker import deduplicate_by_track

VARIANTS: Dict[str, Dict[str, Any]] = {
    "A_baseline": {"track": False, "count_mode": "overlap"},
    "B1_augment": {"track": False, "count_mode": "overlap"},
    "B2_track": {"track": True, "count_mode": "overlap"},
    "B3_center_roi": {"track": False, "count_mode": "center"},
    "B_all": {"track": True, "count_mode": "center"},
}


class PersonCountingPipeline:
    def __init__(
        self,
        cfg: Dict[str, Any],
        variant: str = "A_baseline",
        weights: str | None = None,
        detect_overrides: Dict[str, Any] | None = None,
    ) -> None:
        model_cfg = cfg["model"]
        roi_cfg = cfg["roi"]
        track_cfg = cfg["track"]

        variant_cfg = VARIANTS.get(variant, VARIANTS["A_baseline"])
        self.variant = variant
        self.track_enabled = variant_cfg["track"] and track_cfg.get("enabled", True)
        self.count_mode = variant_cfg["count_mode"]
        self.display_cfg = cfg.get("display", {})

        det_kwargs: Dict[str, Any] = dict(
            weights=weights or model_cfg["weights"],
            conf=float(model_cfg["conf"]),
            iou=float(model_cfg["iou"]),
            imgsz=int(model_cfg["imgsz"]),
            device=model_cfg.get("device", "auto"),
            classes=tuple(model_cfg.get("classes", [0])),
            augment=bool(model_cfg.get("augment", False)),
            nms_iou=float(model_cfg.get("nms_iou", 0.55)),
        )
        if detect_overrides:
            det_kwargs.update(detect_overrides)
        self.detector = PersonDetector(**det_kwargs)

        self.counter = SeatROICounter(
            cfg=cfg,
            mode=self.count_mode,
            overlap_threshold=float(roi_cfg.get("overlap_threshold", 0.30)),
            min_area_ratio=float(roi_cfg.get("min_area_ratio", 0.002)),
        )
        self.tracker_name = track_cfg.get("tracker", "bytetrack.yaml")
        self.seen_ids: set[int] = set()
        self._frame_times: list[float] = []

    def reset(self) -> None:
        self.seen_ids.clear()
        self._frame_times.clear()

    def _fps(self) -> float:
        if not self._frame_times:
            return 0.0
        avg = sum(self._frame_times) / len(self._frame_times)
        return round(1.0 / avg, 2) if avg > 0 else 0.0

    def process(
        self, frame: np.ndarray, draw: bool = True
    ) -> tuple[np.ndarray | None, Dict[str, Any]]:
        t0 = time.perf_counter()
        if self.track_enabled:
            detections = self.detector.track(frame, tracker=self.tracker_name)
            detections, self.seen_ids = deduplicate_by_track(detections, self.seen_ids)
        else:
            detections = self.detector.detect(frame)

        result = self.counter.count(detections, frame.shape)
        result["variant"] = self.variant

        elapsed = time.perf_counter() - t0
        self._frame_times.append(elapsed)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        result["fps"] = self._fps()
        result["frame_ms"] = round(elapsed * 1000, 1)

        if draw:
            from .visualize import draw_result

            annotated = draw_result(
                frame, detections, result,
                show_track=self.track_enabled,
                display_cfg=self.display_cfg,
            )
            return annotated, result
        return None, result

    def process_image(self, image: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        self.reset()
        annotated, result = self.process(image, draw=True)
        return annotated, result

    def process_video(
        self, video_path: str, output_path: str | None = None, max_frames: int = 0
    ) -> Dict[str, Any]:
        self.reset()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        peak_people = 0
        peak_occupied = 0
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, result = self.process(frame, draw=True)
            peak_people = max(peak_people, result["total_persons"])
            peak_occupied = max(peak_occupied, result["occupied_count"])
            if writer is not None:
                writer.write(annotated)
            frame_idx += 1
            if max_frames and frame_idx >= max_frames:
                break

        cap.release()
        if writer is not None:
            writer.release()
        return {
            "frames": frame_idx,
            "peak_persons": peak_people,
            "peak_occupied": peak_occupied,
            "fps": self._fps(),
            "output": output_path,
        }
