"""完整推理流水线（v2.2）：检测 -> 融合去重 -> (跟踪) -> 区域归属 -> 座位关联 -> 可视化。

v2.2 新增：
- **切片推理**：图片走"全图 + 2×2 切片"的高精度模式，视频/摄像头走单程快速模式，
  两者共用同一套后处理，指标口径可分别核算；
- **指定区域**：检测结果里标注区域、统计区域内人数，可选丢弃区域外误检；
- 结果新增 tiling / regions / inside_persons 字段，方便界面与评测核对。
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import cv2
import numpy as np

from .detector import PersonDetector
from .region import Region, annotate_detections, filter_outside, parse_regions, region_summary
from .roi_counter import SeatROICounter
from .tracker import deduplicate_by_track
from .utils import resolve_weights

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
        tiling: bool | None = None,
    ) -> None:
        model_cfg = cfg["model"]
        roi_cfg = cfg["roi"]
        track_cfg = cfg["track"]

        variant_cfg = VARIANTS.get(variant, VARIANTS["A_baseline"])
        self.variant = variant
        self.track_enabled = variant_cfg["track"] and track_cfg.get("enabled", True)
        self.count_mode = variant_cfg["count_mode"]
        self.display_cfg = cfg.get("display", {})

        tiling_cfg = dict(model_cfg.get("tiling") or {})
        if tiling is False:
            tiling_cfg["enabled"] = False

        # 多尺度推理：图片默认开（精度优先），视频默认关（帧率优先）
        scales_cfg = [int(s) for s in (model_cfg.get("scales") or []) if int(s) > 0]

        det_kwargs: Dict[str, Any] = dict(
            weights=resolve_weights(weights or model_cfg["weights"]),
            conf=float(model_cfg["conf"]),
            iou=float(model_cfg["iou"]),
            imgsz=int(model_cfg["imgsz"]),
            device=model_cfg.get("device", "auto"),
            classes=tuple(model_cfg.get("classes", [0])),
            augment=bool(model_cfg.get("augment", False)),
            nms_iou=float(model_cfg.get("nms_iou", 0.55)),
            tiling=tiling_cfg,
            merge=str(model_cfg.get("merge", "nms")),
            tile_merge=str(tiling_cfg.get("merge", "wbf")),
            scales=scales_cfg if model_cfg.get("scales_image", True) else None,
            ios_merge=float(model_cfg.get("ios_merge", 0.90)),
        )
        if detect_overrides:
            det_kwargs.update(detect_overrides)
        self.detector = PersonDetector(**det_kwargs)

        # 图片默认走高精度切片，视频/实时默认走快速单程（可在配置里改）
        self.tiling_image = bool(tiling_cfg.get("enabled", False)) and bool(
            tiling_cfg.get("image", True)
        )
        self.tiling_video = bool(tiling_cfg.get("enabled", False)) and bool(
            tiling_cfg.get("video", False)
        )

        # 多尺度推理：图片默认开（精度优先），视频默认关（帧率优先）
        self.scales_image = scales_cfg if model_cfg.get("scales_image", True) else []
        self.scales_video = scales_cfg if model_cfg.get("scales_video", False) else []

        # 指定区域
        region_cfg = cfg.get("regions") or {}
        self.regions: List[Region] = parse_regions(region_cfg)
        self.region_min_ratio = float(region_cfg.get("min_inside_ratio", 0.60))
        self.region_filter = bool(region_cfg.get("filter_outside", False))

        self.counter = SeatROICounter(
            cfg=cfg,
            mode=self.count_mode,
            overlap_threshold=float(roi_cfg.get("overlap_threshold", 0.30)),
            min_area_ratio=float(roi_cfg.get("min_area_ratio", 0.002)),
            aspect_range=tuple(roi_cfg.get("aspect_range", [0.30, 5.0])),
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
        self,
        frame: np.ndarray,
        draw: bool = True,
        tiling: bool | None = None,
        mode: str = "image",
    ) -> tuple[np.ndarray | None, Dict[str, Any]]:
        t0 = time.perf_counter()
        use_tiling = self.tiling_image if tiling is None else bool(tiling)
        scales = self.scales_image if mode == "image" else self.scales_video
        if self.track_enabled:
            detections = self.detector.track(frame, tracker=self.tracker_name)
        else:
            detections = self.detector.detect(frame, tiling=use_tiling, scales=scales or None)

        # 区域归属先算，再决定是否按区域过滤
        detections = annotate_detections(
            detections, self.regions, frame.shape, self.region_min_ratio
        )
        if self.regions and self.region_filter:
            detections = filter_outside(detections)

        result = self.counter.count(detections, frame.shape)
        result["variant"] = self.variant
        result["tiling"] = bool(
            use_tiling
            and self.detector.tiling_enabled
            and min(frame.shape[:2]) >= self.detector.min_image_side
        )
        result["scales"] = list(scales) if len(scales) > 0 else [self.detector.imgsz]
        if self.regions:
            # 只统计通过有效性过滤的框，保证"区域内 + 区域外 = 总人数"三者自洽
            summary, outside = region_summary(result["detections"], self.regions)
            result["regions"] = summary
            result["inside_persons"] = int(sum(r["count"] for r in summary))
            result["outside_persons"] = int(outside)
        else:
            result["regions"] = []
            result["inside_persons"] = int(result["total_persons"])
            result["outside_persons"] = 0

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
        annotated, result = self.process(image, draw=True, tiling=self.tiling_image)
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
        peak_inside = 0
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, result = self.process(frame, draw=True, tiling=self.tiling_video, mode="video")
            peak_people = max(peak_people, result["total_persons"])
            peak_occupied = max(peak_occupied, result["occupied_count"])
            peak_inside = max(peak_inside, result.get("inside_persons", 0))
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
            "peak_inside": peak_inside,
            "fps": self._fps(),
            "output": output_path,
        }
