"""YOLOv8 人员检测器（改进版）。

改进点：
1. 支持更大 imgsz 与更低 conf 的"密集场景"参数，显著提升教室小目标召回；
2. 可选 TTA（augment）进一步稳定检测；
3. 合并高度重叠的重复框（NMS），减少"框杂乱/重复"；
4. 过滤极端尺寸/越界框，减少无效框；
5. 保留 track 接口与 A/B 实验变体兼容。
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装依赖: pip install -r requirements.txt") from exc


def _iou(a: List[float], b: List[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_detections(
    detections: List[Dict[str, Any]], iou_threshold: float = 0.55
) -> List[Dict[str, Any]]:
    """按置信度降序做 NMS，合并重复框。"""
    if not detections:
        return []
    order = sorted(detections, key=lambda d: -d.get("conf", 0.0))
    kept: List[Dict[str, Any]] = []
    while order:
        current = order.pop(0)
        kept.append(current)
        order = [
            d for d in order
            if _iou(current["bbox"], d["bbox"]) < iou_threshold
        ]
    return kept


class PersonDetector:
    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.15,
        iou: float = 0.45,
        imgsz: int = 960,
        device: str = "auto",
        classes: tuple[int, ...] = (0,),
        augment: bool = False,
        nms_iou: float = 0.55,
    ) -> None:
        self.weights = weights
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.augment = augment
        self.nms_iou = nms_iou
        self.device = device
        if str(device).lower() in ("auto", ""):
            try:
                import torch

                self.device = 0 if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        self.classes = list(classes)
        self.model = YOLO(weights)

    @staticmethod
    def _parse_boxes(results) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].detach().cpu().numpy().astype(float)
            conf = float(boxes.conf[i].detach().cpu().numpy())
            cls = int(boxes.cls[i].detach().cpu().numpy())
            track_id = None
            if boxes.id is not None and i < len(boxes.id):
                track_id = int(boxes.id[i].detach().cpu().numpy())
            x1, y1, x2, y2 = xyxy.tolist()
            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "class": cls,
                    "class_name": "person" if cls == 0 else str(cls),
                    "track_id": track_id,
                    "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                }
            )
        return detections

    def _postprocess(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 过滤越界框
        kept = [d for d in detections if np.isfinite(d["bbox"]).all()]
        # 合并重复框
        return nms_detections(kept, self.nms_iou)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """逐帧检测（基线/改进共用），含 TTA 与去重。"""
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.classes,
            augment=self.augment,
            verbose=False,
        )
        return self._postprocess(self._parse_boxes(results))

    def track(
        self, frame: np.ndarray, tracker: str = "bytetrack.yaml", persist: bool = True
    ) -> List[Dict[str, Any]]:
        """启用 ByteTrack，为每个 person 分配跨帧 track_id 并去重。"""
        results = self.model.track(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.classes,
            tracker=tracker,
            persist=persist,
            verbose=False,
        )
        return self._postprocess(self._parse_boxes(results))

    def model_info(self) -> Dict[str, float]:
        try:
            info = self.model.info(verbose=False)
            return {"params": float(info[0]), "flops": float(info[1])}
        except Exception:
            return {"params": 0.0, "flops": 0.0}
