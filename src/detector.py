"""YOLOv8 人员检测器（v2.2 高精度版）。

相对 v2.1 的改动：
1. **切片推理（Tiled / Sliced Inference）**：把一张大图切成带重叠的子图分别推理，
   小目标在子图里像素更多，密集教室后排的"小人头"召回明显提升；子图坐标会映射回
   原图再融合。
2. **框融合 WBF（Weighted Box Fusion）**：切片与全图会产生同一人的多个框，WBF 按
   置信度加权合并成一个框，而不是像 NMS 那样直接丢掉低分框 —— 既减少重复框（误检），
   又不会把相邻的两个人误删（漏检）。
3. **Soft-NMS**：可选的高斯衰减抑制，替代硬阈值 NMS，密集场景更平滑。
4. **可配置融合策略**：nms / soft_nms / wbf 三选一，便于做消融。
5. 保留 track 接口与 A/B 实验变体兼容。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装依赖: pip install -r requirements.txt") from exc

Box = Tuple[float, float, float, float]


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _with_center(det: Dict[str, Any]) -> Dict[str, Any]:
    x1, y1, x2, y2 = det["bbox"]
    det["center"] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return det


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


def soft_nms_detections(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.55,
    sigma: float = 0.5,
    score_thr: float = 0.01,
) -> List[Dict[str, Any]]:
    """高斯 Soft-NMS：重叠框按 IoU 指数衰减置信度，而不是直接删除。

    密集场景里两个人框天然重叠，硬 NMS 容易误删；Soft-NMS 用衰减代替删除，
    让"真目标"即使被压分也还能留在阈值之上。
    """
    dets = [dict(d) for d in detections]
    for d in dets:
        d["conf"] = float(d.get("conf", 0.0))
    dets.sort(key=lambda d: -d["conf"])
    kept: List[Dict[str, Any]] = []
    while dets:
        cur = dets.pop(0)
        if cur["conf"] < score_thr:
            continue
        kept.append(cur)
        for d in dets:
            ov = _iou(cur["bbox"], d["bbox"])
            if ov > iou_threshold:
                d["conf"] *= math.exp(-(ov * ov) / max(sigma, 1e-6))
        dets.sort(key=lambda d: -d["conf"])
    return kept


def wbf_detections(
    detections: List[Dict[str, Any]], iou_threshold: float = 0.55
) -> List[Dict[str, Any]]:
    """Weighted Box Fusion：把同一目标的多个框按置信度加权融合成一个。

    - 坐标 = 簇内各框的置信度加权平均（比任取一个更稳，边界抖动更小）；
    - 置信度 = 簇内最大值（保守，不虚高，避免为了好看而虚抬指标）；
    - 融合后的框带 merged 字段，便于排查"一个框由几个框合成"。
    """
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: -float(d.get("conf", 0.0)))
    clusters: List[Dict[str, Any]] = []
    for det in dets:
        conf = float(det.get("conf", 0.0))
        best_i, best_iou = -1, 0.0
        for i, cl in enumerate(clusters):
            ov = _iou(cl["bbox"], det["bbox"])
            if ov > best_iou:
                best_i, best_iou = i, ov
        if best_i >= 0 and best_iou >= iou_threshold:
            members = clusters[best_i]["members"] + [det]
            weights = [max(float(m.get("conf", 0.0)), 1e-6) for m in members]
            total = sum(weights)
            fused: Box = tuple(  # type: ignore[assignment]
                sum(m["bbox"][k] * w for m, w in zip(members, weights)) / total
                for k in range(4)
            )
            clusters[best_i] = {
                "bbox": fused,
                "members": members,
                "conf": max(weights),
            }
        else:
            clusters.append({"bbox": tuple(det["bbox"]), "members": [det], "conf": conf})

    fused_dets: List[Dict[str, Any]] = []
    for cl in clusters:
        members = cl["members"]
        base = dict(members[0])
        base["bbox"] = cl["bbox"]
        base["conf"] = float(cl["conf"])
        base["merged"] = len(members)
        fused_dets.append(_with_center(base))
    return fused_dets


def ioS(a: Sequence[float], b: Sequence[float]) -> float:
    """交集占较小框面积的比例（Intersection over Smaller area）。

    IoU 对"大框包小框"不敏感：一个只框住头的框和框住全身的框 IoU 可能不到 0.3，
    但它们显然是同一个人。IoS 用较小框面积做分母，专门用来发现这种嵌套重复框。
    """
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def suppress_contained(
    detections: List[Dict[str, Any]], ios_threshold: float = 0.90
) -> List[Dict[str, Any]]:
    """抑制"被更大框包住"的重复框（如只框住头/腿的碎片框）。

    按置信度降序保留；若某框有 ≥ threshold 的面积落在已保留框内，则视为重复。
    与 NMS 的区别：NMS 看 IoU（大小相近的框），这里看 IoS（嵌套的框）。
    """
    if ios_threshold <= 0 or not detections:
        return detections
    kept: List[Dict[str, Any]] = []
    for det in sorted(detections, key=lambda d: -float(d.get("conf", 0.0))):
        if any(ioS(det["bbox"], k["bbox"]) >= ios_threshold for k in kept):
            continue
        kept.append(det)
    return kept


def tile_windows(
    width: int, height: int, rows: int, cols: int, overlap: float = 0.2
) -> List[Tuple[int, int, int, int]]:
    """生成带重叠的切片窗口 (x0, y0, x1, y1)，覆盖整幅画面。

    overlap 是相邻切片的重叠比例（0~0.5），重叠区给 WBF 提供融合余量，
    避免目标正好被切在缝上时两个半身框都判不出来。
    """
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    overlap = float(min(max(overlap, 0.0), 0.5))
    tw = width / (cols - (cols - 1) * overlap) if cols > 1 else float(width)
    th = height / (rows - (rows - 1) * overlap) if rows > 1 else float(height)
    step_x = tw * (1 - overlap) if cols > 1 else 0.0
    step_y = th * (1 - overlap) if rows > 1 else 0.0
    windows: List[Tuple[int, int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            x0 = int(round(c * step_x))
            y0 = int(round(r * step_y))
            x1 = min(width, int(round(x0 + tw)))
            y1 = min(height, int(round(y0 + th)))
            x0 = max(0, min(x0, x1 - 2))
            y0 = max(0, min(y0, y1 - 2))
            windows.append((x0, y0, x1, y1))
    return windows


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
        tiling: Dict[str, Any] | None = None,
        merge: str = "nms",
        tile_merge: str = "wbf",
        scales: Sequence[int] | None = None,
        ios_merge: float = 0.90,
    ) -> None:
        self.weights = weights
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.augment = augment
        self.nms_iou = nms_iou
        self.device = device
        self.merge = merge
        self.tile_merge = tile_merge
        self.ios_merge = ios_merge
        self.tiling = dict(tiling or {})
        self.tiling_enabled = bool(self.tiling.get("enabled", False))
        self.min_image_side = int(self.tiling.get("min_image_side", 900))
        self.scales = [int(s) for s in (scales or []) if int(s) > 0] or [int(imgsz)]
        if str(device).lower() in ("auto", ""):
            try:
                import torch

                self.device = 0 if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        self.classes = list(classes)
        self.model = YOLO(weights)

    # ------------------------------------------------------------------ 推理
    def _predict(self, frame: np.ndarray, imgsz: int, conf: float) -> List[Dict[str, Any]]:
        results = self.model.predict(
            source=frame,
            conf=conf,
            iou=self.iou,
            imgsz=int(imgsz),
            device=self.device,
            classes=self.classes,
            augment=self.augment,
            verbose=False,
        )
        return self._parse_boxes(results)

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

    def _detect_tiled(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """全图 + 多切片推理，框映射回原图后交给融合阶段。"""
        h, w = frame.shape[:2]
        rows = int(self.tiling.get("rows", 2))
        cols = int(self.tiling.get("cols", 2))
        overlap = float(self.tiling.get("overlap", 0.2))
        tile_imgsz = int(self.tiling.get("tile_imgsz", 640))
        delta = float(self.tiling.get("conf_delta", 0.03))
        tile_conf = max(0.01, self.conf - delta)

        raw: List[Dict[str, Any]] = []
        if self.tiling.get("full_frame", True):
            raw.extend(self._predict(frame, self.imgsz, self.conf))

        for (x0, y0, x1, y1) in tile_windows(w, h, rows, cols, overlap):
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            for det in self._predict(crop, tile_imgsz, tile_conf):
                bx1, by1, bx2, by2 = det["bbox"]
                det["bbox"] = (
                    max(0.0, min(w - 1.0, bx1 + x0)),
                    max(0.0, min(h - 1.0, by1 + y0)),
                    max(0.0, min(w - 1.0, bx2 + x0)),
                    max(0.0, min(h - 1.0, by2 + y0)),
                )
                det["source"] = "tile"
                _with_center(det)
                if det["bbox"][2] - det["bbox"][0] >= 2 and det["bbox"][3] - det["bbox"][1] >= 2:
                    raw.append(det)
        return raw

    # -------------------------------------------------------------- 后处理
    def _postprocess(
        self, detections: List[Dict[str, Any]], tiled: bool = False
    ) -> List[Dict[str, Any]]:
        kept = [d for d in detections if np.isfinite(np.asarray(d["bbox"], dtype=float)).all()]
        if not kept:
            return []
        merge = self.tile_merge if tiled else self.merge
        if merge == "wbf":
            out = wbf_detections(kept, self.nms_iou)
        elif merge == "soft_nms":
            out = soft_nms_detections(kept, self.nms_iou)
        else:
            out = nms_detections(kept, self.nms_iou)
        return suppress_contained(out, self.ios_merge)

    def detect(
        self,
        frame: np.ndarray,
        tiling: bool | None = None,
        scales: Sequence[int] | None = None,
    ) -> List[Dict[str, Any]]:
        """逐帧检测。tiling=None 时按配置默认（图片开、视频关由流水线决定）。

        低分辨率图（最短边 < min_image_side）会自动放弃切片：小图上切片等于
        反复放大噪声，实测精确率会从 0.87 掉到 0.57，只该用于大图/高分辨率帧。

        scales 给多个分辨率时做多尺度推理 + WBF 融合：不同尺度各自能捞到
        对方漏掉的小目标，融合后召回提升明显（见 docs/EVAL_v2.2.md）。
        """
        use_tiling = self.tiling_enabled if tiling is None else bool(tiling and self.tiling_enabled)
        if use_tiling and min(frame.shape[:2]) < self.min_image_side:
            use_tiling = False
        if use_tiling:
            return self._postprocess(self._detect_tiled(frame), tiled=True)

        scale_list = [int(s) for s in (scales or self.scales)]
        if len(scale_list) > 1:
            raw: List[Dict[str, Any]] = []
            for s in scale_list:
                raw.extend(self._predict(frame, s, self.conf))
            # 多尺度结果来自同一模型，坐标已在同一张图上，直接用 WBF 融合
            return suppress_contained(wbf_detections(raw, self.nms_iou), self.ios_merge)
        return self._postprocess(self._predict(frame, scale_list[0], self.conf), tiled=False)

    def track(
        self, frame: np.ndarray, tracker: str = "bytetrack.yaml", persist: bool = True
    ) -> List[Dict[str, Any]]:
        """启用 ByteTrack，为每个 person 分配跨帧 track_id 并去重。

        跟踪不做切片（跨帧 ID 依赖整帧上下文），保证视频端速度与 ID 稳定。
        """
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
        return self._postprocess(self._parse_boxes(results), tiled=False)

    def model_info(self) -> Dict[str, float]:
        try:
            info = self.model.info(verbose=False)
            return {"params": float(info[0]), "flops": float(info[1])}
        except Exception:
            return {"params": 0.0, "flops": 0.0}
