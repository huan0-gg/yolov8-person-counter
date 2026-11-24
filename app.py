#!/usr/bin/env python3
"""Gradio Web 界面：让项目"能看"。

支持三种输入：图片 / 视频 / 摄像头；可切换 A/B 方案；
并提供检测调优（灵敏度、分辨率）与座位网格（行列、模式）控制，
用于适配不同教室视角、减少错框与漏检。
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == "scripts" else Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2
import gradio as gr
import numpy as np

from src.pipeline import VARIANTS, PersonCountingPipeline
from src.utils import load_config

BASE_CFG = load_config(ROOT / "configs" / "config.yaml")


def build_cfg(
    conf: float,
    imgsz: int,
    grid_enabled: bool,
    rows: int,
    cols: int,
    show_person: bool,
    show_regions: bool = True,
    use_tiling: bool = True,
    dim_outside: bool = False,
) -> dict:
    """按界面参数生成配置副本，避免污染全局配置。"""
    cfg = copy.deepcopy(BASE_CFG)
    cfg["model"]["conf"] = float(conf)
    cfg["model"]["imgsz"] = int(imgsz)
    cfg["model"].setdefault("tiling", {})
    cfg["model"]["tiling"]["enabled"] = bool(use_tiling)
    cfg["model"]["tiling"]["image"] = bool(use_tiling)
    cfg["seat_grid"]["enabled"] = bool(grid_enabled)
    cfg["seat_grid"]["rows"] = int(rows)
    cfg["seat_grid"]["cols"] = int(cols)
    cfg["display"]["show_person_boxes"] = bool(show_person)
    cfg["display"]["show_regions"] = bool(show_regions)
    cfg["display"]["dim_outside"] = bool(dim_outside)
    return cfg


def _mode_text(result: dict) -> str:
    """把推理策略说清楚：切片 / 多尺度 / 单程，避免"到底跑的是哪套"说不明白。"""
    parts = []
    if result.get("tiling"):
        parts.append("切片")
    scales = result.get("scales") or []
    if len(scales) > 1:
        parts.append("多尺度 " + "/".join(str(s) for s in scales))
    return " + ".join(parts) if parts else "单程"


def result_text(result: dict) -> str:
    seats = "  ".join(
        f"S{i + 1}:{'占用' if occupied else '空闲'}"
        for i, occupied in enumerate(result["occupied"])
    )
    lines = [
        f"方案：{result['variant']}",
        f"检测人数：{result['total_persons']}（原始 {result.get('raw_persons', result['total_persons'])}，过滤 {result.get('filtered_out', 0)}）",
        f"占用座位：{result['occupied_count']}/{result['total_seats']}",
        f"座位框来源：{result.get('seat_source', 'n/a')}　判定：{result['count_mode']}",
        f"推理模式：{_mode_text(result)}",
        f"处理耗时：{result.get('frame_ms', 0)} ms（约 {result['fps']} FPS）",
    ]
    if result.get("regions"):
        region_lines = [
            f"  · {r['label']}（{r['name']}）：{r['count']} 人" for r in result["regions"]
        ]
        lines += ["", f"指定区域人数：{result.get('inside_persons', 0)}"] + region_lines
        lines.append(f"区域外人数：{result.get('outside_persons', 0)}")
    lines += ["", "座位状态：", seats]
    if result.get("note"):
        lines += ["", f"提示：{result['note']}"]
    return "\n".join(lines)


def _run(cfg: dict, variant: str, weights: str, image=None, video=None, webcam=False, state=None):
    pipe = state if state is not None else PersonCountingPipeline(cfg, variant=variant, weights=weights)
    if webcam:
        # Gradio 传出的是 RGB，流水线按 OpenCV 的 BGR 处理，这里统一通道顺序
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image is not None else image
        annotated, result = pipe.process(frame, draw=True)
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), result_text(result), pipe
    if video:
        out_dir = ROOT / "assets"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"annotated_{variant}.mp4")
        stats = pipe.process_video(video, output_path=output_path)
        info = (
            f"方案：{variant}\n处理帧数：{stats['frames']}\n"
            f"峰值人数：{stats['peak_persons']}\n峰值占用座位：{stats['peak_occupied']}\n"
            f"平均速度：{stats['fps']} FPS\n结果视频：{output_path}"
        )
        return output_path, info, pipe
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image is not None else image
    annotated, result = pipe.process_image(frame)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), result_text(result), pipe


def infer_image(image, variant, weights, conf, imgsz, grid_enabled, rows, cols, show_person,
                show_regions, use_tiling, dim_outside):
    if image is None:
        return None, "请先上传一张图片"
    cfg = build_cfg(conf, imgsz, grid_enabled, rows, cols, show_person,
                    show_regions, use_tiling, dim_outside)
    out, txt, _ = _run(cfg, variant, weights, image=image)
    return out, txt


def infer_video(video, variant, weights, conf, imgsz, grid_enabled, rows, cols, show_person,
                show_regions, use_tiling, dim_outside):
    if not video:
        return None, "请先上传一段视频"
    cfg = build_cfg(conf, imgsz, grid_enabled, rows, cols, show_person,
                    show_regions, use_tiling, dim_outside)
    out, txt, _ = _run(cfg, variant, weights, video=video)
    return out, txt


# 课堂演示用：不需要自己找图，点一下就能跑通完整链路
# v2.2 起改用 AI 生成的同风格场景图（assets/ai/），不再依赖旧版截图
DEMO_IMAGES = {
    "密集教室（示例）": "ai/classroom_dense.jpg",
    "中等密度教室（示例）": "ai/classroom_seat.jpg",
    "空教室（示例）": "ai/classroom_empty.jpg",
    "会议室（示例）": "ai/meeting_room.jpg",
}


def infer_demo(which, variant, weights, conf, imgsz, grid_enabled, rows, cols, show_person,
               show_regions, use_tiling, dim_outside):
    """载入仓库自带示例图并直接出结果，失败时给出明确原因而不是空白。"""
    name = DEMO_IMAGES.get(which, "classroom_crowded.jpg")
    path = ROOT / "assets" / name
    img = cv2.imread(str(path))
    if img is None:
        return None, f"未找到示例图：{path}，请确认 assets/ 目录完整", None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out, txt = infer_image(rgb, variant, weights, conf, imgsz, grid_enabled, rows, cols,
                           show_person, show_regions, use_tiling, dim_outside)
    return out, txt, rgb


def infer_webcam(frame, variant, weights, conf, imgsz, grid_enabled, rows, cols, show_person,
                 show_regions, use_tiling, dim_outside, state):
    if frame is None:
        return None, "", state
    cfg = build_cfg(conf, imgsz, grid_enabled, rows, cols, show_person,
                    show_regions, use_tiling, dim_outside)
    if state is None:
        state = PersonCountingPipeline(cfg, variant=variant, weights=weights)
    out, txt, state = _run(cfg, variant, weights, image=frame, webcam=True, state=state)
    return out, txt, state


def training_panel():
    """v2.0：读取真实训练产物，生成展示面板（指标表 + 曲线 + 数据样例）。"""
    results_dir = ROOT / "experiments" / "results"
    parts = [
        "## v2.0 训练与验证（真实运行结果）\n",
        "> 数据来自 `data/DATA_CARD.md` 所述可复现数据集；指标由训练脚本真实产出，未手工填写。\n",
    ]

    # 训练对比表
    a = results_dir / "A_baseline_train_result.json"
    b = results_dir / "B1_augment_train_result.json"
    if a.exists() and b.exists():
        import json

        ra, rb = json.loads(a.read_text(encoding="utf-8")), json.loads(b.read_text(encoding="utf-8"))
        parts += [
            "| 实验 | 增强策略 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | 耗时 |",
            "|---|---|---|---|---|---|---|",
            f"| 基线 A | {ra.get('augment')} | {ra.get('map50')} | {ra.get('map50_95')} | "
            f"{ra.get('precision')} | {ra.get('recall')} | {ra.get('train_seconds')}s |",
            f"| B1 增强 | {rb.get('augment')} | {rb.get('map50')} | {rb.get('map50_95')} | "
            f"{rb.get('precision')} | {rb.get('recall')} | {rb.get('train_seconds')}s |",
            "",
            "**解读**：B1 精确率更高但召回略降——增强策略的真实权衡，详见 "
            "`experiments/results/training_verification.md`。\n",
        ]
    else:
        parts.append("⚠️ 未找到训练结果，请先运行 `scripts/train.py`。\n")

    verification = results_dir / "training_verification.md"
    if verification.exists():
        parts += ["### 验证报告", f"```\n{verification.read_text(encoding='utf-8')[:1600]}\n```\n"]
    return "\n".join(parts)


def build_ui() -> gr.Blocks:
    try:
        theme = gr.themes.Soft(primary_hue="teal", neutral_hue="slate")
    except Exception:
        theme = None
    with gr.Blocks(title="YOLOv8 人员统计与座位占用检测", theme=theme) as demo:
        gr.Markdown(
            "# YOLOv8 人员到岗统计与座位占用检测\n"
            "能看 · 能跑 · 能对比 · 能复述 ｜ 支持图片 / 视频 / 摄像头，可切换 A/B 实验方案\n\n"
            "> 第一次用？点 **一键载入示例并统计** 即可跑通完整链路，不需要先准备图片。"
        )

        with gr.Row():
            variant = gr.Dropdown(choices=list(VARIANTS.keys()), value="A_baseline", label="实验方案")
            weights = gr.Textbox(value=BASE_CFG["model"]["weights"], label="模型权重路径", scale=2)

        with gr.Accordion("检测与座位调优（漏检/错框时可调）", open=False):
            with gr.Row():
                conf = gr.Slider(0.05, 0.5, value=float(BASE_CFG["model"]["conf"]), step=0.01, label="灵敏度阈值（越低检人越多）")
                imgsz = gr.Slider(480, 1280, value=int(BASE_CFG["model"]["imgsz"]), step=160, label="推理分辨率（越高小目标越准）")
            with gr.Row():
                grid_enabled = gr.Checkbox(value=True, label="启用透视座位网格（关=旧固定坐标）")
                rows = gr.Slider(2, 8, value=int(BASE_CFG["seat_grid"]["rows"]), step=1, label="座位行数")
                cols = gr.Slider(2, 8, value=int(BASE_CFG["seat_grid"]["cols"]), step=1, label="座位列数")
                show_person = gr.Checkbox(value=True, label="显示人员框")
            with gr.Row():
                use_tiling = gr.Checkbox(
                    value=False, label="切片推理（仅 4K/超大图，小图实测增误检）"
                )
                show_regions = gr.Checkbox(value=True, label="标注指定区域")
                dim_outside = gr.Checkbox(value=False, label="区域外压暗")

        tune = [conf, imgsz, grid_enabled, rows, cols, show_person,
                show_regions, use_tiling, dim_outside]

        with gr.Tabs():
            with gr.Tab("图片检测"):
                with gr.Row():
                    demo_pick = gr.Dropdown(
                        choices=list(DEMO_IMAGES.keys()),
                        value="密集教室（示例）",
                        label="示例图（无需准备素材）",
                        scale=3,
                    )
                    demo_btn = gr.Button("一键载入示例并统计", variant="secondary", scale=1)
                with gr.Row():
                    img_in = gr.Image(type="numpy", label="上传教室/会议室图片")
                    img_out = gr.Image(type="numpy", label="检测结果")
                img_btn = gr.Button("开始统计", variant="primary")
                img_txt = gr.Textbox(label="统计明细", lines=11)
                img_btn.click(infer_image, inputs=[img_in, variant, weights] + tune, outputs=[img_out, img_txt])
                demo_btn.click(
                    infer_demo,
                    inputs=[demo_pick, variant, weights] + tune,
                    outputs=[img_out, img_txt, img_in],
                )

            with gr.Tab("视频检测"):
                with gr.Row():
                    vid_in = gr.Video(label="上传视频")
                    vid_out = gr.Video(label="标注结果视频")
                vid_btn = gr.Button("开始处理", variant="primary")
                vid_txt = gr.Textbox(label="统计明细", lines=8)
                vid_btn.click(infer_video, inputs=[vid_in, variant, weights] + tune, outputs=[vid_out, vid_txt])

            with gr.Tab("摄像头实时"):
                try:
                    cam_in = gr.Image(sources=["webcam"], type="numpy", streaming=True, label="摄像头画面")
                except TypeError:
                    cam_in = gr.Webcam(type="numpy", label="摄像头画面")
                cam_out = gr.Image(type="numpy", label="实时检测结果")
                cam_txt = gr.Textbox(label="实时统计", lines=11)
                cam_state = gr.State(None)
                reset_btn = gr.Button("重置跟踪状态")
                cam_in.stream(
                    infer_webcam,
                    inputs=[cam_in, variant, weights] + tune + [cam_state],
                    outputs=[cam_out, cam_txt, cam_state],
                )
                reset_btn.click(lambda: None, inputs=None, outputs=cam_state)

            with gr.Tab("训练与验证 v2.0"):
                gr.Markdown(training_panel)
                curve_a = ROOT / "experiments" / "results" / "A_baseline_curves.png"
                curve_b = ROOT / "experiments" / "results" / "B1_augment_curves.png"
                if curve_a.exists() and curve_b.exists():
                    with gr.Row():
                        gr.Image(value=str(curve_a), label="基线 A 训练曲线", interactive=False)
                        gr.Image(value=str(curve_b), label="B1 增强训练曲线", interactive=False)
                samples = ROOT / "experiments" / "results" / "dataset_samples.jpg"
                if samples.exists():
                    gr.Image(value=str(samples), label="数据集标注样例（抽检）", interactive=False)

        gr.Markdown(
            "**展示规则**：青绿描边并带色块填充的是**指定区域**（标签上直接写区域内人数）；"
            "红色座位框=占用、绿色=空闲；黄色框=区域内人员、灰色=区域外人员（附序号与座位号）。"
            "座位框由透视网格生成，始终贴合画面；若与实际课桌不符，请调「行数/列数」或修改 "
            "`configs/config.yaml` 的 `seat_grid.quad`；区域范围改 `regions.items` 即可。"
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, share=False)
