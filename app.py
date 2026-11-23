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
    conf: float, imgsz: int, grid_enabled: bool, rows: int, cols: int, show_person: bool
) -> dict:
    """按界面参数生成配置副本，避免污染全局配置。"""
    cfg = copy.deepcopy(BASE_CFG)
    cfg["model"]["conf"] = float(conf)
    cfg["model"]["imgsz"] = int(imgsz)
    cfg["seat_grid"]["enabled"] = bool(grid_enabled)
    cfg["seat_grid"]["rows"] = int(rows)
    cfg["seat_grid"]["cols"] = int(cols)
    cfg["display"]["show_person_boxes"] = bool(show_person)
    return cfg


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
        f"处理耗时：{result.get('frame_ms', 0)} ms（约 {result['fps']} FPS）",
        "",
        "座位状态：",
        seats,
    ]
    if result.get("note"):
        lines += ["", f"提示：{result['note']}"]
    return "\n".join(lines)


def _run(cfg: dict, variant: str, weights: str, image=None, video=None, webcam=False, state=None):
    pipe = state if state is not None else PersonCountingPipeline(cfg, variant=variant, weights=weights)
    if webcam:
        annotated, result = pipe.process(image, draw=True)
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
    annotated, result = pipe.process_image(image)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), result_text(result), pipe


def infer_image(image, variant, weights, conf, imgsz, grid_enabled, rows, cols, show_person):
    if image is None:
        return None, "请先上传一张图片"
    cfg = build_cfg(conf, imgsz, grid_enabled, rows, cols, show_person)
    out, txt, _ = _run(cfg, variant, weights, image=image)
    return out, txt


def infer_video(video, variant, weights, conf, imgsz, grid_enabled, rows, cols, show_person):
    if not video:
        return None, "请先上传一段视频"
    cfg = build_cfg(conf, imgsz, grid_enabled, rows, cols, show_person)
    out, txt, _ = _run(cfg, variant, weights, video=video)
    return out, txt


def infer_webcam(frame, variant, weights, conf, imgsz, grid_enabled, rows, cols, show_person, state):
    if frame is None:
        return None, "", state
    cfg = build_cfg(conf, imgsz, grid_enabled, rows, cols, show_person)
    if state is None:
        state = PersonCountingPipeline(cfg, variant=variant, weights=weights)
    out, txt, state = _run(cfg, variant, weights, image=frame, webcam=True, state=state)
    return out, txt, state


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="YOLOv8 人员统计与座位占用检测") as demo:
        gr.Markdown(
            "# YOLOv8 人员到岗统计与座位占用检测\n"
            "能看 · 能跑 · 能对比 · 能复述 ｜ 支持图片 / 视频 / 摄像头，可切换 A/B 实验方案"
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

        tune = [conf, imgsz, grid_enabled, rows, cols, show_person]

        with gr.Tabs():
            with gr.Tab("图片检测"):
                with gr.Row():
                    img_in = gr.Image(type="numpy", label="上传教室/会议室图片")
                    img_out = gr.Image(type="numpy", label="检测结果")
                img_btn = gr.Button("开始统计", variant="primary")
                img_txt = gr.Textbox(label="统计明细", lines=11)
                img_btn.click(infer_image, inputs=[img_in, variant, weights] + tune, outputs=[img_out, img_txt])

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

        gr.Markdown(
            "**展示规则**：红色座位框=占用、绿色=空闲；黄色框=检测到的人员（附序号与座位号）。"
            "座位框由透视网格生成，始终贴合画面；若与实际课桌不符，请调「行数/列数」或修改 "
            "`configs/config.yaml` 的 `seat_grid.quad` 四个角点。"
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, share=False)
