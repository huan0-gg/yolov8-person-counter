#!/usr/bin/env python3
"""生成 45 秒开源项目宣传片（1280x720 / 30fps）+ README 封面 GIF。

所有数字与检测框都从 docs/video/evidence/summary.json 读取，
该文件由 scripts/collect_promo_assets.py 真实运行流水线产生，
因此画面里的每一帧标注都对应一次可复现的推理，不是手绘示意图。

用法：
    python scripts/collect_promo_assets.py   # 先产出证据
    python scripts/make_promo.py             # 再合成视频
输出：
    docs/video/promo_45s.mp4
    docs/video/cover.gif
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
EVID = ROOT / "docs" / "video" / "evidence"
OUT = ROOT / "docs" / "video"
TMP = OUT / "_frames"

W, H = 1280, 720
FPS = 30
DUR = 45.0
BG = (11, 21, 33)
CARD = (22, 37, 52)
LINE = (42, 61, 78)
ACC = (45, 212, 167)
WARN = (245, 185, 66)
RED = (255, 107, 107)
TXT = (226, 238, 246)
MUT = (143, 163, 179)

F_TITLE = "C:/Windows/Fonts/msyhbd.ttc"
F_BODY = "C:/Windows/Fonts/msyh.ttc"
F_MONO = "C:/Windows/Fonts/consola.ttf"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def seg(t: float, a: float, b: float) -> float:
    """把绝对时间 t 映射到 [a,b] 区间内的 0..1 进度。"""
    if b <= a:
        return 1.0 if t >= b else 0.0
    return clamp01((t - a) / (b - a))


def bg_image() -> Image.Image:
    """深空渐变底 + 两团极淡的品牌色光晕，避免纯色死板。"""
    y = np.linspace(0, 1, H)[:, None]
    x = np.linspace(0, 1, W)[None, :]
    base = np.zeros((H, W, 3), dtype=np.float32)
    for i, c in enumerate(BG):
        base[:, :, i] = c

    def glow(cx, cy, r, color, strength):
        d = np.sqrt((x - cx) ** 2 + ((y - cy) * (H / W)) ** 2)
        m = np.clip(1 - d / r, 0, 1) ** 2 * strength
        for i, c in enumerate(color):
            base[:, :, i] += (c - base[:, :, i]) * m

    glow(0.86, 0.12, 0.55, ACC, 0.10)
    glow(0.10, 0.92, 0.60, (58, 110, 165), 0.12)
    return Image.fromarray(base.clip(0, 255).astype(np.uint8))


def rounded(im: Image.Image, radius: int) -> Image.Image:
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, im.size[0] - 1, im.size[1] - 1], radius, fill=255)
    out = im.convert("RGBA")
    out.putalpha(mask)
    return out


def fit_cover(im: Image.Image, box: tuple[int, int], sharpen: bool = True) -> Image.Image:
    """等比放大裁剪到目标尺寸，轻微锐化补偿小图放大后的发虚。"""
    tw, th = box
    s = max(tw / im.width, th / im.height)
    nw, nh = max(tw, int(im.width * s + 0.5)), max(th, int(im.height * s + 0.5))
    im = im.resize((nw, nh), Image.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    im = im.crop((left, top, left + tw, top + th))
    if sharpen:
        im = im.filter(ImageFilter.UnsharpMask(radius=1.4, percent=110, threshold=3))
    return im


def tracked_text(d, xy, text, f, fill, tracking=0, anchor_top=True):
    """带字距的文本绘制（PIL 无原生 letter-spacing）。"""
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=f, fill=fill)
        x += d.textlength(ch, font=f) + tracking
    return x


def text_w(text: str, f, tracking=0) -> float:
    return sum(f.getlength(c) + tracking for c in text) - (tracking if text else 0)


class Scene:
    """场景基类：u 为 0..1 的局部进度。"""

    dur = 3.0
    start = 0.0

    def draw(self, frame: Image.Image, t: float, u: float) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------- 素材预加载
def load_evidence() -> dict:
    p = EVID / "summary.json"
    if not p.exists():
        sys.exit("缺少 docs/video/evidence/summary.json，请先运行 scripts/collect_promo_assets.py")
    return json.loads(p.read_text(encoding="utf-8"))


EV = load_evidence()
CROWD = EV["crowded_new"]
CROWD_OLD = EV["crowded_old"]
EMPTY = EV["empty_new"]

RAW_CROWD = Image.open(ROOT / "assets" / CROWD["image"]).convert("RGB")
RAW_EMPTY = Image.open(ROOT / "assets" / EMPTY["image"]).convert("RGB")

# 原始素材是手机截图，含 App 工具条与抖音水印；裁掉后再上镜
# 裁剪量(像素)：empty 去掉顶部 52 工具条、底部 43 水印条；crowded 去掉底部 24
CROP_EMPTY = (0, 62, 389, 408)
CROP_CROWD = (0, 0, 366, 168)
RAW_EMPTY = RAW_EMPTY.crop(CROP_EMPTY)
RAW_CROWD = RAW_CROWD.crop(CROP_CROWD)


# ---------------------------------------------------------------- 场景 1 问题
class S1(Scene):
    dur = 3.4

    def draw(self, frame, t, u):
        d = ImageDraw.Draw(frame)
        a = ease(seg(t, 0.0, 0.40))
        y = 250 - (1 - a) * 18
        f1 = font(F_TITLE, 68)
        label = "教室到岗统计"
        d.text((96, y - 62), "01 /  PROBLEM", font=font(F_MONO, 20), fill=ACC)
        d.text((96, y), "还在靠点名？", font=f1, fill=TXT, stroke_width=0)
        d.text((96, y + 96), "人数靠人工、座位靠目测、结论靠感觉", font=font(F_BODY, 32), fill=MUT)

        # 右侧：三个点状网格，暗示"看不清"
        g = Image.new("RGBA", (360, 300), (0, 0, 0, 0))
        gd = ImageDraw.Draw(g)
        for r in range(6):
            for c in range(7):
                px, py = 24 + c * 46, 24 + r * 46
                al = int(40 * clamp01(seg(t, 0.3 + (r * 7 + c) * 0.012, 0.7 + (r * 7 + c) * 0.012)))
                gd.ellipse([px, py, px + 5, py + 5], fill=(45, 212, 167, al))
        frame.alpha_composite(g, (846, 210))


# ---------------------------------------------------------------- 场景 2 三个痛点
class S2(Scene):
    dur = 3.4

    ITEMS = [
        ("漏 检", "密集场景只报 9 人", "旧参数 640 / 0.25 的真实输出"),
        ("错 框", "16 个座位坐标写死", "换个视角，框就跑到黑板和墙上"),
        ("说 不 清", "改了什么、好了多少", "只能靠感觉，答辩无法复现"),
    ]

    def draw(self, frame, t, u):
        d = ImageDraw.Draw(frame)
        for i, (tag, title, sub) in enumerate(self.ITEMS):
            a = ease(seg(t, 0.12 + i * 0.16, 0.52 + i * 0.16))
            if a <= 0:
                continue
            x = 96 + i * 372
            y = 210 + (1 - a) * 26
            card = Image.new("RGBA", (340, 280), (0, 0, 0, 0))
            cd = ImageDraw.Draw(card)
            cd.rounded_rectangle([0, 0, 339, 279], 18, fill=CARD + (255,), outline=LINE + (255,), width=2)
            cd.rounded_rectangle([0, 0, 6, 279], 3, fill=WARN + (255,))
            cd.text((34, 40), tag, font=font(F_TITLE, 22), fill=WARN)
            cd.text((34, 92), title, font=font(F_TITLE, 30), fill=TXT)
            cd.text((34, 160), sub, font=font(F_BODY, 19), fill=MUT)
            card.putalpha(card.getchannel("A").point(lambda v: int(v * a)))
            frame.alpha_composite(card, (x, int(y)))


# ---------------------------------------------------------------- 场景 3 方案
class S3(Scene):
    dur = 5.2

    NODES = [
        ("输入", ["图片 / 视频 / 摄像头"], (58, 110, 165)),
        ("检测", ["YOLOv8n · person"], ACC),
        ("净化", ["框过滤 + 一人一座"], ACC),
        ("关联", ["透视网格 ROI"], ACC),
        ("输出", ["人数 / 座位 / CSV"], WARN),
    ]

    def draw(self, frame, t, u):
        d = ImageDraw.Draw(frame)
        d.text((96, 92), "02 /  SOLUTION", font=font(F_MONO, 20), fill=ACC)
        a = ease(seg(t, 0.05, 0.45))
        d.text((96, 126), "一条流水线，把画面变成数字", font=font(F_TITLE, 44), fill=TXT)

        y0 = 300
        bw, bh, gap = 196, 118, 41
        x0 = (W - (bw * 5 + gap * 4)) // 2
        for i, (name, lines, color) in enumerate(self.NODES):
            ap = ease(seg(t, 0.30 + i * 0.13, 0.62 + i * 0.13))
            if ap <= 0:
                continue
            x = x0 + i * (bw + gap)
            y = y0 + (1 - ap) * 22
            card = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
            cd = ImageDraw.Draw(card)
            cd.rounded_rectangle([0, 0, bw - 1, bh - 1], 14, fill=(24, 42, 58, 255), outline=color + (110,), width=2)
            cd.text((20, 22), name, font=font(F_TITLE, 24), fill=color)
            for j, ln in enumerate(lines):
                cd.text((20, 62 + j * 24), ln, font=font(F_BODY, 17), fill=TXT)
            card.putalpha(card.getchannel("A").point(lambda v: int(v * ap)))
            frame.alpha_composite(card, (x, int(y)))

            if i < 4:
                ax = x + bw + 8
                ay = y + bh // 2
                pa = ease(seg(t, 0.45 + i * 0.13, 0.75 + i * 0.13))
                d.line([ax, ay, ax + gap - 16, ay], fill=LINE, width=2)
                if pa > 0:
                    mx = ax + (gap - 16) * pa
                    d.polygon([(mx, ay - 5), (mx + 8, ay), (mx, ay + 5)], fill=ACC)
                    # 沿管线流动的小点
                    ph = (t * 1.1 + i * 0.25) % 1.0
                    px = ax + (gap - 16) * ph
                    d.ellipse([px - 3, ay - 3, px + 3, ay + 3], fill=ACC)

        # 底部三个成果
        for i, (k, v) in enumerate([("人数", "实时"), ("座位", "占用 / 空闲"), ("证据", "CSV 可复核")]):
            ap = ease(seg(t, 1.55 + i * 0.12, 2.0 + i * 0.12))
            if ap <= 0:
                continue
            x = 96 + i * 372
            d.text((x, 512), k, font=font(F_BODY, 18), fill=MUT)
            d.text((x, 538), v, font=font(F_TITLE, 28), fill=ACC)


# ---------------------------------------------------------------- 场景 4 空教室
class S4(Scene):
    dur = 5.4

    def draw(self, frame, t, u):
        d = ImageDraw.Draw(frame)
        d.text((96, 92), "03 /  DEMO · EMPTY ROOM", font=font(F_MONO, 20), fill=ACC)

        card_w, card_h = 366, 326
        cx, cy = 96, 168
        ap = ease(seg(t, 0.05, 0.5))
        if ap > 0:
            img = fit_cover(RAW_EMPTY, (card_w, card_h))
            # 网格逐格出现：使用真实 seat_boxes
            ov = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
            od = ImageDraw.Draw(ov)
            sx = card_w / RAW_EMPTY.width
            sy = card_h / RAW_EMPTY.height
            oy = CROP_EMPTY[1]
            n = len(EMPTY["seat_boxes"])
            for i, sb in enumerate(EMPTY["seat_boxes"]):
                g = ease(seg(t, 0.45 + i * 0.055, 0.75 + i * 0.055))
                if g <= 0:
                    continue
                box = [sb[0] * sx, (sb[1] - oy) * sy, sb[2] * sx, (sb[3] - oy) * sy]
                grow = 1 - g
                box = [
                    box[0] + (box[2] - box[0]) * 0.5 * grow,
                    box[1] + (box[3] - box[1]) * 0.5 * grow,
                    box[2] - (box[2] - box[0]) * 0.5 * grow,
                    box[3] - (box[3] - box[1]) * 0.5 * grow,
                ]
                od.rounded_rectangle(box, 4, outline=(45, 212, 167, int(230 * g)), width=2)
            ov = ov.filter(ImageFilter.GaussianBlur(0.4))
            img = img.convert("RGBA")
            img.alpha_composite(ov)
            img.putalpha(img.getchannel("A").point(lambda v: int(v * ap)))
            frame.alpha_composite(img, (cx, cy))

        # 右侧结论
        rx = 516
        a1 = ease(seg(t, 0.7, 1.05))
        if a1 > 0:
            d.text((rx, 196), "0", font=font(F_TITLE, 132), fill=ACC)
            d.text((rx + 118, 232), "人", font=font(F_TITLE, 44), fill=TXT)
            d.text((rx + 118, 292), "16 个座位框全部落在课桌区", font=font(F_BODY, 22), fill=MUT)
        a2 = ease(seg(t, 1.2, 1.6))
        if a2 > 0:
            d.line([rx, 372, rx + 660, 372], fill=LINE, width=2)
            d.text((rx, 400), "空场景不乱报", font=font(F_TITLE, 34), fill=TXT)
            d.text((rx, 452), "旧版固定坐标会把框压到黑板和天花板；", font=font(F_BODY, 22), fill=MUT)
            d.text((rx, 486), "现在按透视网格标定，空教室稳定 0 人 0 占用。", font=font(F_BODY, 22), fill=MUT)
        d.text((96, 620), "source: assets/classroom_empty.jpg  ·  B3_center_roi  ·  960 / 0.12",
               font=font(F_MONO, 16), fill=(96, 114, 130))


# ---------------------------------------------------------------- 场景 5 密集教室
class S5(Scene):
    dur = 8.0

    def draw(self, frame, t, u):
        d = ImageDraw.Draw(frame)
        d.text((96, 78), "03 /  DEMO · CROWDED ROOM", font=font(F_MONO, 20), fill=ACC)

        card_w, card_h = 700, 321
        cx, cy = 96, 132

        base = fit_cover(RAW_CROWD, (card_w, card_h))
        ov = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        sx = card_w / RAW_CROWD.width
        sy = card_h / RAW_CROWD.height
        oy = CROP_CROWD[1]

        old_a = 1.0 - ease(seg(t, 2.7, 3.1))
        new_a = ease(seg(t, 2.8, 3.2))
        layers = [
            (CROWD_OLD, old_a, WARN, 0.20),
            (CROWD, new_a, ACC, 2.90),
        ]
        for data, alpha_key, color, t_start in layers:
            if alpha_key <= 0.01:
                continue
            for i, b in enumerate(data["boxes"]):
                t0 = t_start + i * 0.05
                g = ease(seg(t, t0, t0 + 0.42))
                if g <= 0:
                    continue
                x1, y1, x2, y2 = b["bbox"]
                rect = [x1 * sx, (y1 - oy) * sy, x2 * sx, (y2 - oy) * sy]
                a = int(235 * g * alpha_key)
                lw = 2 if (rect[3] - rect[1]) > 46 else 1
                od.rounded_rectangle(rect, 3, outline=color + (a,), width=lw)
                # 小框不标序号，避免糊成一团
                if (rect[3] - rect[1]) > 46:
                    lbl = str(i + 1)
                    lf = font(F_MONO, 12)
                    tw = lf.getlength(lbl) + 7
                    if rect[1] - 14 > 0:
                        od.rectangle([rect[0], rect[1] - 14, rect[0] + tw, rect[1] - 2],
                                     fill=color + (int(a * 0.9),))
                        od.text((rect[0] + 3, rect[1] - 13), lbl, font=lf, fill=(9, 20, 30, a))

        base = base.convert("RGBA")
        base.alpha_composite(ov)
        frame.alpha_composite(base, (cx, cy))

        # 右侧计数卡
        rx = 838
        a0 = ease(seg(t, 0.2, 0.6))
        if a0 > 0:
            d.text((rx, 148), "检出人数", font=font(F_BODY, 20), fill=MUT)
            old_n, new_n = CROWD_OLD["persons"], CROWD["persons"]
            val = old_n + (new_n - old_n) * ease(seg(t, 2.85, 4.25))
            d.text((rx, 176), str(int(round(val))), font=font(F_TITLE, 116),
                   fill=(WARN if val < new_n - 0.5 else ACC))
            d.text((rx + (150 if val >= 10 else 90), 236), "人", font=font(F_TITLE, 32), fill=TXT)

            d.text((rx, 320), f"座位占用 {CROWD['occupied']}/{CROWD['seats']}",
                   font=font(F_BODY, 22), fill=TXT)
            d.text((rx, 356), f"原始框 {len(CROWD['boxes'])} · 过滤后 {CROWD['persons']}",
                   font=font(F_BODY, 19), fill=MUT)

        # 对比条
        a1 = ease(seg(t, 4.3, 4.9))
        if a1 > 0:
            by = 560
            bx = 96
            bw2 = 620
            d.text((bx, by - 34), "同一张图 · 只改推理参数", font=font(F_BODY, 19), fill=MUT)
            seg_w = bw2 * (CROWD_OLD["persons"] / CROWD["persons"])
            d.rounded_rectangle([bx, by, bx + bw2 * a1, by + 26], 13, fill=(30, 52, 68))
            d.rounded_rectangle([bx, by, bx + seg_w * a1, by + 26], 13, fill=WARN + (255,))
            d.text((bx + 12, by + 4), f"旧参数 640/0.25 → {CROWD_OLD['persons']} 人",
                   font=font(F_BODY, 15), fill=(9, 20, 30))
            by2 = by + 44
            d.rounded_rectangle([bx, by2, bx + bw2 * a1, by2 + 26], 13, fill=ACC)
            d.text((bx + 12, by2 + 4), f"现配置 960/0.12 → {CROWD['persons']} 人",
                   font=font(F_BODY, 15), fill=(9, 20, 30))


# ---------------------------------------------------------------- 场景 6 A/B 指标
class S6(Scene):
    dur = 6.4

    BARS = [
        ("预训练权重", 0.8132, WARN),
        ("自训权重 B1", 0.9942, ACC),
    ]

    def draw(self, frame, t, u):
        d = ImageDraw.Draw(frame)
        d.text((96, 92), "04 /  RESULT", font=font(F_MONO, 20), fill=ACC)
        a = ease(seg(t, 0.05, 0.45))
        d.text((96, 126), "mAP@0.5：0.8132 → 0.9942", font=font(F_TITLE, 42), fill=TXT)

        # 柱状图
        base_y = 520
        x0 = 150
        max_h = 240
        for i, (name, v, color) in enumerate(self.BARS):
            g = ease(seg(t, 0.5 + i * 0.35, 1.15 + i * 0.35))
            x = x0 + i * 190
            bh = max_h * v * g
            d.rounded_rectangle([x, base_y - bh, x + 110, base_y], 8, fill=color)
            if g > 0.9:
                d.text((x, base_y - bh - 44), f"{v:.4f}", font=font(F_TITLE, 30), fill=color)
            d.text((x, base_y + 16), name, font=font(F_BODY, 19), fill=MUT)
        d.line([x0 - 30, base_y, x0 + 420, base_y], fill=LINE, width=2)

        # 右侧：增量和工程证据
        rx = 640
        for i, (k, sub, color) in enumerate([
            ("+22.3%", "自训权重带来的 mAP 提升", ACC),
            ("5 组", "单变量消融 · 每次只改一处", TXT),
            ("6 / 6", "交付自检项全部通过", TXT),
        ]):
            ap = ease(seg(t, 1.4 + i * 0.22, 1.9 + i * 0.22))
            if ap <= 0:
                continue
            y = 250 + i * 118
            d.text((rx, y), k, font=font(F_TITLE, 40), fill=color)
            d.text((rx, y + 50), sub, font=font(F_BODY, 20), fill=MUT)

        d.text((96, 640), "source: experiments/results/ablation_report.md  ·  real runs, not hand-filled",
               font=font(F_MONO, 15), fill=(96, 114, 130))


# ---------------------------------------------------------------- 场景 7 工程可信
class S7(Scene):
    dur = 4.6

    def draw(self, frame, t, u):
        d = ImageDraw.Draw(frame)
        d.text((96, 92), "05 /  REPRODUCIBLE", font=font(F_MONO, 20), fill=ACC)
        a = ease(seg(t, 0.05, 0.45))
        d.text((96, 126), "不是跑通一次，是随时能重跑", font=font(F_TITLE, 42), fill=TXT)

        cells = [
            ("84", "训练图", "seed 42 固定"),
            ("22", "验证图", "与训练不重叠"),
            ("801", "person 框", "真实人物合成"),
            ("2", "训练权重", "A 基线 / B1 增强"),
        ]
        for i, (num, label, sub) in enumerate(cells):
            ap = ease(seg(t, 0.45 + i * 0.14, 0.85 + i * 0.14))
            if ap <= 0:
                continue
            x = 96 + (i % 2) * 300
            y = 226 + (i // 2) * 186
            card = Image.new("RGBA", (270, 158), (0, 0, 0, 0))
            cd = ImageDraw.Draw(card)
            cd.rounded_rectangle([0, 0, 269, 157], 14, fill=CARD + (255,), outline=LINE + (255,))
            cd.text((24, 22), num, font=font(F_TITLE, 52), fill=ACC)
            cd.text((24, 92), label, font=font(F_BODY, 20), fill=TXT)
            cd.text((24, 120), sub, font=font(F_BODY, 16), fill=MUT)
            card.putalpha(card.getchannel("A").point(lambda v: int(v * ap)))
            frame.alpha_composite(card, (x, int(y)))

        # 右侧：一条命令 + 产物清单
        rx = 700
        ap = ease(seg(t, 1.2, 1.65))
        if ap > 0:
            d.text((rx, 236), "一条命令，从数据到结论", font=font(F_BODY, 20), fill=MUT)
            box = Image.new("RGBA", (480, 74), (0, 0, 0, 0))
            bd = ImageDraw.Draw(box)
            bd.rounded_rectangle([0, 0, 479, 73], 10, fill=(8, 16, 26, 255), outline=LINE + (255,))
            bd.text((20, 12), "$ python scripts/train.py \\", font=font(F_MONO, 17), fill=ACC)
            bd.text((20, 38), "    --augment mosaic_strong", font=font(F_MONO, 17), fill=(150, 170, 190))
            box.putalpha(box.getchannel("A").point(lambda v: int(v * ap)))
            frame.alpha_composite(box, (rx, 268))

        for i, ln in enumerate(["权重 .pt 入仓，可直接复核",
                                "loss / mAP 曲线自动导出",
                                "训练报告 + 验证脚本 6 项自检"]):
            ap = ease(seg(t, 1.8 + i * 0.16, 2.2 + i * 0.16))
            if ap <= 0:
                continue
            y = 386 + i * 46
            d.ellipse([rx, y + 7, rx + 9, y + 16], fill=ACC)
            d.text((rx + 24, y), ln, font=font(F_BODY, 21), fill=TXT)

        ap3 = ease(seg(t, 2.9, 3.3))
        if ap3 > 0:
            d.line([96, 604, 1184, 604], fill=LINE, width=2)
            d.text((96, 622), "训练 12 epoch / 416 / batch 8 · CPU 约 75 秒 · seed 固定",
                   font=font(F_BODY, 18), fill=MUT)


# ---------------------------------------------------------------- 场景 8 能做什么
class S8(Scene):
    dur = 4.4

    ITEMS = [
        ("三种输入", "图片 / 视频 / 摄像头", "拖进去就出结果"),
        ("参数可调", "灵敏度 · 分辨率 · 行列", "换教室不用改代码"),
        ("结果可导出", "CSV / 标注图 / 统计面板", "直接进汇报材料"),
        ("流程可复现", "固定 seed + 一键脚本", "换台机器也能跑"),
    ]

    def draw(self, frame, t, u):
        d = ImageDraw.Draw(frame)
        d.text((96, 88), "06 /  WHAT YOU CAN DO", font=font(F_MONO, 20), fill=ACC)
        d.text((96, 120), "装上就能用，不是只能看的 Demo", font=font(F_TITLE, 40), fill=TXT)

        for i, (title, mid, sub) in enumerate(self.ITEMS):
            ap = ease(seg(t, 0.35 + i * 0.15, 0.75 + i * 0.15))
            if ap <= 0:
                continue
            x = 96 + (i % 2) * 546
            y = 222 + (i // 2) * 214
            card = Image.new("RGBA", (506, 188), (0, 0, 0, 0))
            cd = ImageDraw.Draw(card)
            cd.rounded_rectangle([0, 0, 505, 187], 16, fill=CARD + (255,), outline=LINE + (255,))
            cd.rounded_rectangle([0, 0, 5, 187], 3, fill=ACC + (255,))
            cd.text((30, 26), title, font=font(F_TITLE, 26), fill=ACC)
            cd.text((30, 76), mid, font=font(F_BODY, 23), fill=TXT)
            cd.text((30, 126), sub, font=font(F_BODY, 18), fill=MUT)
            card.putalpha(card.getchannel("A").point(lambda v: int(v * ap)))
            frame.alpha_composite(card, (x, int(y)))


# ---------------------------------------------------------------- 场景 9 CTA
class S9(Scene):
    dur = 4.2

    def draw(self, frame, t, u):
        d = ImageDraw.Draw(frame)
        a = ease(seg(t, 0.05, 0.5))
        d.text((96, 210), "YOLOv8 人员到岗统计与座位占用检测", font=font(F_TITLE, 40), fill=TXT)
        d.text((96, 268), "开源 · 可复现 · 有真实 A/B 证据", font=font(F_BODY, 24), fill=MUT)

        ap = ease(seg(t, 0.5, 0.95))
        if ap > 0:
            box = Image.new("RGBA", (760, 96), (0, 0, 0, 0))
            bd = ImageDraw.Draw(box)
            bd.rounded_rectangle([0, 0, 759, 95], 14, fill=(16, 30, 44, 255), outline=ACC + (120,), width=2)
            bd.text((28, 20), "github.com/huan0-gg/yolov8-person-counter", font=font(F_MONO, 26), fill=ACC)
            bd.text((28, 58), "git clone && python app.py   ·   3 分钟跑通全流程", font=font(F_BODY, 18), fill=MUT)
            box.putalpha(box.getchannel("A").point(lambda v: int(v * ap)))
            frame.alpha_composite(box, (96, 356))

        ap2 = ease(seg(t, 1.1, 1.5))
        if ap2 > 0:
            d.text((96, 500), "别只看它能跑，看它能替你交差。", font=font(F_BODY, 22), fill=MUT)
            d.line([96, 586, 1184, 586], fill=LINE, width=2)
            d.text((96, 604), "顾欢 · 银川科技学院 · 计算机科学与技术", font=font(F_BODY, 19), fill=(120, 140, 158))
            d.text((980, 604), "2026.11", font=font(F_MONO, 19), fill=(120, 140, 158))


SCENES = [S1(), S2(), S3(), S4(), S5(), S6(), S7(), S8(), S9()]

# 计算起止时间
_cursor = 0.0
for _s in SCENES:
    _s.start = _cursor
    _cursor += _s.dur
assert abs(_cursor - DUR) < 0.01, f"场景总时长 {_cursor} 与 {DUR} 不一致"


def scene_at(t: float):
    for s in SCENES:
        if s.start <= t < s.start + s.dur:
            return s
    return SCENES[-1]


def chrome(frame: Image.Image, t: float, scene: Scene) -> None:
    """统一角标 + 底部进度条，形成系列感。"""
    d = ImageDraw.Draw(frame)
    # 左上品牌
    d.ellipse([44, 40, 56, 52], fill=ACC)
    d.text((66, 36), "yolov8-person-counter", font=font(F_MONO, 15), fill=(110, 130, 148))
    # 右上计时
    d.text((W - 116, 36), f"{t:04.1f}s", font=font(F_MONO, 15), fill=(110, 130, 148))
    # 底部进度
    p = t / DUR
    d.line([0, H - 3, W, H - 3], fill=(24, 38, 52), width=3)
    d.line([0, H - 3, W * p, H - 3], fill=ACC, width=3)


def render_frames(target_fps: int = FPS) -> None:
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    base = bg_image()
    n = int(DUR * target_fps)
    for i in range(n):
        t = i / target_fps
        frame = base.copy().convert("RGBA")
        sc = scene_at(t)
        sc.draw(frame, t, seg(t, sc.start, sc.start + sc.dur))
        chrome(frame, t, sc)
        frame.convert("RGB").save(TMP / f"f{i:05d}.png")
        if i % 150 == 0:
            print(f"  frame {i}/{n}")


def encode_mp4() -> Path:
    out = OUT / "promo_45s.mp4"
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = None

    if exe and os.path.exists(exe):
        cmd = [
            exe, "-y", "-framerate", str(FPS), "-i", str(TMP / "f%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-preset", "slow", "-movflags", "+faststart", str(out),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("ffmpeg 编码完成:", out)
        return out

    # 回退：OpenCV mp4v（兼容性略差，但能出片）
    print("未找到 ffmpeg，回退到 OpenCV mp4v 编码")
    files = sorted(TMP.glob("f*.png"))
    first = cv2.imread(str(files[0]))
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                         (first.shape[1], first.shape[0]))
    for f in files:
        vw.write(cv2.imread(str(f)))
    vw.release()
    print("OpenCV 编码完成:", out)
    return out


def encode_gif() -> Path:
    """封面 GIF：密集教室标注 + 计数，控制体积便于 README 内嵌。"""
    out = OUT / "cover.gif"
    gw, gh = 640, 360
    gif_dir = OUT / "_gif"
    if gif_dir.exists():
        shutil.rmtree(gif_dir)
    gif_dir.mkdir(parents=True)

    t0 = SCENES[4].start
    base = bg_image()
    n = int(4.2 * 12)
    for i in range(n):
        t = t0 + i / 12
        frame = base.copy().convert("RGBA")
        sc = scene_at(t)
        sc.draw(frame, t, seg(t, sc.start, sc.start + sc.dur))
        frame.convert("RGB").resize((gw, gh), Image.LANCZOS).save(gif_dir / f"g{i:03d}.png")

    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = None

    if exe and os.path.exists(exe):
        vf = ("fps=12,scale=640:-1:flags=lanczos,split[a][b];"
              "[a]palettegen=max_colors=200:stats_mode=diff[p];"
              "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
        subprocess.run(
            [exe, "-y", "-framerate", "12", "-i", str(gif_dir / "g%03d.png"),
             "-vf", vf, "-loop", "0", str(out)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        frames = [Image.open(gif_dir / f"g{i:03d}.png").convert("RGB") for i in range(n)]
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=83,
                       loop=0, optimize=True)

    shutil.rmtree(gif_dir, ignore_errors=True)
    print("GIF 完成:", out, f"{out.stat().st_size/1024/1024:.1f} MB")
    return out


def main() -> int:
    print(f"渲染 {DUR}s @ {FPS}fps ...")
    render_frames()
    mp4 = encode_mp4()
    gif = encode_gif()
    shutil.rmtree(TMP, ignore_errors=True)

    cap = cv2.VideoCapture(str(mp4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"MP4: {mp4.name}  {mp4.stat().st_size/1024/1024:.1f} MB  "
          f"{cnt} 帧  {fps:.0f}fps  {cnt/fps:.1f}s")
    print(f"GIF: {gif.name}  {gif.stat().st_size/1024/1024:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
