#!/usr/bin/env python3
"""渲染若干关键帧为 PNG，用于上线前逐帧检查排版。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_promo as mp  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "video" / "preview"
OUT.mkdir(parents=True, exist_ok=True)

base = mp.bg_image()
for label, t in [
    ("s1", 2.0),
    ("s2", 3.0),
    ("s3", 9.5),
    ("s4", 16.0),
    ("s5_old", 19.5),
    ("s5_new", 23.5),
    ("s5_bar", 24.8),
    ("s6", 28.5),
    ("s7", 34.0),
    ("s8", 40.0),
    ("s9", 43.5),
]:
    frame = base.copy().convert("RGBA")
    sc = mp.scene_at(t)
    sc.draw(frame, t, mp.seg(t, sc.start, sc.start + sc.dur))
    mp.chrome(frame, t, sc)
    frame.convert("RGB").save(OUT / f"{label}.png")
    print("ok", label, t, type(sc).__name__)
