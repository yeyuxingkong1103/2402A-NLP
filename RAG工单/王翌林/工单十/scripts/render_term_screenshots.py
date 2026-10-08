#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 工单十：将终端输出渲染为 PNG 截图
import os
from PIL import Image, ImageDraw, ImageFont

TERM_DIR = "/tmp/v10_term"
OUT_DIR = "/home/dabaie/code/工单/工单十/docs/screenshots"

FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
]
FONT_SIZE = 14
LINE_H = 20
PAD_X = 20
PAD_Y = 20
BG = (30, 30, 46)
FG = (220, 220, 220)

def get_font():
    for p in FONT_PATHS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, FONT_SIZE)
            except Exception:
                pass
    return ImageFont.load_default()

def render_text_to_png(txt_path, out_path):
    with open(txt_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    font = get_font()
    # 计算宽度
    dummy = Image.new("RGB", (100, 100))
    dd = ImageDraw.Draw(dummy)
    max_w = 0
    for ln in lines:
        bbox = dd.textbbox((0, 0), ln, font=font)
        w = bbox[2] - bbox[0]
        if w > max_w:
            max_w = w
    img_w = max_w + PAD_X * 2
    img_h = len(lines) * LINE_H + PAD_Y * 2
    img = Image.new("RGB", (img_w, img_h), BG)
    draw = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        draw.text((PAD_X, PAD_Y + i * LINE_H), ln, fill=FG, font=font)
    img.save(out_path)
    print(f"saved {out_path} ({img_w}x{img_h})")

mapping = {
    "01_docker_ps.txt": "04_docker_ps.png",
    "02_docker_logs_api.txt": "05_docker_logs_api.png",
    "03_docker_logs_ui.txt": "06_docker_logs_ui.png",
    "04_docker_volume.txt": "07_docker_volume.png",
    "05_docker_network.txt": "08_docker_network.png",
    "06_acceptance_result.txt": "09_acceptance_result.png",
}

os.makedirs(OUT_DIR, exist_ok=True)
for src, dst in mapping.items():
    sp = os.path.join(TERM_DIR, src)
    if os.path.exists(sp):
        render_text_to_png(sp, os.path.join(OUT_DIR, dst))
print("done")
