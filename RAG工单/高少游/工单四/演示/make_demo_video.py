# -*- coding: utf-8 -*-
"""演示视频合成脚本（图像内容解析及检索优化版）
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化

把演示截图序列合成为一段带标题卡与说明字幕的 MP4 演示视频。

素材目录（默认 演示/shots/）中的 PNG/JPG 按文件名排序后依次出场，
每张图配一条字幕（由文件名或 sidecar .txt 提供），并生成片头/片尾标题卡。

依赖：opencv-python（cv2）、Pillow（中文渲染）

用法：
    python make_demo_video.py --shots ../演示/shots --out ../演示/演示视频.mp4
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 900
FPS = 30
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]
BG = (18, 24, 38)
FG = (240, 244, 250)
ACCENT = (56, 189, 248)


def _font(size: int):
    for p in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _canvas() -> Image.Image:
    return Image.new("RGB", (W, H), BG)


def _center(draw, text, font, y, fill=FG, width=W):
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text(((width - (bbox[2] - bbox[0])) / 2, y), text, font=font, fill=fill)


def title_card(title: str, subtitle: str = "") -> np.ndarray:
    img = _canvas()
    d = ImageDraw.Draw(img)
    d.rectangle([0, H // 2 - 150, W, H // 2 - 146], fill=ACCENT)
    _center(d, title, _font(60), H // 2 - 120)
    if subtitle:
        _center(d, subtitle, _font(28), H // 2 + 10, fill=(180, 195, 215))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def shot_frame(image_path: Path, caption: str) -> np.ndarray:
    """一张截图 + 顶部标题条 + 底部字幕条。"""
    img = _canvas()
    d = ImageDraw.Draw(img)
    # 顶栏
    d.rectangle([0, 0, W, 84], fill=(28, 36, 54))
    _center(d, caption or image_path.stem, _font(34), 22)
    # 截图区域
    try:
        pic = Image.open(image_path).convert("RGB")
        max_w, max_h = W - 80, H - 200
        ratio = min(max_w / pic.width, max_h / pic.height)
        pic = pic.resize((int(pic.width * ratio), int(pic.height * ratio)))
        x = (W - pic.width) // 2
        y = 100 + (max_h - pic.height) // 2
        img.paste(pic, (x, y))
        d.rectangle([x - 1, y - 1, x + pic.width, y + pic.height], outline=(70, 90, 120))
    except Exception:
        _center(d, f"[无法加载 {image_path.name}]", _font(28), H // 2)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def caption_for(p: Path) -> str:
    side = p.with_suffix(".txt")
    if side.exists():
        return side.read_text(encoding="utf-8").strip()
    return re.sub(r"^\d+[_\-]?", "", p.stem).replace("_", " ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="../演示/shots")
    ap.add_argument("--out", default="../演示/演示视频.mp4")
    ap.add_argument("--hold", type=float, default=3.0, help="每张截图停留秒数")
    ap.add_argument("--title", default="招股说明书图像内容解析问答系统")
    ap.add_argument("--subtitle", default="工单编号：人工智能 NLP-RAG-图像内容解析及检索优化")
    args = ap.parse_args()

    shots_dir = Path(args.shots)
    files = sorted([p for p in shots_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out), fourcc, FPS, (W, H))
    hold = int(FPS * args.hold)

    def write(frame, n):
        for _ in range(n):
            vw.write(frame)

    write(title_card(args.title, args.subtitle), int(FPS * 4))
    for i, p in enumerate(files, 1):
        write(shot_frame(p, caption_for(p)), hold)
    write(title_card("演示结束", "感谢观看 · 系统准确率与响应时间详见测试报告"), int(FPS * 3))

    vw.release()
    print(f"演示视频已生成: {out}  ({len(files)} 张截图, 时长约 "
          f"{(4 + len(files) * args.hold + 3):.1f}s)")


if __name__ == "__main__":
    main()