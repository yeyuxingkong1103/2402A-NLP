# -*- coding: utf-8 -*-
"""演示视频生成：把界面截图合成为带字幕的 MP4。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务

依赖 opencv（cv2.VideoWriter）。用法：
    python tools/make_demo_video.py --shots 演示/shots --out 演示/演示视频.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap(draw, text, font, max_w):
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=font) <= max_w:
            cur += ch
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def frame_from_image(path: Path, size=(1280, 720), caption="", index=0, total=1):
    img = Image.open(path).convert("RGB")
    img.thumbnail((size[0] - 40, size[1] - 190))
    canvas = Image.new("RGB", size, "#0f172a")
    x = (size[0] - img.width) // 2
    y = 60 + (size[1] - 190 - img.height) // 2
    canvas.paste(img, (x, y))
    d = ImageDraw.Draw(canvas)
    # 顶部标题
    d.rectangle([0, 0, size[0], 52], fill="#1e293b")
    d.text((24, 12), "招股说明书多轮检索问答系统 · 功能演示", font=_font(24), fill="#e2e8f0")
    # 底部字幕
    d.rectangle([0, size[1] - 130, size[0], size[1]], fill="#1e293b")
    f = _font(22)
    lines = _wrap(d, caption, f, size[0] - 60)[:4]
    ty = size[1] - 118
    for ln in lines:
        d.text((30, ty), ln, font=f, fill="#f1f5f9")
        ty += 30
    d.text((size[0] - 120, size[1] - 40), f"{index}/{total}", font=_font(18), fill="#94a3b8")
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="演示/shots")
    ap.add_argument("--out", default="演示/演示视频.mp4")
    ap.add_argument("--seconds", type=float, default=3.5, help="每张截图停留秒数")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    shots_dir = Path(args.shots)
    images = sorted([p for p in shots_dir.glob("*.png")])
    if not images:
        print("未找到截图，请先运行 tools/capture_shots.py")
        return 1

    captions = {}
    cap_file = shots_dir / "captions.txt"
    if cap_file.exists():
        for line in cap_file.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                captions[k.strip()] = v.strip()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out), fourcc, args.fps, (1280, 720))

    per = int(args.seconds * args.fps)
    total = len(images)
    for i, img in enumerate(images, 1):
        cap = captions.get(img.stem, img.stem)
        frame = frame_from_image(img, caption=cap, index=i, total=total)
        for _ in range(per):
            vw.write(frame)
    vw.release()
    print(f"演示视频已生成 → {out}（{total} 张截图，约 {total * args.seconds:.1f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())