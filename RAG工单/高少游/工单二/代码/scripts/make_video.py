# -*- coding: utf-8 -*-
"""系统演示视频生成脚本
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化

用真实运行截图与评估图表合成演示视频（MP4）：
    1. 用 PIL 渲染 1920x1080 幻灯片（含中文标题/说明 + 截图/图表）；
    2. 用 OpenCV 逐帧写入，幻灯片之间做交叉淡入淡出。

用法（项目根目录）：
    python scripts/make_video.py
产物：
    output/demo_video.mp4
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

W, H = 1920, 1080
FPS = 30
DUR = 4.0          # 每张幻灯片时长（秒）
TRANS = 0.6        # 交叉淡入时长（秒）

FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"
FONT_REG = r"C:\Windows\Fonts\msyh.ttc"

BG = (17, 24, 39)
FG = (243, 244, 246)
ACCENT = (34, 197, 94)
MUTED = (148, 163, 184)
CARD = (31, 41, 55)


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _center_text(d: ImageDraw.ImageDraw, y: int, text: str, font, fill) -> int:
    w = d.textlength(text, font=font)
    d.text(((W - w) / 2, y), text, font=font, fill=fill)
    return y + font.size


def _slide_cover(title: str, subtitle: str, footer: str) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 14], fill=ACCENT)
    y = _center_text(d, 300, title, _font(FONT_BOLD, 78), FG)
    y = _center_text(d, y + 40, subtitle, _font(FONT_REG, 38), MUTED)
    _center_text(d, y + 90, footer, _font(FONT_REG, 30), ACCENT)
    return img


def _slide_content(kicker: str, title: str, note: str, image_path: Path | None) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 14], fill=ACCENT)

    d.text((90, 70), kicker, font=_font(FONT_REG, 30), fill=ACCENT)
    d.text((90, 120), title, font=_font(FONT_BOLD, 56), fill=FG)

    if image_path and Path(image_path).exists():
        box_w, box_h = W - 300, 620
        top = 300
        src = Image.open(image_path).convert("RGB")
        ratio = min(box_w / src.width, box_h / src.height)
        new = src.resize((int(src.width * ratio), int(src.height * ratio)), Image.LANCZOS)
        x = (W - new.width) // 2
        d.rectangle([x - 14, top - 14, x + new.width + 14, top + new.height + 14],
                    fill=CARD, outline=(55, 65, 81), width=2)
        img.paste(new, (x, top))

    if note:
        _center_text(d, H - 150, note, _font(FONT_REG, 30), MUTED)
    return img


def build_slides() -> list[Image.Image]:
    fig = config.FIGURE_DIR
    deliver = Path(r"C:\Users\30274\Desktop\实训一工单\工单二\设计")

    home = deliver / "06_Streamlit首页截图.png"
    single = deliver / "07_Streamlit优化后问答截图.png"
    compare = deliver / "08_Streamlit前后对比截图.png"

    scenes = [
        ("cover", "基于 PDF 文档的问答系统优化（RAG）",
         "工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化",
         "目标：问答准确率 ≥ 90% ｜ 响应时间 ≤ 3 s"),
        ("content", "系统架构", "离线建库 → 在线问答：解析 / 分块 / 检索 / 答案合成 四层协同",
         "RAG 全链路架构与数据流", fig / "04_pipeline_flow.png"),
        ("content", "功能演示 ①", "Streamlit 演示界面：优化后 / 优化前 / 前后对比 / 批量评估",
         "演示界面首页", home),
        ("content", "功能演示 ②", "单题问答：答案 + 页码引用 + 可折叠依据片段（预热后 2.108 s）",
         "优化后单题问答", single),
        ("content", "功能演示 ③", "同一问题前后对比：基线 16.95 s 未命中；优化后 2.139 s 命中",
         "优化前后对比（Q531 法定代表人）", compare),
        ("content", "优化点总览", "PDF 解析 · 分块 · 检索 · 答案合成 四层优化点与解决的问题",
         "四层优化点总览", fig / "06_optimization_layers.png"),
        ("content", "分块优化", "结构感知分块 + 表格原子块 + 父子块 vs 固定长度切片",
         "分块策略对比", fig / "05_chunking_compare.png"),
        ("content", "效果对比 ①", "答案命中率 60% → 100%，平均响应 3.532 s → 1.122 s",
         "核心指标对比", fig / "01_metrics_before_after.png"),
        ("content", "效果对比 ②", "10 道题逐题命中情况：优化前 6/10 → 优化后 10/10",
         "逐题命中对比", fig / "02_per_question_hit.png"),
        ("content", "效果对比 ③", "逐题响应时间：优化后最大 2.181 s，全部满足 ≤ 3 s",
         "响应时间对比", fig / "03_latency_compare.png"),
        ("cover", "结论：准确率 100%，最大响应 2.181 s",
         "答案零幻觉、可溯源（附页码引用）｜ 满足全部验收标准",
         "工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化"),
    ]

    slides: list[Image.Image] = []
    for sc in scenes:
        if sc[0] == "cover":
            slides.append(_slide_cover(sc[1], sc[2], sc[3]))
        else:
            slides.append(_slide_content(sc[1], sc[2], sc[3], sc[4]))
    return slides


def main() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUTPUT_DIR / "demo_video.mp4"

    slides = [np.array(s.convert("RGB"))[:, :, ::-1] for s in build_slides()]  # RGB→BGR
    n = len(slides)
    print(f"生成幻灯片 {n} 张")

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise SystemExit("无法创建视频写入器（mp4v 编解码器不可用）")

    hold = int((DUR - TRANS) * FPS)
    t_frames = int(TRANS * FPS)
    total = 0
    for i, img in enumerate(slides):
        n_hold = hold if i < n - 1 else int(DUR * FPS)
        for _ in range(n_hold):
            writer.write(img)
            total += 1
        if i < n - 1:
            nxt = slides[i + 1]
            for k in range(t_frames):
                a = k / t_frames
                frame = cv2.addWeighted(img, 1.0 - a, nxt, a, 0.0)
                writer.write(frame)
                total += 1
    writer.release()

    size = out.stat().st_size / 1024 / 1024 if out.exists() else 0
    print(f"完成：{out}（{size:.1f} MB，{total} 帧，约 {total / FPS:.1f} 秒）")


if __name__ == "__main__":
    main()