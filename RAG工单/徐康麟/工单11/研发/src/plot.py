# 工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务
# -*- coding: utf-8 -*-
"""PIL 折线图渲染（本机无 matplotlib，故自实现）。

只画**真实产物**（train_log.jsonl 等）里的数值，禁止手填数据。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]
_FONT_CACHE: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}

COLORS = [(28, 96, 200), (196, 48, 48), (22, 140, 74), (150, 80, 200),
          (200, 120, 20), (0, 140, 160), (110, 110, 110)]


def _font_path() -> str:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return p
    raise SystemExit("!! 未找到中文字体")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key not in _FONT_CACHE:
        p = r"C:\Windows\Fonts\msyhbd.ttc" if bold and Path(r"C:\Windows\Fonts\msyhbd.ttc").exists() else _font_path()
        _FONT_CACHE[key] = ImageFont.truetype(p, size)
    return _FONT_CACHE[key]


def line_chart(
    out_path: str | Path,
    *,
    title: str,
    series: Sequence[tuple[str, Sequence[float], Sequence[float]]],
    xlabel: str = "step",
    ylabel: str = "loss",
    width: int = 1400,
    height: int = 760,
    subtitle: str = "",
    markers: Sequence[tuple[float, str]] = (),
    y_min: float | None = None,
    y_max: float | None = None,
) -> Path:
    """绘制多序列折线图。

    ``series`` = [(名称, xs, ys), ...]；``markers`` = [(x, 文本), ...] 画竖直参考线。
    """
    img = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, width, 88], fill=(22, 40, 74))
    d.text((30, 20), title, font=font(28, True), fill=(255, 255, 255))
    if subtitle:
        d.text((30, 58), subtitle, font=font(16), fill=(196, 212, 236))

    left, right, top, bottom = 110, width - 40, 130, height - 90
    xs_all = [x for _n, xs, _ys in series for x in xs]
    ys_all = [y for _n, _xs, ys in series for y in ys]
    if not xs_all or not ys_all:
        d.text((left, top), "（无数据）", font=font(22), fill=(196, 48, 48))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        return Path(out_path)
    x0, x1 = min(xs_all), max(xs_all)
    if x1 <= x0:
        x1 = x0 + 1
    lo = y_min if y_min is not None else min(ys_all)
    hi = y_max if y_max is not None else max(ys_all)
    if hi <= lo:
        hi = lo + 1e-6
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad

    def px(x: float) -> float:
        return left + (x - x0) / (x1 - x0) * (right - left)

    def py(y: float) -> float:
        return bottom - (y - lo) / (hi - lo) * (bottom - top)

    # 网格 + Y 轴刻度
    for i in range(6):
        yv = lo + (hi - lo) * i / 5
        yy = py(yv)
        d.line([left, yy, right, yy], fill=(226, 232, 240))
        d.text((18, yy - 10), f"{yv:.4f}", font=font(15), fill=(108, 116, 128))
    # X 轴刻度
    for i in range(6):
        xv = x0 + (x1 - x0) * i / 5
        xx = px(xv)
        d.line([xx, bottom, xx, bottom + 6], fill=(150, 158, 170))
        d.text((xx - 22, bottom + 10), f"{xv:.0f}", font=font(15), fill=(108, 116, 128))
    d.line([left, top, left, bottom], fill=(120, 130, 145))
    d.line([left, bottom, right, bottom], fill=(120, 130, 145))
    d.text((right - 60, bottom + 34), xlabel, font=font(16), fill=(60, 70, 85))
    d.text((left - 90, top - 30), ylabel, font=font(16), fill=(60, 70, 85))

    for (x, label) in markers:
        if x0 <= x <= x1:
            xx = px(x)
            d.line([xx, top, xx, bottom], fill=(220, 150, 60), width=2)
            d.text((xx + 4, top + 4), label, font=font(14), fill=(180, 110, 20))

    for i, (name, xs, ys) in enumerate(series):
        color = COLORS[i % len(COLORS)]
        pts = [(px(x), py(y)) for x, y in zip(xs, ys)]
        if len(pts) > 1:
            d.line(pts, fill=color, width=3, joint="curve")
        for (xx, yy) in pts:
            d.ellipse([xx - 2.5, yy - 2.5, xx + 2.5, yy + 2.5], fill=color)
        lx = right - 320
        ly = top + 12 + i * 28
        d.line([lx, ly + 8, lx + 26, ly + 8], fill=color, width=3)
        last = ys[-1]
        d.text((lx + 34, ly), f"{name}（末值 {last:.4f}）", font=font(16), fill=(40, 48, 60))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[折线图] {out_path} ({width}x{height})")
    return Path(out_path)


def bar_chart(
    out_path: str | Path,
    *,
    title: str,
    labels: Sequence[str],
    groups: Sequence[tuple[str, Sequence[float]]],
    width: int = 1400,
    height: int = 720,
    subtitle: str = "",
    value_fmt: str = "{:.4f}",
) -> Path:
    """分组柱状图（每个 label 一组，组内多个系列并排）。"""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, width, 88], fill=(22, 40, 74))
    d.text((30, 20), title, font=font(28, True), fill=(255, 255, 255))
    if subtitle:
        d.text((30, 58), subtitle, font=font(16), fill=(196, 212, 236))
    left, right, top, bottom = 110, width - 40, 140, height - 110
    vals = [v for _n, vs in groups for v in vs]
    hi = max(vals) * 1.18 if vals else 1.0
    lo = 0.0

    def py(y: float) -> float:
        return bottom - (y - lo) / (hi - lo) * (bottom - top)

    for i in range(6):
        yv = lo + (hi - lo) * i / 5
        yy = py(yv)
        d.line([left, yy, right, yy], fill=(226, 232, 240))
        d.text((18, yy - 10), f"{yv:.3f}", font=font(15), fill=(108, 116, 128))
    d.line([left, bottom, right, bottom], fill=(120, 130, 145))

    n = max(1, len(labels))
    slot = (right - left) / n
    ng = max(1, len(groups))
    bw = slot * 0.62 / ng
    for gi, (gname, vs) in enumerate(groups):
        color = COLORS[gi % len(COLORS)]
        for i, v in enumerate(vs):
            cx = left + slot * (i + 0.5) - slot * 0.31 + bw * (gi + 0.5)
            y = py(v)
            d.rectangle([cx - bw / 2, y, cx + bw / 2, bottom], fill=color)
            d.text((cx - bw * 0.45, y - 20), value_fmt.format(v), font=font(13), fill=(40, 48, 60))
        lx = right - 330
        ly = top + 12 + gi * 26
        d.rectangle([lx, ly + 2, lx + 20, ly + 16], fill=color)
        d.text((lx + 28, ly), gname, font=font(16), fill=(40, 48, 60))
    for i, lab in enumerate(labels):
        cx = left + slot * (i + 0.5)
        d.text((cx - slot * 0.4, bottom + 12), lab[:14], font=font(15), fill=(60, 70, 85))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[柱状图] {out_path} ({width}x{height})")
    return Path(out_path)
