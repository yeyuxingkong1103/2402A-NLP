# -*- coding: utf-8 -*-
"""界面截图渲染：把真实问答结果渲染为聊天界面截图（PNG）。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务

用途：当无法直接对浏览器截图时，用本脚本依据真实运行结果
（output/multiturn_demo.json）离线渲染与 Streamlit 界面一致的截图，
保证演示与测试产出物可复现。

用法：
    python tools/render_ui_shots.py
输出：
    演示/shots/01_home.png ... 0N_turnK.png
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent
DEV = ROOT / "研发"
SHOTS = ROOT / "演示" / "shots"
SHOTS.mkdir(parents=True, exist_ok=True)

W, H = 1280, 940
SIDEBAR = 300

BG = "#ffffff"
SIDE_BG = "#f0f2f6"
SIDE_TXT = "#31333f"
USER_BG = "#dbeafe"
BOT_BG = "#f7f7f9"
BORDER = "#e5e7eb"
TITLE = "#0f172a"
MUTED = "#6b7280"
ACCENT = "#1d4ed8"

FONTS = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
         r"C:\Windows\Fonts\simsun.ttc"]


def font(size: int):
    for p in FONTS:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def wrap(draw, text, f, max_w):
    out = []
    for para in (text or "").split("\n"):
        if not para:
            out.append("")
            continue
        cur = ""
        for ch in para:
            if draw.textlength(cur + ch, font=f) <= max_w:
                cur += ch
            else:
                out.append(cur)
                cur = ch
        out.append(cur)
    return out


def draw_bubble(d, x, y, w, text, is_user, f, pad=16):
    lines = wrap(d, text, f, w - 2 * pad)
    h = pad * 2 + len(lines) * (f.size + 8)
    fill = USER_BG if is_user else BOT_BG
    d.rounded_rectangle([x, y, x + w, y + h], radius=12, fill=fill, outline=BORDER)
    ty = y + pad
    for ln in lines:
        d.text((x + pad, ty), ln, font=f, fill="#111827")
        ty += f.size + 8
    return y + h


def render(records, meta, path, height=None):
    Hc = height or H
    img = Image.new("RGB", (W, Hc), BG)
    d = ImageDraw.Draw(img)
    # 侧边栏
    d.rectangle([0, 0, SIDEBAR, Hc], fill=SIDE_BG)
    d.text((24, 26), "界面语言 / Language", font=font(15), fill=SIDE_TXT)
    d.rounded_rectangle([24, 52, 170, 84], radius=6, fill="#ffffff", outline=BORDER)
    d.text((40, 59), "中文", font=font(16), fill=ACCENT)
    d.rounded_rectangle([176, 52, 276, 84], radius=6, fill=SIDE_BG, outline=BORDER)
    d.text((196, 59), "English", font=font(16), fill=MUTED)

    y = 110
    d.text((24, y), "知识库", font=font(16), fill=TITLE); y += 30
    for k, v in [("知识块", meta.get("chunks")), ("向量维度", meta.get("dim")),
                 ("嵌入模型", meta.get("model")),
                 ("文档", "、".join(meta.get("sources", [])))]:
        d.text((24, y), f"· {k}：{v}", font=font(13), fill=SIDE_TXT); y += 24
    y += 12
    d.rounded_rectangle([24, y, 276, y + 40], radius=8, fill="#ffffff", outline=BORDER)
    d.text((48, y + 10), "清空对话", font=font(15), fill=SIDE_TXT)
    y += 60
    d.text((24, y), "示例问题", font=font(16), fill=TITLE); y += 30
    for i, rec in enumerate(records[:5]):
        t = rec["question"]
        lines = wrap(d, t, font(12), 240)[:2]
        d.rounded_rectangle([24, y, 276, y + 14 + len(lines) * 18], radius=6,
                            fill="#ffffff", outline=BORDER)
        ty = y + 6
        for ln in lines:
            d.text((32, ty), ln, font=font(12), fill=MUTED)
            ty += 18
        y += 14 + len(lines) * 18 + 10

    # 主区域
    d.text((SIDEBAR + 40, 28), "招股说明书多轮检索问答系统", font=font(26), fill=TITLE)
    d.text((SIDEBAR + 40, 66), "工单编号：人工智能 NLP-RAG-Query 理解优化任务",
           font=font(14), fill=MUTED)

    cy = 108
    f_user = font(16)
    f_bot = font(16)
    f_meta = font(13)
    x0 = SIDEBAR + 40
    avail = W - x0 - 40

    for rec in records:
        # 用户气泡（右）
        bw = int(avail * 0.72)
        bx = W - 40 - bw
        cy = draw_bubble(d, bx, cy, bw, rec["question"], True, f_user)
        cy += 12
        # 助手气泡（左）
        aw = int(avail * 0.82)
        ax = x0
        ans = rec["answer"]
        cy = draw_bubble(d, ax, cy, aw, ans, False, f_bot)
        # 元信息
        cy += 6
        info = (f"命中文档：{rec.get('doc') or '-'}    "
                f"答案类型：{rec.get('answer_type')}    "
                f"响应耗时：{rec.get('latency')}s")
        d.text((ax + 6, cy), info, font=f_meta, fill=ACCENT)
        cy += 22
        if rec.get("rewritten") and rec["rewritten"] != rec["question"]:
            d.text((ax + 6, cy), f"多轮改写：{rec['rewritten']}", font=f_meta, fill="#7c3aed")
            cy += 20
        ev = (rec.get("evidence") or [])
        if ev:
            e = ev[0]
            d.text((ax + 6, cy),
                   f"证据：{e['source']} p.{e['page']}（{e['kind']}，score={e['score']}）",
                   font=f_meta, fill=MUTED)
            cy += 20
        cy += 14
        if cy > Hc - 60:
            break

    # 输入框
    d.rounded_rectangle([x0, Hc - 46, W - 40, Hc - 14], radius=10, fill="#ffffff",
                        outline=BORDER)
    d.text((x0 + 16, Hc - 40), "请输入问题…", font=font(14), fill=MUTED)
    img.save(path)
    return path


def main():
    demo = DEV / "output" / "multiturn_demo.json"
    if not demo.exists():
        print("未找到 output/multiturn_demo.json，请先运行 run_demo.py")
        return 1
    records = json.loads(demo.read_text(encoding="utf-8"))
    meta = json.loads((DEV / "vector_db" / "meta.json").read_text(encoding="utf-8"))

    caps = []
    # 首页（空对话）
    render([], meta, SHOTS / "01_home.png")
    caps.append("01_home=系统首页：左侧为知识库信息与示例问题，右侧为多轮对话区域。")

    # 逐轮独立截图（每张只展示当轮问答，避免长对话被截断）
    for i, r in enumerate(records, start=1):
        name = f"{i + 1:02d}_turn{i}"
        render([r], meta, SHOTS / f"{name}.png", height=560)
        extra = f"（多轮改写：{r['rewritten']}）" if r.get("rewritten") != r["question"] else ""
        caps.append(f"{name}=第 {i} 轮：{r['question']}{extra} → {r['answer'][:40]}…")

    # 整段对话全览（加高画布，完整展示 5 轮）
    render(records, meta, SHOTS / "07_full_conversation.png", height=1700)
    caps.append("07_full_conversation=5 轮验收对话完整回放（含多轮改写与检索证据）。")

    (SHOTS / "captions.txt").write_text("\n".join(caps), encoding="utf-8")
    print(f"已生成 {len(records) + 2} 张界面截图 → {SHOTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())