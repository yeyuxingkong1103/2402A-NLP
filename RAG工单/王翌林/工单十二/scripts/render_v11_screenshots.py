#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 工单十一：渲染终端输出为 PNG 截图 + 绘制训练 loss 曲线
import os
import json
from PIL import Image, ImageDraw, ImageFont

TERM_DIR = "/tmp/v11_term"
OUT_DIR = "/home/dabaie/code/工单/工单十一/docs/screenshots"

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

def get_font(size=FONT_SIZE):
    for p in FONT_PATHS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()

def render_text_to_png(txt_path, out_path):
    with open(txt_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    font = get_font()
    dummy = Image.new("RGB", (100, 100))
    dd = ImageDraw.Draw(dummy)
    max_w = 0
    for ln in lines:
        bbox = dd.textbbox((0, 0), ln, font=font)
        w = bbox[2] - bbox[0]
        if w > max_w:
            max_w = w
    img_w = max(max_w + PAD_X * 2, 600)
    img_h = len(lines) * LINE_H + PAD_Y * 2
    img = Image.new("RGB", (img_w, img_h), BG)
    draw = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        draw.text((PAD_X, PAD_Y + i * LINE_H), ln, fill=FG, font=font)
    img.save(out_path)
    print(f"saved {out_path} ({img_w}x{img_h})")

mapping = {
    "01_finetune_metrics.txt": "01_finetune_metrics.png",
    "02_train_loss.txt": "02_train_loss_log.png",
    "03_dataset.txt": "03_dataset.png",
    "04_model_dir.txt": "04_model_dir.png",
    "05_unit_tests.txt": "05_unit_tests.png",
    "06_gen_report.txt": "06_gen_report.png",
}

os.makedirs(OUT_DIR, exist_ok=True)
for src, dst in mapping.items():
    sp = os.path.join(TERM_DIR, src)
    if os.path.exists(sp):
        render_text_to_png(sp, os.path.join(OUT_DIR, dst))

# ========== 绘制训练 loss 曲线图 ==========
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.load(open("/home/dabaie/code/工单/工单十一/docs/finetune_v11_results.json"))
    steps = [h["step"] for h in d["train"]["log_history"]]
    losses = [h["loss"] for h in d["train"]["log_history"]]
    lrs = [h["learning_rate"] for h in d["train"]["log_history"]]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = "#e74c3c"
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(steps, losses, "o-", color=color, linewidth=2, markersize=5, label="Training Loss")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = "#3498db"
    ax2.set_ylabel("Learning Rate", color=color)
    ax2.plot(steps, lrs, "s--", color=color, linewidth=1.5, markersize=4, label="Learning Rate")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("工单十一 · bge-m3 微调训练 Loss 曲线\n(MultipleNegativesRankingLoss, epochs=2, batch=8, freeze bottom 12 layers)")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "07_train_loss_curve.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"saved {out_path}")
except Exception as e:
    print(f"matplotlib plot skipped: {e}")

# ========== 绘制微调前后指标对比柱状图 ==========
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    d = json.load(open("/home/dabaie/code/工单/工单十一/docs/finetune_v11_results.json"))
    metrics = [c["metric"] for c in d["compare"]]
    before = [c["before"] for c in d["compare"]]
    after = [c["after"] for c in d["compare"]]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, before, width, label="微调前 (Before)", color="#95a5a6")
    bars2 = ax.bar(x + width/2, after, width, label="微调后 (After)", color="#2ecc71")

    ax.set_ylabel("Score")
    ax.set_title("工单十一 · 微调前后检索指标对比 (7/7 项全部提升)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "08_metrics_comparison.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"saved {out_path}")
except Exception as e:
    print(f"metrics plot skipped: {e}")

print("done")
