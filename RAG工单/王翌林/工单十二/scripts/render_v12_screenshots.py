#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 工单十二：渲染终端输出为 PNG 截图 + 绘制 RAG vs LightRAG 对比图
import os
import json
from PIL import Image, ImageDraw, ImageFont

TERM_DIR = "/tmp/v12_term"
OUT_DIR = "/home/dabaie/code/工单/工单十二/docs/screenshots"

# 工单十二：含中文输出，优先使用黑体
FONT_PATHS = [
    "/home/dabaie/.fonts/simhei.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
FONT_SIZE = 15
LINE_H = 22
PAD_X = 20
PAD_Y = 20
BG = (30, 30, 46)
FG = (220, 220, 220)
GREEN = (130, 220, 130)


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
        color = GREEN if ln.startswith("[") or ln.startswith("===") else FG
        draw.text((PAD_X, PAD_Y + i * LINE_H), ln, fill=color, font=font)
    img.save(out_path)
    print(f"saved {out_path} ({img_w}x{img_h})")


mapping = {
    "01_graph_stats.txt": "01_graph_stats.png",
    "02_comparison.txt": "02_comparison_summary.png",
    "03_unit_tests.txt": "03_unit_tests.png",
    "04_storage.txt": "04_storage.png",
}

os.makedirs(OUT_DIR, exist_ok=True)
for src, dst in mapping.items():
    sp = os.path.join(TERM_DIR, src)
    if os.path.exists(sp):
        render_text_to_png(sp, os.path.join(OUT_DIR, dst))

# ========== 生成示例答案对比截图 ==========
try:
    res = json.load(open("/home/dabaie/code/工单/工单十二/docs/v12_comparison_results.json"))
    lines = ["工单十二：RAG vs LightRAG 检索答案示例对比", "=" * 60, ""]
    for r in res["results"][:3]:
        lines.append(f"[问题 {r['id']}] {r['question']}")
        lines.append("-" * 60)
        lines.append(f"RAG 答案      : {r['rag']['answer'][:120]}...")
        lines.append(f"LightRAG 答案 : {r['lightrag']['answer'][:120]}...")
        rs, ls = r["rag"]["scores"], r["lightrag"]["scores"]
        lines.append(f"RAG      relevancy={rs['answer_relevancy']:.3f} recall={rs['context_recall']:.3f}")
        lines.append(f"LightRAG relevancy={ls['answer_relevancy']:.3f} recall={ls['context_recall']:.3f}")
        lines.append("")
    sample_txt = "/tmp/v12_term/05_sample_answers.txt"
    with open(sample_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    render_text_to_png(sample_txt, os.path.join(OUT_DIR, "05_sample_answers.png"))
except Exception as e:
    print(f"示例答案截图失败: {e}")

# ========== 绘制 RAGAS 指标对比图 ==========
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 工单十二：注册黑体以支持中文
    font_manager.fontManager.addfont("/home/dabaie/.fonts/simhei.ttf")
    plt.rcParams["font.family"] = "SimHei"
    plt.rcParams["axes.unicode_minus"] = False

    d = json.load(open("/home/dabaie/code/工单/工单十二/docs/v12_comparison_results.json"))
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    labels = ["Faithfulness\n(忠实度)", "Answer Relevancy\n(答案相关性)",
              "Context Precision\n(上下文精确率)", "Context Recall\n(上下文召回率)"]
    rag_vals = [d["rag"][f"avg_{m}"] for m in metrics]
    lr_vals = [d["lightrag"][f"avg_{m}"] for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    x = range(len(metrics))
    w = 0.35
    ax1 = axes[0]
    b1 = ax1.bar([i - w / 2 for i in x], rag_vals, w, label="RAG (v6 混合检索)", color="#4C72B0")
    b2 = ax1.bar([i + w / 2 for i in x], lr_vals, w, label="LightRAG (知识图谱)", color="#55A868")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("得分")
    ax1.set_ylim(0, 1.15)
    ax1.set_title("RAGAS 四项指标对比（16 题平均）")
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", alpha=0.3)
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.3f}",
                 ha="center", fontsize=9)

    ax2 = axes[1]
    systems = ["RAG (v6)", "LightRAG"]
    times = [d["rag"]["avg_time_s"], d["lightrag"]["avg_time_s"]]
    bars = ax2.bar(systems, times, color=["#4C72B0", "#55A868"], width=0.45)
    ax2.set_ylabel("平均响应时间（秒）")
    ax2.set_title("检索响应时间对比")
    ax2.grid(axis="y", alpha=0.3)
    for b in bars:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 0.2, f"{h:.2f}s",
                 ha="center", fontsize=11)

    plt.suptitle("工单十二：RAG vs LightRAG 对比评估", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "06_rag_vs_lightrag_chart.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
except Exception as e:
    print(f"图表绘制失败: {e}")
