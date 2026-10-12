# -*- coding: utf-8 -*-
"""生成工单十六的设计类截图：架构图、技术流程图、思维导图、训练loss曲线、评估对比柱状图。"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

FONT = "/home/dabaie/.fonts/simhei.ttf"
if os.path.isfile(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/home/dabaie/code/工单/工单十六/docs/screenshots"
os.makedirs(OUT, exist_ok=True)


def box(ax, x, y, w, h, text, fc="#E8F1FF", ec="#2E6BD6", fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, color="#555", lw=1.4))


# 1. 技术架构图
def draw_arch():
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")
    ax.set_title("工单十六：专用视觉语言模型（VLM）领域适应技术架构", fontsize=15, fontweight="bold")
    # 数据层
    box(ax, 0.4, 7.3, 2.6, 1.1, "IMDR 工业语料\n专利PDF + QA对", fc="#FFF1E0", ec="#E08A2E")
    box(ax, 3.4, 7.3, 2.6, 1.1, "页面渲染\n图文块抽取", fc="#FFF1E0", ec="#E08A2E")
    box(ax, 6.4, 7.3, 2.6, 1.1, "SFT 数据集\ntrain/eval JSONL", fc="#FFF1E0", ec="#E08A2E")
    box(ax, 9.4, 7.3, 2.2, 1.1, "数据增强\n术语回填", fc="#FFF1E0", ec="#E08A2E")
    arrow(ax, 3.0, 7.85, 3.4, 7.85); arrow(ax, 6.0, 7.85, 6.4, 7.85); arrow(ax, 9.0, 7.85, 9.4, 7.85)
    # 模型层
    box(ax, 0.4, 5.4, 3.4, 1.2, "基座 VLM\nQwen2-VL-2B-Instruct\n(fp16 / 4bit 量化)", fc="#E8F1FF", ec="#2E6BD6")
    box(ax, 4.3, 5.4, 3.4, 1.2, "LoRA 适配器\nr=16, alpha=32\nq/k/v/o/gate/up/down", fc="#E8F1FF", ec="#2E6BD6")
    box(ax, 8.2, 5.4, 3.4, 1.2, "训练策略\ngrad_checkpointing\n冻结视觉塔 / batch=1", fc="#E8F1FF", ec="#2E6BD6")
    arrow(ax, 3.8, 6.0, 4.3, 6.0); arrow(ax, 7.7, 6.0, 8.2, 6.0)
    arrow(ax, 7.7, 7.3, 6.0, 6.6)  # 数据 -> 模型
    # 评估层
    box(ax, 0.4, 3.3, 2.7, 1.1, "通用指标\nBLEU / ROUGE-L", fc="#EAFBEA", ec="#2E9E4A")
    box(ax, 3.5, 3.3, 2.7, 1.1, "工业术语\n准确性/覆盖率", fc="#EAFBEA", ec="#2E9E4A")
    box(ax, 6.6, 3.3, 2.7, 1.1, "图纸推理\n选择题命中率", fc="#EAFBEA", ec="#2E9E4A")
    box(ax, 9.7, 3.3, 1.9, 1.1, "失败案例\n分析", fc="#EAFBEA", ec="#2E9E4A")
    arrow(ax, 2.1, 5.4, 2.1, 4.4); arrow(ax, 4.9, 5.4, 4.9, 4.4); arrow(ax, 7.9, 5.4, 7.9, 4.4)
    # 输出
    box(ax, 2.5, 1.3, 3.2, 1.1, "LoRA 权重\neval_baseline/finetuned.json", fc="#F3E8FF", ec="#8A2BE2")
    box(ax, 6.3, 1.3, 3.2, 1.1, "评估报告 + 超参建议\n部署脚本(start/stop)", fc="#F3E8FF", ec="#8A2BE2")
    arrow(ax, 3.1, 3.3, 3.6, 2.4); arrow(ax, 8.0, 3.3, 7.6, 2.4)
    ax.text(6, 0.5, "RTX 5060 Laptop 8GB · transformers+peft+accelerate", ha="center", fontsize=10, color="#666")
    fig.tight_layout(); fig.savefig(f"{OUT}/01_architecture.png", dpi=130); plt.close(fig)
    print("saved 01_architecture.png")


# 2. 技术流程图
def draw_flow():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("微调与评估流程", fontsize=15, fontweight="bold")
    steps = [
        ("① 准备微调数据", "build_finetune_data.py\nQA对+图文块→JSONL\nimage/question/answer"),
        ("② 配置 LoRA", "lora_sft.yaml\nr=16 alpha=32\nlr=1e-4 冻结视觉塔"),
        ("③ 启动微调", "finetune.py\nfp16+grad_checkpoint\n保存适配器"),
        ("④ 专业评估", "evaluate.py\nBLEU/ROUGE+术语+图纸\nbaseline对比"),
        ("⑤ 产出交付", "评估报告+失败案例\n超参建议+部署脚本"),
    ]
    xs = [0.5, 2.4, 4.3, 6.2, 8.1]
    for (t, d), x in zip(steps, xs):
        box(ax, x, 6.6, 1.7, 1.0, t, fc="#DFF3FF", ec="#1C7ED6", fs=11)
        ax.text(x + 0.85, 5.4, d, ha="center", va="top", fontsize=8.5)
    for i in range(4):
        arrow(ax, xs[i] + 1.7, 7.1, xs[i + 1], 7.1)
    # 反馈环
    ax.text(5, 3.5, "迭代反馈：loss 不降 / 术语命中低 → 调 lr、增 epoch、扩充数据、强化术语样本",
            ha="center", fontsize=11, color="#B00020",
            bbox=dict(boxstyle="round", fc="#FFF0F0", ec="#B00020"))
    arrow(ax, 8.95, 6.6, 5.5, 4.2); arrow(ax, 4.5, 4.2, 1.35, 6.6)
    fig.tight_layout(); fig.savefig(f"{OUT}/02_flow.png", dpi=130); plt.close(fig)
    print("saved 02_flow.png")


# 3. 思维导图
def draw_mind():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("工单十六 · 思维导图", fontsize=15, fontweight="bold")
    box(ax, 4.8, 4.4, 2.4, 1.0, "调试与微调\n专用 VLM", fc="#FFE9B0", ec="#D9822B", fs=13)
    nodes = [
        (0.4, 7.5, "数据准备\n专利渲染\nQA→JSONL\n102/18 划分"),
        (0.4, 4.5, "模型微调\nQwen2-VL-2B\nLoRA r=16\nfp16+ckpt"),
        (0.4, 1.5, "显存优化\n4bit量化\n冻结视觉塔\nbatch=1"),
        (8.6, 7.5, "专业评估\nBLEU/ROUGE\n术语准确\n图纸推理"),
        (8.6, 4.5, "交付物\n适配器权重\n评估报告\n部署脚本"),
        (8.6, 1.5, "迭代优化\n超参建议\n失败案例\n术语增强"),
    ]
    for x, y, t in nodes:
        box(ax, x, y, 2.8, 1.3, t, fc="#EAF4FF", ec="#2E6BD6", fs=10)
        sx = 6 if x < 6 else 6
        sy = 4.9
        arrow(ax, 6, sy, x + 1.4, y + 0.65)
    fig.tight_layout(); fig.savefig(f"{OUT}/03_mindmap.png", dpi=130); plt.close(fig)
    print("saved 03_mindmap.png")


# 4. 训练 loss 曲线
def draw_loss():
    p = "/home/dabaie/code/工单/工单十六/finetune_output/train_history.json"
    h = json.load(open(p))
    pts = [(x["epoch"], x["loss"]) for x in h if "loss" in x]
    if not pts:
        return
    ep, ls = zip(*pts)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ep, ls, marker="o", color="#2E6BD6", lw=2)
    ax.set_xlabel("训练进度 (epoch)"); ax.set_ylabel("训练 Loss")
    ax.set_title("LoRA 微调训练 Loss 曲线（1 epoch，26 steps）")
    ax.grid(alpha=0.3)
    for i, (e, l) in enumerate(zip(ep, ls)):
        if i in (0, len(ep) - 1):
            ax.annotate(f"{l:.2f}", (e, l), textcoords="offset points", xytext=(0, 10), ha="center")
    fig.tight_layout(); fig.savefig(f"{OUT}/04_train_loss.png", dpi=130); plt.close(fig)
    print("saved 04_train_loss.png")


# 5. 评估对比柱状图（若评估结果存在）
def draw_eval():
    import os
    base = f"{OUT}/eval_baseline.json"
    fine = f"{OUT}/eval_finetuned.json"
    if not (os.path.isfile(base) and os.path.isfile(fine)):
        print("eval json not ready, skip 05")
        return
    b = json.load(open(base))["summary"]; f = json.load(open(fine))["summary"]
    metrics = ["avg_bleu", "avg_rouge_l", "exact_acc", "image_exact_acc"]
    labels = ["BLEU", "ROUGE-L", "准确率", "图纸推理"]
    bv = [b[m] for m in metrics]; fv = [f[m] for m in metrics]
    import numpy as np
    x = np.arange(len(metrics)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, bv, w, label="基线", color="#9BB7E8")
    ax.bar(x + w / 2, fv, w, label="微调后", color="#2E6BD6")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("分数")
    ax.set_title("基线 vs 微调后 评估指标对比")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(fv):
        ax.text(i + w / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/05_eval_compare.png", dpi=130); plt.close(fig)
    print("saved 05_eval_compare.png")


if __name__ == "__main__":
    draw_arch(); draw_flow(); draw_mind(); draw_loss(); draw_eval()
    print("ALL DIAGRAMS DONE")
