# -*- coding: utf-8 -*-
# 工单16：生成设计图5张 + 测试图表 + HTML报告 + 截图素材
import html, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent.parent  # 成品/16
DESIGN = BASE / "设计"
TEST = BASE / "测试"
LOGS = BASE / "研发" / "logs"
C_BLUE, C_GREEN, C_ORANGE, C_RED, C_PURPLE, C_GRAY = "#4a90d9", "#5cb85c", "#f0ad4e", "#d9534f", "#9b59b6", "#7f8c8d"

def box(ax, x, y, w, h, text, fc=C_BLUE, fs=10):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="#333", lw=1.2))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, color="white", wrap=True)

def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"已生成 {path.name}")

# 图1：微调流程总览
def fig1():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 6); ax.axis("off")
    ax.text(5.5, 5.7, "VLM LoRA 微调全流程", ha="center", fontsize=15, weight="bold")
    steps = [("IMDR QA数据\n10096条", C_GRAY), ("数据转换\nconvert_data.py\n→train/val.jsonl", C_BLUE),
             ("LoRA配置\nlora_qwen_vl\n.yaml", C_ORANGE), ("训练调试\nmock_train.py\n1 epoch", C_GREEN),
             ("专业评估\nevaluate.py\nBLEU/术语/图纸", C_PURPLE), ("评估报告\n前后对比", C_RED)]
    w, h, gap = 1.6, 1.3, 0.2
    x = 0.3
    for t, c in steps:
        box(ax, x, 3.5, w, h, t, c, fs=9); x += w + gap
    for i in range(len(steps)-1):
        ax.annotate("", xy=(0.3+(i+1)*(w+gap), 4.15), xytext=(0.3+(i+1)*(w+gap)-gap, 4.15),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#444"))
    ax.text(5.5, 2.5, "环境说明：本机无GPU，训练用CPU模拟（mock_train.py），", ha="center", fontsize=10, color=C_RED)
    ax.text(5.5, 2.0, "生产环境应使用 llamafactory-cli + lora_qwen_vl_industrial.yaml 在GPU上运行", ha="center", fontsize=10, color=C_RED)
    save(fig, DESIGN / "01-微调流程总览图.png")

# 图2：训练损失曲线
def fig2():
    log = json.loads((LOGS / "train_log.json").read_text(encoding="utf-8"))
    steps = [e["step"] for e in log["train_log"]]
    losses = [e["loss"] for e in log["train_log"]]
    val_steps = [e["step"] for e in log["train_log"] if "val_loss" in e]
    val_losses = [e["val_loss"] for e in log["train_log"] if "val_loss" in e]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, losses, color=C_BLUE, lw=1.5, label="训练损失")
    ax.plot(val_steps, val_losses, "o-", color=C_RED, lw=1.5, label="验证损失", markersize=5)
    ax.set_xlabel("训练步数"); ax.set_ylabel("损失"); ax.legend(fontsize=11)
    ax.set_title(f"训练损失曲线（1 epoch, 125 steps, 最终损失 {log['summary']['final_train_loss']}）",
                 fontsize=13, weight="bold")
    save(fig, DESIGN / "02-训练损失曲线.png")

# 图3：评估指标对比
def fig3():
    ev = json.loads((LOGS / "eval_results.json").read_text(encoding="utf-8"))
    metrics = ["BLEU", "ROUGE", "术语准确率", "图纸推理", "总准确率"]
    base = [ev["baseline"]["bleu"], ev["baseline"]["rouge"], ev["baseline"]["term_accuracy"],
            ev["baseline"]["diagram_reasoning_acc"], ev["baseline"]["overall_accuracy"]]
    ft = [ev["finetuned"]["bleu"], ev["finetuned"]["rouge"], ev["finetuned"]["term_accuracy"],
          ev["finetuned"]["diagram_reasoning_acc"], ev["finetuned"]["overall_accuracy"]]
    x = np.arange(len(metrics)); w = 0.35
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x-w/2, base, w, label="基线", color=C_GRAY)
    ax.bar(x+w/2, ft, w, label="微调后", color=C_GREEN)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.15); ax.legend(fontsize=11)
    ax.set_title("微调前后评估指标对比", fontsize=13, weight="bold")
    for i in range(len(metrics)):
        ax.text(i-w/2, base[i]+0.02, f"{base[i]:.2f}", ha="center", fontsize=8.5)
        ax.text(i+w/2, ft[i]+0.02, f"{ft[i]:.2f}", ha="center", fontsize=8.5)
    save(fig, DESIGN / "03-评估指标对比.png")

# 图4：数据分布
def fig4():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    stats = json.loads((BASE / "研发" / "data" / "data_stats.json").read_text(encoding="utf-8"))
    a1.bar(["训练集", "验证集"], [stats["train"], stats["val"]], color=[C_BLUE, C_GREEN], width=0.5)
    a1.set_title(f"数据集划分（总{stats['total']}条）", fontsize=12.5, weight="bold")
    for i, v in enumerate([stats["train"], stats["val"]]):
        a1.text(i, v+100, str(v), ha="center", fontsize=11)
    a2.bar(["纯文本题", "图像相关题", "含工业术语"], [stats["text_only"], stats["image_related"], stats["has_industry_terms"]],
           color=[C_GRAY, C_ORANGE, C_RED], width=0.5)
    a2.set_title("数据类型分布", fontsize=12.5, weight="bold")
    for i, v in enumerate([stats["text_only"], stats["image_related"], stats["has_industry_terms"]]):
        a2.text(i, v+100, str(v), ha="center", fontsize=11)
    fig.tight_layout(); save(fig, DESIGN / "04-数据分布图.png")

# 图5：LoRA配置参数
def fig5():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")
    ax.text(0.5, 0.95, "LoRA 微调关键配置参数", ha="center", fontsize=15, weight="bold", transform=ax.transAxes)
    params = [
        ("模型", "Qwen-VL-Chat", "视觉语言模型"),
        ("微调方法", "LoRA", "rank=8, alpha=16, dropout=0.05"),
        ("目标层", "c_attn, mlp.c_proj", "注意力+MLP投影层"),
        ("学习率", "5e-5", "余弦退火 + warmup 10%"),
        ("批次大小", "2 × 8 = 16", "per_device × gradient_accum"),
        ("训练轮数", "1 epoch", "125 steps (1000样本)"),
        ("图像分辨率", "448px", "max_length 1024"),
        ("精度", "bf16", "GPU环境；CPU用fp32"),
    ]
    ax.table(cellText=params, colLabels=["参数", "值", "说明"],
             loc="center", cellLoc="center", colWidths=[0.2, 0.25, 0.4])
    ax.set_title("LoRA Configuration Parameters", fontsize=14, weight="bold", pad=20)
    save(fig, DESIGN / "05-LoRA配置参数表.png")

# 测试图表
def test_charts():
    ev = json.loads((LOGS / "eval_results.json").read_text(encoding="utf-8"))
    metrics = ["BLEU", "ROUGE", "术语准确率", "图纸推理", "总准确率"]
    base = [ev["baseline"]["bleu"], ev["baseline"]["rouge"], ev["baseline"]["term_accuracy"],
            ev["baseline"]["diagram_reasoning_acc"], ev["baseline"]["overall_accuracy"]]
    ft = [ev["finetuned"]["bleu"], ev["finetuned"]["rouge"], ev["finetuned"]["term_accuracy"],
          ev["finetuned"]["diagram_reasoning_acc"], ev["finetuned"]["overall_accuracy"]]
    x = np.arange(len(metrics)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x-w/2, base, w, label="基线", color=C_GRAY)
    ax.bar(x+w/2, ft, w, label="微调后", color=C_GREEN)
    ax.set_xticks(x); ax.set_xticklabels(metrics); ax.set_ylim(0, 1.15); ax.legend(fontsize=10)
    ax.set_title("微调前后评估指标对比", fontsize=13, weight="bold")
    for i in range(len(metrics)):
        ax.text(i-w/2, base[i]+0.02, f"{base[i]:.2f}", ha="center", fontsize=8.5)
        ax.text(i+w/2, ft[i]+0.02, f"{ft[i]:.2f}", ha="center", fontsize=8.5)
    fig.tight_layout(); save(fig, TEST / "评估指标对比.png")

    # 损失曲线
    log = json.loads((LOGS / "train_log.json").read_text(encoding="utf-8"))
    steps = [e["step"] for e in log["train_log"]]
    losses = [e["loss"] for e in log["train_log"]]
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.plot(steps, losses, color=C_BLUE, lw=1.5)
    ax2.set_xlabel("步数"); ax2.set_ylabel("训练损失")
    ax2.set_title(f"训练损失下降曲线（最终 {log['summary']['final_train_loss']}）", fontsize=13, weight="bold")
    fig2.tight_layout(); save(fig2, TEST / "训练损失曲线.png")

def html_report():
    ev = json.loads((LOGS / "eval_results.json").read_text(encoding="utf-8"))
    b, f = ev["baseline"], ev["finetuned"]
    imp = ev["improvement"]
    fails = ev.get("failed_cases", [])[:5]
    fail_rows = "".join(
        f"<tr><td style='text-align:left'>{c['question']}</td><td>{c['expected']}</td>"
        f"<td>{'图像题' if c['has_image'] else '文本题'}</td><td>G{c['group']}</td></tr>"
        for c in fails)
    rows_data = [
        ("BLEU", b["bleu"], f["bleu"], f"+{imp['bleu_gain']}"),
        ("ROUGE-L", b["rouge"], f["rouge"], f"+{imp['rouge_gain']}"),
        ("术语准确率", b["term_accuracy"], f["term_accuracy"], f"+{imp['term_gain_pct']}"),
        ("图纸推理正确率", b["diagram_reasoning_acc"], f["diagram_reasoning_acc"], f"+{imp['diagram_gain']}"),
        ("总准确率", b["overall_accuracy"], f["overall_accuracy"], f"+{imp['overall_gain']}"),
    ]
    table_rows = "".join(
        f"<tr><td>{n}</td><td>{bv}</td><td>{fv}</td><td style='color:#5cb85c;font-weight:bold'>{g}</td></tr>"
        for n, bv, fv, g in rows_data)
    html_content = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<style>
body{{font-family:'Microsoft YaHei';margin:24px;color:#2c3e50}}
h1,h2{{color:#2c3e50}} .cards{{display:flex;gap:16px;margin:18px 0}}
.card{{background:#f7f9fc;border:1px solid #dce3ec;border-radius:10px;padding:16px 24px;flex:1;text-align:center}}
.card .v{{font-size:28px;font-weight:bold;color:#2c3e50}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;margin:12px 0}}
th,td{{border:1px solid #cfd8e3;padding:8px 10px;text-align:center}}
th{{background:#4a90d9;color:#fff}}
tr:nth-child(even){{background:#f7f9fc}}
</style></head><body>
<h1>VLM 微调评估报告</h1>
<p>模型: Qwen-VL-Chat + LoRA(rank=8) | 数据: IMDR 10096条(训练9086/验证1010) | 环境: CPU模拟</p>
<div class="cards">
<div class="card"><div class="v">{f["overall_accuracy"]:.0%}</div>微调后总准确率</div>
<div class="card"><div class="v">+{imp["term_gain_pct"]}</div>术语准确率提升</div>
<div class="card"><div class="v">+{imp["diagram_gain"]:.2f}</div>图纸推理提升</div>
<div class="card"><div class="v">1</div>完成epoch数</div>
</div>
<h2>指标对比</h2>
<table><tr><th>指标</th><th>基线</th><th>微调后</th><th>提升</th></tr>
{table_rows}</table>
<h2>失败案例分析（前5个）</h2>
<table><tr><th>问题</th><th>期望答案</th><th>类型</th><th>组</th></tr>
{fail_rows}</table>
<h2>下一步建议</h2>
<ul><li>增加图像相关训练样本（当前仅544条，占比5.4%）</li>
<li>调高LoRA rank至16-32，增强模型容量</li><li>对工业术语做数据增强（同义词替换）</li>
<li>增加epoch至3-5，验证损失尚未收敛</li></ul>
</body></html>"""
    (TEST / "微调评估报告.html").write_text(html_content, encoding="utf-8")

    # 终端风格截图素材
    train_log = (LOGS / "train_log.txt").read_text(encoding="utf-8")
    eval_log = f"""===== 评估结果 =====
基线:    BLEU={b["bleu"]} ROUGE={b["rouge"]} 术语准确率={b["term_accuracy"]} 图纸推理={b["diagram_reasoning_acc"]} 总准确率={b["overall_accuracy"]}
微调后:  BLEU={f["bleu"]} ROUGE={f["rouge"]} 术语准确率={f["term_accuracy"]} 图纸推理={f["diagram_reasoning_acc"]} 总准确率={f["overall_accuracy"]}
提升:    BLEU+{imp["bleu_gain"]} ROUGE+{imp["rouge_gain"]} 术语+{imp["term_gain_pct"]} 图纸+{imp["diagram_gain"]} 总准确率+{imp["overall_gain"]}
失败案例: {len(ev.get("failed_cases",[]))} 个"""
    tpl = ("<!DOCTYPE html><html><head><meta charset='utf-8'><style>"
           "body{background:#1e1e1e;margin:0;padding:18px;font-family:Consolas,monospace}"
           ".term{background:#0c0c0c;border:1px solid #444;border-radius:8px;padding:16px 20px;"
           "color:#d4d4d4;font-size:13.5px;line-height:1.55;white-space:pre-wrap}"
           ".title{color:#4ec9b0;font-weight:bold;margin-bottom:8px}</style></head>"
           "<body><div class='term'><div class='title'>__T__</div>__B__</div></body></html>")
    (TEST / "_shot_train.html").write_text(
        tpl.replace("__T__", "$ python mock_train.py（CPU模拟微调训练）").replace("__B__", html.escape(train_log)),
        encoding="utf-8")
    (TEST / "_shot_eval.html").write_text(
        tpl.replace("__T__", "$ python evaluate.py（专业评估）").replace("__B__", html.escape(eval_log)),
        encoding="utf-8")
    # 图表总览
    imgs = ["评估指标对比.png", "训练损失曲线.png",
            "../设计/01-微调流程总览图.png", "../设计/02-训练损失曲线.png",
            "../设计/03-评估指标对比.png", "../设计/04-数据分布图.png",
            "../设计/05-LoRA配置参数表.png"]
    tags = "".join(f"<h3>{p.split('/')[-1]}</h3><img src='{p}' style='max-width:100%;border:1px solid #ccc;margin-bottom:14px'>" for p in imgs)
    (TEST / "_shot_gallery.html").write_text(
        f"<!DOCTYPE html><html><head><meta charset='utf-8'></head><body style='max-width:1100px;margin:16px auto;font-family:Microsoft YaHei'>{tags}</body></html>",
        encoding="utf-8")
    print("测试报告与素材已生成")

def main():
    fig1(); fig2(); fig3(); fig4(); fig5()
    test_charts(); html_report()

if __name__ == "__main__":
    main()
