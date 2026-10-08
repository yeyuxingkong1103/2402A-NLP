# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-Embedding模型微调任务
"""生成5张设计PNG：微调技术路线图、数据生成流程图、训练架构图、评估指标体系图、功能思维导图。"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = r"D:\作业\6-专高NLP 作业\成品\11\设计"
os.makedirs(OUT, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def box(ax, x, y, w, h, text, color, fs=9, tc='#333'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                facecolor=color, edgecolor='#666', linewidth=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc)


def arrow(ax, x1, y1, x2, y2, color='#555'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))


def create_roadmap():
    """微调技术路线图"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis('off')
    ax.set_title('Embedding 模型微调技术路线图', fontsize=16, fontweight='bold', pad=20)

    stages = [
        ("①数据准备\n9份金融年报", 0.5, '#E8F4FD'),
        ("②段落采样\n360段", 3.2, '#D1ECF1'),
        ("③LLM生成QA\n问答对", 5.9, '#FFF3CD'),
        ("④负例挖掘\nBM25困难负例", 8.6, '#F8D7DA'),
        ("⑤模型微调\nMNRL+套娃损失", 11.3, '#D4EDDA'),
    ]
    for text, x, color in stages:
        box(ax, x, 4.5, 2.3, 1.3, text, color, fs=10)
    for x in [2.8, 5.5, 8.2, 10.9]:
        arrow(ax, x, 5.15, x + 0.4, 5.15)

    # 下方评估闭环
    box(ax, 2, 1.5, 3, 1.1, "微调前评估\nRecall/MRR基线", '#E2E3F3')
    box(ax, 5.5, 1.5, 3, 1.1, "每epoch回调评估\n检索指标监控", '#E2E3F3')
    box(ax, 9, 1.5, 3, 1.1, "微调后评估\n对比验证提升", '#E2E3F3')
    arrow(ax, 3.5, 2.6, 4, 4.5)
    arrow(ax, 7, 2.6, 7, 4.5)
    arrow(ax, 10.5, 2.6, 10.5, 4.5)

    plt.tight_layout()
    plt.savefig(f"{OUT}\\01-微调技术路线图.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("01-微调技术路线图.png")


def create_data_flow():
    """数据生成流程图"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis('off')
    ax.set_title('微调数据集生成流程图', fontsize=16, fontweight='bold', pad=20)

    # 主流程
    box(ax, 1, 6, 2.6, 1.2, "年报JSONL\n(txt目录9份)", '#E8F4FD')
    box(ax, 4.5, 6, 2.6, 1.2, "段落提取\n长度/中文比例过滤", '#D1ECF1')
    box(ax, 8, 6, 2.6, 1.2, "均匀采样\n每份40段=360", '#FFF3CD')
    box(ax, 11.5, 6, 2.2, 1.2, "passages\n.jsonl", '#FFF')
    for x in [3.6, 7.1, 10.6]:
        arrow(ax, x, 6.6, x + 0.5, 6.6)

    # QA 生成
    box(ax, 4.5, 3.5, 2.6, 1.2, "DeepSeek API\n20线程并发生成", '#F8D7DA')
    box(ax, 8, 3.5, 2.6, 1.2, "问答对\nquestion+answer", '#D4EDDA')
    arrow(ax, 12.6, 6, 7.1, 4.7)

    # 划分
    box(ax, 1, 1, 3, 1.1, "train ~300条\n(query,正例,困难负例)", '#E2E3F3')
    box(ax, 5.5, 1, 3, 1.1, "eval 60条\n(query,正例)", '#E2E3F3')
    box(ax, 10, 1, 3, 1.1, "corpus 360段\n评估候选语料", '#E2E3F3')
    arrow(ax, 5.8, 3.5, 3, 2.1)
    arrow(ax, 7, 3.5, 7, 2.1)
    arrow(ax, 8.5, 3.5, 11, 2.1)

    plt.tight_layout()
    plt.savefig(f"{OUT}\\02-数据生成流程图.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("02-数据生成流程图.png")


def create_train_arch():
    """训练架构与损失函数图"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis('off')
    ax.set_title('微调训练架构与损失函数', fontsize=16, fontweight='bold', pad=20)

    # 输入三元组
    box(ax, 0.5, 5.5, 2.4, 1.3, "锚点 query\n用户问题", '#E8F4FD')
    box(ax, 0.5, 3.5, 2.4, 1.3, "正例 positive\n对应年报段落", '#D4EDDA')
    box(ax, 0.5, 1.5, 2.4, 1.3, "困难负例\nBM25高分非正例", '#F8D7DA')

    # 模型
    box(ax, 4.5, 3, 3, 2, "bge-base-en-v1.5\n12层Transformer\nmax_seq=256", '#FFF3CD', fs=10)
    arrow(ax, 2.9, 6.15, 4.5, 4.5)
    arrow(ax, 2.9, 4.15, 4.5, 4)
    arrow(ax, 2.9, 2.15, 4.5, 3.5)

    # 损失
    box(ax, 9, 4.5, 4.2, 1.2, "多重负例排序损失 MNRL\n+in-batch negatives", '#E2E3F3')
    box(ax, 9, 2.5, 4.2, 1.2, "套娃损失 Matryoshka\n[768,512,256,128,64]", '#E2E3F3')
    arrow(ax, 7.5, 4.5, 9, 5)
    arrow(ax, 11.1, 4.5, 11.1, 3.7)

    # 训练参数标注
    box(ax, 4.5, 0.5, 8.7, 1,
        "训练参数：epochs=2 | batch=16 | lr=2e-5 | warmup=10% | 优化器 AdamW | 设备 CPU",
        '#343A40', fs=10, tc='white')

    plt.tight_layout()
    plt.savefig(f"{OUT}\\03-训练架构与损失函数图.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("03-训练架构与损失函数图.png")


def create_metrics():
    """评估指标体系图"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis('off')
    ax.set_title('微调评估指标体系', fontsize=16, fontweight='bold', pad=20)

    box(ax, 5.5, 6, 3, 1, "Embedding 检索评估", '#343A40', fs=12, tc='white')

    groups = [
        ("召回类", ["Recall@1", "Recall@3", "Recall@5", "Recall@10"], 1, '#D4EDDA'),
        ("排序类", ["MRR@10", "倒数排名均值"], 6, '#FFF3CD'),
        ("准确率类", ["Top1准确率", "命中率"], 11, '#F8D7DA'),
    ]
    for title, items, x, color in groups:
        box(ax, x - 1, 4.2, 2.8, 0.8, title, color, fs=11)
        for i, it in enumerate(items):
            box(ax, x - 1, 3 - i * 0.7, 2.8, 0.55, it, '#F0F0F0', fs=9)
        arrow(ax, 7, 6, x + 0.4, 5)

    # 评估方式说明
    box(ax, 1.5, 0.3, 11, 0.9,
        "评估方式：60 条金融查询 × 360 段候选语料 → 向量编码 → 余弦相似度排序 → 统计正例排名",
        '#E2E3F3', fs=10)

    plt.tight_layout()
    plt.savefig(f"{OUT}\\04-评估指标体系图.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("04-评估指标体系图.png")


def create_mindmap():
    """功能思维导图"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis('off')
    ax.set_title('Embedding 微调任务思维导图', fontsize=16, fontweight='bold', pad=20)

    box(ax, 5.5, 4.2, 3, 1, "Embedding微调", '#343A40', fs=13, tc='white')

    branches = [
        ("数据生成", ["年报解析", "段落采样", "LLM问答对", "训练/评估划分"], 2, 7.5, '#E8F4FD'),
        ("损失函数", ["三元组损失", "对比损失", "MNRL", "套娃损失"], 7, 7.8, '#D4EDDA'),
        ("训练参数", ["epochs=2", "batch=16", "lr=2e-5", "warmup"], 12, 7.5, '#FFF3CD'),
        ("评估器", ["Recall@K", "MRR@10", "Top1准确率", "epoch回调"], 2, 2, '#F8D7DA'),
        ("前后对比", ["基线评估", "微调后评估", "提升量化", "报告输出"], 7, 1.2, '#E2E3F3'),
        ("产出物", ["微调模型", "数据集", "训练过程", "过程记录"], 12, 2, '#D1ECF1'),
    ]
    for label, items, x, y, color in branches:
        box(ax, x - 1.1, y, 2.2, 0.7, label, color, fs=10)
        for i, it in enumerate(items):
            iy = y - 0.55 - i * 0.42
            box(ax, x - 0.95, iy, 1.9, 0.32, it, 'white', fs=8)
        ax.plot([7, x], [5, y + 0.35], '--', color='#999', lw=1.2)

    plt.tight_layout()
    plt.savefig(f"{OUT}\\05-功能思维导图.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("05-功能思维导图.png")


if __name__ == "__main__":
    create_roadmap()
    create_data_flow()
    create_train_arch()
    create_metrics()
    create_mindmap()
    print(f"\n全部生成完成！目录: {OUT}")
