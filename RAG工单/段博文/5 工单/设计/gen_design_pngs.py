# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Query理解优化任务
"""
设计图生成脚本：生成 5 张 PNG（技术组件、技术架构、思维导图、接口文档、流程图）。
使用 matplotlib 绘制，确保中文正常显示。
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"saved: {path}")


# ==================== 1. 技术组件图 ====================
def gen_tech_components():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('RAG-PDF 问答系统（Query理解优化版）- 技术组件', fontsize=18, fontweight='bold', pad=20)

    components = [
        (1.5, 6.5, 3.0, 1.2, '前端界面\n(Web UI)', '#4C78A8'),
        (5.5, 6.5, 3.0, 1.2, 'FastAPI 服务', '#F58518'),
        (9.5, 6.5, 3.0, 1.2, '会话管理\nSessionManager', '#54A24B'),
        (1.5, 4.5, 3.0, 1.2, 'Query 改写器\nQueryRewriter', '#E45756'),
        (5.5, 4.5, 3.0, 1.2, '混合检索\nHybrid Retrieval', '#72B7B2'),
        (9.5, 4.5, 3.0, 1.2, '重排模型\nbge-reranker', '#EECA3B'),
        (1.5, 2.5, 3.0, 1.2, '向量数据库\nMilvus', '#FF9DA6'),
        (5.5, 2.5, 3.0, 1.2, '嵌入模型\nbge-m3', '#9D755D'),
        (9.5, 2.5, 3.0, 1.2, 'LLM 生成\nDeepSeek API', '#B279A2'),
        (5.5, 0.5, 3.0, 1.2, 'PDF 解析器\n(文本/表格/图像)', '#59A14F'),
    ]

    for x, y, w, h, label, color in components:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                             facecolor=color, edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=11, color='white', fontweight='bold', linespacing=1.4)

    # 箭头
    arrows = [
        ((3.0, 7.1), (5.5, 7.1)),
        ((7.0, 6.5), (7.0, 5.7)),
        ((4.5, 5.1), (5.5, 5.1)),
        ((7.0, 4.5), (7.0, 3.7)),
        ((4.5, 3.1), (5.5, 3.1)),
        ((8.5, 3.1), (9.5, 3.1)),
        ((7.0, 2.5), (7.0, 1.7)),
        ((10.0, 4.5), (10.0, 3.7)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))

    save_fig(fig, '01-技术组件图.png')


# ==================== 2. 技术架构图 ====================
def gen_architecture():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('RAG-PDF 问答系统（Query理解优化版）- 技术架构', fontsize=18, fontweight='bold', pad=20)

    # 分层
    layers = [
        (1, 8.5, 14, 1.2, '接入层', '#E8F4F8'),
        (1, 6.5, 14, 1.5, '应用层', '#D1E7DD'),
        (1, 4.0, 14, 2.0, '服务层', '#FFF3CD'),
        (1, 1.5, 14, 2.0, '数据层', '#F8D7DA'),
    ]
    for x, y, w, h, label, color in layers:
        rect = mpatches.Rectangle((x, y), w, h, facecolor=color, edgecolor='#999', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.2, y + h - 0.3, label, fontsize=12, fontweight='bold', color='#333')

    # 接入层内容
    ax.text(8, 9.1, 'Web 前端 (HTML/JS)  |  FastAPI REST API  |  SSE 流式响应', ha='center', fontsize=11)

    # 应用层内容
    app_items = [
        (2.5, 7.2, '会话管理\nSession Manager', '#54A24B'),
        (6.0, 7.2, 'Query 改写\nQuery Rewriter', '#E45756'),
        (9.5, 7.2, '缓存管理\nCache', '#72B7B2'),
        (13.0, 7.2, '健康检查\nHealth', '#F58518'),
    ]
    for x, y, label, color in app_items:
        box = FancyBboxPatch((x, y), 2.2, 0.9, boxstyle="round,pad=0.03", facecolor=color, edgecolor='white')
        ax.add_patch(box)
        ax.text(x + 1.1, y + 0.45, label, ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    # 服务层内容
    svc_items = [
        (2.5, 5.2, '混合检索\nVector + BM25', '#4C78A8'),
        (5.5, 5.2, '重排\nbge-reranker', '#EECA3B'),
        (8.5, 5.2, 'LLM 生成\nDeepSeek', '#B279A2'),
        (11.5, 5.2, 'PDF 解析\n文本/表格/图像', '#59A14F'),
    ]
    for x, y, label, color in svc_items:
        box = FancyBboxPatch((x, y), 2.2, 0.9, boxstyle="round,pad=0.03", facecolor=color, edgecolor='white')
        ax.add_patch(box)
        ax.text(x + 1.1, y + 0.45, label, ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    svc_items2 = [
        (2.5, 4.2, '指代消解\nCoreference', '#E45756'),
        (5.5, 4.2, '主语继承\nSubject Inherit', '#9D755D'),
        (8.5, 4.2, '主语切换\nSubject Switch', '#FF9DA6'),
        (11.5, 4.2, 'Query 扩展\nExpansion', '#72B7B2'),
    ]
    for x, y, label, color in svc_items2:
        box = FancyBboxPatch((x, y), 2.2, 0.9, boxstyle="round,pad=0.03", facecolor=color, edgecolor='white')
        ax.add_patch(box)
        ax.text(x + 1.1, y + 0.45, label, ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    # 数据层内容
    data_items = [
        (2.5, 2.5, 'Milvus\n向量库', '#4C78A8'),
        (5.5, 2.5, 'bge-m3\n嵌入模型', '#9D755D'),
        (8.5, 2.5, 'bge-reranker\n重排模型', '#EECA3B'),
        (11.5, 2.5, 'DeepSeek API\n大模型', '#B279A2'),
    ]
    for x, y, label, color in data_items:
        box = FancyBboxPatch((x, y), 2.2, 0.9, boxstyle="round,pad=0.03", facecolor=color, edgecolor='white')
        ax.add_patch(box)
        ax.text(x + 1.1, y + 0.45, label, ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    save_fig(fig, '02-技术架构图.png')


# ==================== 3. 思维导图 ====================
def gen_mindmap():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Query 理解优化任务 - 思维导图', fontsize=18, fontweight='bold', pad=20)

    # 中心节点
    center = FancyBboxPatch((6.5, 4.5), 3, 1, boxstyle="round,pad=0.1", facecolor='#4C78A8', edgecolor='white', linewidth=2)
    ax.add_patch(center)
    ax.text(8, 5, 'Query 理解优化\n多轮对话', ha='center', va='center', fontsize=13, color='white', fontweight='bold')

    # 一级分支
    branches = [
        (1.5, 7.5, '会话管理', '#54A24B', [
            'Session 存储',
            '历史窗口',
            '实体追踪',
            'LRU 过期',
        ]),
        (1.5, 2.5, 'Query 改写', '#E45756', [
            '指代消解',
            '主语继承',
            '主语切换',
            'LLM 改写',
        ]),
        (8.5, 7.5, '检索增强', '#F58518', [
            '改写后检索',
            '混合检索',
            '重排优化',
            '缓存加速',
        ]),
        (8.5, 2.5, '前端交互', '#B279A2', [
            '多轮展示',
            '改写提示',
            '会话徽章',
            '新会话',
        ]),
    ]

    for bx, by, blabel, bcolor, leaves in branches:
        # 分支节点
        box = FancyBboxPatch((bx, by), 2.5, 0.8, boxstyle="round,pad=0.05", facecolor=bcolor, edgecolor='white')
        ax.add_patch(box)
        ax.text(bx + 1.25, by + 0.4, blabel, ha='center', va='center', fontsize=11, color='white', fontweight='bold')
        # 连线到中心
        ax.annotate('', xy=(6.5 if bx < 5 else 9.5, 5), xytext=(bx + (2.5 if bx < 5 else 0), by + 0.4),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
        # 叶子
        for i, leaf in enumerate(leaves):
            ly = by - 0.8 - i * 0.5
            ax.text(bx + 0.2, ly, f'• {leaf}', fontsize=9, color='#333')

    save_fig(fig, '03-思维导图.png')


# ==================== 4. 接口文档图 ====================
def gen_api_doc():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('RAG-PDF 问答系统 - API 接口文档', fontsize=18, fontweight='bold', pad=20)

    apis = [
        ('POST', '/api/upload', '上传 PDF 文件', 'file: PDF', '{filename, saved_path, size}'),
        ('POST', '/api/ingest', '解析入库', '{files?: [name]}', '{processed, ingested, collection_count}'),
        ('POST', '/api/chat', '问答（支持多轮）', '{query, session_id?, use_rewrite?, stream?}', '{answer, rewritten_query, references}'),
        ('GET', '/api/search', '纯检索', '?query=&top_k=', '{results: [...], total}'),
        ('GET', '/api/health', '健康检查', '', '{status, collection_count, features}'),
        ('GET', '/api/session/{id}', '获取会话历史', '', '{session_id, history, last_entity}'),
        ('DELETE', '/api/session/{id}', '清空会话', '', '{session_id, reset}'),
    ]

    y = 8.5
    for method, path, desc, req, resp in apis:
        color = '#4C78A8' if method == 'GET' else ('#54A24B' if method == 'POST' else '#E45756')
        # 方法标签
        box = FancyBboxPatch((0.5, y), 1.2, 0.5, boxstyle="round,pad=0.03", facecolor=color, edgecolor='white')
        ax.add_patch(box)
        ax.text(1.1, y + 0.25, method, ha='center', va='center', fontsize=10, color='white', fontweight='bold')
        # 路径
        ax.text(2.0, y + 0.25, path, fontsize=11, fontweight='bold', va='center')
        # 描述
        ax.text(5.5, y + 0.25, desc, fontsize=10, va='center', color='#333')
        # 请求/响应
        ax.text(8.5, y + 0.25, f'请求: {req}', fontsize=8, va='center', color='#666')
        ax.text(11.5, y + 0.25, f'响应: {resp}', fontsize=8, va='center', color='#666')
        y -= 1.2

    save_fig(fig, '04-接口文档.png')


# ==================== 5. 流程图 ====================
def gen_flowchart():
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_title('Query 理解优化 - 多轮对话处理流程', fontsize=18, fontweight='bold', pad=20)

    steps = [
        (6, 12.5, '用户提问\n(可能含指代/省略)', '#4C78A8'),
        (6, 11.0, '加载会话历史\nSession History', '#54A24B'),
        (6, 9.5, 'Query 改写\nQuery Rewriter', '#E45756'),
        (3.5, 8.0, '规则改写\n(主语切换)', '#F58518'),
        (8.5, 8.0, 'LLM 改写\n(指代消解)', '#B279A2'),
        (6, 6.5, '改写后问题\nStandalone Query', '#72B7B2'),
        (6, 5.0, '混合检索\nVector + BM25', '#9D755D'),
        (6, 3.5, '重排 + 过滤\nRerank', '#EECA3B'),
        (6, 2.0, 'LLM 生成答案\nDeepSeek', '#FF9DA6'),
        (6, 0.5, '写回会话历史\n返回响应', '#4C78A8'),
    ]

    for x, y, label, color in steps:
        w = 3.0 if x == 6 else 2.5
        box = FancyBboxPatch((x - w/2, y - 0.4), w, 0.8, boxstyle="round,pad=0.05", facecolor=color, edgecolor='white')
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    # 主流程箭头
    main_flow = [12.5, 11.0, 9.5, 6.5, 5.0, 3.5, 2.0, 0.5]
    for i in range(len(main_flow) - 1):
        y1, y2 = main_flow[i], main_flow[i + 1]
        ax.annotate('', xy=(6, y2 + 0.4), xytext=(6, y1 - 0.4),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=2))

    # 分支箭头
    ax.annotate('', xy=(3.5, 8.4), xytext=(5.0, 9.1), arrowprops=dict(arrowstyle='->', color='#F58518', lw=1.5))
    ax.annotate('', xy=(8.5, 8.4), xytext=(7.0, 9.1), arrowprops=dict(arrowstyle='->', color='#B279A2', lw=1.5))
    ax.annotate('', xy=(5.0, 6.9), xytext=(3.5, 7.6), arrowprops=dict(arrowstyle='->', color='#F58518', lw=1.5))
    ax.annotate('', xy=(7.0, 6.9), xytext=(8.5, 7.6), arrowprops=dict(arrowstyle='->', color='#B279A2', lw=1.5))

    save_fig(fig, '05-流程图.png')


if __name__ == '__main__':
    gen_tech_components()
    gen_architecture()
    gen_mindmap()
    gen_api_doc()
    gen_flowchart()
    print('All design PNGs generated.')
