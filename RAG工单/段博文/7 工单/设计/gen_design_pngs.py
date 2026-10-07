# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估
"""生成5张设计PNG：技术架构图、混合检索流程图、功能思维导图、接口文档、技术组件。"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = r"D:\作业\6-专高NLP 作业\成品\7\设计"
os.makedirs(OUT, exist_ok=True)

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_tech_architecture():
    """技术架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('RAG 系统技术架构图（功能测试及评估）', fontsize=16, fontweight='bold', pad=20)

    layers = [
        ("用户层", ["Web 前端", "API 客户端", "多轮对话"], 8.5, '#E8F4FD'),
        ("应用层", ["FastAPI", "SSE 流式", "会话管理", "Query 改写"], 6.5, '#FFF3CD'),
        ("检索层", ["向量检索", "BM25 全文", "RRF 融合", "CrossEncoder 重排"], 4.5, '#D4EDDA'),
        ("数据层", ["Milvus 向量库", "BM25 倒排索引", "LRU 缓存"], 2.5, '#F8D7DA'),
        ("模型层", ["bge-m3 向量", "bge-reranker", "DeepSeek LLM", "RapidOCR"], 0.5, '#E2E3F3'),
    ]

    for label, items, y, color in layers:
        # 层标签
        ax.add_patch(FancyBboxPatch((0.2, y), 1.5, 1.2, boxstyle="round,pad=0.05",
                                     facecolor='#343A40', edgecolor='none'))
        ax.text(0.95, y + 0.6, label, ha='center', va='center', fontsize=11,
                fontweight='bold', color='white')

        # 层内容
        x_start = 2.2
        for i, item in enumerate(items):
            w = 2.2
            x = x_start + i * (w + 0.3)
            ax.add_patch(FancyBboxPatch((x, y), w, 1.2, boxstyle="round,pad=0.05",
                                         facecolor=color, edgecolor='#666', linewidth=1.5))
            ax.text(x + w/2, y + 0.6, item, ha='center', va='center', fontsize=9)

    # 箭头
    for y in [7.3, 5.3, 3.3, 1.3]:
        ax.annotate('', xy=(7, y), xytext=(7, y + 0.5),
                   arrowprops=dict(arrowstyle='->', color='#666', lw=2))

    plt.tight_layout()
    plt.savefig(f"{OUT}\\01-技术架构图.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("01-技术架构图.png")


def create_retrieval_flow():
    """混合检索流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('混合检索流程图', fontsize=16, fontweight='bold', pad=20)

    steps = [
        ("用户查询", 1, 6, '#E8F4FD'),
        ("Query 改写\n(指代消解/主语继承)", 3.5, 6, '#FFF3CD'),
        ("向量检索\n(bge-m3)", 6, 7.2, '#D4EDDA'),
        ("BM25 检索\n(jieba 分词)", 6, 4.8, '#D4EDDA'),
        ("RRF 融合\n(倒数排名融合)", 8.5, 6, '#F8D7DA'),
        ("CrossEncoder\n重排", 11, 6, '#E2E3F3'),
        ("Top-K 结果", 13, 6, '#D1ECF1'),
    ]

    for text, x, y, color in steps:
        ax.add_patch(FancyBboxPatch((x-0.9, y-0.5), 1.8, 1, boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='#666', linewidth=1.5))
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

    # 箭头
    arrows = [
        ((1.9, 6), (2.6, 6)),
        ((4.4, 6), (5.1, 6.8)),
        ((4.4, 6), (5.1, 5.2)),
        ((6.9, 7.2), (7.6, 6.3)),
        ((6.9, 4.8), (7.6, 5.7)),
        ((9.4, 6), (10.1, 6)),
        ((11.9, 6), (12.1, 6)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='#666', lw=2))

    plt.tight_layout()
    plt.savefig(f"{OUT}\\02-混合检索流程图.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("02-混合检索流程图.png")


def create_mindmap():
    """功能思维导图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('RAG 系统功能思维导图', fontsize=16, fontweight='bold', pad=20)

    # 中心节点
    ax.add_patch(FancyBboxPatch((5.5, 4.5), 3, 1, boxstyle="round,pad=0.1",
                                 facecolor='#343A40', edgecolor='none'))
    ax.text(7, 5, 'RAG 问答系统', ha='center', va='center', fontsize=14,
            fontweight='bold', color='white')

    branches = [
        ("文档解析", ["PDF 文本提取", "表格 Markdown", "图像 OCR", "语义生成"], 2, 8, '#E8F4FD'),
        ("检索策略", ["向量检索", "BM25 全文", "混合检索", "布尔查询", "短语匹配", "模糊查询"], 7, 8.5, '#D4EDDA'),
        ("重排算法", ["CrossEncoder", "TF-IDF", "LLM 重排"], 12, 8, '#F8D7DA'),
        ("融合算法", ["RRF", "加权融合", "投票融合"], 1.5, 3, '#FFF3CD'),
        ("会话管理", ["多轮对话", "Query 改写", "指代消解", "主语继承"], 7, 1.5, '#E2E3F3'),
        ("评估指标", ["准确率", "召回率", "RAGAS 指标", "响应时间"], 12, 3, '#D1ECF1'),
    ]

    for label, items, x, y, color in branches:
        # 分支节点
        ax.add_patch(FancyBboxPatch((x-1.2, y-0.4), 2.4, 0.8, boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='#666', linewidth=1.5))
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')

        # 子项
        for i, item in enumerate(items):
            item_y = y - 0.8 - i * 0.5
            ax.add_patch(FancyBboxPatch((x-1, item_y-0.15), 2, 0.3, boxstyle="round,pad=0.02",
                                         facecolor='white', edgecolor='#999', linewidth=1))
            ax.text(x, item_y, item, ha='center', va='center', fontsize=8)

        # 连线
        ax.annotate('', xy=(x, y + 0.4), xytext=(7, 5.5),
                   arrowprops=dict(arrowstyle='-', color='#999', lw=1.5, linestyle='--'))

    plt.tight_layout()
    plt.savefig(f"{OUT}\\03-功能思维导图.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("03-功能思维导图.png")


def create_api_doc():
    """接口文档图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('API 接口文档', fontsize=16, fontweight='bold', pad=20)

    apis = [
        ("POST /api/upload", "上传 PDF 文件", "multipart/form-data", "filename, saved_path, size"),
        ("POST /api/ingest", "解析+切分+入库", "JSON: files[]", "processed, ingested, total_chunks"),
        ("POST /api/chat", "问答（检索+LLM）", "JSON: query, top_k, stream, session_id", "answer, sources, session_id"),
        ("GET /api/search", "纯检索", "query, top_k, strategy, fusion, reranker", "results[], total, elapsed"),
        ("GET /api/health", "健康检查", "-", "status, collection_count, cache"),
        ("GET /api/stats", "系统统计", "-", "collection_count, cache, llm_cache"),
        ("GET /api/retrieve_config", "获取检索配置", "-", "strategy, fusion, reranker, weights"),
        ("POST /api/retrieve_config", "更新检索配置", "JSON: strategy, fusion, reranker", "updated config"),
        ("GET /api/session/{sid}", "获取会话历史", "session_id", "history, last_entity"),
        ("DELETE /api/session/{sid}", "清空会话", "session_id", "reset: true/false"),
    ]

    y = 9
    for method_path, desc, req, resp in apis:
        # 方法标签
        color = '#28A745' if 'GET' in method_path else '#007BFF' if 'POST' in method_path else '#DC3545'
        ax.add_patch(FancyBboxPatch((0.3, y-0.35), 2.5, 0.7, boxstyle="round,pad=0.03",
                                     facecolor=color, edgecolor='none'))
        ax.text(1.55, y, method_path, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white')

        # 描述
        ax.text(3.2, y + 0.15, desc, fontsize=9, fontweight='bold', va='center')
        ax.text(3.2, y - 0.2, f"请求: {req[:40]}...", fontsize=7, va='center', color='#666')
        ax.text(8, y - 0.2, f"响应: {resp[:40]}...", fontsize=7, va='center', color='#666')

        y -= 0.95

    plt.tight_layout()
    plt.savefig(f"{OUT}\\04-接口文档.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("04-接口文档.png")


def create_tech_components():
    """技术组件图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('技术组件清单', fontsize=16, fontweight='bold', pad=20)

    components = [
        ("FastAPI", "Web 框架", "0.110+", '#007BFF'),
        ("Uvicorn", "ASGI 服务器", "0.29+", '#28A745'),
        ("Milvus", "向量数据库", "2.4+", '#DC3545'),
        ("bge-m3", "向量模型", "BAAI", '#6F42C1'),
        ("bge-reranker", "重排模型", "BAAI", '#E83E8C'),
        ("DeepSeek", "LLM", "deepseek-chat", '#FD7E14'),
        ("BM25", "全文检索", "rank_bm25", '#20C997'),
        ("jieba", "中文分词", "0.42+", '#17A2B8'),
        ("LangChain", "RAG 框架", "0.2+", '#FFC107'),
        ("PyMuPDF", "PDF 解析", "1.24+", '#6C757D'),
        ("RapidOCR", "图像 OCR", "1.3+", '#343A40'),
        ("SSE", "流式传输", "原生", '#17A2B8'),
    ]

    cols = 4
    for i, (name, desc, version, color) in enumerate(components):
        row, col = i // cols, i % cols
        x = 1.5 + col * 3.2
        y = 6.5 - row * 2.2

        ax.add_patch(FancyBboxPatch((x-1.3, y-0.7), 2.6, 1.4, boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='none', alpha=0.15))
        ax.add_patch(FancyBboxPatch((x-1.3, y+0.2), 2.6, 0.5, boxstyle="round,pad=0.02",
                                     facecolor=color, edgecolor='none'))
        ax.text(x, y + 0.45, name, ha='center', va='center', fontsize=11,
                fontweight='bold', color='white')
        ax.text(x, y - 0.15, desc, ha='center', va='center', fontsize=9)
        ax.text(x, y - 0.5, version, ha='center', va='center', fontsize=8, color='#666')

    plt.tight_layout()
    plt.savefig(f"{OUT}\\05-技术组件.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("05-技术组件.png")


if __name__ == "__main__":
    create_tech_architecture()
    create_retrieval_flow()
    create_mindmap()
    create_api_doc()
    create_tech_components()
    print(f"\n全部生成完成！目录: {OUT}")
