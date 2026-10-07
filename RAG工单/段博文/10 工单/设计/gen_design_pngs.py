# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-金融问答系统部署
"""生成部署相关设计 PNG：Docker 部署架构图、容器编排图、数据流图、网络拓扑图、技术组件图。"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = r"D:\作业\6-专高NLP 作业\成品\10\设计"
os.makedirs(OUT, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_deploy_architecture():
    """Docker 部署架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_title('Docker 部署架构图', fontsize=16, fontweight='bold', pad=20)

    # Docker 网络边界
    ax.add_patch(FancyBboxPatch((0.5, 0.5), 13, 7.5, boxstyle="round,pad=0.1",
                                 facecolor='#F0F4F8', edgecolor='#343A40', linewidth=2, linestyle='--'))
    ax.text(7, 7.7, 'Docker Network: rag-net', ha='center', fontsize=12, fontweight='bold', color='#343A40')

    # 宿主机
    ax.add_patch(FancyBboxPatch((1, 6.3), 4, 0.6, boxstyle="round,pad=0.05",
                                 facecolor='#343A40', edgecolor='none'))
    ax.text(3, 6.6, '宿主机 (Linux Server)', ha='center', va='center', fontsize=11, color='white', fontweight='bold')

    # 应用容器
    ax.add_patch(FancyBboxPatch((1, 4.5), 4, 1.5, boxstyle="round,pad=0.05",
                                 facecolor='#E8F4FD', edgecolor='#007BFF', linewidth=2))
    ax.text(3, 5.5, 'rag-app 容器', ha='center', fontsize=11, fontweight='bold', color='#007BFF')
    ax.text(3, 5.1, 'FastAPI + Uvicorn', ha='center', fontsize=8, color='#555')
    ax.text(3, 4.8, '端口: 8000', ha='center', fontsize=8, color='#555')

    # Milvus 容器
    ax.add_patch(FancyBboxPatch((7, 4.5), 4, 1.5, boxstyle="round,pad=0.05",
                                 facecolor='#D4EDDA', edgecolor='#28A745', linewidth=2))
    ax.text(9, 5.5, 'rag-milvus 容器', ha='center', fontsize=11, fontweight='bold', color='#28A745')
    ax.text(9, 5.1, 'Milvus v2.4.0', ha='center', fontsize=8, color='#555')
    ax.text(9, 4.8, '端口: 19530/9091', ha='center', fontsize=8, color='#555')

    # etcd 容器
    ax.add_patch(FancyBboxPatch((7, 2.5), 1.8, 1.2, boxstyle="round,pad=0.05",
                                 facecolor='#FFF3CD', edgecolor='#FFC107', linewidth=2))
    ax.text(7.9, 3.1, 'etcd', ha='center', fontsize=10, fontweight='bold', color='#FFC107')
    ax.text(7.9, 2.8, '元数据存储', ha='center', fontsize=7, color='#555')

    # MinIO 容器
    ax.add_patch(FancyBboxPatch((9.2, 2.5), 1.8, 1.2, boxstyle="round,pad=0.05",
                                 facecolor='#F8D7DA', edgecolor='#DC3545', linewidth=2))
    ax.text(10.1, 3.1, 'MinIO', ha='center', fontsize=10, fontweight='bold', color='#DC3545')
    ax.text(10.1, 2.8, '对象存储', ha='center', fontsize=7, color='#555')

    # 数据卷
    volumes = [
        ('rag-data', 1.5, '#6F42C1'),
        ('rag-models', 3.5, '#6F42C1'),
        ('milvus-data', 7, '#20C997'),
        ('rag-logs', 10, '#17A2B8'),
    ]
    for name, x, color in volumes:
        ax.add_patch(FancyBboxPatch((x, 0.8), 1.5, 0.6, boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='none', alpha=0.3))
        ax.text(x + 0.75, 1.1, name, ha='center', va='center', fontsize=8, fontweight='bold')

    # 箭头
    arrows = [
        ((5, 5.25), (7, 5.25), 'API 调用'),
        ((9, 4.5), (7.9, 3.7), '元数据'),
        ((9, 4.5), (10.1, 3.7), '对象存储'),
        ((3, 4.5), (2.25, 1.4), 'PDF/日志'),
        ((3, 4.5), (4.25, 1.4), '模型'),
        ((9, 4.5), (7.75, 1.4), '向量数据'),
    ]
    for (x1, y1), (x2, y2), label in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label, ha='center', fontsize=7, color='#666')

    plt.tight_layout()
    plt.savefig(f"{OUT}\\01-Docker部署架构图.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("01-Docker部署架构图.png")


def create_container_orchestration():
    """容器编排图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Docker Compose 容器编排图', fontsize=16, fontweight='bold', pad=20)

    # 编排层
    ax.add_patch(FancyBboxPatch((5.5, 6.5), 3, 0.8, boxstyle="round,pad=0.05",
                                 facecolor='#343A40', edgecolor='none'))
    ax.text(7, 6.9, 'docker-compose.yml', ha='center', va='center', fontsize=12, color='white', fontweight='bold')

    services = [
        ('rag-app', 'FastAPI 应用', 1, 4.5, '#007BFF'),
        ('rag-milvus', '向量数据库', 5, 4.5, '#28A745'),
        ('rag-etcd', '元数据存储', 9, 4.5, '#FFC107'),
        ('rag-minio', '对象存储', 12, 4.5, '#DC3545'),
    ]

    for name, desc, x, y, color in services:
        ax.add_patch(FancyBboxPatch((x - 1, y), 2, 1.2, boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='none', alpha=0.2))
        ax.add_patch(FancyBboxPatch((x - 1, y + 0.7), 2, 0.5, boxstyle="round,pad=0.02",
                                     facecolor=color, edgecolor='none'))
        ax.text(x, y + 0.95, name, ha='center', va='center', fontsize=10, color='white', fontweight='bold')
        ax.text(x, y + 0.35, desc, ha='center', va='center', fontsize=8)

    # 依赖关系
    deps = [
        ((2, 4.5), (6, 4.5), 'depends_on'),
        ((6, 4.5), (10, 4.5), 'depends_on'),
        ((6, 4.5), (13, 4.5), 'depends_on'),
    ]
    for (x1, y1), (x2, y2), label in deps:
        ax.annotate('', xy=(x2 - 1, y1), xytext=(x1 + 1, y1),
                   arrowprops=dict(arrowstyle='->', color='#666', lw=2))

    # 数据卷
    ax.text(7, 2.5, 'Docker Volumes', ha='center', fontsize=12, fontweight='bold')
    vols = ['milvus-data', 'etcd-data', 'minio-data', 'rag-data', 'rag-models', 'rag-logs']
    for i, v in enumerate(vols):
        x = 1.5 + i * 2
        ax.add_patch(FancyBboxPatch((x - 0.8, 1.5), 1.6, 0.6, boxstyle="round,pad=0.03",
                                     facecolor='#6F42C1', edgecolor='none', alpha=0.3))
        ax.text(x, 1.8, v, ha='center', va='center', fontsize=7, fontweight='bold')

    # 网络
    ax.add_patch(FancyBboxPatch((5.5, 0.3), 3, 0.6, boxstyle="round,pad=0.05",
                                 facecolor='#17A2B8', edgecolor='none'))
    ax.text(7, 0.6, 'Network: rag-net', ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUT}\\02-容器编排图.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("02-容器编排图.png")


def create_data_flow():
    """数据流图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('容器数据流图', fontsize=16, fontweight='bold', pad=20)

    steps = [
        ("用户请求", 1, 6, '#E8F4FD'),
        ("rag-app 容器", 4, 6, '#FFF3CD'),
        ("Milvus 向量检索", 7.5, 7, '#D4EDDA'),
        ("BM25 全文检索", 7.5, 5, '#D4EDDA'),
        ("RRF 融合+重排", 10.5, 6, '#F8D7DA'),
        ("LLM 生成答案", 13, 6, '#E2E3F3'),
    ]

    for text, x, y, color in steps:
        ax.add_patch(FancyBboxPatch((x - 0.9, y - 0.4), 1.8, 0.8, boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='#666', linewidth=1.5))
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

    # 箭头
    arrows = [
        ((1.9, 6), (3.1, 6)),
        ((4.9, 6), (6.6, 7)),
        ((4.9, 6), (6.6, 5)),
        ((8.4, 7), (9.6, 6.3)),
        ((8.4, 5), (9.6, 5.7)),
        ((11.4, 6), (12.1, 6)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='#666', lw=2))

    # 数据持久化标注
    ax.text(4, 3.5, '数据持久化', ha='center', fontsize=11, fontweight='bold')
    persist = [
        ('rag-data\nPDF/索引', 2, 2.5),
        ('milvus-data\n向量数据', 5, 2.5),
        ('rag-models\n模型权重', 8, 2.5),
        ('rag-logs\n运行日志', 11, 2.5),
    ]
    for text, x, y in persist:
        ax.add_patch(FancyBboxPatch((x - 0.9, y - 0.4), 1.8, 0.8, boxstyle="round,pad=0.05",
                                     facecolor='#6F42C1', edgecolor='none', alpha=0.2))
        ax.text(x, y, text, ha='center', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUT}\\03-数据流图.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("03-数据流图.png")


def create_network_topology():
    """网络拓扑图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('容器网络拓扑图', fontsize=16, fontweight='bold', pad=20)

    # 外部用户
    ax.add_patch(FancyBboxPatch((1, 6), 2, 0.8, boxstyle="round,pad=0.05",
                                 facecolor='#343A40', edgecolor='none'))
    ax.text(2, 6.4, '外部用户', ha='center', va='center', fontsize=11, color='white', fontweight='bold')

    # 主机端口映射
    ax.add_patch(FancyBboxPatch((5, 6), 4, 0.8, boxstyle="round,pad=0.05",
                                 facecolor='#6C757D', edgecolor='none'))
    ax.text(7, 6.4, '主机端口映射\n8000:8000 / 19530:19530', ha='center', va='center', fontsize=9, color='white')

    # 网络边界
    ax.add_patch(FancyBboxPatch((0.5, 0.5), 13, 5, boxstyle="round,pad=0.1",
                                 facecolor='#F0F4F8', edgecolor='#17A2B8', linewidth=2, linestyle='--'))
    ax.text(7, 5.2, 'Docker Network: rag-net (bridge)', ha='center', fontsize=11, fontweight='bold', color='#17A2B8')

    # 容器
    containers = [
        ('rag-app', '8000', 2.5, 3, '#007BFF'),
        ('rag-milvus', '19530', 6, 3, '#28A745'),
        ('rag-etcd', '2379', 9, 3, '#FFC107'),
        ('rag-minio', '9000', 11.5, 3, '#DC3545'),
    ]
    for name, port, x, y, color in containers:
        ax.add_patch(FancyBboxPatch((x - 0.8, y - 0.5), 1.6, 1, boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='none', alpha=0.2))
        ax.text(x, y + 0.15, name, ha='center', fontsize=10, fontweight='bold')
        ax.text(x, y - 0.25, f'port:{port}', ha='center', fontsize=8, color='#555')

    # 连接
    conns = [
        ((2.5, 3.5), (6, 3.5)),
        ((6, 3.5), (9, 3.5)),
        ((6, 3.5), (11.5, 3.5)),
    ]
    for (x1, y1), (x2, y2) in conns:
        ax.annotate('', xy=(x2, y1), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='<->', color='#666', lw=2))

    # 外部服务（RTMP 示例）
    ax.add_patch(FancyBboxPatch((9, 1), 3, 0.6, boxstyle="round,pad=0.05",
                                 facecolor='#FD7E14', edgecolor='none'))
    ax.text(10.5, 1.3, '外部服务 (RTMP)', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    ax.annotate('', xy=(10.5, 1.6), xytext=(10.5, 2.5),
               arrowprops=dict(arrowstyle='->', color='#FD7E14', lw=2, linestyle='--'))

    # 用户到主机
    ax.annotate('', xy=(5, 6.4), xytext=(3, 6.4),
               arrowprops=dict(arrowstyle='->', color='#343A40', lw=2))
    # 主机到容器
    ax.annotate('', xy=(2.5, 4), xytext=(7, 5.6),
               arrowprops=dict(arrowstyle='->', color='#6C757D', lw=2))

    plt.tight_layout()
    plt.savefig(f"{OUT}\\04-网络拓扑图.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("04-网络拓扑图.png")


def create_tech_components():
    """技术组件图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('部署技术组件清单', fontsize=16, fontweight='bold', pad=20)

    components = [
        ("Docker", "容器运行时", "≥20.10", '#2496ED'),
        ("Docker Compose", "容器编排", "≥2.0", '#00BFFF'),
        ("Milvus", "向量数据库", "v2.4.0", '#28A745'),
        ("etcd", "元数据存储", "v3.5.5", '#FFC107'),
        ("MinIO", "对象存储", "RELEASE", '#DC3545'),
        ("Python", "运行环境", "3.10", '#3776AB'),
        ("FastAPI", "Web 框架", "0.110+", '#009688'),
        ("Uvicorn", "ASGI 服务器", "0.29+", '#6B46C1'),
        ("bge-m3", "向量模型", "BAAI", '#6F42C1'),
        ("bge-reranker", "重排模型", "BAAI", '#E83E8C'),
        ("DeepSeek", "LLM", "API", '#FD7E14'),
        ("Nginx", "反向代理", "可选", '#009639'),
    ]

    cols = 4
    for i, (name, desc, version, color) in enumerate(components):
        row, col = i // cols, i % cols
        x = 1.5 + col * 3.2
        y = 6.5 - row * 2

        ax.add_patch(FancyBboxPatch((x - 1.3, y - 0.7), 2.6, 1.4, boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='none', alpha=0.15))
        ax.add_patch(FancyBboxPatch((x - 1.3, y + 0.2), 2.6, 0.5, boxstyle="round,pad=0.02",
                                     facecolor=color, edgecolor='none'))
        ax.text(x, y + 0.45, name, ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        ax.text(x, y - 0.15, desc, ha='center', va='center', fontsize=9)
        ax.text(x, y - 0.5, version, ha='center', va='center', fontsize=8, color='#666')

    plt.tight_layout()
    plt.savefig(f"{OUT}\\05-技术组件图.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("05-技术组件图.png")


if __name__ == "__main__":
    create_deploy_architecture()
    create_container_orchestration()
    create_data_flow()
    create_network_topology()
    create_tech_components()
    print(f"\n全部生成完成！目录: {OUT}")
