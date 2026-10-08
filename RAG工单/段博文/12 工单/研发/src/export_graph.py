# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-LightRAG优化
"""
知识图谱导出与可视化模块（产出物 2：知识图谱）。

功能：
    1. 读取 LightRAG 的 NetworkX 图存储（graphml）；
    2. 生成图谱统计：实体类型分布、节点度分布、Top 实体；
    3. pyvis 生成交互式 HTML 知识图谱（按节点度取前 300 个实体）；
    4. matplotlib 生成图谱统计 PNG。
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from config import WORK_DIR, RESULT_DIR
from logger import get_logger

logger = get_logger(__name__)

OUT_DIR = RESULT_DIR
# 测试目录（HTML 与 PNG 同时输出到 测试\知识图谱\）
# src/results 向上三层：results -> src -> 研发 -> 成品12
TEST_KG_DIR = OUT_DIR.parent.parent.parent / "测试" / "知识图谱"


def find_graphml() -> Path:
    """定位 graphml 图存储文件。"""
    for pattern in ("*.graphml", "**/*.graphml"):
        found = list(WORK_DIR.glob(pattern))
        if found:
            return found[0]
    raise FileNotFoundError(f"未找到 graphml 文件（{WORK_DIR}）")


def load_graph() -> nx.Graph:
    g = nx.read_graphml(find_graphml())
    logger.info(f"图谱已加载：{g.number_of_nodes()} 节点 / {g.number_of_edges()} 边")
    return g


def graph_summary(g: nx.Graph) -> dict:
    """图谱统计信息。"""
    types = Counter()
    for _, data in g.nodes(data=True):
        t = data.get("entity_type", "未知")
        types[t] += 1
    degrees = dict(g.degree())
    top_entities = sorted(degrees.items(), key=lambda x: -x[1])[:20]
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "entity_type_dist": dict(types.most_common()),
        "top_entities": [
            {"entity": name, "degree": d,
             "type": g.nodes[name].get("entity_type", ""),
             "description": str(g.nodes[name].get("description", ""))[:120]}
            for name, d in top_entities
        ],
    }


def make_pyvis_html(g: nx.Graph, max_nodes: int = 300) -> Path:
    """pyvis 交互式知识图谱 HTML。"""
    from pyvis.network import Network

    degrees = dict(g.degree())
    keep = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
    sub = g.subgraph(keep).copy()

    # 类型→颜色
    palette = {
        "公司": "#e74c3c", "人员": "#3498db", "股东": "#9b59b6",
        "产品": "#f39c12", "产品与服务": "#f39c12", "行业": "#16a085",
        "财务指标": "#27ae60", "募投项目": "#d35400", "项目": "#d35400",
        "技术标准": "#2980b9", "工程奖项": "#c0392b", "技术与资质": "#8e44ad",
        "机构": "#7f8c8d", "地点": "#2c3e50", "事件": "#e67e22",
    }

    net = Network(height="760px", width="100%", bgcolor="#ffffff",
                  font_color="#333333", notebook=False, cdn_resources="in_line")
    net.force_atlas_2based(gravity=-40, central_gravity=0.01,
                           spring_length=120, damping=0.9)
    for node, data in sub.nodes(data=True):
        et = data.get("entity_type", "未知")
        color = None
        for key, c in palette.items():
            if key in et:
                color = c
                break
        deg = degrees.get(node, 1)
        title = (f"<b>{node}</b><br>类型: {et}<br>度: {deg}<br>"
                 f"{str(data.get('description', ''))[:200]}")
        net.add_node(node, label=node[:14], title=title,
                     size=min(10 + deg * 2, 48),
                     color=color or "#95a5a6")
    for u, v, data in sub.edges(data=True):
        net.add_edge(u, v, title=str(data.get("description", ""))[:150],
                     width=1.2, arrows="to")

    TEST_KG_DIR.mkdir(parents=True, exist_ok=True)
    out = TEST_KG_DIR / "知识图谱可视化.html"
    # 300 节点超过 improvedLayout 阈值，关闭它并减少 stabilization 迭代，
    # 使加载进度条(loadingBar)能及时消失
    net.set_options(json.dumps({
        "layout": {"improvedLayout": False},
        "physics": {"stabilization": {"enabled": True, "iterations": 200,
                                      "updateInterval": 50}},
        "interaction": {"hover": True, "navigationButtons": True},
    }, ensure_ascii=False))
    net.save_graph(str(out))
    logger.info(f"交互式知识图谱已生成：{out}")
    return out


def make_stats_png(g: nx.Graph) -> Path:
    """图谱统计 PNG（类型分布 + Top 实体 + 度分布）。"""
    summary = graph_summary(g)
    TEST_KG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    fig.suptitle("招股说明书知识图谱统计（LightRAG 构建）", fontsize=16, fontweight="bold")

    # 1) 实体类型分布
    items = list(summary["entity_type_dist"].items())[:12]
    axes[0].barh([k for k, _ in items][::-1], [v for _, v in items][::-1], color="#3498db")
    axes[0].set_title("实体类型分布 Top12")
    axes[0].set_xlabel("实体数量")
    for i, (_, v) in enumerate(items[::-1]):
        axes[0].text(v + 2, i, str(v), va="center", fontsize=9)

    # 2) Top15 实体（按度）
    tops = summary["top_entities"][:15]
    names = [t["entity"][:12] for t in tops][::-1]
    vals = [t["degree"] for t in tops][::-1]
    axes[1].barh(names, vals, color="#e67e22")
    axes[1].set_title("核心实体 Top15（按关联度）")
    axes[1].set_xlabel("关联边数")

    # 3) 节点度分布
    degs = [d for _, d in g.degree()]
    axes[2].hist(degs, bins=40, color="#27ae60", edgecolor="white")
    axes[2].set_title("节点度分布")
    axes[2].set_xlabel("节点度")
    axes[2].set_ylabel("节点数")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = TEST_KG_DIR / "知识图谱统计图.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"图谱统计图已生成：{out}")

    (OUT_DIR / "graph_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main():
    g = load_graph()
    print(json.dumps(graph_summary(g), ensure_ascii=False, indent=2)[:1500])
    make_pyvis_html(g)
    make_stats_png(g)


if __name__ == "__main__":
    main()
