# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于Graph RAG实现金融问答
知识图谱可视化：
  1. render_png()  —— networkx + matplotlib 渲染静态图谱（按实体类型着色，无需联网）；
  2. render_html() —— 生成 vis-network 交互式 HTML（可拖拽/缩放，浏览器直接打开）。
"""
import os

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import GRAPH_IMG, GRAPH_HTML

# 实体类型 -> 颜色
TYPE_COLOR = {
    "公司": "#e74c3c", "指标": "#3498db", "数值": "#2ecc71", "人物": "#9b59b6",
    "职位": "#f39c12", "策略": "#16a085", "领域": "#1abc9c", "年份": "#95a5a6",
    "机构": "#d35400", "产品": "#8e44ad", "未知": "#7f8c8d",
}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def to_nx(graph_data, max_nodes=80):
    """把 graph.json 转为 networkx 图；节点过多时按度排序截断，保证可视化可读"""
    G = nx.DiGraph()
    for n in graph_data["nodes"]:
        G.add_node(n["id"], type=n.get("type", "未知"))
    type_of = {n["id"]: n.get("type", "未知") for n in graph_data["nodes"]}
    for e in graph_data["edges"]:
        if e["source"] in G and e["target"] in G:
            G.add_edge(e["source"], e["target"], relation=e.get("relation", ""))
    if G.number_of_nodes() > max_nodes:
        keep = sorted(G.degree, key=lambda x: -x[1])[:max_nodes]
        G = G.subgraph([n for n, _ in keep]).copy()
    return G, type_of


def render_png(graph_data, save_path=GRAPH_IMG, max_nodes=60):
    """静态图谱：按实体类型着色，公司节点放大"""
    G, type_of = to_nx(graph_data, max_nodes)
    if G.number_of_nodes() == 0:
        return None
    pos = nx.spring_layout(G, k=0.9, seed=42, iterations=120)
    colors = [TYPE_COLOR.get(type_of.get(n, "未知"), "#7f8c8d") for n in G.nodes]
    sizes = [1400 if type_of.get(n) == "公司" else 700 for n in G.nodes]
    plt.figure(figsize=(16, 11))
    nx.draw_networkx_edges(G, pos, alpha=0.35, arrows=True, arrowsize=10, edge_color="#888")
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.9,
                           edgecolors="white", linewidths=1.2)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family=plt.rcParams["font.sans-serif"][0])
    nx.draw_networkx_edge_labels(G, pos, font_size=6,
                                 edge_labels={(u, v): d.get("relation", "") for u, v, d in G.edges(data=True)})
    plt.title("金融年报知识图谱（实体-关系）", fontsize=16)
    plt.axis("off")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()
    return save_path


def render_html(graph_data, save_path=GRAPH_HTML, max_nodes=200):
    """交互式 HTML：vis-network（CDN 引入）"""
    G, type_of = to_nx(graph_data, max_nodes)
    nodes = [{"id": n, "label": n, "color": TYPE_COLOR.get(type_of.get(n, "未知"), "#7f8c8d"),
              "title": f"类型：{type_of.get(n, '未知')}",
              "size": 26 if type_of.get(n) == "公司" else 14} for n in G.nodes]
    edges = [{"from": u, "to": v, "label": d.get("relation", ""),
              "title": d.get("relation", "")} for u, v, d in G.edges(data=True)]
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>金融年报知识图谱</title>
<script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>html,body{{margin:0;height:100%}}#g{{width:100%;height:100vh}}</style></head><body>
<div id="g"></div><script>
var nodes = new vis.DataSet({nodes});
var edges = new vis.DataSet({edges});
new vis.Network(document.getElementById('g'), {{nodes:nodes, edges:edges}}, {{
  physics: {{solver:'forceAtlas2Based', forceAtlas2Based:{{gravitationalConstant:-60}}, stabilization:true}},
  edges: {{arrows:'to', font:{{size:11, align:'middle'}}, color:{{color:'#999'}}}},
  nodes: {{shape:'dot', font:{{size:14}}}},
  interaction: {{hover:true, tooltipDelay:120}}
}});
</script></body></html>"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)
    return save_path


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    from graph_builder import load_graph
    d = load_graph()
    print(render_png(d))
    print(render_html(d))
