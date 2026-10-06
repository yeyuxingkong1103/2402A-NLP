# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG 项目-LightRAG 优化任务
# 模块：lightrag_rag/viz_lightrag —— 把 LightRAG 建好的知识图谱导出为交互式 HTML（交付物「知识图谱」）
# 说明：读取 LightRAG 的图存储 `data/lightrag_store/graph_chunk_entity_relation.graphml`，
#       按实体类型着色、按关系标签加边注，用 pyvis 生成可直接打开的单文件 HTML；
#       同时输出图谱统计（节点/边/类型分布/度中心 top）为 JSON，供文档引用。
# 用法（.venv_lightrag 下）：python lightrag_rag\viz_lightrag.py
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.stdout.reconfigure(encoding="utf-8")

GRAPH = os.path.join(ROOT, "data", "lightrag_store", "graph_chunk_entity_relation.graphml")
OUT_HTML = os.path.join(ROOT, "static", "lightrag_graph.html")
OUT_JSON = os.path.join(HERE, "graph_stats.json")

PALETTE = ["#4C8DFF", "#FF7A45", "#36CFC9", "#9254DE", "#FADB14", "#73D13D", "#FF85C0",
           "#597EF7", "#FFA940", "#5CDBD3", "#B37FEB", "#D3ADF7"]


def render(g, out_html):
    """networkx 图 → pyvis 交互式 HTML（中文用 UTF-8 写盘）。"""
    from pyvis.network import Network
    types = Counter((a.get("entity_type") or "unknown") for _, a in g.nodes(data=True))
    cmap = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(sorted(types))}
    deg = dict(g.degree())
    net = Network(height="720px", width="100%", directed=True, bgcolor="#11151c",
                  font_color="#e6ebf5", notebook=False)
    net.barnes_hut(gravity=-9000, central_gravity=0.3, spring_length=160)
    for n, a in g.nodes(data=True):
        t = a.get("entity_type") or "unknown"
        d = a.get("description") or ""
        net.add_node(n, label=n[:24], title="%s\n[%s]\ndegree=%d\n%s" % (n, t, deg.get(n, 0), d[:200]),
                     color=cmap[t], size=8 + min(22, deg.get(n, 0) * 1.6))
    for u, v, a in g.edges(data=True):
        lbl = (a.get("keywords") or a.get("description") or "")[:40]
        net.add_edge(u, v, label=lbl, title=lbl, arrows="to", color="#5b6b82")
    try:
        html = net.generate_html(notebook=False)
    except Exception:                                     # noqa: BLE001
        html = net.html or ""
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return types


def stats_of(g):
    types = Counter((a.get("entity_type") or "unknown") for _, a in g.nodes(data=True))
    top = sorted(dict(g.degree()).items(), key=lambda x: -x[1])[:15]
    return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
            "entity_types": dict(types.most_common()),
            "isolated": sum(1 for _, d in g.degree() if d == 0),
            "top_degree": [{"entity": k, "degree": v} for k, v in top]}


def main():
    import networkx as nx

    g = nx.read_graphml(GRAPH)
    types = render(g, OUT_HTML)
    top = sorted(dict(g.degree()).items(), key=lambda x: -x[1])[:15]

    stats = {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
             "entity_types": dict(types.most_common()),
             "top_degree": [{"entity": k, "degree": v} for k, v in top]}
    json.dump(stats, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("图谱：节点 %d / 边 %d ｜ 实体类型 %d 种" % (stats["nodes"], stats["edges"], len(types)))
    for t, c in types.most_common(8):
        print("   %-28s %d" % (t[:28], c))
    print("度最高：", ", ".join("%s(%d)" % (k, v) for k, v in top[:6]))
    print("saved ->", OUT_HTML, "|", OUT_JSON)


if __name__ == "__main__":
    main()
