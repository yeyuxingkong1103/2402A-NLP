# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
模块：知识图谱可视化
功能：用 PyVis 生成交互式 HTML 图谱
"""

import json
import networkx as nx
from pyvis.network import Network


class GraphVisualizer:
    TYPE_COLORS = {
        "公司": "#4A90E2", "人物": "#F5A623", "机构": "#7B68EE",
        "指标": "#50C878", "数值": "#FF6B6B", "年份": "#95A5A6",
        "地点": "#E67E22", "产品": "#9B59B6", "报告": "#3498DB",
        "未知": "#BDC3C7",
    }

    def __init__(self, graph_path: str):
        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data, edges="links")
        print(f"✅ 加载图谱：{self.graph.number_of_nodes()} 节点 / {self.graph.number_of_edges()} 边")

    def render_html(self, output_path: str = "graph.html", max_nodes: int = 200):
        if self.graph.number_of_nodes() > max_nodes:
            degrees = dict(self.graph.degree())
            top_nodes = sorted(degrees.keys(), key=lambda x: -degrees[x])[:max_nodes]
            subgraph = self.graph.subgraph(top_nodes).copy()
        else:
            subgraph = self.graph

        net = Network(height="750px", width="100%", bgcolor="#1a1a1a",
                       font_color="white", directed=True, notebook=False)

        for node, attrs in subgraph.nodes(data=True):
            ntype = attrs.get("type", "未知")
            color = self.TYPE_COLORS.get(ntype, self.TYPE_COLORS["未知"])
            size = 10 + subgraph.degree(node) * 2
            net.add_node(node, label=str(node)[:30],
                          title=f"{node}\n类型：{ntype}\n度数：{subgraph.degree(node)}",
                          color=color, size=min(size, 50))

        for src, dst, attrs in subgraph.edges(data=True):
            rel = attrs.get("relation", "")
            net.add_edge(src, dst, label=rel, title=rel)

        net.set_options("""
        {
          "physics": {"barnesHut": {"gravitationalConstant": -8000, "centralGravity": 0.3,
            "springLength": 200, "springConstant": 0.04}, "solver": "barnesHut",
            "stabilization": {"iterations": 150}},
          "interaction": {"hover": true, "navigationButtons": true, "keyboard": true},
          "edges": {"smooth": {"type": "continuous"},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}}
        }
        """)

        net.write_html(output_path, notebook=False, open_browser=False)
        print(f"✅ 已保存到 {output_path}（{subgraph.number_of_nodes()} 节点）")
        return output_path


if __name__ == "__main__":
    import sys
    graph_path = sys.argv[1] if len(sys.argv) > 1 else "graph_test.json"
    viz = GraphVisualizer(graph_path)
    viz.render_html("graph.html", max_nodes=200)
