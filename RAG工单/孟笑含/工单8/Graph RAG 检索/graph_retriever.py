# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
模块：Graph RAG 检索器
功能：基于知识图谱的实体链接 + 子图召回 + 文本融合
"""

import json
import networkx as nx
from typing import List, Dict, Any


class GraphRetriever:
    """Graph RAG 检索器"""

    RELATION_WEIGHTS = {
        "持股": 1.0, "控股": 1.0, "任职": 0.9,
        "实现": 0.8, "报告": 0.7, "拥有": 0.8,
        "属于": 0.6, "同比": 0.7, "包含": 0.6,
    }

    def __init__(self, graph_path: str):
        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data, edges="links")
        print(f"✅ GraphRetriever 加载：{self.graph.number_of_nodes()} 节点 / {self.graph.number_of_edges()} 边")
        self.entities = list(self.graph.nodes())

    def find_entities(self, query: str) -> List[str]:
        found = []
        for ent in self.entities:
            if ent in query and len(ent) >= 3:
                found.append(ent)
        return sorted(set(found), key=lambda x: -len(x))

    def get_subgraph(self, entity: str, hops: int = 2):
        if entity not in self.graph:
            return nx.MultiDiGraph()
        undirected = self.graph.to_undirected()
        nodes = list(nx.single_source_shortest_path_length(undirected, entity, cutoff=hops).keys())
        return self.graph.subgraph(nodes).copy()

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        entities = self.find_entities(query)
        if not entities:
            return []
        all_triples = []
        for ent in entities:
            subgraph = self.get_subgraph(ent, hops=2)
            for src, dst, attrs in subgraph.edges(data=True):
                rel = attrs.get("relation", "")
                weight = self.RELATION_WEIGHTS.get(rel, 0.5)
                if src in entities or dst in entities:
                    all_triples.append({
                        "head": src,
                        "relation": rel,
                        "tail": dst,
                        "doc": attrs.get("doc", ""),
                        "page": attrs.get("page", 0),
                        "score": weight,
                    })
        seen = set()
        unique = []
        for t in all_triples:
            key = (t["head"], t["relation"], t["tail"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(t)
        unique.sort(key=lambda x: -x["score"])
        return unique[:top_k]

    def triples_to_text(self, triples: List[Dict]) -> str:
        lines = []
        for t in triples:
            lines.append(f"• {t['head']} —{t['relation']}→ {t['tail']}（{t['doc']} 第{t['page']}页）")
        return "\n".join(lines)

    def get_stats(self) -> Dict:
        return {"nodes": self.graph.number_of_nodes(), "edges": self.graph.number_of_edges()}


if __name__ == "__main__":
    gr = GraphRetriever("graph_test.json")
    print("\n" + "=" * 70)
    print("Graph RAG 检索测试")
    print("=" * 70)
    test_queries = [
        "武汉兴图新科电子股份有限公司注册资本是多少？",
        "武汉力源信息技术股份有限公司法定代表人是谁？",
    ]
    for q in test_queries:
        print(f"\n【Query】{q}")
        entities = gr.find_entities(q)
        print(f"  识别实体：{entities}")
        triples = gr.retrieve(q, top_k=5)
        print(f"  召回三元组：{len(triples)} 条")
        for t in triples[:5]:
            print(f"    {t['head']} —{t['relation']}→ {t['tail']} (score={t['score']})")
