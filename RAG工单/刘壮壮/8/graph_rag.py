# -*- coding: utf-8 -*-
"""
graph_rag.py — Graph RAG 知识图谱构建与检索
工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

用金融研报构建知识图谱（实体、关系），并基于图谱做检索：
  1. 实体/关系抽取：优先 LLM 结构化抽取，回退到规则抽取
  2. 知识图谱构建：networkx 有向图（实体=节点，关系=边）
  3. 图谱检索：Query 实体链接 → 邻域扩展 → 召回关联文本块
  4. 可视化：导出 GraphML / 用 matplotlib 渲染 PNG
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

import networkx as nx

import rag_core

# 规则抽取：公司 / 人物 / 机构实体
_COMPANY = re.compile(r"[一-龥]{2,20}(?:股份有限公司|有限责任公司|有限公司|集团|银行|证券|保险|基金)")
_PERSON = re.compile(r"[一-龥]{2,4}(?:董事长|总经理|行长|总裁|法定代表人|监事|独立董事)")
_RELATIONS = ["控股", "参股", "投资", "设立", "合并", "担任", "持有", "收购", "发行", "增资", "合作"]


def extract_triples_rule(text: str) -> List[Tuple[str, str, str]]:
    """规则抽取三元组：同句内共现的「实体-关系-实体」。"""
    triples = []
    sentences = re.split(r"[。\n；;]", text)
    for s in sentences:
        companies = _COMPANY.findall(s)
        persons = _PERSON.findall(s)
        ents = list(dict.fromkeys(companies + persons))
        if len(ents) < 2:
            continue
        rel = next((r for r in _RELATIONS if r in s), "关联")
        # 相邻实体两两建边
        for i in range(len(ents) - 1):
            triples.append((ents[i], rel, ents[i + 1]))
    return triples


def extract_triples_llm(text: str, llm) -> List[Tuple[str, str, str]]:
    """LLM 结构化抽取三元组（需配置 LLM 接口）。"""
    prompt = (
        "从下面金融文本中抽取知识三元组，输出 JSON 数组，每个元素为 [头实体, 关系, 尾实体]。"
        "只抽取真实存在的实体关系，不要编造。\n"
        f"文本：{text[:1200]}\n三元组："
    )
    try:
        out = llm.generate(prompt)
        m = re.search(r"\[\[.*?\]\]", out, re.S) or re.search(r"\[.*\]", out, re.S)
        if m:
            import json
            triples = json.loads(m.group(0))
            return [(str(t[0]), str(t[1]), str(t[2])) for t in triples if len(t) >= 3]
    except Exception:
        pass
    return extract_triples_rule(text)


class GraphRAG:
    """Graph RAG：图谱构建 + 检索。"""

    def __init__(self, llm=None):
        self.llm = llm
        self.graph = nx.DiGraph()
        self.entity_chunks: Dict[str, List[int]] = defaultdict(list)  # 实体 -> chunk 下标
        self.chunks: List[str] = []
        self.triples: List[Tuple[str, str, str]] = []

    def build(self, chunks: List[str], use_llm: bool = False):
        self.chunks = chunks
        extract = (lambda t: extract_triples_llm(t, self.llm)) if (use_llm and self.llm) else extract_triples_rule
        for idx, c in enumerate(chunks):
            for h, r, t in extract(c):
                self.triples.append((h, r, t))
                self.graph.add_node(h, type="entity")
                self.graph.add_node(t, type="entity")
                self.graph.add_edge(h, t, relation=r)
                self.entity_chunks[h].append(idx)
                self.entity_chunks[t].append(idx)

    def query_entities(self, query: str) -> List[str]:
        """从问题中识别实体（图谱内匹配）。"""
        return [e for e in self.graph.nodes if e and e in query]

    def retrieve(self, query: str, top_k: int = 5, hop: int = 1) -> List[Dict[str, Any]]:
        """图谱检索：实体链接 → 邻域扩展 → 召回文本块。"""
        seed = self.query_entities(query)
        if not seed:
            # 退化为关键词匹配
            seed = [e for e in self.graph.nodes if any(k in e for k in query[:8])][:3]
        entities = set(seed)
        frontier = set(seed)
        for _ in range(hop):
            nxt = set()
            for e in frontier:
                for _, nb in self.graph.edges(e):
                    nxt.add(nb)
                for nb, _ in self.graph.in_edges(e):
                    nxt.add(nb)
            entities |= nxt
            frontier = nxt

        # 召回关联文本块
        chunk_scores: Dict[int, float] = defaultdict(float)
        for e in entities:
            for cidx in self.entity_chunks.get(e, []):
                chunk_scores[cidx] += 1.0
        order = sorted(chunk_scores.items(), key=lambda x: -x[1])[:top_k]
        hits = []
        for cidx, s in order:
            hits.append({"index": cidx, "text": self.chunks[cidx], "score": s})
        return hits

    def subgraph(self, query: str, depth: int = 1) -> nx.DiGraph:
        """返回与 query 相关的子图（用于可视化）。"""
        seed = self.query_entities(query)
        nodes = set(seed)
        frontier = set(seed)
        for _ in range(depth):
            nxt = set()
            for e in frontier:
                for _, nb in self.graph.edges(e):
                    nxt.add(nb)
                for nb, _ in self.graph.in_edges(e):
                    nxt.add(nb)
            nodes |= nxt
            frontier = nxt
        return self.graph.subgraph(nodes)

    def stats(self) -> Dict[str, Any]:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "triples": len(self.triples),
            "entity_types": dict(self.graph.nodes(data="type")) if False else {},
        }

    def export_graphml(self, path: str):
        nx.write_graphml(self.graph, path)

    def render(self, path: str, query: Optional[str] = None, max_nodes: int = 40):
        """用 matplotlib 渲染图谱（可选，需安装 matplotlib）。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        g = self.subgraph(query) if query else self.graph
        if g.number_of_nodes() > max_nodes:
            g = g.subgraph(list(g.nodes)[:max_nodes])
        pos = nx.spring_layout(g, seed=42, k=0.6)
        plt.figure(figsize=(12, 9))
        nx.draw_networkx_nodes(g, pos, node_size=500, node_color="#4C8BF5")
        nx.draw_networkx_edges(g, pos, alpha=0.4, arrows=True)
        nx.draw_networkx_labels(g, pos, font_size=8, font_family="SimHei")
        edge_labels = {(u, v): d.get("relation", "") for u, v, d in g.edges(data=True)}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=6)
        plt.title("金融知识图谱")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()
        return path
