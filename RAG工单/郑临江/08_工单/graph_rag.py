# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于Graph RAG实现金融问答
Graph RAG 检索模块：从问题抽取实体 → 图谱邻居扩展 → 检索相关文本块 → 生成答案。
"""
import config
from graph_builder import extract_entities


class GraphRAG:
    def __init__(self, graph, docs):
        self.graph = graph
        self.docs = docs

    def _query_entities(self, query):
        return [name for _, name in extract_entities(query)]

    def retrieve_subgraph(self, query, hops=2):
        """返回与问题相关的子图（节点、边）。"""
        ents = self._query_entities(query)
        if not ents:
            ents = [n for n in self.graph.nodes][:5]
        nodes = set()
        edges = []
        for e in ents:
            nodes |= self.graph.neighbors(e, hops)
            nodes.add(e)
        for (s, r, t), w in self.graph.edges.items():
            if s in nodes and t in nodes:
                edges.append((s, r, t, w))
        return nodes, edges

    def retrieve_context(self, query, top_k=3):
        """结合图谱邻居检索相关文本块。"""
        ents = self._query_entities(query)
        scored = []
        for i, doc in enumerate(self.docs):
            score = sum(doc.count(e) for e in ents if e)
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda x: -x[1])
        return [self.docs[i] for i, _ in scored[:top_k]]

    def answer(self, query, llm):
        nodes, edges = self.retrieve_subgraph(query)
        ctx = self.retrieve_context(query)
        prompt = (
            f"你是金融年报问答助手。参考下方知识图谱结构与文档片段回答问题。\n"
            f"相关知识图谱实体：{'、'.join(list(nodes)[:20])}\n"
            f"相关文档片段：\n" + "\n".join(c[:400] for c in ctx) +
            f"\n\n用户问题：{query}\n请给出答案："
        )
        answer = llm.generate(prompt)
        return {"answer": answer, "subgraph_nodes": list(nodes), "subgraph_edges": edges}
