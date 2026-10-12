# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务
Graph RAG 检索：从问题实体出发做邻居扩展，返回相关文本块作为上下文。
"""
import config
from graph_builder import neighbors, _rule_entities


class GraphRAG:
    def __init__(self, graph, docs):
        self.graph = graph
        self.docs = docs

    def _query_entities(self, query):
        return [n for _, n in _rule_entities(query)]

    def retrieve(self, query, top_k=3, hops=2):
        ents = self._query_entities(query)
        related = set()
        for e in ents:
            related |= neighbors(self.graph, e, hops)
        scored = []
        for i, doc in enumerate(self.docs):
            score = sum(doc.count(e) for e in related if e) + sum(doc.count(e) for e in ents if e)
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda x: -x[1])
        return [self.docs[i] for i, _ in scored[:top_k]]
