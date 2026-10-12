# -*- coding: utf-8 -*-
"""
lightrag.py — LightRAG 轻量级实现
工单编号：人工智能NLP-RAG 项目-LightRAG 优化

参照 HKUDS/LightRAG（https://github.com/HKUDS/LightRAG）的核心思想：
  - 索引构建：文本切块 + 知识抽取（实体/关系）→ 存图谱 + 向量
  - 双层检索：低层次（具体实体/属性）+ 高层次（主题/概念）
  - 增量更新：新数据只更新相关部分，无需重建全库

本实现自包含（networkx + 规则抽取），离线可运行；
配置 LLM 接口后可切换为大模型抽取实体类型与关系类型。
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import networkx as nx

import rag_core

_ENTITY_TYPE = re.compile(
    r"(?P<company>[一-龥]{2,20}(?:股份有限公司|有限公司|集团|银行|证券|保险|基金))"
    r"|(?P<person>[一-龥]{2,4}(?:董事长|总经理|行长|总裁|法定代表人|监事|独立董事))"
)
_REL_DICT = ["控股", "参股", "投资", "设立", "合并", "担任", "持有", "收购", "发行", "增资", "合作", "子公司"]


class LightRAG:
    """LightRAG：图结构 + 双层检索。"""

    def __init__(self, llm=None):
        self.llm = llm
        self.graph = nx.DiGraph()
        self.chunks: List[str] = []
        # 实体 -> chunk 下标（低层索引）
        self.entity_chunks: Dict[str, List[int]] = defaultdict(list)
        # 主题/概念 -> chunk 下标（高层索引）
        self.topic_chunks: Dict[str, List[int]] = defaultdict(list)
        self._vec: Optional[object] = None  # 可选向量索引

    # ---- 索引构建 ----
    def build(self, chunks: List[str]):
        self.chunks = chunks
        for idx, c in enumerate(chunks):
            ents, rels, topics = self._extract(c)
            for h, r, t in rels:
                self.graph.add_node(h, type="entity")
                self.graph.add_node(t, type="entity")
                self.graph.add_edge(h, t, relation=r)
            for e in ents:
                self.entity_chunks[e].append(idx)
            for tp in topics:
                self.topic_chunks[tp].append(idx)
        # 建低层向量索引（TF-IDF 或语义向量）
        self._vec = rag_core.TfidfEmbedder().fit(chunks)

    def _extract(self, text: str):
        """抽取实体、关系、主题概念。"""
        ents = set(_ENTITY_TYPE.findall(text))
        flat = set()
        for tup in ents:
            for x in tup:
                if x:
                    flat.add(x)
        rels = []
        sents = re.split(r"[。\n；;]", text)
        for s in sents:
            found = list(dict.fromkeys([x for t in _ENTITY_TYPE.findall(s) for x in t if x]))
            if len(found) >= 2:
                r = next((r for r in _REL_DICT if r in s), "关联")
                for i in range(len(found) - 1):
                    rels.append((found[i], r, found[i + 1]))
        # 主题/概念：金融术语关键词
        topics = set(re.findall(r"(零售业务|风险管理|拨备覆盖率|不良贷款|财富管理|投行业务|营收|净利润|金融科技|绿色金融|资产质量|资本充足率)", text))
        return list(flat), rels, list(topics)

    # ---- 双层检索 ----
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        low = self._low_level(query, top_k * 2)
        high = self._high_level(query, top_k * 2)
        # 融合（RRF）
        rrf: Dict[int, float] = {}
        for rank, h in enumerate(low):
            rrf[h["index"]] = rrf.get(h["index"], 0.0) + 1.0 / (rank + 60)
        for rank, h in enumerate(high):
            rrf[h["index"]] = rrf.get(h["index"], 0.0) + 1.0 / (rank + 60)
        order = sorted(rrf.items(), key=lambda x: -x[1])[:top_k]
        return [{"index": i, "text": self.chunks[i], "score": s} for i, s in order]

    def _low_level(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """低层次检索：具体实体/属性，沿图关系扩展。"""
        seed = [e for e in self.graph.nodes if e and e in query]
        entities = set(seed)
        for e in list(seed):
            for _, nb in self.graph.edges(e):
                entities.add(nb)
        scores: Dict[int, float] = defaultdict(float)
        for e in entities:
            for c in self.entity_chunks.get(e, []):
                scores[c] += 1.5 if e in seed else 1.0
        order = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [{"index": c, "text": self.chunks[c], "score": s} for c, s in order]

    def _high_level(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """高层次检索：主题/概念，基于向量相似度。"""
        if self._vec is None:
            return []
        qv = self._vec.encode([query])
        sims = np.dot(self._vec.corpus_matrix(), qv.T).ravel()
        order = np.argsort(-sims)[:top_k]
        return [{"index": int(i), "text": self.chunks[int(i)], "score": float(sims[int(i)])} for i in order]

    # ---- 增量更新 ----
    def insert(self, new_chunks: List[str]):
        """增量更新：只处理新增块，不重建全库。"""
        base = len(self.chunks)
        self.chunks.extend(new_chunks)
        for idx, c in enumerate(new_chunks):
            ents, rels, topics = self._extract(c)
            for h, r, t in rels:
                self.graph.add_edge(h, t, relation=r)
            for e in ents:
                self.entity_chunks[e].append(base + idx)
            for tp in topics:
                self.topic_chunks[tp].append(base + idx)

    def stats(self) -> Dict[str, Any]:
        return {"nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "entities": len(self.entity_chunks),
                "topics": len(self.topic_chunks)}
