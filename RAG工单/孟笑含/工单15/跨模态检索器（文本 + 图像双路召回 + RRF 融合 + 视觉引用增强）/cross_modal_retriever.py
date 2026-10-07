# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-优化技术图纸与文本的跨模态检索流程工单 - 跨模态检索器"""

import re
import json
from typing import List, Dict, Any
from vector_retriever import VectorRetriever


class CrossModalRetriever:
    """跨模态检索器（文本 + 图像双路召回）"""

    def __init__(self, chunks: List[Dict[str, Any]], model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.chunks = chunks
        self.retriever = VectorRetriever(model_name=model_name)
        self.retriever.build_index(chunks)

    def extract_visual_references(self, query: str) -> Dict[str, Any]:
        """提取问题中的视觉引用"""
        refs = {}
        m = re.search(r'图\s*(\d+)', query)
        if m:
            refs["figure"] = int(m.group(1))
        m = re.search(r'第\s*(\d+)\s*页', query)
        if m:
            refs["page"] = int(m.group(1))
        ids = re.findall(r'编号\s*(\d+)', query)
        parts = re.findall(r'部件\s*(\d+)', query)
        refs["component_ids"] = list(set([int(i) for i in ids] + [int(p) for p in parts]))
        return refs

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """跨模态检索主入口"""
        refs = self.extract_visual_references(query)
        print(f"  [视觉引用] {refs}")

        # 1. 原始查询
        results1 = self.retriever.retrieve(query, top_k=20)

        # 2. 视觉增强查询
        if refs:
            enhanced = query
            if "figure" in refs:
                enhanced += f" 图{refs['figure']}"
            if "page" in refs:
                enhanced += f" 第{refs['page']}页"
            if refs.get("component_ids"):
                enhanced += " " + " ".join(f"部件{i}" for i in refs["component_ids"])
            print(f"  [增强查询] {enhanced}")
            results2 = self.retriever.retrieve(enhanced, top_k=20)
        else:
            results2 = []

        # 3. RRF 融合
        merged = self._rrf_fusion(results1, results2)

        # 4. 视觉页加权
        if "page" in refs:
            for item in merged:
                if item.get("page") == refs["page"]:
                    item["rrf_score"] *= 2.0
            merged.sort(key=lambda x: -x.get("rrf_score", 0))

        return merged[:top_k]

    def _rrf_fusion(self, results1, results2, k: int = 60):
        scores = {}
        chunk_map = {}
        for rank, item in enumerate(results1):
            key = item.get("chunk_id", item["content"][:30])
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            chunk_map[key] = item
        for rank, item in enumerate(results2):
            key = item.get("chunk_id", item["content"][:30])
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            if key not in chunk_map:
                chunk_map[key] = item
        sorted_keys = sorted(scores, key=lambda x: -scores[x])
        return [dict(chunk_map[k], rrf_score=scores[k]) for k in sorted_keys]


if __name__ == "__main__":
    with open("cn506_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    cmr = CrossModalRetriever(chunks)
    test_q = "在文件中第11页图3中，编号13的部件相对于编号12的部件的位置关系是？"
    results = cmr.retrieve(test_q, top_k=5)
    print(f"\n===== 检索结果 =====")
    for i, r in enumerate(results, 1):
        print(f"[Top-{i}] 第{r['page']}页 rrf={r.get('rrf_score', 0):.4f}")
        print(f"    {r['content'][:150]}")
