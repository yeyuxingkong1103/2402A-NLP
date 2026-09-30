# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
模块：全文检索（BM25 增强版）
功能：基于关键词的全文检索，支持布尔查询、短语匹配
"""

import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from jieba import cut


class FulltextSearcher:
    """全文检索器（BM25 + 布尔查询）"""

    def __init__(self):
        self.chunks = []
        self.bm25 = None
        self.tokenized_corpus = []

    def build_index(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        texts = [c["content"] for c in chunks]
        self.tokenized_corpus = [list(cut(t)) for t in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"✅ 全文索引构建完成：{len(chunks)} 块")

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        if self.bm25 is None:
            return []
        tokens = list(cut(query))
        scores = self.bm25.get_scores(tokens)
        idx = scores.argsort()[::-1][:top_k]
        results = []
        for i in idx:
            if scores[i] > 0:
                item = dict(self.chunks[i])
                item["fulltext_score"] = float(scores[i])
                results.append(item)
        return results

    def boolean_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        results = []
        if " AND " in query.upper():
            keywords = re.split(r"\s+AND\s+", query, flags=re.IGNORECASE)
            require_all = True
        elif " OR " in query.upper():
            keywords = re.split(r"\s+OR\s+", query, flags=re.IGNORECASE)
            require_all = False
        else:
            keywords = [query]
            require_all = False
        keywords = [list(cut(k.strip())) for k in keywords]
        for i, tokens in enumerate(self.tokenized_corpus):
            hits = sum(1 for kws in keywords if any(kw in tokens for kw in kws))
            matched = (hits == len(keywords)) if require_all else (hits > 0)
            if matched:
                item = dict(self.chunks[i])
                item["boolean_score"] = float(hits)
                results.append(item)
        results.sort(key=lambda x: -x["boolean_score"])
        return results[:top_k]

    def phrase_search(self, phrase: str, top_k: int = 20) -> List[Dict[str, Any]]:
        results = []
        for chunk in self.chunks:
            count = chunk["content"].count(phrase)
            if count > 0:
                item = dict(chunk)
                item["phrase_score"] = float(count)
                results.append(item)
        results.sort(key=lambda x: -x["phrase_score"])
        return results[:top_k]


if __name__ == "__main__":
    from pdf_parser import PDFParser
    parser = PDFParser("./data/招股说明书1.pdf")
    pages = parser.extract_text()
    chunks = parser.chunk_text(pages, chunk_size=300, overlap=80)
    searcher = FulltextSearcher()
    searcher.build_index(chunks)

    print("\n===== BM25 检索：'注册资本' =====")
    for r in searcher.search("注册资本", top_k=3):
        print(f"  第{r['page']}页 分数={r['fulltext_score']:.2f}")

    print("\n===== 布尔查询：'注册资本 AND 5,225' =====")
    for r in searcher.boolean_search("注册资本 AND 5,225", top_k=3):
        print(f"  第{r['page']}页 命中数={r['boolean_score']:.0f}")
