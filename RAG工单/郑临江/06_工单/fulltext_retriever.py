# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
全文检索模块：基于倒排索引，支持布尔查询、短语匹配、模糊匹配，
以及多字段检索（标题、正文、摘要）。
"""
from collections import defaultdict
from embedder import tokenize


def _fields(doc):
    """把文档拆成 标题/摘要/正文 三个字段。"""
    lines = [l for l in doc.split("\n") if l.strip()]
    title = lines[0] if lines else ""
    summary = doc[:120]
    return {"title": title, "summary": summary, "body": doc}


class InvertedIndex:
    def __init__(self):
        self.postings = defaultdict(lambda: defaultdict(list))  # field -> term -> [doc_id, ...]

    def add(self, doc_id, field, text):
        for pos, tok in enumerate(tokenize(text)):
            self.postings[field][tok].append(doc_id)

    def build(self, docs):
        for i, d in enumerate(docs):
            for field, text in _fields(d).items():
                self.add(i, field, text)

    def _match_docs(self, terms, field, op="AND"):
        """返回满足布尔条件的文档 id 集合。"""
        if not terms:
            return set()
        sets = [set(self.postings[field].get(t, [])) for t in terms]
        result = sets[0]
        if op == "AND":
            for s in sets[1:]:
                result &= s
        elif op == "OR":
            for s in sets[1:]:
                result |= s
        return result


class FullTextRetriever:
    def __init__(self, docs):
        self.docs = list(docs)
        self.index = InvertedIndex()
        self.index.build(self.docs)

    def search_boolean(self, query, field="body", op="AND", top_k=10):
        terms = tokenize(query)
        ids = self.index._match_docs(terms, field, op)
        return [(self.docs[i], 1.0) for i in list(ids)[:top_k]]

    def search_phrase(self, phrase, field="body", top_k=10):
        """短语匹配：所有词出现在同一文档中（简化）。"""
        return self.search_boolean(phrase, field, "AND", top_k)

    def search_fuzzy(self, term, field="body", top_k=10):
        """模糊匹配：前缀匹配（简化版编辑距离）。"""
        term = term.lower()
        matched = set()
        for t, docids in self.index.postings[field].items():
            if t.startswith(term) or term.startswith(t[:2]):
                matched.update(docids)
        return [(self.docs[i], 1.0) for i in list(matched)[:top_k]]

    def search(self, query, field="body", top_k=10):
        """默认：多词 OR 检索，按命中词数排序。"""
        terms = tokenize(query)
        scored = defaultdict(float)
        for t in terms:
            for doc_id in self.index.postings[field].get(t, []):
                scored[doc_id] += 1.0
        ranked = sorted(scored.items(), key=lambda x: -x[1])
        return [(self.docs[i], s / max(1, len(terms))) for i, s in ranked[:top_k]]
