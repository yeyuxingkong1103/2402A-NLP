# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-混合检索任务
# 关联工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化 | 人工智能NLP-RAG-图像内容解析及检索优化 | 人工智能NLP-RAG-Query理解优化任务 | 人工智能NLP-RAG-混合检索任务
# 模块：fts —— 全文检索（倒排索引 + 布尔查询 + 短语匹配 + 模糊匹配 + 多字段）
# 说明：为工单 06《混合检索任务》提供「全文检索」这一路。
#   · 倒排索引：term -> {doc_id: 各字段词频}
#   · 布尔查询：支持 AND / OR / NOT（默认空格 = AND），必要时用小括号
#   · 短语匹配："引号内的短语" 要求按序相邻出现
#   · 模糊匹配：词尾加 ~ 做编辑距离近似（也支持前缀 *）
#   · 多字段：标题(title) / 正文(body) / 摘要(summary) 各自权重可调
#   打分沿用 BM25（按字段加权求和），保证与系统既有 BM25 路同源可比。

import math
import re
import difflib
from collections import defaultdict, Counter

from bm25 import tokenize

FIELD_WEIGHTS = {"title": 3.0, "body": 1.0, "summary": 1.5}


class InvertedIndex:
    """多字段倒排索引 + BM25 打分 + 布尔/短语/模糊查询。"""

    def __init__(self, k1=1.5, b=0.75, field_weights=None):
        self.k1, self.b = k1, b
        self.field_weights = dict(FIELD_WEIGHTS)
        if field_weights:
            self.field_weights.update(field_weights)
        self.docs = []                      # 原始文档（dict）
        self.tokens = []                    # list[dict[field -> list[term]]]
        self.flat = []                      # 去空白后的正文字符串（供短语匹配）
        self.postings = defaultdict(lambda: defaultdict(dict))  # term -> field -> {doc: tf}
        self.df = defaultdict(Counter)      # term -> field -> df
        self.field_len = defaultdict(list)  # field -> [len]
        self.N = 0
        self.vocab = set()

    # ---------- 构建 ----------
    def build(self, docs, fields=("title", "body", "summary")):
        self.docs, self.tokens, self.flat = [], [], []
        self.postings = defaultdict(lambda: defaultdict(dict))
        self.df = defaultdict(Counter)
        self.field_len = defaultdict(list)
        self.vocab = set()
        for d in docs:
            toks = {f: tokenize(str(d.get(f, "") or "")) for f in fields}
            i = len(self.docs)
            self.docs.append(d)
            self.tokens.append(toks)
            self.flat.append(re.sub(r"\s+", "", str(d.get("body", "") or "")))
            for f, tl in toks.items():
                self.field_len[f].append(len(tl))
                for t, tf in Counter(tl).items():
                    self.postings[t][f][i] = tf
                    self.df[t][f] += 1
                    self.vocab.add(t)
        self.N = len(self.docs)
        return self

    # ---------- 打分 ----------
    def _avgdl(self, f):
        ls = self.field_len.get(f) or [1]
        return sum(ls) / len(ls) or 1.0

    def _bm25(self, term, f, i):
        tf = self.postings.get(term, {}).get(f, {}).get(i, 0)
        if not tf:
            return 0.0
        n = self.df[term][f]
        idf = math.log(1 + (self.N - n + 0.5) / (n + 0.5))
        dl = self.field_len[f][i] or 1
        avg = self._avgdl(f)
        return idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / avg))

    def _score_term(self, term, i, w=1.0):
        # 单字权重降为 0.3：分词时给中文词补了单字，若不降权会压过真正的词
        if w == 1.0 and len(term) == 1 and re.fullmatch(r"[\u4e00-\u9fff]", term):
            w = 0.3
        return w * sum(self.field_weights.get(f, 1.0) * self._bm25(term, f, i)
                       for f in self.field_len)

    # ---------- 查询解析 ----------
    @staticmethod
    def parse_query(q):
        """返回 [(op, kind, value)]；op ∈ {AND, OR, NOT}，kind ∈ {term, phrase, prefix, fuzzy}。
        中文自然语言问句会被分词成多个 term 子句（默认 AND；检索端在 auto 模式下可退化为 OR）。"""
        q = (q or "").strip()
        if not q:
            return []
        parts = re.findall(r'"[^"]+"|\S+', q)
        clauses, pending = [], None

        def emit(kind, val, neg):
            nonlocal pending
            op = "NOT" if neg else (pending or "AND")
            pending = None
            clauses.append((op, kind, val))

        for p in parts:
            up = p.upper()
            if up in ("AND", "OR", "NOT"):
                pending = up
                continue
            neg = p.startswith("-") or p.startswith("!")
            if neg:
                p = p[1:]
            if len(p) >= 2 and p.startswith('"') and p.endswith('"'):
                emit("phrase", p[1:-1].strip(), neg)
            elif p.endswith("*") and len(p) > 1:
                emit("prefix", p[:-1], neg)
            elif p.endswith("~") and len(p) > 1:
                emit("fuzzy", p[:-1], neg)
            else:
                toks = list(dict.fromkeys(tokenize(p))) or [p.lower()]
                for t in toks:
                    emit("term", t, neg)
                    neg = False   # 同一片段内的后续词不再取 NOT
        return clauses

    def _expand_fuzzy(self, term, max_n=8):
        if term in self.vocab:
            return [term]
        close = difflib.get_close_matches(term, list(self.vocab), n=max_n, cutoff=0.7)
        return close or []

    def expand(self, kind, val):
        """把子句展开成参与打分的索引词（供 _match_docs 与打分共用）。"""
        val = val.lower() if kind in ("term", "prefix", "fuzzy") else val
        if kind == "prefix":
            return [t for t in self.vocab if t.startswith(val)]
        if kind == "fuzzy":
            return self._expand_fuzzy(val)
        if kind == "phrase":
            return list(dict.fromkeys(tokenize(val)))
        return list(dict.fromkeys(tokenize(val))) or [val]

    def _term_docs(self, term):
        s = set()
        for f, d in self.postings.get(term, {}).items():
            s |= set(d)
        return s

    def _match_docs(self, kind, val):
        """返回满足该子句的候选集合。"""
        val = val.lower() if kind in ("term", "prefix", "fuzzy") else val
        if kind == "prefix":
            s = set()
            for t, per in self.postings.items():
                if t.startswith(val):
                    for f, d in per.items():
                        s |= set(d)
            return s
        if kind == "fuzzy":
            s = set()
            for t in self._expand_fuzzy(val):
                for f, d in self.postings.get(t, {}).items():
                    s |= set(d)
            return s
        if kind == "phrase":
            uniq = list(dict.fromkeys(tokenize(val)))
            if not uniq:
                return set()
            base = None
            for t in uniq:
                s = set()
                for f, d in self.postings.get(t, {}).items():
                    s |= set(d)
                base = s if base is None else (base & s)
            needle = re.sub(r"\s+", "", val)
            return {i for i in (base or set()) if needle in self.flat[i]}
        # term：自由文本 → 先用分词拆成词，再取交集（等价于词级 AND）
        toks = list(dict.fromkeys(tokenize(val))) or [val]
        s = None
        for t in toks:
            sd = self._term_docs(t)
            s = sd if s is None else (s & sd)
        return s or set()

    # ---------- 检索 ----------
    def search(self, query, k=10, mode="boolean"):
        """mode: boolean（布尔，默认）/ any（OR 宽松）。返回 [(doc_idx, score)]。"""
        clauses = self.parse_query(query)
        if not clauses:
            return []
        result, matched_terms = None, []
        for op, kind, val in clauses:
            s = self._match_docs(kind, val)
            if op == "NOT":
                result = (set(range(self.N)) if result is None else result) - s
                continue
            if result is None:
                result = set(s)
            elif op == "OR" or mode == "any":
                result |= s
            else:
                result &= s
            matched_terms.extend(self.expand(kind, val))
        if result is None:
            return []
        scored = []
        for i in result:
            sc = sum(self._score_term(t, i) for t in set(matched_terms)) or 0.0
            if sc <= 0:      # 仅靠前缀/短语命中也给一个基础分
                sc = 0.05
            scored.append((i, sc))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


def build_index(chunks):
    """用系统 chunks 建全文索引：title=章节/表标题，body=正文，summary=首 80 字摘要。"""
    docs = []
    for c in chunks:
        summary = c.get("text", "")[:80]
        docs.append({"title": c.get("section") or "", "body": c.get("text", ""),
                     "summary": summary, "chunk_id": c.get("id")})
    return InvertedIndex().build(docs)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from parse import load_blocks          # noqa
    from clean import clean_blocks         # noqa
    from chunk import build_chunks         # noqa
    ch = build_chunks(clean_blocks(load_blocks()))
    idx = build_index(ch)
    print("docs:", idx.N, "vocab:", len(idx.vocab))
    for q in ["注册资本", '"国家科技进步一等奖"', "军用 AND 收入", "注册资~", "C4IS*"]:
        hits = idx.search(q, k=3)
        print("Q=%-24s -> %s" % (q, [(ch[i]["page"], round(s, 2)) for i, s in hits]))
