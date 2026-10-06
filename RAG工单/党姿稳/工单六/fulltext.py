# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
全文检索模块（倒排索引）：
  - 多字段索引：标题(title) / 正文(body) / 摘要(summary)，支持字段级检索与字段加权
  - 布尔查询：AND / OR / NOT（支持括号）
  - 短语匹配："xxx yyy" 引号短语，按位置连续性匹配
  - 模糊匹配：word~ / 编辑距离(Levenshtein)，容错检索
  - 结果按 BM25 打分，返回 (块索引, 分数)
"""
import re
import math
from collections import defaultdict


def tokenize(text):
    """中英文混合分词：英文/数字按词，中文按 2-gram（无需外部分词器）"""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    zh = re.findall(r"[一-鿿]+", text)
    for seg in zh:
        if len(seg) == 1:
            tokens.append(seg)
        for i in range(len(seg) - 1):
            tokens.append(seg[i:i + 2])
    return tokens


def edit_distance(a, b, max_d=2):
    """带阈值剪枝的编辑距离，超过 max_d 直接返回 max_d+1"""
    if abs(len(a) - len(b)) > max_d:
        return max_d + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > max_d:
            return max_d + 1
        prev = cur
    return prev[-1]


class FullTextIndex:
    """多字段倒排索引 + 布尔/短语/模糊检索"""

    def __init__(self, chunks, field_weights=None, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.field_weights = field_weights or {"title": 2.0, "summary": 1.2, "body": 1.0}
        self.fields = list(self.field_weights.keys())
        # field -> token -> {doc_id: 词频}
        self.postings = {f: defaultdict(dict) for f in self.fields}
        # field -> token -> [(doc_id, position), ...]
        self.positions = {f: defaultdict(list) for f in self.fields}
        self.doc_len = {f: [] for f in self.fields}
        self.vocab = set()
        self._build()

    # ---------- 字段切分 ----------
    @staticmethod
    def _fields_of(chunk):
        """从块中抽取 标题 / 摘要 / 正文 三个字段"""
        text = chunk.get("text", "")
        title = chunk.get("caption") or chunk.get("title") or ""
        summary = text[:80]
        return {"title": title, "summary": summary, "body": text}

    def _build(self):
        for doc_id, c in enumerate(self.chunks):
            fs = self._fields_of(c)
            for f in self.fields:
                toks = tokenize(fs[f])
                self.doc_len[f].append(len(toks))
                for pos, t in enumerate(toks):
                    self.postings[f][t][doc_id] = self.postings[f][t].get(doc_id, 0) + 1
                    self.positions[f][t].append((doc_id, pos))
                    self.vocab.add(t)
        self.avg_len = {f: (sum(self.doc_len[f]) / max(len(self.doc_len[f]), 1)) for f in self.fields}
        self.N = len(self.chunks)

    # ---------- 打分 ----------
    def _bm25_field(self, field_name, tokens, doc_id):
        """某字段内单个文档的 BM25 分数"""
        score = 0.0
        dl = self.doc_len[field_name][doc_id] or 1
        for t in tokens:
            post = self.postings[field_name].get(t)
            if not post or doc_id not in post:
                continue
            df = len(post)
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            tf = post[doc_id]
            score += idf * (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * dl / self.avg_len[field_name]))
        return score

    def _score_docs(self, tokens, doc_ids, phrase_tokens=None):
        """对候选文档集合按多字段加权 BM25 打分"""
        out = []
        for d in doc_ids:
            s = sum(self.field_weights[f] * self._bm25_field(f, tokens, d) for f in self.fields)
            if phrase_tokens:
                s += 2.0 * self._phrase_score(phrase_tokens, d)
            out.append((d, s))
        return sorted(out, key=lambda x: x[1], reverse=True)

    # ---------- 短语匹配 ----------
    def _phrase_score(self, phrase_tokens, doc_id):
        """统计短语在文档中连续出现的次数（跨字段），作为加分项"""
        cnt = 0
        for f in self.fields:
            if not phrase_tokens:
                continue
            head = phrase_tokens[0]
            head_pos = [p for (d, p) in self.positions[f].get(head, []) if d == doc_id]
            for p0 in head_pos:
                ok = True
                for k, t in enumerate(phrase_tokens[1:], 1):
                    if (doc_id, p0 + k) not in self.positions[f].get(t, []):
                        ok = False
                        break
                if ok:
                    cnt += 1
        return cnt

    # ---------- 模糊匹配 ----------
    def _expand_fuzzy(self, token, max_d=1):
        """把 token~ 扩展为词表中编辑距离 <= max_d 的所有词"""
        cands = [token]
        for v in self.vocab:
            if abs(len(v) - len(token)) > max_d:
                continue
            if edit_distance(token, v, max_d) <= max_d:
                cands.append(v)
        return set(cands)

    # ---------- 词项解析 ----------
    def _expand(self, term):
        """term -> token 集合；支持 term~ 模糊、term* 前缀通配"""
        if term.endswith("~"):
            return self._expand_fuzzy(term[:-1], 1)
        if term.endswith("*"):
            pre = term[:-1]
            return {v for v in self.vocab if v.startswith(pre)}
        if term in self.vocab:
            return {term}
        # 未登录词：中文词尝试按 2-gram 拆分
        return set(tokenize(term))

    def _match_docs(self, term):
        """单个词项（已展开）命中的文档集合"""
        toks = self._expand(term)
        docs = set()
        for t in toks:
            for f in self.fields:
                docs |= set(self.postings[f].get(t, {}).keys())
        return docs

    # ---------- 布尔查询解析 ----------
    def _parse(self, query):
        """把查询串解析为 (表达式列表, 短语列表)；支持 AND/OR/NOT/括号/引号短语"""
        phrases = re.findall(r'"([^"]+)"', query)
        q = re.sub(r'"[^"]+"', " PHRASE ", query)
        q = q.replace("（", "(").replace("）", ")")
        q = re.sub(r"(\S+)\s*(AND|OR|NOT|and|or|not)\s*(\S+)", r"\1 \2 \3", q)
        q = re.sub(r"\bAND\b|\band\b|且|并且", " AND ", q)
        q = re.sub(r"\bOR\b|\bor\b|或", " OR ", q)
        q = re.sub(r"\bNOT\b|\bnot\b|\bNO\b|非|不包含", " NOT ", q)
        q = q.replace("(", " ( ").replace(")", " ) ")
        return q.split(), phrases

    def _eval(self, tokens, phrases):
        """递归下降求值布尔表达式，返回命中文档集合"""
        pos = [0]
        phrase_vals = [set() for _ in phrases]

        def phrase_docs(i):
            if not phrase_vals[i]:
                toks = tokenize(phrases[i])
                docs = set()
                for d in range(self.N):
                    if self._phrase_score(toks, d) > 0:
                        docs.add(d)
                phrase_vals[i] = docs
            return phrase_vals[i]

        def peek():
            return tokens[pos[0]] if pos[0] < len(tokens) else None

        def parse_primary():
            t = peek()
            if t == "(":
                pos[0] += 1
                r = parse_or()
                if peek() == ")":
                    pos[0] += 1
                return r
            if t == "PHRASE":
                pos[0] += 1
                idx = len([1 for x in tokens[:pos[0]] if x == "PHRASE"]) - 1
                return phrase_docs(min(idx, len(phrases) - 1))
            pos[0] += 1
            return self._match_docs(t) if t else set()

        def parse_not():
            if peek() == "NOT":
                pos[0] += 1
                return set(range(self.N)) - parse_not()
            return parse_primary()

        def parse_and():
            left = parse_not()
            while peek() == "AND":
                pos[0] += 1
                left = left & parse_not()
            return left

        def parse_or():
            left = parse_and()
            while peek() == "OR":
                pos[0] += 1
                left = left | parse_and()
            return left

        return parse_or()

    # ---------- 对外接口 ----------
    def search(self, query, top_k=5, mode="auto"):
        """
        全文检索入口。
        mode: auto（自动识别布尔/短语）| boolean | phrase | fuzzy
        返回 [(块索引, 分数), ...]
        """
        if mode == "fuzzy":
            raw = tokenize(query)
            docs, score_tokens = set(), []
            for t in raw:
                expanded = self._expand(t + "~")     # 模糊展开：编辑距离 ≤1 的词
                score_tokens.extend(expanded)
                for e in expanded:
                    docs |= self._match_docs(e)
            return self._score_docs(list(set(score_tokens)), docs)[:top_k]

        tokens, phrases = self._parse(query)
        if mode == "phrase" and not phrases:
            phrases = [query]
            tokens = ["PHRASE"]
        doc_ids = self._eval(tokens, phrases)

        # 打分词项：所有非逻辑词
        score_tokens = []
        for t in tokens:
            if t in ("AND", "OR", "NOT", "(", ")", "PHRASE"):
                continue
            score_tokens.extend(self._expand(t))
        for p in phrases:
            score_tokens.extend(tokenize(p))
        if not doc_ids and score_tokens:
            # 布尔结果为空时降级为 OR 检索，保证召回
            for t in score_tokens:
                doc_ids |= self._match_docs(t)
        return self._score_docs(list(score_tokens), doc_ids)[:top_k]


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    from kb import load_all_chunks
    allc, _, _ = load_all_chunks()
    idx = FullTextIndex(allc)
    for q in ["法定代表人 AND 兴图新科", '"国家科技进步一等奖"', "军拥 收入", "资产负寨率"]:
        print(f"\n查询: {q}")
        for i, (d, s) in enumerate(idx.search(q, 3), 1):
            print(f"  [{i}] 页{allc[d]['page']} 分数{s:.3f}: {allc[d]['text'][:80]}")
