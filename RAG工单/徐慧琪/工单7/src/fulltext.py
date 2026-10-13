# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估
# （上游来源：工单06 混合检索；本工单复用于 ccf_competition 语料的功能测试与评估）
"""
工单编号：人工智能NLP-RAG-混合检索任务
（工单 PDF 原文：人工智能NLP-RAG-混合检索任务）

模块：全文检索（自研倒排索引）

为什么自研而不用 Whoosh/Elasticsearch：
  · 工单允许「Whoosh、Elasticsearch 或自研倒排索引」三选一；
  · 本机未安装 Whoosh / 没有 Docker，且禁用联网下载；
  · 招股书语料只有 8 千余块，自研**位置倒排索引**完全够用，且行为完全可控、
    可测试、可解释（演示视频要讲清"全文检索用的是什么技术"）。

索引结构（倒排表 = 词 → 文档列表，位置信息用于短语匹配）：
    postings[field][term] = { doc_idx: (pos1, pos2, ...) }
    doc_len[field][doc_idx] = 该字段词数
    df[field][term]         = 文档频率

字段（多字段检索，工单技术要求）：
    title   标题   —— heading_path 各级标题 + 表格标题
    body    正文   —— 块正文（text）
    summary 摘要   —— 表格列表头 / 父块开头（短摘要）

查询语法（工单要求：布尔查询、短语匹配、模糊匹配）：
    布尔   ：A AND B、A OR B、NOT C、括号分组 (A OR B) AND C、
             等价简写 +A -B；中文连接词「与/或/非」同样接受
    短语   ：「"募集资金投资项目"」—— 引号内词序**相邻**才算命中
    模糊   ：term~ 或 term~1 —— 编辑距离 ≤1 的词也命中（如"法定代理人"→"法定代表人"）
    字段限定：title:注册资本、body:军用领域、summary:持股比例

打分：BM25（k1=1.5, b=0.75），多字段加权求和（标题 2.5 / 正文 1.0 / 摘要 1.2），
     短语命中额外加成。命中结果返回原始 payload + fulltext_score + 命中字段明细。

约束：只读本地数据，零网络请求；索引可持久化到 data/index/fulltext.pkl。
"""
from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import math
import os
import pickle
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from src import config
from src.tokenizer import tokenize, tokenize_query

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
FIELDS = ("title", "body", "summary")
FIELD_WEIGHTS = {"title": 2.5, "body": 1.0, "summary": 1.2}
BM25_K1 = 1.5
BM25_B = 0.75
PHRASE_BOOST = 1.6          # 短语命中加成（在成分词 BM25 之上）
MAX_FUZZY_LEN = 12          # 参与模糊匹配的词长上限（控制内存）

_OPERATORS = {"AND", "OR", "NOT", "与", "或", "非"}
_FIELD_ALIASES = {
    "title": "title", "标题": "title",
    "body": "body", "正文": "body",
    "summary": "summary", "摘要": "summary",
}


# ---------------------------------------------------------------------------
# 查询解析（词法 → 语法树）
# ---------------------------------------------------------------------------
@dataclass
class Term:
    """叶子节点：一个词项（或短语）。"""
    text: str                       # 原始词面
    terms: list[str]                # 分词结果
    phrase: bool = False            # 引号短语
    field: str | None = None        # 字段限定（title/body/summary）
    fuzzy: int = 0                  # 模糊编辑距离（0 表示精确）


@dataclass
class Not:
    node: object


@dataclass
class And:
    left: object
    right: object


@dataclass
class Or:
    left: object
    right: object


_QUERY_TOKEN_RE = re.compile(
    r'"(?P<phrase>[^"]+)"'                       # 短语
    r'|(?P<field>[A-Za-z]+):(?P<fvalue>"[^"]+"|\S+)'   # 字段限定
    r'|(?P<op>AND|OR|NOT)'                       # 英文操作符（大写）
    r'|(?P<cnop>[与或非])'                        # 中文操作符
    r'|(?P<fuzzy>[^\s()"]+~[0-9]?)'               # 模糊词
    r'|(?P<word>[^\s()"]+)'                      # 普通词
    r'|(?P<lparen>\()|(?P<rparen>\))'
)


def parse_query(query: str) -> object | None:
    """把查询串解析成语法树；无法解析/无有效词项时返回 None。

    支持的写法：
        注册资本 AND 法定代表人
        "军用领域" OR 军用收入
        (武汉博润 OR 上海博润) AND 关联方
        NOT 招股说明书
        法定代理人~            （模糊，编辑距离 1）
        title:组织结构图
    """
    raw = (query or "").strip()
    if not raw:
        return None

    tokens: list[tuple[str, object]] = []
    for m in _QUERY_TOKEN_RE.finditer(raw):
        if m.group("phrase") is not None:
            text = m.group("phrase")
            # 短语必须用**完整分词**（不丢停用词）：位置相邻判定要求查询侧词序列
            # 与索引侧一致。实测教训：用 tokenize_query 会把 "存在控制关系的关联方"
            # 里的 "的" 丢掉，词位置不再相邻，短语永远匹配不上。
            tokens.append(("term", Term(text=text, terms=tokenize(text),
                                        phrase=True)))
        elif m.group("field") is not None:
            fname = _FIELD_ALIASES.get(m.group("field").lower())
            fval = m.group("fvalue")
            if fname is None:              # 不是字段名 → 当作普通词
                tokens.append(("term", Term(text=m.group(0),
                                            terms=tokenize_query(m.group(0)))))
                continue
            if fval.startswith('"') and fval.endswith('"'):
                text = fval[1:-1]
                tokens.append(("term", Term(text=text, terms=tokenize(text),
                                            phrase=True, field=fname)))
            else:
                tokens.append(("term", Term(text=fval, terms=tokenize_query(fval),
                                            field=fname)))
        elif m.group("op") is not None:
            tokens.append(("op", m.group("op")))
        elif m.group("cnop") is not None:
            cn = m.group("cnop")
            tokens.append(("op", {"与": "AND", "或": "OR", "非": "NOT"}[cn]))
        elif m.group("fuzzy") is not None:
            text = m.group("fuzzy").rstrip("~0123456789")
            eds = m.group("fuzzy").split("~")[1]
            if not text:
                continue
            # 模糊词的词面是用户**整体指定**的一个词（如 "法定代理人~"），
            # 不能交给 jieba 再切碎 —— 实测：切碎后 "注册资木" 变成
            # ["注册","资","木"]，模糊扩展作用在碎片上，永远命不中 "注册资本"。
            terms = [text] if (text and " " not in text) else tokenize_query(text)
            tokens.append(("term", Term(text=text, terms=terms,
                                        fuzzy=int(eds) if eds else 1)))
        elif m.group("word") is not None:
            w = m.group("word")
            if w.upper() in _OPERATORS and w.isascii():
                tokens.append(("op", w.upper()))
                continue
            # 前缀简写：+词 = 必须，-词 = 排除（沿用搜索引擎习惯）
            if w.startswith("+") and len(w) > 1:
                tokens.append(("op", "AND"))
                w = w[1:]
            elif w.startswith("-") and len(w) > 1:
                tokens.append(("op", "NOT"))
                w = w[1:]
            terms = tokenize_query(w) or tokenize(w)
            if terms:
                tokens.append(("term", Term(text=w, terms=terms)))
        elif m.group("lparen") is not None:
            tokens.append(("(", "("))
        elif m.group("rparen") is not None:
            tokens.append((")", ")"))

    if not any(kind == "term" for kind, _ in tokens):
        return None
    parser = _Parser(tokens)
    node = parser.parse_expr()
    return node


class _Parser:
    """递归下降解析：expr := or_expr ; or_expr := and_expr (OR and_expr)* ;
    and_expr := unary (AND? unary)* ; unary := NOT unary | '(' expr ')' | term。

    相邻词项之间没有显式操作符时按 **OR** 组合（搜索引擎默认语义），
    分数为各词得分之和 —— 命中词越多的文档分越高，自然向"隐性 AND"倾斜。
    """

    def __init__(self, tokens: list[tuple[str, object]]):
        self.tokens = tokens
        self.i = 0

    def _peek(self) -> tuple[str, object] | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def _next(self) -> tuple[str, object] | None:
        tok = self._peek()
        self.i += 1
        return tok

    def parse_expr(self) -> object | None:
        return self._parse_or()

    def _parse_or(self) -> object | None:
        node = self._parse_and()
        while True:
            tok = self._peek()
            if tok and tok[0] == "op" and tok[1] == "OR":
                self._next()
                rhs = self._parse_and()
                if rhs is not None:
                    node = rhs if node is None else Or(node, rhs)
            else:
                return node

    def _parse_and(self) -> object | None:
        node = self._parse_unary()
        while True:
            tok = self._peek()
            if tok is None or tok[0] == ")":
                return node
            if tok[0] == "op" and tok[1] in ("AND", "OR"):
                if tok[1] == "OR":
                    return node            # 交给上层 _parse_or
                self._next()
                rhs = self._parse_unary()
                if rhs is not None:
                    node = rhs if node is None else And(node, rhs)
            elif tok[0] == "term" or tok[0] == "(":
                # 隐式 OR：相邻词项
                rhs = self._parse_unary()
                if rhs is not None:
                    node = rhs if node is None else Or(node, rhs)
            else:
                return node

    def _parse_unary(self) -> object | None:
        tok = self._peek()
        if tok is None:
            return None
        if tok[0] == "op" and tok[1] == "NOT":
            self._next()
            inner = self._parse_unary()
            return Not(inner) if inner is not None else None
        if tok[0] == "(":
            self._next()
            inner = self._parse_or()
            nxt = self._peek()
            if nxt and nxt[0] == ")":
                self._next()
            return inner
        if tok[0] == "term":
            self._next()
            return tok[1]
        self._next()
        return None


# ---------------------------------------------------------------------------
# 模糊匹配（SymSpell 式删除索引，编辑距离 ≤1）
# ---------------------------------------------------------------------------
def _deletes(term: str, ) -> set[str]:
    out = {term}
    if 2 < len(term) <= MAX_FUZZY_LEN:
        for i in range(len(term)):
            out.add(term[:i] + term[i + 1:])
    return out


def _levenshtein_le1(a: str, b: str) -> bool:
    """判断编辑距离是否 ≤1（O(n) 早停，不建矩阵）。"""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:                       # 只允许替换
        diff = 0
        for x, y in zip(a, b):
            if x != y:
                diff += 1
                if diff > 1:
                    return False
        return diff == 1
    if la > lb:                        # 保证 a 是短的那个
        a, b = b, a
        la, lb = lb, la
    i = j = 0
    skipped = False
    while i < la and j < lb:           # 只允许一次插入
        if a[i] == b[j]:
            i += 1
            j += 1
        elif skipped:
            return False
        else:
            skipped = True
            j += 1
    return True


# ---------------------------------------------------------------------------
# 倒排索引
# ---------------------------------------------------------------------------
@dataclass
class InvertedIndex:
    """位置倒排索引 + BM25 打分。"""

    n_docs: int = 0
    # postings[field][term] = {doc_idx: (positions...)}
    postings: dict[str, dict[str, dict[int, tuple[int, ...]]]] = field(default_factory=dict)
    doc_len: dict[str, dict[int, int]] = field(default_factory=dict)
    avgdl: dict[str, float] = field(default_factory=dict)
    # 文档侧元数据：与 payloads 一一对应，持久化时只存定位键
    doc_keys: list[str] = field(default_factory=list)
    _fuzzy_map: dict[str, set[str]] = field(default_factory=dict)
    built_at: float = 0.0
    payloads: list[dict] = field(default_factory=list)

    # -- 构建 ---------------------------------------------------------------
    def build(self, payloads: Sequence[dict]) -> "InvertedIndex":
        t0 = time.time()
        postings: dict[str, dict[str, dict[int, tuple[int, ...]]]] = {
            f: {} for f in FIELDS}
        doc_len: dict[str, dict[int, int]] = {f: {} for f in FIELDS}
        doc_keys: list[str] = []

        for idx, p in enumerate(payloads):
            doc_keys.append(_doc_key(p))
            for fname, text in _doc_fields(p).items():
                toks = tokenize(text)
                doc_len[fname][idx] = len(toks)
                pos: dict[str, list[int]] = {}
                for i, t in enumerate(toks):
                    pos.setdefault(t, []).append(i)
                fpost = postings[fname]
                for t, positions in pos.items():
                    fpost.setdefault(t, {})[idx] = tuple(positions)

        self.n_docs = len(payloads)
        self.postings = postings
        self.doc_len = doc_len
        self.doc_keys = doc_keys
        self.payloads = list(payloads)
        self.avgdl = {
            f: (sum(doc_len[f].values()) / max(len(doc_len[f]), 1)) for f in FIELDS
        }
        # 模糊索引：只对不太长的词建删除表（控制内存）
        fmap: dict[str, set[str]] = {}
        for f in FIELDS:
            for term in postings[f]:
                if 2 < len(term) <= MAX_FUZZY_LEN:
                    for d in _deletes(term):
                        fmap.setdefault(d, set()).add(term)
        self._fuzzy_map = fmap
        self.built_at = time.time()
        return self

    # -- 统计 ---------------------------------------------------------------
    def stats(self) -> dict:
        vocab = {f: len(self.postings.get(f, {})) for f in FIELDS}
        return {
            "docs": self.n_docs,
            "vocab": vocab,
            "avgdl": {f: round(v, 1) for f, v in self.avgdl.items()},
            "built_at": self.built_at,
        }

    # -- 打分 ---------------------------------------------------------------
    def _bm25(self, field_name: str, term: str, doc_idx: int) -> float:
        fpost = self.postings[field_name].get(term)
        if not fpost:
            return 0.0
        tf = len(fpost.get(doc_idx, ()))
        if tf <= 0:
            return 0.0
        df = len(fpost)
        n = max(self.n_docs, 1)
        idf = max(0.0, (n - df + 0.5) / (df + 0.5))
        idf = float(math.log(1.0 + idf))
        dl = self.doc_len[field_name].get(doc_idx, 0)
        avg = self.avgdl.get(field_name) or 1.0
        denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / avg)
        return idf * tf * (BM25_K1 + 1) / denom * FIELD_WEIGHTS[field_name]

    def _expand(self, term: str, fuzzy: int) -> list[str]:
        """模糊扩展：返回索引里真实存在的、与 term 编辑距离 ≤fuzzy 的词。"""
        if fuzzy <= 0:
            return [term]
        out: list[str] = []
        seen: set[str] = set()
        for d in _deletes(term):
            for cand in self._fuzzy_map.get(d, ()):
                if cand in seen:
                    continue
                seen.add(cand)
                if _levenshtein_le1(term, cand):
                    out.append(cand)
        if term not in seen and any(term in self.postings[f] for f in FIELDS):
            out.append(term)
        return out[:8]                 # 单个词最多扩展 8 个候选，防止查询爆炸

    def _search_term(self, node: Term, fields: Iterable[str]) -> dict[int, float]:
        """单词项/短语项 → {doc_idx: score}。"""
        target_fields = [node.field] if node.field else list(fields)
        scores: dict[int, float] = {}

        if node.phrase and len(node.terms) >= 2:
            # 短语匹配：所有成分词在同一字段的**相邻位置**出现。
            # 分数口径与普通词项一致（各字段 BM25 求和），整体再乘短语加成 ——
            # 若各字段取 max，短语分数会低于同词的不带引号查询（实测），
            # 那是口径不一致造成的假象，不是短语更差。
            for f in target_fields:
                cand: set[int] | None = None
                for t in node.terms:
                    docs = set(self.postings[f].get(t, {}))
                    cand = docs if cand is None else (cand & docs)
                    if not cand:
                        break
                if not cand:
                    continue
                for d in cand:
                    pos_lists = [self.postings[f].get(t, {}).get(d, ()) for t in node.terms]
                    if _adjacent(pos_lists):
                        base = sum(self._bm25(f, t, d) for t in node.terms)
                        scores[d] = scores.get(d, 0.0) + base * PHRASE_BOOST
            return scores

        terms = node.terms or tokenize_query(node.text)
        for t in terms:
            variants = self._expand(t, node.fuzzy)
            for f in target_fields:
                for v in variants:
                    fpost = self.postings[f].get(v)
                    if not fpost:
                        continue
                    for d in fpost:
                        s = self._bm25(f, v, d)
                        if s > 0:
                            scores[d] = scores.get(d, 0.0) + s
        return scores

    def _eval(self, node: object, fields: list[str]) -> dict[int, float]:
        if node is None:
            return {}
        if isinstance(node, Term):
            return self._search_term(node, fields)
        if isinstance(node, Not):
            inner = self._eval(node.node, fields)
            return {d: 0.0 for d in range(self.n_docs) if d not in inner}
        if isinstance(node, And):
            left = self._eval(node.left, fields)
            right = self._eval(node.right, fields)
            return {d: left[d] + right[d] for d in left.keys() & right.keys()}
        if isinstance(node, Or):
            left = self._eval(node.left, fields)
            right = self._eval(node.right, fields)
            out = dict(left)
            for d, s in right.items():
                out[d] = out.get(d, 0.0) + s
            return out
        return {}

    # -- 查询 ---------------------------------------------------------------
    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """执行全文检索，返回 payload 列表（附 fulltext_score 与命中信息）。"""
        node = parse_query(query)
        if node is None:
            return []
        t0 = time.time()
        scored = self._eval(node, list(FIELDS))
        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)

        out: list[dict] = []
        for doc_idx, score in ranked:
            if score <= 0 or len(out) >= top_k:
                break
            if doc_idx >= len(self.payloads):
                continue
            p = dict(self.payloads[doc_idx])
            p["fulltext_score"] = float(score)
            out.append(p)
        elapsed_ms = round((time.time() - t0) * 1000, 2)
        for p in out:
            p["fulltext_ms"] = elapsed_ms
        return out


def _doc_key(p: dict) -> str:
    return f"{p.get('source', '')}#{p.get('chunk_id', '')}"


def _doc_fields(p: dict) -> dict[str, str]:
    """payload → 三个可检索字段的文本。"""
    title_bits = list(p.get("heading_path") or [])
    if p.get("table_caption"):
        title_bits.append(str(p["table_caption"]))
    if p.get("org_node"):
        title_bits.append(str(p["org_node"]))
    title = " ".join(str(b) for b in title_bits)

    body = p.get("text", "") or ""

    if p.get("block_type") in ("table_row", "table_header") and p.get("table_header"):
        summary = str(p["table_header"])
    else:
        parent = p.get("parent_text") or ""
        summary = parent[:160] if parent and parent != body else body[:160]
    return {"title": title, "body": body, "summary": summary}


def _adjacent(pos_lists: Sequence[Sequence[int]]) -> bool:
    """多词短语判定：存在一条路径使各词位置严格递增且相邻。"""
    first = pos_lists[0]
    if not first:
        return False
    for start in first:
        cur = start
        ok = True
        for pl in pos_lists[1:]:
            nxt = None
            for cand in pl:
                if cand == cur + 1:
                    nxt = cand
                    break
                if cand > cur + 1:
                    break
            if nxt is None:
                ok = False
                break
            cur = nxt
        if ok:
            return True
    return False


# ---------------------------------------------------------------------------
# 全局单例（懒加载 + 持久化）
# ---------------------------------------------------------------------------
_index: InvertedIndex | None = None
_lock = threading.Lock()

_INDEX_PATH = os.path.join(config.DATA_DIR, "index", "fulltext.pkl")


def _load_payloads(name: str | None = None) -> list[dict]:
    """取全量 payload：优先 Qdrant（单一数据源），失败回退 chunks.json。"""
    try:
        from src import vector_store
        payloads = vector_store.scroll_all(name=name)
        if payloads:
            return payloads
    except Exception as exc:  # noqa: BLE001
        print(f"[fulltext] Qdrant 取数失败，回退 chunks.json：{exc}", flush=True)
    path = os.path.join(config.CHUNK_DIR, "chunks.json")
    import json
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build(force: bool = False, name: str | None = None,
          persist: bool = True) -> InvertedIndex:
    """构建（或复用）倒排索引。线程安全。"""
    global _index
    if _index is not None and not force:
        return _index
    with _lock:
        if _index is not None and not force:
            return _index
        payloads = _load_payloads(name)
        if not payloads:
            raise RuntimeError("语料为空：请先执行 python build_index.py")
        t0 = time.time()
        idx = InvertedIndex().build(payloads)
        print(f"[fulltext] 倒排索引就绪：{idx.n_docs} 篇 / "
              f"词表 {sum(len(idx.postings[f]) for f in FIELDS)} 词 / "
              f"用时 {time.time() - t0:.1f}s", flush=True)
        _index = idx
        if persist:
            try:
                os.makedirs(os.path.dirname(_INDEX_PATH), exist_ok=True)
                with open(_INDEX_PATH, "wb") as fh:
                    pickle.dump({"postings": idx.postings, "doc_len": idx.doc_len,
                                 "avgdl": idx.avgdl, "doc_keys": idx.doc_keys,
                                 "payloads": idx.payloads, "n_docs": idx.n_docs,
                                 "built_at": idx.built_at}, fh, protocol=4)
                print(f"[fulltext] 索引已持久化：{_INDEX_PATH}"
                      f"（{os.path.getsize(_INDEX_PATH) / 1e6:.1f} MB）", flush=True)
            except Exception as exc:  # noqa: BLE001 —— 持久化失败不影响检索
                print(f"[fulltext] 索引持久化失败（不影响检索）：{exc}", flush=True)
        return _index


def refresh(name: str | None = None) -> InvertedIndex:
    """建库后强制重建。"""
    return build(force=True, name=name)


def search(query: str, top_k: int | None = None,
           name: str | None = None) -> list[dict]:
    """全文检索对外入口。"""
    idx = build(name=name)
    return idx.search(query, top_k=top_k or config.FULLTEXT_TOP_K)


def stats() -> dict:
    idx = build()
    s = idx.stats()
    s["index_file"] = _INDEX_PATH
    s["persisted"] = os.path.isfile(_INDEX_PATH)
    return s


def _cli() -> None:
    import sys

    q = " ".join(sys.argv[1:]) or '"军用领域" AND 收入'
    t0 = time.time()
    hits = search(q, top_k=5)
    print(f"查询：{q}\n命中 {len(hits)} 条（{time.time() - t0:.2f}s）")
    for i, h in enumerate(hits, start=1):
        print(f"\n[{i}] score={h['fulltext_score']:.3f} "
              f"第{int(h.get('page_idx', 0)) + 1}页 {h.get('block_type')} "
              f"{' › '.join(h.get('heading_path') or [])}")
        print("   ", (h.get("text") or "")[:150].replace("\n", " "))


if __name__ == "__main__":  # 冒烟：python -m src.fulltext '"军用领域" AND 收入'
    bootstrap.run_with_large_stack(_cli)
