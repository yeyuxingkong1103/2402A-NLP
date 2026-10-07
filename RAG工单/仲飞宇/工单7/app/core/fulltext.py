# 工单编号：人工智能NLP-RAG-混合检索任务
# 工单06 - 混合检索（向量检索 / 全文检索 / 混合检索）
"""
全文检索：应用层倒排索引 + BM25F + 布尔/短语/模糊/多字段查询。

【为什么不用 Milvus 的稀疏向量做全文检索】
工单06 对全文检索逐条点名了四种能力：**布尔查询、短语匹配、模糊匹配、多字段**。
Milvus 的稀疏向量只能表达"一袋词"（bag of terms）并按内积打分 ——
它既不能说 AND/NOT，也不知道词的位置（做不了短语），更做不了模糊扩展。
所以这四种能力必须由一个真正的倒排索引来提供。

Milvus 的稀疏路（工单01 建表时预留的 `sparse` 字段）没有浪费：它作为
**"向量库原生"混合检索**保留（`--fusion-impl milvus`），与本模块形成对照。
两者共用 `app/core/text_analysis.py` 的同一套切词，所以命中可以互相印证。

【索引结构】
    postings: {term: {field: {doc_idx: [position, ...]}}}
位置是必需的：短语匹配（`"军用领域的收入"`）要求这些词**相邻**，
没有位置就只能退化成布尔 AND —— 那是另一种能力，工单把它们分开点名了。

【字段】语料里没有现成的"摘要"字段，所以三个字段这样定义（写进文档）：
    title    = `section_path` 的最后一段（章节标题），权重最高
    abstract = `content` 的前 120 字（表格/图块即其首行与题注）
    body     = `content` 全文

【查询语法】（演示与调试用；检索主链路传的是自然语言问句）
    布尔    `注册资本 AND 5,520` · `A OR B` · `NOT C` · `-C`
    短语    `"军用领域的收入"`
    模糊    `法定代表~`（距离 1）· `法定代表~2`
    多字段  `title:募集资金` · `body:注册资本`
    裸词    默认 OR（与 Lucene 一致，见下方说明）
"""

from __future__ import annotations

import hashlib
import logging
import math
import pickle
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.config import settings
from app.core.text_analysis import is_numeric_token, tokenize_positions
from app.core.vectorstore import SearchHit, VectorStore

log = logging.getLogger("rag.fulltext")

FIELDS: tuple[str, ...] = ("title", "abstract", "body")
FIELD_WEIGHTS: dict[str, float] = {"title": 3.0, "abstract": 2.0, "body": 1.0}
FIELD_B: dict[str, float] = {"title": 0.3, "abstract": 0.5, "body": 0.75}
ABSTRACT_CHARS = 120


# ======================================================================
# 查询 AST
# ======================================================================
@dataclass(frozen=True)
class QTerm:
    term: str
    field: str | None = None


@dataclass(frozen=True)
class QPhrase:
    """短语：terms 必须**相邻**依次出现。"""

    terms: tuple[str, ...]
    field: str | None = None


@dataclass(frozen=True)
class QFuzzy:
    term: str
    max_edits: int = 1


@dataclass(frozen=True)
class QAnd:
    left: Any
    right: Any


@dataclass(frozen=True)
class QOr:
    left: Any
    right: Any


@dataclass(frozen=True)
class QNot:
    child: Any


@dataclass(frozen=True)
class _Tok:
    kind: str          # lparen | rparen | phrase | op | field | colon | fuzzy | word
    text: str
    extra: int = 1     # fuzzy 的距离


# 词法：**运算符必须大写**（`AND/OR/NOT`）。
# 【为什么不做大小写不敏感】英文问句里 "and" 是普通词（"registered capital and
# total shares…"），把它当运算符会把整句解析歪。Lucene 也是这个约定。
_LEX_RE = re.compile(
    r"""
    (?P<lparen>\() |
    (?P<rparen>\)) |
    (?P<phrase>"[^"]*") |
    (?P<op>AND|OR|NOT) |
    (?P<field>title|abstract|body)(?=:) |
    (?P<colon>:) |
    (?P<fuzzy>[^\s()":~]+)~(?P<fuzzy_d>\d*) |
    (?P<word>[^\s()":~]+)
    """,
    re.VERBOSE,
)


def _lex(query: str) -> list[_Tok]:
    out: list[_Tok] = []
    for m in _LEX_RE.finditer(query):
        if m.group("lparen"):
            out.append(_Tok("lparen", "("))
        elif m.group("rparen"):
            out.append(_Tok("rparen", ")"))
        elif m.group("phrase") is not None:
            out.append(_Tok("phrase", m.group("phrase")))
        elif m.group("op"):
            out.append(_Tok("op", m.group("op")))
        elif m.group("field"):
            out.append(_Tok("field", m.group("field")))
        elif m.group("colon"):
            out.append(_Tok("colon", ":"))
        elif m.group("fuzzy") is not None:
            d = m.group("fuzzy_d") or ""
            out.append(_Tok("fuzzy", m.group("fuzzy"), max(1, int(d)) if d else 1))
        else:
            out.append(_Tok("word", m.group("word")))
    return out


def _term_nodes(text: str, field_name: str | None) -> Any:
    """裸词/短语里的文本 → AST 节点（**必须走分词**）。

    【为什么裸词也要分词】用户写 `军用领域` 时，倒排索引里存的是 `军用` 与 `领域`
    两个词条。拿整串去查 postings 会**一个都查不到**（而这不报错，只是没结果）。
    """
    toks = [t for t, _ in tokenize_positions(text)]
    if not toks:
        return QTerm(text, field_name)
    if len(toks) == 1:
        return QTerm(toks[0], field_name)
    # 多词 → 隐式 OR（与整体默认一致），BM25 会让"全都命中"的块排前面
    node: Any = QTerm(toks[0], field_name)
    for t in toks[1:]:
        node = QOr(node, QTerm(t, field_name))
    return node


def parse(query: str) -> Any:
    """解析查询串 → AST。空查询返回 None。

    优先级（低到高）：OR  <  AND  <  NOT  <  括号/字段/短语/模糊/词
    相邻的两个项没有运算符时，按 **OR** 连接 —— 见下方理由。
    """
    # `-词` 是 NOT 的简写
    toks = _lex(re.sub(r"(?<=\s)-(?=\S)", " NOT ", (query or "").strip()))
    pos = 0

    def peek() -> _Tok | None:
        return toks[pos] if pos < len(toks) else None

    def take() -> _Tok:
        nonlocal pos
        t = toks[pos]
        pos += 1
        return t

    def parse_or() -> Any:
        node = parse_and()
        while (t := peek()) and t.kind == "op" and t.text == "OR":
            take()
            node = QOr(node, parse_and())
        return node

    def parse_and() -> Any:
        node = parse_unary()
        while True:
            t = peek()
            if t is None or t.kind == "rparen":
                break
            if t.kind == "op" and t.text == "OR":
                break
            if t.kind == "op" and t.text == "AND":
                take()
                # 显式 AND：右侧若是否定，仍按 AND 处理（求值时会做差集）
                node = QAnd(node, parse_unary())
                continue
            # 【隐式运算符 = OR】检索入口拿到的是自然语言问句
            # （"报告期内，公司来自军用领域的收入分别是多少？"）。若相邻词默认 AND，
            # 一句十几个词要求全部同时命中 → 绝大多数查询直接零召回。
            # 所以要严格匹配就显式写 AND（与 Lucene 的默认一致）。
            node = QOr(node, parse_unary())
        return node

    def parse_unary() -> Any:
        t = peek()
        if t and t.kind == "op" and t.text == "NOT":
            take()
            return QNot(parse_unary())
        return parse_primary()

    def parse_primary() -> Any:
        t = take()
        if t.kind == "lparen":
            node = parse_or()
            if peek() and peek().kind == "rparen":
                take()
            return node
        if t.kind == "field":
            if peek() and peek().kind == "colon":
                take()
            return _with_field(parse_primary(), t.text)
        if t.kind == "phrase":
            # 【必须真的建 QPhrase】引号里的内容若退化成"各词 OR"，短语匹配就没了 ——
            # 实测 `"军用收入领域"`（顺序错）会与 `"军用领域的收入"` 返回**完全相同**的结果，
            # 而且不报错。短语的唯一价值就是"相邻且有序"，退化等于没做。
            terms = tuple(x for x, _ in tokenize_positions(t.text.strip('"')))
            if not terms:
                return QTerm(t.text.strip('"'))
            return QTerm(terms[0]) if len(terms) == 1 else QPhrase(terms)
        if t.kind == "fuzzy":
            return QFuzzy(t.text, t.extra)
        return _term_nodes(t.text, None)

    return parse_or() if toks else None


def _with_field(node: Any, field_name: str) -> Any:
    if isinstance(node, QTerm):
        return QTerm(node.term, field_name)
    if isinstance(node, QOr):
        return QOr(_with_field(node.left, field_name),
                   _with_field(node.right, field_name))
    if isinstance(node, QAnd):
        return QAnd(_with_field(node.left, field_name),
                    _with_field(node.right, field_name))
    if isinstance(node, QFuzzy):
        return node                       # 模糊暂不支持字段限定（低频，写明限度）
    return node


# ======================================================================
# 倒排索引
# ======================================================================
@dataclass
class DocMeta:
    chunk_id: int
    content: str
    page_no: int
    page_label: str
    chunk_type: str
    section_path: str
    doc_name: str
    chunk_index: int
    doc_id: str

    def to_hit(self, score: float, *, retrieval_src: str = "keyword") -> SearchHit:
        return SearchHit(
            chunk_id=self.chunk_id, score=score, content=self.content,
            page_no=self.page_no, page_label=self.page_label,
            chunk_type=self.chunk_type, section_path=self.section_path,
            doc_name=self.doc_name, chunk_index=self.chunk_index,
            doc_id=self.doc_id,
            score_kind="bm25", retrieval_src=retrieval_src,
        )


def _title_of(section_path: str) -> str:
    """章节路径的最后一段当"标题"。

    路径形如 `第五节 业务与技术 > 一、发行人主营业务 > （一）…`，
    最后一段最贴近这一块的内容。
    """
    if not section_path:
        return ""
    parts = [p for p in re.split(r"\s*[>＞＞|｜/]\s*", section_path.strip()) if p]
    return parts[-1] if parts else section_path


def _levenshtein(a: str, b: str, cap: int = 2) -> int:
    """编辑距离（带上限提前退出）。候选词通常只有几个，够快。"""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


class InvertedIndex:
    """三字段倒排索引 + BM25F。"""

    def __init__(self, *, k1: float = 1.2, fuzzy_min_len: int = 2,
                 fuzzy_penalty: float = 0.5, fuzzy_max_expansions: int = 32) -> None:
        self.k1 = k1
        self.fuzzy_min_len = fuzzy_min_len
        self.fuzzy_penalty = fuzzy_penalty
        self.fuzzy_max_expansions = fuzzy_max_expansions

        self.docs: list[DocMeta] = []
        self.postings: dict[str, dict[str, dict[int, list[int]]]] = {}
        self.field_len: dict[str, dict[int, int]] = {f: {} for f in FIELDS}
        self.field_avg: dict[str, float] = {f: 1.0 for f in FIELDS}
        self.field_weights = dict(FIELD_WEIGHTS)
        self.field_b = dict(FIELD_B)
        # SymSpell 式"删除字典"：删 1 字符后的键 → 候选词集合（模糊匹配用）
        self.deletes: dict[str, set[str]] = defaultdict(set)
        self.n_docs = 0

    # ------------------------------------------------------------------
    @classmethod
    def build(cls, rows: Iterable[dict], **kw) -> "InvertedIndex":
        idx = cls(**kw)
        for r in rows:
            content = r.get("content") or ""
            section_path = r.get("section_path") or ""
            fields = {
                "title": _title_of(section_path),
                "abstract": content[:ABSTRACT_CHARS],
                "body": content,
            }
            d = len(idx.docs)
            idx.docs.append(DocMeta(
                chunk_id=int(r.get("id") or 0), content=content,
                page_no=int(r.get("page_no") or -1),
                page_label=r.get("page_label") or "",
                chunk_type=r.get("chunk_type") or "text",
                section_path=section_path,
                doc_name=r.get("doc_name") or "",
                chunk_index=int(r.get("chunk_index") or 0),
                doc_id=r.get("doc_id") or "",
            ))
            for fname, text in fields.items():
                toks = tokenize_positions(text)
                idx.field_len[fname][d] = len(toks)
                per_term: dict[str, list[int]] = defaultdict(list)
                for term, p in toks:
                    per_term[term].append(p)
                for term, positions in per_term.items():
                    idx._remember_fuzzy(term)
                    idx.postings.setdefault(term, {}).setdefault(fname, {})[d] = positions

        idx.n_docs = len(idx.docs)
        for f in FIELDS:
            lens = list(idx.field_len[f].values())
            idx.field_avg[f] = (sum(lens) / len(lens)) if lens else 1.0
        return idx

    def _remember_fuzzy(self, term: str) -> None:
        """为一个词登记它的"删 1 字符"变体。

        【为什么用删除字典而不是逐词算编辑距离】库里有 2 万多个词，每个查询词
        都对全词典跑一遍 Levenshtein 会把检索从毫秒拖到秒级。预先建好删除键，
        查询侧只要生成自己的删除变体去查表（O(1)），再对少量候选校验真实距离。
        """
        if len(term) < self.fuzzy_min_len or is_numeric_token(term):
            return
        for i in range(len(term)):
            self.deletes[term[:i] + term[i + 1:]].add(term)

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "InvertedIndex":
        with path.open("rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # BM25F
    # ------------------------------------------------------------------
    def _term_scores(self, term: str, field_name: str | None,
                     allowed: set[int] | None) -> dict[int, float]:
        """单个 term 的 BM25F 贡献：doc_idx -> 分数。"""
        by_field = self.postings.get(term)
        if not by_field:
            return {}
        fnames = (field_name,) if field_name else FIELDS

        docs_union: set[int] = set()
        for f in fnames:
            docs_union |= set(by_field.get(f, {}).keys())
        if allowed is not None:
            docs_union &= allowed
        df = len(docs_union)
        if df == 0:
            return {}
        idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

        # 先把各字段的归一化词频加起来（BM25F 的"F"：跨字段合并后再饱和）
        tf_tilde: dict[int, float] = {}
        for f in fnames:
            post = by_field.get(f, {})
            if not post:
                continue
            w_f, b_f = self.field_weights[f], self.field_b[f]
            avg = self.field_avg.get(f) or 1.0
            for d, positions in post.items():
                if d not in docs_union:
                    continue
                norm = 1.0 - b_f + b_f * (self.field_len[f].get(d, 0) / avg)
                tf_tilde[d] = tf_tilde.get(d, 0.0) + w_f * len(positions) / max(norm, 1e-6)

        return {d: idf * (self.k1 + 1) * t / (self.k1 + t) for d, t in tf_tilde.items()}

    def _phrase_scores(self, terms: Sequence[str], field_name: str | None,
                       allowed: set[int] | None) -> dict[int, float]:
        """短语匹配：terms 必须**相邻**依次出现（位置差恒为 1）。"""
        if not terms:
            return {}
        fnames = (field_name,) if field_name else FIELDS
        result: dict[int, float] = {}
        for f in fnames:
            posts: list[dict[int, list[int]]] = []
            for t in terms:
                p = self.postings.get(t, {}).get(f)
                if not p:
                    posts = []
                    break
                posts.append({d: set(v) for d, v in p.items()})
            if not posts:
                continue
            common = set(posts[0])
            for p in posts[1:]:
                common &= set(p)
            if allowed is not None:
                common &= allowed
            for d in common:
                starts = set(posts[0][d])
                # 【偏移必须随词序递增】第 k 个词要落在 `起点 + k` 上。
                # 若每轮都拿 `起点 + 1` 去比，三词及以上的短语永远匹配不到
                # —— 而两词短语正常，所以这个 bug 只会在长短语上暴露。
                for k, p in enumerate(posts[1:], start=1):
                    starts = {s for s in starts if (s + k) in p[d]}
                    if not starts:
                        break
                if not starts:
                    continue
                # 短语当成一个整体：分数 = 各词 BM25F 之和 × 命中位置数
                s = sum(self._term_scores(t, f, {d}).get(d, 0.0) for t in terms)
                result[d] = result.get(d, 0.0) + s * len(starts)
        return result

    def _fuzzy_expand(self, raw: str, max_edits: int) -> list[tuple[str, float]]:
        """模糊扩展 → [(候选词, 折扣)]。

        【为什么先分词】倒排索引里存的是**词条**。用户写 `法定代表~` 时，
        库里其实是 `法定` / `代表` 两个词条 —— 拿整串 `法定代表` 去查删除字典
        一个候选都找不到（而且不报错，只是"模糊怎么搜都没有"）。
        所以先切词，再逐词做模糊扩展；原词本身按折扣 1.0 一起收进来
        （模糊查询不该排除精确命中）。

        【数字绝不参与】`5,520` 与 `5,530` 编辑距离是 1，但招股书里数字**就是答案**，
        模糊命中等于把答案改错。所以数字只做归一化后的精确比较。
        """
        terms = [t for t, _ in tokenize_positions(raw)] or [raw]
        out: list[tuple[str, float]] = []
        for term in terms:
            if term in self.postings:
                out.append((term, 1.0))
            if max_edits <= 0 or len(term) < self.fuzzy_min_len or is_numeric_token(term):
                continue
            cands: set[str] = set()
            for i in range(len(term)):
                cands |= self.deletes.get(term[:i] + term[i + 1:], set())
            for c in sorted(cands)[: self.fuzzy_max_expansions]:
                if c != term and _levenshtein(term, c, max_edits) <= max_edits:
                    out.append((c, self.fuzzy_penalty))
        return out

    # ------------------------------------------------------------------
    def _eval(self, node: Any, allowed: set[int] | None
              ) -> tuple[set[int], dict[int, float], set[int]]:
        """求值 → (命中集, 分数, 排除集)。"""
        if node is None:
            return set(), {}, set()
        if isinstance(node, QNot):
            docs, _, _ = self._eval(node.child, allowed)
            return set(), {}, docs
        if isinstance(node, QTerm):
            sc = self._term_scores(node.term, node.field, allowed)
            return set(sc), sc, set()
        if isinstance(node, QPhrase):
            sc = self._phrase_scores(node.terms, node.field, allowed)
            return set(sc), sc, set()
        if isinstance(node, QFuzzy):
            sc: dict[int, float] = {}
            for cand, discount in self._fuzzy_expand(node.term, node.max_edits):
                for d, s in self._term_scores(cand, None, allowed).items():
                    sc[d] = sc.get(d, 0.0) + s * discount
            return set(sc), sc, set()
        if isinstance(node, (QAnd, QOr)):
            ld, ls, le = self._eval(node.left, allowed)
            rd, rs, re_ = self._eval(node.right, allowed)
            if isinstance(node, QAnd):
                # `A AND NOT B`：右侧是纯排除项时做差集。
                # **绝不对全库取补** —— 那会把 1848 行全返回，"看起来有结果"的静默错误。
                if not rd and re_:
                    return ld - re_, dict(ls), le | re_
                if not ld and le:
                    return rd - le, dict(rs), le | re_
                docs = ld & rd
            else:
                docs = ld | rd
            scores = {d: ls.get(d, 0.0) + rs.get(d, 0.0) for d in docs}
            return docs, scores, le | re_
        return set(), {}, set()

    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 5, *,
               doc_name: str | None = None) -> list[SearchHit]:
        """全文检索。`doc_name` 用于**路由过滤**（工单03 的文档隔离）。

        收结构化字段而不是 Milvus 的 `expr` 字符串 —— 在 Python 里解析
        `doc_name == "招股说明书2.pdf"` 太脆。
        """
        allowed = None
        if doc_name:
            allowed = {i for i, d in enumerate(self.docs) if d.doc_name == doc_name}
            if not allowed:
                return []
        docs, scores, excluded = self._eval(parse(query), allowed)
        docs -= excluded
        ranked = sorted(docs, key=lambda d: -scores.get(d, 0.0))[:k]
        return [self.docs[d].to_hit(scores.get(d, 0.0)) for d in ranked]

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "n_docs": self.n_docs,
            "n_terms": len(self.postings),
            "n_delete_keys": len(self.deletes),
            "field_avg_len": {f: round(v, 1) for f, v in self.field_avg.items()},
        }


# ======================================================================
# 索引的构建与缓存
# ======================================================================
_INDEX_CACHE: dict[str, InvertedIndex] = {}


def _fingerprint(store: VectorStore) -> tuple[int, str]:
    """(行数, 内容指纹)。**指纹必须含内容** —— 只看行数的话，
    重新入库（行数不变、内容变了）会静默沿用旧索引，
    检索结果看起来正常，其实与库不一致。"""
    rows = store.client.query(
        collection_name=store.collection, filter="id >= 0",
        output_fields=["doc_id", "chunk_index", "content_hash"], limit=16384)
    h = hashlib.sha1()
    for r in sorted(rows, key=lambda x: (x.get("doc_id") or "", x.get("chunk_index") or 0)):
        h.update(f"{r.get('doc_id')}|{r.get('chunk_index')}|{r.get('content_hash')}|".encode())
    return len(rows), h.hexdigest()[:12]


def index_path_for(collection: str, *, store: VectorStore | None = None) -> Path:
    store = store or VectorStore(collection=collection)
    n, fp = _fingerprint(store)
    return settings.data_path / "index" / f"fulltext-{collection}-{n}-{fp}.pkl"


def get_index(collection: str | None = None, *, store: VectorStore | None = None,
              rebuild: bool = False) -> InvertedIndex:
    """取（必要时构建）倒排索引。**进程级单例**。

    【为什么必须是单例】`Retriever` 在 `api/chat.py` 是**每个请求新建**的。
    若把建索引放进 `Retriever.__init__`，等于每问一次就重新切词 1848 块
    （实测约 2 秒），TTFT 直接爆。启动时预热一次即可（见 main.py 的 lifespan）。
    """
    name = collection or settings.milvus_collection
    if not rebuild and name in _INDEX_CACHE:
        return _INDEX_CACHE[name]

    store = store or VectorStore(collection=name)
    t0 = time.perf_counter()
    path = index_path_for(name, store=store)
    if path.exists() and not rebuild:
        try:
            idx = InvertedIndex.load(path)
            log.info("全文索引命中缓存：%s（%d 篇 / %d 词 / %.2fs）",
                     path.name, idx.n_docs, len(idx.postings), time.perf_counter() - t0)
            _INDEX_CACHE[name] = idx
            return idx
        except Exception as e:  # noqa: BLE001
            log.warning("全文索引缓存损坏，改为重建：%s", e)

    rows = store.client.query(
        collection_name=name, filter="id >= 0",
        output_fields=["id", "content", "page_no", "page_label", "chunk_type",
                       "section_path", "doc_name", "chunk_index", "doc_id"],
        limit=16384)
    idx = InvertedIndex.build(rows)
    try:
        idx.save(path)
    except Exception as e:  # noqa: BLE001
        log.warning("全文索引写盘失败（不影响本次使用）：%s", e)
    log.info("全文索引已构建：%d 篇 / %d 词 / %.2fs",
             idx.n_docs, len(idx.postings), time.perf_counter() - t0)
    _INDEX_CACHE[name] = idx
    return idx
