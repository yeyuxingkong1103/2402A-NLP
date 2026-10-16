# -*- coding: utf-8 -*-
"""工单6 全文检索 —— 倒排索引 + 布尔 / 短语 / 模糊查询 + 多字段。

工单编号：人工智能NLP-RAG-混合检索任务

工单原文要求
------------
「基于关键词的倒排索引，支持布尔查询、短语匹配、模糊匹配。支持多字段检索（标题、正文、摘要等），
最低需求可只对正文检索。推荐使用 ``jieba`` + ``rank_bm25`` 或 ``Whoosh`` 快速实现。」

本机事实（实测，勿凭提示词臆测）
--------------------------------
* ``rank_bm25`` / ``Whoosh`` **均未安装**（``importlib.util.find_spec`` 未命中），
  且 ``pip install --target`` 在本沙箱内**实测挂死**（180 s 无输出、无退出码）。
  故本模块在 工单4 自实现 BM25（``bm25_index.py``）之上自建**倒排索引 + 布尔求值器**，
  对外提供与 ``Whoosh`` 等价的查询能力，行为可解释、可测试。
* 语料复用 工单4 已建好的块级索引（2676 块 / 词表 23607），**不重建、不改动既有索引文件**。

查询语法（DSL）
---------------
============================  ==================================================
写法                           语义
============================  ==================================================
``甲 乙``                      默认运算符（可配 ``default_op``，默认 OR）
``甲 AND 乙`` / ``甲 && 乙``   布尔与（亦可写 ``且``）
``甲 OR 乙``  / ``甲 || 乙``   布尔或（亦可写 ``或``）
``NOT 甲`` / ``-甲`` / ``非 甲`` 布尔非（**中文 ``非`` 必须与词之间有空格**；
                               ``非甲`` 连写会被分词当成普通词项 —— 见 §已知限制）
``(甲 OR 乙) AND 丙``          括号分组，优先级 NOT > AND > OR
``"发行股数"``                 短语匹配（要求分词后在原文中**位置相邻**）
``军用~`` / ``军用~1``         模糊匹配（词表内编辑距离 ≤1 的近词扩展）
``标题:发行股数``              字段限定，字段 ∈ ``正文`` / ``标题`` / ``关键词`` / ``摘要``
``"标题:军用"``                **字段 + 短语**要写成「字段名也放进引号内」或 ``标题:"军用"``
                               （两种写法都等价于「标题字段内短语匹配」）
============================  ==================================================

字段口径（**如实声明，不夸大**）
--------------------------------
* ``正文``（``content``）：块 ``content`` 的分词（直接复用 ``bm25.doc_tokens``，零额外开销）；
* ``标题``（``title``）：表格/图像块的 ``[表标题]/[图标题]`` 行；否则取块的 ``section``；
* ``关键词``（``keywords``）：块的 ``keywords`` 列表 + 正文中的数字串（提升数字型提问召回）；
* ``摘要``（``summary``）：``content`` 前 220 字（轻量近似，**不是**人工摘要）。

纯负查询（如只输入 ``NOT 军用``）在倒排语义下会命中几乎全库，无检索意义 ⇒
本实现返回空结果并在 ``notes`` 中说明，**不静默返回全量**。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Iterable, Mapping, Sequence

from .text_utils import tokenize

WORK_ORDER = "人工智能NLP-RAG-混合检索任务"

#: 支持的字段（英文 id → 中文名）
FIELD_LABELS: dict[str, str] = {
    "content": "正文",
    "title": "标题",
    "keywords": "关键词",
    "summary": "摘要",
}
#: 界面友好的中文字段别名 → 英文 id
FIELD_ALIASES: dict[str, str] = {
    "正文": "content", "内容": "content", "content": "content", "body": "content",
    "标题": "title", "title": "title", "name": "title",
    "关键词": "keywords", "关键字": "keywords", "keywords": "keywords", "keyword": "keywords",
    "摘要": "summary", "summary": "summary", "abstract": "summary",
}
#: 多字段检索的默认字段集合（工单最低需求 = 只对正文检索；此处默认已超出最低需求）
DEFAULT_FIELDS: tuple[str, ...] = ("content",)

#: 字段命中加成（多字段检索时，标题/关键词命中额外加权；正文为基准 1.0）
DEFAULT_FIELD_BOOSTS: dict[str, float] = {"content": 1.0, "title": 1.35, "keywords": 1.20,
                                          "summary": 0.90}

#: 模糊匹配的默认编辑距离与扩展上限
DEFAULT_FUZZY_DISTANCE = 1
DEFAULT_FUZZY_LIMIT = 8

_OPERATOR_TOKENS = {"AND", "OR", "NOT", "&&", "||", "且", "或", "非"}
_FIELD_PREFIX_RE = re.compile(r"^([\u4e00-\u9fffA-Za-z_]+)\s*:\s*(.+)$", re.S)
#: 「字段:」前缀本身（后面还没跟值）—— 用于把紧邻引号的前缀与短语合成一个 token
_FIELD_PREFIX_ONLY_RE = re.compile(r"^[\u4e00-\u9fffA-Za-z_]+\s*:\s*$")
_TITLE_LINE_RE = re.compile(r"\[(?:表标题|图标题|标题)\]\s*(.+)")
_NUMBER_RE = re.compile(r"\d[\d,\.]*%?")


# ---------------------------------------------------------------------------
# ① 查询 AST
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Node:
    """查询 AST 基类。"""

    def positive_terms(self) -> list[str]:
        """该子树里所有**正向**词项（用于 BM25 打分；被 NOT 包住的词不算）。"""
        return []

    def describe(self) -> str:
        return "?"


@dataclass(slots=True)
class Term(Node):
    """单词项（可带字段限定与模糊距离）。"""

    text: str
    field: str = "content"
    fuzzy_distance: int = 0
    expanded: list[str] = dc_field(default_factory=list)

    def positive_terms(self) -> list[str]:
        return [self.text, *self.expanded]

    def describe(self) -> str:
        tag = f"{self.field}:" if self.field != "content" else ""
        fuzzy = f"~{self.fuzzy_distance}" if self.fuzzy_distance else ""
        return f"{tag}{self.text}{fuzzy}"


@dataclass(slots=True)
class Phrase(Node):
    """短语项（要求词项在原文中位置相邻）。"""

    text: str
    field: str = "content"

    def positive_terms(self) -> list[str]:
        return tokenize(self.text)

    def describe(self) -> str:
        tag = f"{self.field}:" if self.field != "content" else ""
        return f'{tag}"{self.text}"'


@dataclass(slots=True)
class And(Node):
    """布尔与。"""

    children: list[Node] = dc_field(default_factory=list)

    def positive_terms(self) -> list[str]:
        out: list[str] = []
        for child in self.children:
            out.extend(child.positive_terms())
        return out

    def describe(self) -> str:
        return "(" + " AND ".join(c.describe() for c in self.children) + ")"


@dataclass(slots=True)
class Or(Node):
    """布尔或。"""

    children: list[Node] = dc_field(default_factory=list)

    def positive_terms(self) -> list[str]:
        out: list[str] = []
        for child in self.children:
            out.extend(child.positive_terms())
        return out

    def describe(self) -> str:
        return "(" + " OR ".join(c.describe() for c in self.children) + ")"


@dataclass(slots=True)
class Not(Node):
    """布尔非：**不贡献正向词项**（否则被排除的词反而会给分）。"""

    child: Node | None = None

    def positive_terms(self) -> list[str]:
        return []

    def describe(self) -> str:
        return f"NOT {self.child.describe() if self.child else '?'}"


# ---------------------------------------------------------------------------
# ② 查询解析（递归下降；优先级 NOT > AND > OR）
# ---------------------------------------------------------------------------
def _tokenize_query(text: str) -> list[str]:
    """把查询串切成「字面量 / 运算符 / 括号」序列。

    * 引号内整体作为一个 LITERAL（保留短语语义）；
    * ``字段:值`` 作为一个 LITERAL（值可能带引号）；
    * ``-词`` 切出 ``NOT`` + ``词``；
    * 其余按空白切分。
    """
    tokens: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            chunk = "".join(buf).strip()
            if chunk:
                tokens.append(chunk)
            buf.clear()

    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            # 字段前缀紧邻引号 ⇒ 必须合成**一个** token。
            # 否则 ``标题:"军用"`` 会被切成 ``标题:`` + ``"军用"`` 两个 token，
            # 解析成 ``(标题 OR "军用")`` —— 字段限定被静默丢掉（实测陷阱：命中数 662 vs 正确 7）。
            prefix = "".join(buf).strip()
            if prefix and _FIELD_PREFIX_ONLY_RE.match(prefix):
                buf.clear()
                j = i + 1
                while j < n and text[j] != '"':
                    j += 1
                tokens.append(prefix + '"' + text[i + 1:j] + '"')
                i = j + 1
                continue
            flush()
            j = i + 1
            while j < n and text[j] != '"':
                j += 1
            tokens.append('"' + text[i + 1:j] + '"')
            i = j + 1
            continue
        if ch == "(" or ch == ")":
            flush()
            tokens.append(ch)
            i += 1
            continue
        if ch.isspace():
            flush()
            i += 1
            continue
        buf.append(ch)
        i += 1
    flush()

    # 二次切分：把「字段:值」「-词」「词~N」从空白块里拆出来
    out: list[str] = []
    for token in tokens:
        if token in ("(", ")") or token.startswith('"'):
            out.append(token)
            continue
        if token.upper() in _OPERATOR_TOKENS:
            out.append(token.upper() if token.isascii() else token)
            continue
        work = token
        # 前导 '-'（非字段名的一部分）视作 NOT
        while work.startswith("-") and len(work) > 1:
            out.append("NOT")
            work = work[1:]
        if work:
            out.append(work)
    return out


def _analyze_text(text: str) -> list[str]:
    """把字面量切成与**文档侧同口径**的词项（jieba + 数字保留）。

    ⚠️ 这是本模块最容易踩的坑（实测踩过一次）：查询侧若不做与文档侧相同的分词，
    中文自然语言问句（无空格、无运算符）会整体变成一个「词」去查倒排表 ⇒ **恒定 0 命中**。
    例：问句「…本次发行股数是多少，占发行后总股本的比例是多少？」整串查表必空；
    经此分析后变为 ``发行 / 股数 / 发行 / 后 / 总股本 / 比例 / 多少 …`` 的 OR 查询，命中正常。

    退化策略：分词结果为空（例如单字、纯符号）时，保留整串作为唯一词项，
    以免把用户输入**静默丢掉**。
    """
    tokens = [t for t in tokenize(text) if t.strip()]
    if tokens:
        return tokens
    raw = str(text or "").strip()
    return [raw] if raw else []


def _term_from_literal(literal: str, *, default_field: str, fuzzy_distance: int,
                       default_op: str = "OR", analyze: bool = True) -> Node | None:
    """把单个字面量转成 ``Term`` / ``Phrase`` / 其布尔组合。

    处理顺序（不可颠倒）：引号（短语）→ ``~N``（模糊）→ ``字段:``（字段限定）→ 分词分析。
    多词项时按 ``default_op`` 组合（默认 OR）—— 与 Lucene 的 QueryParser 行为一致。
    """
    text = literal
    is_phrase = False
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1]
        is_phrase = True
    distance = 0
    if not is_phrase:
        m = re.match(r"^(.*?)(~(\d*))$", text, re.S)
        if m and m.group(1):
            text = m.group(1)
            distance = int(m.group(3)) if m.group(3) else max(1, int(fuzzy_distance))
    node_field = default_field
    fm = _FIELD_PREFIX_RE.match(text)
    if fm and fm.group(1) in FIELD_ALIASES:
        node_field = FIELD_ALIASES[fm.group(1)]
        text = fm.group(2).strip()
        if text.startswith('"') and text.endswith('"') and len(text) >= 2:
            text = text[1:-1]
            is_phrase = True
    if is_phrase:
        return Phrase(text=text, field=node_field)
    tokens = _analyze_text(text) if analyze else ([text.strip()] if text.strip() else [])
    nodes = [Term(text=token, field=node_field, fuzzy_distance=distance) for token in tokens]
    if not nodes:
        return None
    if len(nodes) == 1:
        return nodes[0]
    return And(children=nodes) if str(default_op).upper() == "AND" else Or(children=nodes)


class _Parser:
    """递归下降解析器。"""

    def __init__(self, tokens: Sequence[str], *, default_field: str, default_op: str,
                 fuzzy_distance: int, analyze: bool = True) -> None:
        self.tokens = list(tokens)
        self.pos = 0
        self.default_field = default_field
        self.default_op = (default_op or "OR").upper()
        self.fuzzy_distance = int(fuzzy_distance)
        self.analyze = bool(analyze)

    # -- 工具 --
    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _next(self) -> str | None:
        token = self._peek()
        if token is not None:
            self.pos += 1
        return token

    def _is_and(self, token: str | None) -> bool:
        return token is not None and token in ("AND", "&&", "且")

    def _is_or(self, token: str | None) -> bool:
        return token is not None and token in ("OR", "||", "或")

    def _is_not(self, token: str | None) -> bool:
        return token is not None and token in ("NOT", "非")

    # -- 文法 --
    def parse(self) -> Node | None:
        node = self._parse_or()
        if self.pos != len(self.tokens):     # 多余 token：按 OR 兜底并入，绝不静默丢弃
            rest = self._parse_or()
            if rest is not None and node is not None:
                node = Or(children=[node, rest])
        return node

    def _parse_or(self) -> Node | None:
        left = self._parse_and()
        while self._is_or(self._peek()):
            self._next()
            right = self._parse_and()
            if right is None:
                break
            if isinstance(left, Or):
                left.children.append(right)
            else:
                left = Or(children=[left, right] if left is not None else [right])
        return left

    def _parse_and(self) -> Node | None:
        left = self._parse_not()
        while True:
            token = self._peek()
            explicit = self._is_and(token)
            if explicit:
                self._next()
            elif token is None or token == ")" or self._is_or(token):
                break
            elif self.default_op == "AND":
                pass                       # 隐式 AND
            else:
                break                      # 隐式 OR：交给上层 _parse_or
            right = self._parse_not()
            if right is None:
                break
            if isinstance(left, And):
                left.children.append(right)
            else:
                left = And(children=[left, right] if left is not None else [right])
        return left

    def _parse_not(self) -> Node | None:
        if self._is_not(self._peek()):
            self._next()
            child = self._parse_not()
            return Not(child=child) if child is not None else None
        return self._parse_atom()

    def _parse_atom(self) -> Node | None:
        token = self._next()
        if token is None:
            return None
        if token == "(":
            inner = self._parse_or()
            if self._peek() == ")":
                self._next()
            return inner
        if token == ")":
            return None
        return _term_from_literal(token, default_field=self.default_field,
                                  fuzzy_distance=self.fuzzy_distance,
                                  default_op=self.default_op, analyze=self.analyze)


def parse_fulltext_query(text: str, *, default_field: str = "content",
                         default_op: str = "OR",
                         fuzzy_distance: int = DEFAULT_FUZZY_DISTANCE,
                         analyze: bool = True) -> Node | None:
    """解析查询串为 AST；空串/纯符号返回 ``None``。

    ``analyze=True``（默认）时字面量会经 jieba 分词（与文档侧同口径）；
    置 ``False`` 可做**精确词项**匹配（测试与调试用）。
    """
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    tokens = _tokenize_query(raw)
    if not tokens:
        return None
    return _Parser(tokens, default_field=default_field, default_op=default_op,
                   fuzzy_distance=fuzzy_distance, analyze=analyze).parse()


# ---------------------------------------------------------------------------
# ③ 多字段倒排索引
# ---------------------------------------------------------------------------
def field_text_of(chunk: Mapping[str, Any], field_name: str) -> str:
    """取块在某个字段上的文本（口径见模块 docstring「字段口径」）。"""
    content = str(chunk.get("content") or "")
    if field_name == "title":
        m = _TITLE_LINE_RE.search(content)
        if m:
            return m.group(1).strip()
        section = chunk.get("section")
        if section:
            return str(section)
        return content.split("\n", 1)[0][:60]
    if field_name == "keywords":
        kws = chunk.get("keywords") or []
        nums = _NUMBER_RE.findall(content)[:80]
        return " ".join([str(k) for k in kws] + nums)
    if field_name == "summary":
        return content[:220]
    return content


@dataclass(slots=True)
class FulltextIndex:
    """多字段倒排索引（挂在 ``IndexBundle`` 上的惰性派生产物，不改动磁盘索引）。"""

    index_dir: str
    chunk_ids: list[str]
    postings: dict[str, dict[str, list[int]]] = dc_field(default_factory=dict)   # field → term → rows
    field_tokens: dict[str, list[list[str]]] = dc_field(default_factory=dict)   # field → row → tokens
    vocab: dict[str, set[str]] = dc_field(default_factory=dict)                 # field → 词表
    built_ms: float = 0.0

    # -- 构建 --
    @classmethod
    def build(cls, bundle: Any, *, fields: Sequence[str] = ("content", "title", "keywords", "summary"),
              use_jieba: bool = True, logger: Any = None) -> "FulltextIndex":
        """从 ``IndexBundle`` 构建多字段倒排索引。

        ``content`` 字段**直接复用** BM25 已有的 ``doc_tokens``（零额外分词开销），
        其它字段现场分词（2676 块、4 个字段，**实测 built_ms ≈ 5000 ms**，
        属一次性预热成本，不是单次检索成本 —— 因此由 ``QAEngine.warmup()`` 预热）。
        """
        started = time.perf_counter()
        chunk_ids = [str(cid) for cid in getattr(bundle, "order", [])]
        index = cls(index_dir=str(getattr(bundle, "index_dir", "")), chunk_ids=chunk_ids)
        bm25_doc_tokens = list(getattr(bundle.bm25, "doc_tokens", []) or [])
        for name in fields:
            rows: list[list[str]] = []
            for row, cid in enumerate(chunk_ids):
                chunk = bundle.chunks.get(cid) or {}
                if name == "content" and row < len(bm25_doc_tokens):
                    toks = list(bm25_doc_tokens[row])
                else:
                    toks = tokenize(field_text_of(chunk, name), use_jieba=use_jieba)
                rows.append(toks)
            postings: dict[str, list[int]] = {}
            for row, toks in enumerate(rows):
                for term in set(toks):
                    postings.setdefault(term, []).append(row)
            index.field_tokens[name] = rows
            index.postings[name] = postings
            index.vocab[name] = set(postings.keys())
        index.built_ms = round((time.perf_counter() - started) * 1000, 2)
        if logger is not None:
            logger.log_event("fulltext_index.built", chunks=len(chunk_ids),
                             fields={f: len(index.vocab.get(f, ())) for f in fields},
                             elapsed_ms=index.built_ms)
        return index

    def size(self) -> int:
        return len(self.chunk_ids)

    def to_dict(self) -> dict[str, Any]:
        return {"index_dir": self.index_dir, "chunks": self.size(),
                "fields": {f: {"terms": len(self.vocab.get(f, ()))} for f in self.postings},
                "built_ms": self.built_ms}

    # -- 单词条查询 --
    def rows_with_term(self, term: str, field_name: str) -> set[int]:
        return set(self.postings.get(field_name, {}).get(term, ()))

    def fuzzy_expand(self, term: str, *, field_name: str, max_distance: int = 1,
                     limit: int = DEFAULT_FUZZY_LIMIT) -> list[str]:
        """词表内模糊扩展：返回编辑距离 ≤ ``max_distance`` 的近词（按距离、词频排序）。"""
        vocab = self.vocab.get(field_name) or set()
        if not vocab or max_distance <= 0:
            return []
        length = len(term)
        scored: list[tuple[int, int, str]] = []
        postings = self.postings.get(field_name, {})
        for candidate in vocab:
            if candidate == term:
                continue
            if abs(len(candidate) - length) > max_distance:
                continue
            dist = _bounded_edit_distance(term, candidate, max_distance)
            if dist is None:
                continue
            scored.append((dist, -len(postings.get(candidate, ())), candidate))
        scored.sort()
        return [item[2] for item in scored[: max(1, int(limit))]]


def _bounded_edit_distance(a: str, b: str, limit: int) -> int | None:
    """带上界的编辑距离（超过 ``limit`` 立即返回 ``None``，用于模糊匹配剪枝）。"""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > limit:
        return None
    previous = list(range(lb + 1))
    for i in range(1, la + 1):
        current = [i] + [0] * lb
        best = current[0]
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            if current[j] < best:
                best = current[j]
        if best > limit:
            return None
        previous = current
    return previous[lb] if previous[lb] <= limit else None


# ---------------------------------------------------------------------------
# ④ 布尔求值
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class FulltextEvalResult:
    """布尔求值结果。"""

    rows: set[int] = dc_field(default_factory=set)
    positive_terms: list[str] = dc_field(default_factory=list)
    fuzzy_expansions: dict[str, list[str]] = dc_field(default_factory=dict)
    phrases: list[tuple[str, str]] = dc_field(default_factory=list)   # (短语文本, 字段)
    notes: list[str] = dc_field(default_factory=list)
    matched_by: dict[str, str] = dc_field(default_factory=dict)       # 归一化查询串 → 说明
    #: **词表外词项**：不在倒排词表里的查询词。最常见原因是命中了分词器的停用词表
    #: （工单4 ``text_utils.STOPWORDS`` 含「项目/公司/本次/比例…」等表格高频词）。
    #: 这类词**索引里本来就没有**，因此 AND 上它们必然 0 命中 —— 必须显式告知用户，
    #: 而不是让他看到「查不到」却不知道原因。
    oov_terms: list[str] = dc_field(default_factory=list)


def _phrase_rows(index: FulltextIndex, phrase: Phrase) -> set[int]:
    """短语匹配：分词后在**同一字段的 token 序列**中位置相邻。"""
    terms = tokenize(phrase.text)
    if not terms:
        return set()
    rows = index.field_tokens.get(phrase.field) or []
    if not rows:
        return set()
    hit: set[int] = set()
    need = len(terms)
    for row, toks in enumerate(rows):
        if len(toks) < need:
            continue
        for start in range(len(toks) - need + 1):
            if toks[start:start + need] == terms:
                hit.add(row)
                break
    return hit


def evaluate_query(index: FulltextIndex, node: Node | None, *,
                   enable_fuzzy: bool = True,
                   fuzzy_limit: int = DEFAULT_FUZZY_LIMIT) -> FulltextEvalResult:
    """在倒排索引上求值 AST，返回命中的行号集合与正向词项。"""
    result = FulltextEvalResult()
    if node is None:
        result.notes.append("空查询")
        return result
    total = set(range(index.size()))

    def walk(cur: Node | None, negated: bool = False) -> set[int]:
        """求值子树，返回命中的行号集合。

        ``negated``：当前子树是否位于**奇数层** NOT 之下。被否定的词项**不得**计入
        ``positive_terms`` —— 否则「纯否定查询」守卫恒不成立（实测踩坑：
        ``NOT 军用`` 曾返回 2621/2676 行、全部分值 1e-6、排序退化为 chunk_id 字典序）。
        """
        if cur is None:
            return set()
        if isinstance(cur, Term):
            rows = index.rows_with_term(cur.text, cur.field)
            if cur.text not in (index.vocab.get(cur.field) or set()):
                result.oov_terms.append(cur.text)
            if enable_fuzzy and cur.fuzzy_distance > 0:
                expanded = index.fuzzy_expand(cur.text, field_name=cur.field,
                                              max_distance=cur.fuzzy_distance,
                                              limit=fuzzy_limit)
                if expanded:
                    result.fuzzy_expansions[cur.text] = expanded
                    cur.expanded = expanded
                    for term in expanded:
                        rows |= index.rows_with_term(term, cur.field)
            if not negated:
                result.positive_terms.append(cur.text)
                result.positive_terms.extend(cur.expanded)
            return rows
        if isinstance(cur, Phrase):
            if not negated:
                result.phrases.append((cur.text, cur.field))
                result.positive_terms.extend(cur.positive_terms())
            return _phrase_rows(index, cur)
        if isinstance(cur, And):
            sets = [walk(child, negated) for child in cur.children]
            if not sets:
                return set()
            out = sets[0]
            for item in sets[1:]:
                out &= item
            return out
        if isinstance(cur, Or):
            out: set[int] = set()
            for child in cur.children:
                out |= walk(child, negated)
            return out
        if isinstance(cur, Not):
            return total - walk(cur.child, not negated)
        return set()

    rows = walk(node)
    # 纯负查询保护：AST 里没有任何正向词项/短语 ⇒ 命中语义无意义，返回空并说明
    if not result.positive_terms and not result.phrases:
        result.notes.append("纯否定查询（无正向词项）在倒排索引下会命中几乎全库，已返回空结果")
        return result
    result.rows = rows
    # 去重但保序
    seen: set[str] = set()
    ordered: list[str] = []
    for term in result.positive_terms:
        if term and term not in seen:
            seen.add(term)
            ordered.append(term)
    result.positive_terms = ordered
    return result


# ---------------------------------------------------------------------------
# ⑤ 对外检索入口
# ---------------------------------------------------------------------------
def search_fulltext(bundle: Any, query: str, *, top_k: int = 20,
                    fields: Sequence[str] = DEFAULT_FIELDS,
                    default_op: str = "OR",
                    enable_fuzzy: bool = True,
                    fuzzy_distance: int = DEFAULT_FUZZY_DISTANCE,
                    fuzzy_limit: int = DEFAULT_FUZZY_LIMIT,
                    field_boosts: Mapping[str, float] | None = None,
                    score_threshold: float = 0.0,
                    allowed_ids: set[str] | None = None,
                    index: FulltextIndex | None = None,
                    diag: dict[str, Any] | None = None,
                    logger: Any = None) -> list[tuple[str, float]]:
    """全文检索：布尔/短语/模糊过滤 + BM25 打分 + 多字段加成。

    返回 ``[(chunk_id, score)]`` 降序。与 工单4 ``BM25Index.search`` 的差别：
        * 支持完整查询 DSL（布尔 / 短语 / 模糊 / 字段限定）；
        * 支持多字段（``fields``）与字段加成（``field_boosts``）；
        * 支持分数阈值（``score_threshold``）裁剪低分候选。

    打分口径：以**正向词项**在 ``bm25`` 主索引上算 BM25 分（复用既有倒排表，速度最快），
    再乘「命中字段的最大加成」；短语命中额外 ×1.1（短语比散词更可信）。
    """
    started = time.perf_counter()
    idx = index if index is not None else get_fulltext_index(bundle, logger=logger)
    boosts = dict(DEFAULT_FIELD_BOOSTS)
    if field_boosts:
        boosts.update({str(k): float(v) for k, v in field_boosts.items()})
    fs = [str(f) for f in (fields or DEFAULT_FIELDS)]
    for name in fs:
        if name not in idx.postings:
            raise ValueError(f"未知字段：{name!r}（可选：{', '.join(FIELD_LABELS)}）")

    node = parse_fulltext_query(query, default_field=fs[0], default_op=default_op,
                                fuzzy_distance=fuzzy_distance)
    evaluation = evaluate_query(idx, node, enable_fuzzy=enable_fuzzy, fuzzy_limit=fuzzy_limit)
    if diag is not None:
        diag.update({
            "query_ast": node.describe() if node is not None else None,
            "positive_terms": list(evaluation.positive_terms),
            "fuzzy_expansions": dict(evaluation.fuzzy_expansions),
            "phrases": [p[0] for p in evaluation.phrases],
            "oov_terms": list(evaluation.oov_terms),
            "boolean_hits": len(evaluation.rows),
            "notes": list(evaluation.notes),
            "fields": fs,
        })
    if evaluation.oov_terms and logger is not None:
        logger.log_event("fulltext.oov_terms", level="WARNING",
                         terms=evaluation.oov_terms,
                         reason="词项不在倒排词表中（多为分词停用词，索引中本就不存在）")
    if not evaluation.rows:
        if logger is not None:
            logger.log_event("fulltext.search", query=str(query)[:60], hits=0,
                             notes=evaluation.notes, elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
        return []

    # 允许的语料范围过滤（语料路由）
    rows = evaluation.rows
    if allowed_ids is not None:
        rows = {r for r in rows if idx.chunk_ids[r] in allowed_ids}
        if not rows:
            return []

    # BM25 打分（复用 工单4 的 PureBM25；仅对命中行取值，不倒排全库）
    scores = bundle.bm25.bm25.get_scores(evaluation.positive_terms) if evaluation.positive_terms else [0.0] * idx.size()
    phrase_rows: set[int] = set()
    for phrase_text, phrase_field in evaluation.phrases:
        phrase_rows |= _phrase_rows(idx, Phrase(text=phrase_text, field=phrase_field))

    # 字段加成：预先按「词项 → 命中行」累积每个行的最大字段加成。
    # ⚠️ 性能纪律（实测踩过一次）：**不要**写成 `for row in rows: row in postings[t]` ——
    # postings 的值是 list，`in` 是 O(n)，常见词（如「发行」）有上千个命中行，
    # 于是退化成 O(行数 × 词项数 × 命中数) ≈ 千万级比较。改为**正向遍历倒排表累积**。
    term_set = set(evaluation.positive_terms)
    row_boost: dict[int, float] = {}
    need_boost = any(abs(float(boosts.get(name, 1.0)) - 1.0) > 1e-9 for name in fs)
    if need_boost:
        for name in fs:
            boost_value = float(boosts.get(name, 1.0))
            postings = idx.postings.get(name, {})
            for term in term_set:
                for row in postings.get(term, ()):
                    if row_boost.get(row, 1.0) < boost_value:
                        row_boost[row] = boost_value

    scored: list[tuple[str, float]] = []
    for row in rows:
        base = float(scores[row]) if row < len(scores) else 0.0
        boost = row_boost.get(row, 1.0)
        value = base * boost
        if row in phrase_rows:
            value *= 1.1
        if value <= 0.0:
            # 布尔命中但 BM25 为 0（例如仅模糊扩展命中）⇒ 给一个极小的保底分，保证可排序可检索
            value = 1e-6 * float(boost)
        scored.append((idx.chunk_ids[row], round(value, 8)))
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    if score_threshold and float(score_threshold) > 0:
        scored = [item for item in scored if item[1] >= float(score_threshold)]
    hits = scored[: max(int(top_k), 1)]
    if logger is not None:
        logger.log_event("fulltext.search", query=str(query)[:60], fields=fs,
                         boolean_hits=len(rows), returned=len(hits),
                         fuzzy=evaluation.fuzzy_expansions or None,
                         phrases=[p[0] for p in evaluation.phrases],
                         elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
    return hits


#: ``index_dir`` → ``FulltextIndex`` 的进程内缓存（构建一次、多次复用）
_FULLTEXT_CACHE: dict[str, FulltextIndex] = {}


def get_fulltext_index(bundle: Any, *, fields: Sequence[str] = ("content", "title", "keywords", "summary"),
                       logger: Any = None) -> FulltextIndex:
    """取（或惰性构建）``bundle`` 对应的多字段倒排索引。"""
    key = str(getattr(bundle, "index_dir", "")) or f"bundle-{id(bundle)}"
    cached = _FULLTEXT_CACHE.get(key)
    if cached is not None and all(f in cached.postings for f in fields):
        return cached
    index = FulltextIndex.build(bundle, fields=fields, logger=logger)
    _FULLTEXT_CACHE[key] = index
    return index


def clear_fulltext_cache() -> None:
    """清空倒排索引缓存（测试用）。"""
    _FULLTEXT_CACHE.clear()


def describe_fields() -> list[dict[str, Any]]:
    """给界面/文档用的字段清单。"""
    return [{"id": fid, "label": label, "default_boost": DEFAULT_FIELD_BOOSTS.get(fid, 1.0)}
            for fid, label in FIELD_LABELS.items()]


def query_syntax_help() -> str:
    """界面上的查询语法提示。"""
    return (
        "布尔：甲 AND 乙 / 甲 OR 乙 / NOT 甲（也可写 且/或/非，注意「非」要与词隔开空格；-甲 亦可）　"
        "短语：\"发行股数\"　模糊：军用~ 或 军用~1　字段：标题:发行股数（字段+短语写 \"标题:军用\"）"
    )


__all__ = [
    "WORK_ORDER", "FIELD_LABELS", "FIELD_ALIASES", "DEFAULT_FIELDS", "DEFAULT_FIELD_BOOSTS",
    "Node", "Term", "Phrase", "And", "Or", "Not", "FulltextIndex", "FulltextEvalResult",
    "parse_fulltext_query", "evaluate_query", "search_fulltext", "get_fulltext_index",
    "clear_fulltext_cache", "field_text_of", "describe_fields", "query_syntax_help",
]
