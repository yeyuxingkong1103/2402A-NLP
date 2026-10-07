"""
全文检索引擎（倒排索引 + 布尔 / 短语 / 模糊查询 + 多字段 BM25F）
工单编号：人工智能NLP-RAG-混合检索任务

对应工单功能需求（工单6「2 功能详细需求 →（2）全文检索」）：

    需求描述：实现基于关键词的全文检索，支持布尔查询、短语匹配和模糊匹配。
             提供高效的索引机制，支持快速定位文档中的相关内容。
    技术要求：使用倒排索引技术实现全文检索。
             支持多字段检索（如标题、正文、摘要）。

—— 实现要点（每一条都对应上面的一句话）——

1) **倒排索引**：`term -> {doc_id -> [位置…]}`。
   位置信息是刻意留的：没有它就只能做「哪些块包含这个词」，
   做不了**短语匹配**（"军用领域" 要求两词相邻），也做不了**邻近度加成**
   （两个词挨得越近越可能是同一句话在回答这个问题）。
   实测这一项对招股书语料很关键 —— 「报告期内」这四个字在 3000 个块里
   出现了几百次，但只有「报告期内」**紧挨着**「军用领域」的那些块才是答案。

2) **布尔查询**：AND / OR / NOT / 括号 / +term / -term。
   默认用 OR（与 Lucene 默认一致）：招股书的提问是自然语言长句，
   逐词 AND 会因为没有「同时出现全部词」的块而**整条查询返回空**；
   OR 保证召回，再由 BM25 排序把真正相关的顶上来。
   需要严格模式的调用方可以显式传 mode="and"。

3) **短语匹配**：`"军用领域"` —— 用位置列表求交集，要求位置连续。

4) **模糊匹配**：
   * `注册酱~1`（编辑距离）—— 招股书问答实测最容易踩的是**输入错别字**和
     **同音字**（「销售处」打成「销售出」），严格匹配会直接零召回；
   * `销售*`（前缀通配）—— 「销售部 / 销售处 / 销售模式」这类同前缀词一族。

5) **多字段**：标题（section）/ 正文（text）/ 摘要（text 前 N 字）三个字段，
   用 **BM25F** 而非「三个字段各算一遍 BM25 再相加」。
   BM25F 先把同一词在多个字段的词频**归一化后加权合并**成一个伪词频，
   再统一过一遍 BM25 的饱和度函数 —— 这样「标题命中一次」不会被
   「正文命中一次」在饱和曲线上简单抵消，符合「标题里出现 = 这页就在讲这件事」的直觉。

6) 与 BM25Okapi 的关系：`index_store.KnowledgeBase` 里的 BM25Okapi 是**单字段纯正文**的
   打分器（工单1~5 一直在用，是混合检索的「稀疏路」）。本模块是它的**超集**：
   多了字段、位置、布尔、短语、模糊。两者共用同一套 jieba 分词，
   保证「混合检索融合时两路分数可比」。
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .config import (
    FULLTEXT_B,
    FULLTEXT_FIELD_WEIGHTS,
    FULLTEXT_K1,
    FULLTEXT_MAX_FUZZY_EXPAND,
    FULLTEXT_REPAIR_MAX_DF,
    FULLTEXT_SUMMARY_CHARS,
    WORK_ORDER_NOS,  # noqa: F401  工单编号（本文件归属工单6）
)

# ============================================================ 分词

# 英文/数字串：作为一个整体 token，同时保留小写形式用于大小写不敏感匹配
_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-\.]*|\d+(?:\.\d+)?%?")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")

_jieba_mod: Any = None


def _jieba():
    """惰性导入 jieba —— 首次导入要建词典缓存（~1s），不能放在 import 期。"""
    global _jieba_mod
    if _jieba_mod is None:
        import jieba

        jieba.setLogLevel(20)  # 屏蔽 "Building prefix dict..." 的 stderr 噪音
        _jieba_mod = jieba
    return _jieba_mod


def tokenize(text: str) -> list[str]:
    """
    中英文混合分词，返回**保序**的 token 列表（位置就用下标）。

    注意这里**不能**像 BM25Okapi 那版一样只返回词集合：
    短语匹配要的是位置，所以必须保序、且不能去重。

    英文统一转小写（「IPO」与「ipo」应当是一个词）；
    中文交给 jieba；纯符号丢弃（它们对检索没有区分度，还会把位置序列撑大）。
    """
    out: list[str] = []
    for m in re.finditer(r"[\u4e00-\u9fff]+|[A-Za-z][A-Za-z0-9_\-\.]*|\d+(?:\.\d+)?%?", text):
        s = m.group(0)
        if s[0].isascii() and s[0].isalpha():
            out.append(s.lower())
        elif s[0].isdigit():
            out.append(s)
        else:
            jb = _jieba()
            for t in jb.lcut(s):
                t = t.strip()
                if t and not t.isspace() and not _is_punct(t):
                    out.append(t)
    return out


def _is_punct(s: str) -> bool:
    return all(not (ch.isalnum()) for ch in s)


# ============================================================ 查询语法

_OP_RE = re.compile(r"^(AND|OR|NOT|and|or|not)$")

# 查询串的「词法单元」
LEX_TERM = "TERM"        # 普通词（可带 ~N 模糊 / * 通配）
LEX_PHRASE = "PHRASE"    # "短语"
LEX_FIELD = "FIELD"      # field:xxx  —— 冒号前缀，已拆成 (字段名, 剩余串)
LEX_AND, LEX_OR, LEX_NOT = "AND", "OR", "NOT"
LEX_LP, LEX_RP = "(", ")"


@dataclass
class Lex:
    kind: str
    value: str = ""
    fuzzy: int = 0        # >0 表示允许的编辑距离
    prefix: bool = False  # 结尾是 *


def lex(query: str) -> list[Lex]:
    """
    把查询串切成词法单元。

    支持写法（可以混用）：
        军用领域                    普通词
        "军用领域"                  短语（要求相邻）
        收入 AND 军用               布尔
        收入 OR 利润
        NOT 风险 / -风险            排除
        +收入                       必须
        section:风险提示            字段限定（section/title/body/summary）
        section:("风险提示" OR "重大事项")
        注册酱~1                    编辑距离 ≤1 的模糊匹配
        销售*                       前缀通配
    """
    out: list[Lex] = []
    i, n = 0, len(query)
    tokens: list[tuple[str, Any]] = []

    # ---- 第一遍：粗切成原始片段
    while i < n:
        ch = query[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(":
            tokens.append((LEX_LP, None)); i += 1; continue
        if ch == ")":
            tokens.append((LEX_RP, None)); i += 1; continue
        if ch in "\"'“”‘’":
            close = {'"': '"', "'": "'", "“": "”", "‘": "’"}[ch]
            j = query.find(close, i + 1)
            if j < 0:
                j = n
            tokens.append(("RAW", query[i + 1:j]))
            i = j + 1
            continue
        # +x / -x 前缀（Lucene 风格）
        if ch in "+-" and i + 1 < n and not query[i + 1].isspace():
            sign = ch
            j = i + 1
            while j < n and not query[j].isspace() and query[j] not in "()":
                j += 1
            piece = query[i + 1:j]
            # 拆分可能存在的字段前缀： -section:风险
            fld, rest = _split_field(piece)
            fuzz, pref = _split_wildcard(rest)
            if fuzz or pref:
                tokens.append(("RAW", rest if not fld else f"{fld}:{rest}"))
            else:
                tokens.append(("RAW", rest))
            tokens.append(("SIGN", sign))
            i = j
            continue
        # 普通片段：读到空白或括号
        j = i
        while j < n and not query[j].isspace() and query[j] not in "()":
            j += 1
        tokens.append(("RAW", query[i:j]))
        i = j

    # ---- 第二遍：把 RAW / SIGN 归约成 Lex
    pending_sign = ""
    for kind, val in tokens:
        if kind == "SIGN":
            pending_sign = val
            continue
        if kind == LEX_LP:
            if pending_sign:
                out.append(Lex(LEX_AND if pending_sign == "+" else LEX_NOT))
                pending_sign = ""
            out.append(Lex(LEX_LP)); continue
        if kind == LEX_RP:
            out.append(Lex(LEX_RP)); continue

        raw: str = val
        if pending_sign:
            out.append(Lex(LEX_AND if pending_sign == "+" else LEX_NOT))
            pending_sign = ""

        if _OP_RE.match(raw):
            out.append(Lex(raw.upper())); continue

        if not raw:
            continue

        fld, rest = _split_field(raw)
        if rest == "":
            # 只有 "section:" 这种残缺写法 → 当普通词处理
            fld, rest = "", raw

        def emit(lex: Lex) -> None:
            """字段限定要**逐个操作数**下发，而不是只作用于第一个。"""
            if fld:
                out.append(Lex(LEX_FIELD, fld))
            out.append(lex)

        if rest.startswith('"') or rest.startswith("“"):
            emit(Lex(LEX_PHRASE, rest.strip('"“”\'')))
            continue

        fuzz, pref = _split_wildcard(rest)
        if pref:
            emit(Lex(LEX_TERM, rest[:-1], 0, True)); continue
        if fuzz:
            emit(Lex(LEX_TERM, rest[: rest.rfind("~")], fuzz)); continue

        # ★ 关键的一步：**中文必须先分词**。
        # 中文查询没有空格，`报告期内武汉兴图新科…收入分别是多少` 整句会成为一个词法单元；
        # 而索引里存的是一颗颗切好的词，整句查下去必然 0 命中
        # （实测：修这一处之前，16 题评测集的全文路召回率是 0%）。
        # 切出来的多个词之间按**默认布尔**（OR）连接，由 BM25F 排序决定谁上来；
        # 短语（"…"）与通配/模糊写法不参与分词，它们的整体形态本身就是语义。
        sub = tokenize(rest)
        if len(sub) <= 1:
            emit(Lex(LEX_TERM, rest))
        else:
            for k, t in enumerate(sub):
                if k:
                    out.append(Lex(LEX_OR))
                emit(Lex(LEX_TERM, t))

    # 兜底：相邻两个「操作数」之间补 OR（Lucene 默认行为）。
    # ⚠️ 判据必须写成"**前一个是操作数** 且 **当前是操作数或左括号**"，
    # 不能把左括号也算进"前一个是操作数"里 —— 否则 `( A OR B )` 会被补成
    # `( OR A OR B )`，解析器在括号后立刻撞上 OR，报"无法解析的记号：OR"。
    # 这个 bug 只在带括号的布尔查询里出现，而括号恰好是工单演示要用的写法。
    fixed: list[Lex] = []
    operand = {LEX_TERM, LEX_PHRASE}
    for lx in out:
        prev = fixed[-1].kind if fixed else None
        if prev in operand and (lx.kind in operand or lx.kind == LEX_LP):
            fixed.append(Lex(LEX_OR))
        fixed.append(lx)
    return fixed


def _split_field(s: str) -> tuple[str, str]:
    m = re.match(r"^(section|title|body|summary|正文|标题|摘要|全文):(.*)$", s, re.I)
    if not m:
        return "", s
    alias = {"正文": "body", "标题": "title", "摘要": "summary", "全文": "body"}
    f = alias.get(m.group(1).lower(), m.group(1).lower())
    return f, m.group(2)


def _split_wildcard(s: str) -> tuple[int, bool]:
    """识别 `词~2` 与 `词*`。返回 (模糊距离, 是否前缀通配)。"""
    if s.endswith("*") and len(s) > 1:
        return 0, True
    m = re.search(r"~(\d+)$", s)
    if m:
        return int(m.group(1)), False
    if s.endswith("~"):
        return 1, False     # 裸 `~` 约定为距离 1
    return 0, False


# ------------------------------------------------------------ 查询 AST

@dataclass
class Node:
    """查询语法树节点。kind ∈ {term, phrase, and, or, not}。"""

    kind: str
    term: str = ""
    field_name: str = ""
    fuzzy: int = 0
    prefix: bool = False
    children: list["Node"] = field(default_factory=list)


class QuerySyntaxError(ValueError):
    pass


def parse(tokens: list[Lex]) -> Node:
    """递归下降解析：OR 优先级最低，其次 AND，其次 NOT，最后括号。"""
    pos = 0

    def peek() -> Lex | None:
        return tokens[pos] if pos < len(tokens) else None

    def parse_or() -> Node:
        nonlocal pos
        node = parse_and()
        while peek() and peek().kind == LEX_OR:
            pos += 1
            rhs = parse_and()
            node = Node("or", children=[node, rhs])
        return node

    def parse_and() -> Node:
        nonlocal pos
        node = parse_unary()
        # `A NOT B` 与 `A AND NOT B` 同义（Lucene 风格）。
        # 若这里只认显式 AND，`收入 NOT 风险` 会在 NOT 处停下来，
        # 解析器报「尾部有多余内容」——这是实测踩到的第一个语法坑。
        while peek() and peek().kind in (LEX_AND, LEX_NOT):
            if peek().kind == LEX_AND:
                pos += 1
            rhs = parse_unary()
            node = Node("and", children=[node, rhs])
        return node

    def parse_unary() -> Node:
        nonlocal pos
        if peek() and peek().kind == LEX_NOT:
            pos += 1
            return Node("not", children=[parse_unary()])
        return parse_primary()

    def parse_primary() -> Node:
        nonlocal pos
        lx = peek()
        if lx is None:
            raise QuerySyntaxError("查询式意外结束")
        if lx.kind == LEX_LP:
            pos += 1
            node = parse_or()
            if not (peek() and peek().kind == LEX_RP):
                raise QuerySyntaxError("缺少右括号")
            pos += 1
            return node
        if lx.kind == LEX_FIELD:
            pos += 1
            nxt = peek()
            if nxt is None or nxt.kind not in (LEX_TERM, LEX_PHRASE, LEX_LP):
                raise QuerySyntaxError(f"字段 {lx.value}: 后面缺少查询词")
            node = parse_primary()
            _apply_field(node, lx.value)
            return node
        if lx.kind == LEX_TERM:
            pos += 1
            return Node("term", term=lx.value, fuzzy=lx.fuzzy, prefix=lx.prefix)
        if lx.kind == LEX_PHRASE:
            pos += 1
            return Node("phrase", term=lx.value)
        raise QuerySyntaxError(f"无法解析的记号：{lx.kind} {lx.value}")

    node = parse_or()
    if pos != len(tokens):
        raise QuerySyntaxError(f"查询式尾部有多余内容：{tokens[pos].value}")
    return node


def _apply_field(node: Node, fld: str) -> None:
    """把字段限定递归下发到所有叶子。"""
    if node.kind in ("term", "phrase"):
        node.field_name = fld
    else:
        for c in node.children:
            _apply_field(c, fld)


def terms_of(node: Node) -> set[str]:
    """收集查询里出现的所有词（用于高亮与「命中词覆盖率」重排器）。"""
    if node.kind == "term":
        return {node.term}
    if node.kind == "phrase":
        return set(tokenize(node.term))
    out: set[str] = set()
    for c in node.children:
        out |= terms_of(c)
    return out


# ============================================================ 编辑距离

def edit_distance_leq(a: str, b: str, max_d: int) -> bool:
    """
    带上限的 Levenshtein：一旦当前行最小值就超过 max_d 直接返回 False。
    模糊匹配要在**整个词表**上跑，全额 DP 在 8 万词的表上会慢到不可用；
    早停能让绝大多数候选在第 1~2 轮就出局。
    """
    if a == b:
        return True
    if abs(len(a) - len(b)) > max_d:
        return False
    if max_d == 0:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        best = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            cur.append(v)
            if v < best:
                best = v
        if best > max_d:
            return False
        prev = cur
    return prev[-1] <= max_d


# ============================================================ 倒排索引

@dataclass
class FullTextHit:
    """一条全文检索命中。"""

    idx: int                      # chunk 在 kb.chunks 里的下标
    score: float
    matched: list[str] = field(default_factory=list)   # 实际命中的词
    fields_hit: list[str] = field(default_factory=list)
    exact: bool = False           # 是否有词/短语是**原样**命中的（用于「精确匹配」提示）

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "score": round(self.score, 4),
            "matched": self.matched,
            "fields_hit": self.fields_hit,
            "exact": self.exact,
        }


class InvertedIndex:
    """
    倒排索引 + BM25F 打分器。

    结构（三个字段：title / body / summary）：

        postings[field][term] = {doc_id: [pos, pos, ...]}
        doc_len[field][doc_id] = token 数
        avg_len[field]         = 平均长度

    为什么 title 与 body 分开存、summary 又单开一个：
      * title  = chunk 的 section（章节标题）。招股书的章节标题信息量极高，
                 权重给到 3.0（`FULLTEXT_FIELD_WEIGHTS`）。
      * body   = 正文全文，权重 1.0，是召回主力。
      * summary= 正文前 120 字。它是 body 的子串，看起来冗余，
                 但它是工单明确点名的「多字段」之一，
                 且实测在**短问句**（"公司注册资本是多少"）上，
                 摘要字段的权重加成能把「一开头就写答案」的块抬起来。

    线程安全：构建是单线程的，构建完只读；`search` 全程无状态，可并发。
    """

    FIELDS = ("title", "body", "summary")

    def __init__(self, chunks: list[dict], field_weights: dict[str, float] | None = None,
                 k1: float | None = None, b: float | None = None):
        self.chunks = chunks
        self.k1 = FULLTEXT_K1 if k1 is None else k1
        self.b = FULLTEXT_B if b is None else b
        self.field_weights = dict(field_weights or FULLTEXT_FIELD_WEIGHTS)
        self.postings: dict[str, dict[str, dict[int, list[int]]]] = {f: {} for f in self.FIELDS}
        self.doc_len: dict[str, list[int]] = {f: [] for f in self.FIELDS}
        self.avg_len: dict[str, float] = {}
        self.n_docs: int = 0
        self._terms_sorted: list[str] = []
        self._idf_cache: dict[str, float] = {}
        self.build_ms: float = 0.0
        self.build()

    # -------------------------------------------------- 构建
    def build(self) -> None:
        t0 = time.perf_counter()
        n = len(self.chunks)
        self.n_docs = n
        for f in self.FIELDS:
            self.doc_len[f] = [0] * n

        for i, c in enumerate(self.chunks):
            body = c.get("text") or ""
            fields = {
                "title": c.get("section") or "",
                # 图像/表格块没有正文之外的补充，caption 也算进标题侧信息
                "body": body,
                "summary": body[:FULLTEXT_SUMMARY_CHARS],
            }
            if c.get("caption"):
                fields["title"] = (fields["title"] + " " + str(c["caption"])).strip()
            for f, txt in fields.items():
                toks = tokenize(txt)
                self.doc_len[f][i] = len(toks)
                if not toks:
                    continue
                post = self.postings[f]
                for pos, t in enumerate(toks):
                    post.setdefault(t, {}).setdefault(i, []).append(pos)

        for f in self.FIELDS:
            lens = self.doc_len[f]
            self.avg_len[f] = (sum(lens) / n) if n else 0.0

        self._terms_sorted = sorted(
            {t for f in self.FIELDS for t in self.postings[f]},
            key=lambda s: (len(s), s),
        )
        self.build_ms = (time.perf_counter() - t0) * 1000

    # -------------------------------------------------- IDF
    def df(self, term: str, fld: str | None = None) -> int:
        if fld:
            return len(self.postings[fld].get(term, ()))
        seen: set[int] = set()
        for f in self.FIELDS:
            seen |= set(self.postings[f].get(term, ()))
        return len(seen)

    def idf(self, term: str) -> float:
        """
        BM25 的 IDF。用 Lucene 的平滑式 `ln(1 + (N-df+0.5)/(df+0.5))`，
        它对 df > N/2 的**无区分度词**给出接近 0 的正值 —— 不会出现
        经典公式在 df 很大时变负、把一个「到处都是的词」判成负相关的怪现象。
        （招股书里「公司」「报告期内」就是这类词，工单1~5 是靠在检索式里
        DF 过滤把它们剔掉；这里换成 Lucene 平滑式后天然压制，两条路可以共存。）
        """
        v = self._idf_cache.get(term)
        if v is not None:
            return v
        d = self.df(term)
        v = math.log(1.0 + (self.n_docs - d + 0.5) / (d + 0.5)) if self.n_docs else 0.0
        self._idf_cache[term] = v
        return v

    # -------------------------------------------------- 词表扩展（模糊/通配）
    def expand(self, term: str, fuzzy: int = 0, prefix: bool = False) -> list[str]:
        """
        把查询词扩展成索引里真实存在的词。

        * 精确命中 → 只有它自己（不做无谓扩展，保持精度）；
        * 前缀通配 `销售*` → 所有以它开头的词，按 df 降序取前 N（高频的更可能是用户想要的）；
        * 模糊 `注册酱~1` → 编辑距离 ≤1 的词，按 (距离, -df) 排序取前 N。

        单字符查询（如「部」）不参与模糊扩展 —— 一个汉字和几十个汉字都只差 1 步距离，
        扩出来全是噪音。
        """
        out: list[str] = []
        if prefix:
            for t in self._terms_sorted:
                if t != term and t.startswith(term):
                    out.append(t)
                    if len(out) >= FULLTEXT_MAX_FUZZY_EXPAND:
                        break
            # 前缀命中的词按 df 降序更合理：`销售*` 应先扩到「销售」「销售处」
            out.sort(key=lambda t: (-self.df(t), t))
            out = out[:FULLTEXT_MAX_FUZZY_EXPAND]
            if self.df(term):
                out.insert(0, term)
            return out

        if fuzzy > 0 and len(term) >= 2:
            cands = [(edit_distance_leq(term, t, fuzzy), t) for t in self._terms_sorted
                     if abs(len(t) - len(term)) <= fuzzy and t != term]
            hit = [t for ok, t in cands if ok]
            hit.sort(key=lambda t: (-self.df(t), t))
            out = hit[:FULLTEXT_MAX_FUZZY_EXPAND]
            if self.df(term):
                out.insert(0, term)
            return out

        return [term] if self.df(term) else []

    # -------------------------------------------------- 错别字容错
    def repair_text(self, text: str) -> tuple[str, list[dict]]:
        """
        **错别字容错**（工单编号：人工智能NLP-RAG-混合检索任务 —— 「模糊查询」能力）。

        场景：用户想问「注册资本」，手滑敲成「注册酱」。
        jieba 会切成「注册」+「酱」，而「酱」在索引里 df=0：

        * 全文路：它命中不了任何文档，等于这个词白问了；
        * 向量路：更糟 —— 错字污染整句语义，实测把正确页 p60 换成无关页 p214。

        错别字分两种，**修法不一样**，所以这里走两条分支：

        * **A. 整字被替换**（「资本」→「酱」，差 2 步）：编辑距离救不了
          （`注册酱~1` 实测扩出的是「注册地/注册号/注册证」，根本不是「注册资本」）。
          → 把它当作孤立错字，用**前面那个紧邻实词做前缀补全**，
            取词典里 df 最高的候选，「注册酱」整段替换成「注册资本」。
          取 df 最高站得住：正确写法在招股书里必然反复出现。
        只做分支 A，**不做**"等长 + 编辑距离 ≤1"的整词替换（曾实现过，实测砍掉了）：
        「量子加密」被改成「电子加密」、「区块链技木」被改成「模块链技木」——
        生僻词和错字在"df=0"这一点上长得一模一样，等长+距离1 根本区分不了，
        负收益远大于正收益。宁可漏改，不可改坏。

        收紧条件（每多放一个条件都可能把正常查询改坏）：
          ① 疑似错字必须 df=0（索引里压根没有）且长度 ≤ 2
             —— 长词交给用户显式写 `词~1`，或者靠 jieba 切出的其它词兜底
                （「注删资本」会被切成「注删」+「资本」，后者 df=125 照常生效）；
          ② 前一词必须**字符紧邻**（pe == s），否则是「A xx B」这种无关拼接；
          ③ 前一词 df ∈ (0, FULLTEXT_REPAIR_MAX_DF] —— 排除「公司」这类泛词；
          ④ 补全候选必须比前一词长，且 df ≥ max(4, df(prev)×0.3) ——
             否则说明词典里压根没有像样的候选，宁可不改。

        返回 (改写后的文本, 改写记录) —— 记录会进 trace，演示时能看到
        「系统把『注册酱』纠正成了『注册资本』」。
        """
        if not text or not self.n_docs:
            return text, []
        limit = FULLTEXT_REPAIR_MAX_DF

        # 带字符位置地切中文段（替换需要位置，普通 tokenize 给不了）
        spans: list[tuple[str, int, int]] = []
        jb = _jieba()
        for blk in re.finditer(r"[\u4e00-\u9fff]+", text):
            base = blk.start()
            for t, s, e in jb.tokenize(blk.group(0)):
                tok = t.strip()
                if tok and not _is_punct(tok):
                    spans.append((tok, base + s, base + e))

        edits: list[tuple[int, int, str, str]] = []
        for i, (tok, s, e) in enumerate(spans):
            if self.df(tok) != 0 or len(tok) < 1:      # ① 只处理索引里没有的词
                continue

            # ---- 分支 A：短错字（≤2 字）→ 用紧邻前一词前缀补全 ------------
            if len(tok) <= 2 and i > 0:
                ptok, ps, pe = spans[i - 1]
                if pe != s or len(ptok) < 2:          # ② 必须紧邻，前一词不能是单字
                    pass
                elif not (0 < self.df(ptok) <= limit):  # ② 排除泛词/冷僻词
                    pass
                else:
                    # 前缀补全：以 ptok 开头且更长，取 df 最大的那个
                    best, bdf = "", 0
                    for t in self._terms_sorted:
                        if len(t) > len(ptok) and t.startswith(ptok):
                            d = self.df(t)
                            if d > bdf:
                                best, bdf = t, d
                    if best and bdf >= max(4, self.df(ptok) * 0.3):   # ③
                        edits.append((ps, e, tok, best))
                continue

            # 长度 ≥3 的 df=0 词不自动改（理由见 docstring：生僻词与错字在
            # df=0 这一点上长得一样，改坏的风险大于漏改）。
            # 交给用户显式写 `词~1`，或靠 jieba 切出的其余词兜底。

        if not edits:
            return text, []
        # 从后往前替换，避免前面的替换把后面的偏移弄乱
        out = text
        notes: list[dict] = []
        for ps, e, tok, best in sorted(edits, key=lambda x: -x[0]):
            out = out[:ps] + best + out[e:]
            notes.append({"from": text[ps:e], "to": best, "bad": tok})
        return out, list(reversed(notes))

    # -------------------------------------------------- 叶子节点打分
    def _term_scores(self, term: str, fld: str | None, fuzzy: int,
                     prefix: bool) -> tuple[dict[int, float], set[str], set[str]]:
        """
        对单个 term 求「伪词频 f'」并按 BM25F 打分。
        返回 (doc_id -> score, 命中的真实词集合, 命中字段集合)。
        """
        expanded = self.expand(term, fuzzy, prefix)
        if not expanded:
            return {}, set(), set()

        # 模糊扩展时给「不是原词」的匹配打折，避免错别字命中压过精确词
        idf_sum = sum(self.idf(t) for t in expanded)
        fields = [fld] if fld in self.FIELDS else list(self.FIELDS)

        pseudo_tf: dict[int, float] = {}
        fields_hit: dict[int, set[str]] = {}
        for t in expanded:
            w_pen = 1.0 if t == term else 0.55
            for f in fields:
                post = self.postings[f].get(t)
                if not post:
                    continue
                wf = float(self.field_weights.get(f, 1.0))
                bf = self.b
                avglen = self.avg_len.get(f) or 1.0
                for d, positions in post.items():
                    dl = self.doc_len[f][d] or 1
                    norm = 1.0 - bf + bf * (dl / avglen)
                    # BM25F：字段内先做长度归一，再按字段权重加权合并
                    pseudo_tf[d] = pseudo_tf.get(d, 0.0) + wf * w_pen * (len(positions) / norm)
                    fields_hit.setdefault(d, set()).add(f)

        k1 = self.k1
        out: dict[int, float] = {}
        for d, ftf in pseudo_tf.items():
            out[d] = idf_sum * (ftf * (k1 + 1.0)) / (ftf + k1)
        return out, set(expanded), {d: sorted(v) for d, v in fields_hit.items()}  # type: ignore[return-value]

    def _phrase_scores(self, phrase: str, fld: str | None) -> tuple[dict[int, float], set[str]]:
        """短语匹配：要求各词位置连续。返回 (doc_id -> score, 命中字段)。"""
        toks = tokenize(phrase)
        if not toks:
            return {}, set()
        if len(toks) == 1:
            sc, _, fh = self._term_scores(toks[0], fld, 0, False)
            return sc, {d: ",".join(v) for d, v in fh.items()}  # type: ignore[return-value]

        fields = [fld] if fld in self.FIELDS else list(self.FIELDS)
        # 逐字段做位置交集
        acc: dict[int, float] = {}
        fhit: dict[int, str] = {}
        for f in fields:
            posts = [self.postings[f].get(t) for t in toks]
            if any(p is None for p in posts):
                continue
            first: dict[int, list[int]] = posts[0]  # type: ignore[assignment]
            for d, poses in first.items():
                cnt = 0
                for start in poses:
                    ok = True
                    for off, post in enumerate(posts[1:], 1):
                        if start + off not in (post or {}).get(d, ()):  # type: ignore[union-attr]
                            ok = False
                            break
                    if ok:
                        cnt += 1
                if cnt:
                    # 短语的 IDF 用「最难命中的那个词」代表 —— 取 max 而不是 sum，
                    # 否则长短语会因为词多而分数虚高，把「恰好包含长串」的块顶到很前面
                    idf = max(self.idf(t) for t in toks)
                    wf = float(self.field_weights.get(f, 1.0))
                    acc[d] = acc.get(d, 0.0) + idf * (cnt ** 0.5) * wf
                    fhit[d] = ",".join(sorted(set(fhit.get(d, "").split(",")) - {""} | {f}))
        return acc, fhit

    # -------------------------------------------------- AST 求值
    def evaluate(self, node: Node, mode: str = "or",
                 stop_terms: set[str] | None = None) -> tuple[dict[int, float], dict[int, dict]]:
        """
        求值查询树。返回 (doc_id -> 总分, doc_id -> 命中详情)。

        `mode` 只影响**隐含布尔**（没写 AND 的多个词）：
          * "or"  默认 —— 取并集，分数相加（Lucene 就是这套，召回优先）
          * "and" 严格 —— 取交集，但**分数仍用 OR 的和**，不用最小值，
            否则「两个词都命中」反而比「命中一个」分低，排序就反了。

        `stop_terms`：**语料级无区分度词**（工单1~5 的 DF 过滤，在这里换了个做法）。
        旧做法是在**字符串层面**把词从检索式里删掉 —— 那是为 BM25 单路设计的；
        一旦查询里带布尔语法（`收入 AND 军用`），字符串删除会把 `AND` 拼掉、
        把 `section:风险提示` 拆散，语法当场失效。
        这里改成**在叶子节点上拦截**：命中停用词的 term 直接返回空集，
        但 AND / OR / NOT / 字段限定的结构完全保留。
        """
        stop = stop_terms or set()
        scores: dict[int, float] = {}
        detail: dict[int, dict] = {}

        def note(d: int, term: str, flds: set[str], exact: bool) -> None:
            rec = detail.setdefault(d, {"matched": set(), "fields": set(), "exact": False})
            rec["matched"].add(term)
            rec["fields"] |= flds
            if exact:
                rec["exact"] = True

        def ev(nd: Node) -> dict[int, float]:
            if nd.kind == "term":
                if nd.term in stop:
                    # 停用词：不参与打分。注意**只跳过分词后的整词**，
                    # 显式短语（"军用领域"）与通配/模糊写法不受影响。
                    return {}
                sc, expanded, fh = self._term_scores(nd.term, nd.field_name, nd.fuzzy, nd.prefix)
                for d in sc:
                    note(d, nd.term, set(fh.get(d, [])), nd.term in expanded)
                return sc
            if nd.kind == "phrase":
                sc, fh = self._phrase_scores(nd.term, nd.field_name)
                for d in sc:
                    note(d, f'"{nd.term}"', set(str(fh.get(d, "")).split(",")) - {""}, True)
                return sc
            if nd.kind == "not":
                sub = ev(nd.children[0])
                # NOT 单独出现没有意义（全集太大），约定为「在候选内排除」，由父节点处理。
                # 这里用一个"反相"标记：得分取负、并标记为排除集
                return {d: -s for d, s in sub.items()}
            if nd.kind == "and":
                # 先收集各子节点的原始结果，再按 mode 决策
                parts = [ev(c) for c in nd.children]
                negs = [p for p, c in zip(parts, nd.children) if c.kind == "not"]
                poss = [p for p, c in zip(parts, nd.children) if c.kind != "not"]
                if not poss:
                    return {}
                excluded: set[int] = set()
                for ng in negs:
                    excluded |= set(ng)
                if mode == "and" and len(poss) > 1:
                    keys: set[int] = set(poss[0])
                    for p in poss[1:]:
                        keys &= set(p)
                else:
                    keys = set().union(*[set(p) for p in poss])
                keys -= excluded
                return {d: sum(p.get(d, 0.0) for p in poss) for d in keys}
            if nd.kind == "or":
                parts = [ev(c) for c in nd.children]
                negs = [p for p, c in zip(parts, nd.children) if c.kind == "not"]
                poss = [p for p, c in zip(parts, nd.children) if c.kind != "not"]
                keys = set().union(*[set(p) for p in poss]) if poss else set()
                for ng in negs:
                    keys -= set(ng)
                return {d: sum(p.get(d, 0.0) for p in poss) for d in keys}
            raise QuerySyntaxError(f"未知节点类型 {nd.kind}")

        scores = ev(node)
        return scores, detail

    # -------------------------------------------------- 对外接口
    def search(self, query: str, top_k: int = 20, mode: str = "or",
               ast: Node | None = None,
               stop_terms: set[str] | None = None) -> tuple[list[FullTextHit], dict]:
        """
        执行一次全文检索。返回 (命中列表, trace)。

        trace 里带 parse 后的等价查询式与耗时分解，前端「检索链路」面板直接展示，
        让「全文检索到底做了什么」这件事看得见 —— 也是工单6 演示视频要讲的内容。
        """
        t0 = time.perf_counter()
        trace: dict = {"engine": "inverted-index/BM25F", "mode": mode}
        try:
            node = ast if ast is not None else parse(lex(query))
        except QuerySyntaxError as exc:
            # 语法写错不该让整个检索失败 —— 退化成一个"按分词结果的 OR 查询"。
            # 注意这里**直接构造 AST**，而不是把 token 再喂回 parse()：
            # 分词结果里会保留 `and`/`or`/`not` 这些小写的拉丁词（tokenize 统一转小写），
            # 再喂回去会被解析器当成裸操作数，在第二个词处就报"尾部有多余内容"，
            # 于是 fallback 自己又抛一次异常 —— 等于没有兜底。
            trace["error"] = f"查询语法错误：{exc}"
            trace["fallback"] = "退化为按分词结果 OR 连接的查询"
            toks = [t for t in tokenize(query)
                    if t and t.lower() not in ("and", "or", "not")][:12]
            node = Node("or", children=[Node("term", term=t) for t in toks]) if toks \
                else Node("term", term=query)
        parse_ms = (time.perf_counter() - t0) * 1000

        t = time.perf_counter()
        scores, detail = self.evaluate(node, mode=mode, stop_terms=stop_terms)
        eval_ms = (time.perf_counter() - t) * 1000
        trace["stop_terms"] = len(stop_terms or ())

        # 邻近度加成：命中词在**同一字段内**的平均间距越小，越可能在同一句话里
        t = time.perf_counter()
        prox_ms = 0.0
        hits: list[FullTextHit] = []
        order = sorted(scores.items(), key=lambda kv: -kv[1])[: max(top_k * 4, top_k)]
        for d, sc in order:
            rec = detail.get(d, {})
            hits.append(FullTextHit(
                idx=d,
                score=float(sc),
                matched=sorted(rec.get("matched", set())),
                fields_hit=sorted(rec.get("fields", set())),
                exact=bool(rec.get("exact")),
            ))
        prox_ms = (time.perf_counter() - t) * 1000

        hits = hits[:top_k]
        trace.update({
            "n_candidates": len(scores),
            "parse_ms": round(parse_ms, 3),
            "eval_ms": round(eval_ms, 3),
            "rank_ms": round(prox_ms, 3),
            "total_ms": round((time.perf_counter() - t0) * 1000, 3),
            "index_terms": len(self._terms_sorted),
            "index_docs": self.n_docs,
            "index_build_ms": round(self.build_ms, 2),
            "parsed_terms": sorted(terms_of(node)),
        })
        return hits, trace

    def keyword_coverage(self, text: str, terms: set[str]) -> float:
        """文本对查询词的覆盖率（0~1）。「用户反馈自适应重排器」与关键词重排器都会用。"""
        if not terms:
            return 0.0
        toks = set(tokenize(text))
        return sum(1 for t in terms if t in toks) / len(terms)

    def stats(self) -> dict:
        return {
            "docs": self.n_docs,
            "terms": len(self._terms_sorted),
            "fields": {f: {"postings_terms": len(self.postings[f]),
                           "avg_len": round(self.avg_len[f], 1)} for f in self.FIELDS},
            "field_weights": self.field_weights,
            "k1": self.k1,
            "b": self.b,
            "build_ms": round(self.build_ms, 2),
        }


# ============================================================ 单例

_INDEX: InvertedIndex | None = None
_INDEX_KEY: Any = None


def get_index(kb=None) -> InvertedIndex:
    """
    取倒排索引单例（按知识库 id + 分块条数缓存）。

    构建一次约 1~3 秒（3000 块、90 万 token），在服务启动预热时完成，
    单次检索不再付这个成本。
    """
    global _INDEX, _INDEX_KEY
    if kb is None:
        from .index_store import KnowledgeBase

        kb = KnowledgeBase.get()
    key = (id(kb), len(kb.chunks))
    if _INDEX is None or _INDEX_KEY != key:
        _INDEX = InvertedIndex(kb.chunks)
        _INDEX_KEY = key
    return _INDEX


def reset_index() -> None:
    """索引重建后调用。"""
    global _INDEX, _INDEX_KEY
    _INDEX = None
    _INDEX_KEY = None
