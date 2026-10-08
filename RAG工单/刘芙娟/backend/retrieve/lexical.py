"""BM25 关键词检索。**本特性的"关键词"那一半。**

---

## 为什么自实现而不引入 `rank_bm25`

1. `rank_bm25` 是纯 Python 实现，与自写在性能上无差别，却多一个依赖、多一层
   版本耦合。本项目已有先例：specs/003 的 D2 裁决用环境里已有的 `transformers`
   而非引入 `FlagEmbedding`。装一个用不上的包违反 constitution 原则 I。
2. **IDF 公式各版本有差异**（有些用 `log((N−df+0.5)/(df+0.5))` 且允许负值）。
   自写才能锁定公式，而锁定的公式正是"同问题两次检索结果一致"（SC-006）的前提。
3. 语料只有几十到几千条，公式就是全部成本。

## 为什么用 jieba 而不是字符 2-gram

医疗问句的核心是**药品通用名与疾病分型名**（"盐酸氨氯地平"、"2 级高血压"）。
字符 2-gram 会把它们切成跨词片段，直接削弱"照抄术语能命中"这条本特性存在的
理由（US1）。jieba 无医疗专用词典，但本特性不要求分词边界与人工标注一致 ——
验收以端到端命中率为准（SC-001）。

---

⚠️ **检索链路上 MUST NOT 存在第二个分词入口。**

索引期把「盐酸氨氯地平」切成 A、查询期切成 B，则 BM25 永远返回空 —— 而且
**不报错、不告警**，表现为"关键词检索时灵时不灵"。这是分词类系统最经典的静默
失效，靠"索引与查询共用同一个 `tokenize()`"从结构上消除。

> 说明：`backend/clean/space_merge.py`（S2 的清洗管线）也用 jieba，但那服务于
> "词内空格合并"，与检索**没有任何关系** —— 它的输入是清洗阶段的段落文本，
> 不参与 BM25。两者唯一需要保持一致的只有一件事：**本项目运行时里 jieba 的
> 词典是同一份**（同一个解释器、同一个包），这一点不需要代码维护。
> 本文件"唯一入口"的约束范围因此是 `backend/retrieve/` 内的索引侧与查询侧。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import jieba

from . import DEFAULT_BM25_B, DEFAULT_BM25_K1
from .models import CorpusChunk

# ⚠️ 关掉 jieba 的 `Building prefix dict` / `Dumping model` 等日志。
#
# 不是洁癖：它们打到 stderr，会把服务启动输出冲散，而 `quickstart.md` §5.1
# 要靠那几行判断"关键词索引到底建起来没有"。放在**导入期**而不是 `warm_up()`
# 里，是因为 `selfcheck` 也会走到分词（`_sc_tokenize`）但不调 `warm_up()` ——
# 而它在同样需要干净输出的场合被使用。
#
# `backend/clean/space_merge.py` 出于同样的理由做了同样的事。
jieba.setLogLevel(60)

__all__ = [
    "tokenize",
    "LexicalIndex",
    "build_index",
    "search",
    "coverage",
    "warm_up",
]


def warm_up() -> None:
    """预热 jieba 词典（首次调用会加载，约 0.5–1 s）。

    MUST 在**启动期**调用，MUST NOT 让第一次用户提问承担这个代价 ——
    与 S8 的"启动期加载 BGE-M3 权重"是同一取向。

    日志级别已在导入期设好（见文件头），这里只负责把词典加载进内存。
    """

    jieba.lcut("预热")


def _is_content(token: str) -> bool:
    """token 是否含有实际内容（至少一个字母、数字或汉字）。

    纯标点与空白在建索引时是噪声：它们几乎出现在**每一个** chunk 里，df 接近
    N，IDF 接近 0，对排序没有贡献，却会让倒排表大出一大截。查询侧同理 ——
    用户输入「血压高？？？」时的问号不该参与打分。
    """

    for ch in token:
        if ch.isalnum() or "一" <= ch <= "鿿":
            return True
    return False


def tokenize(text: str) -> list[str]:
    """**索引期与查询期的唯一分词入口。**

    处理顺序：`jieba.lcut` → 小写化 → 丢弃无内容 token。

    小写化是为了让英文药名与缩写大小写不敏感（"ACEI" 与 "acei" 应命中同一条）。
    中文不受影响。
    """

    return [
        lowered
        for token in jieba.lcut(text, cut_all=False)
        if _is_content(token) and (lowered := token.lower())
    ]


@dataclass
class LexicalIndex:
    """倒排索引 + BM25 统计量。

    不变量：`n_docs == len(doc_len) == len(chunk_ids)`；`df[t] == len(postings[t])`。

    `postings` 的值是 `[(doc_index, tf), ...]`。用 `doc_index`（整数）而不是
    `chunk_id`（字符串）：查询期最内层循环里做的是字典查找，整数键更省，
    且 `chunk_ids` 提供了从下标回查标识的唯一映射。
    """

    postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    doc_len: list[int] = field(default_factory=list)
    df: dict[str, int] = field(default_factory=dict)
    chunk_ids: list[str] = field(default_factory=list)
    avgdl: float = 1.0
    k1: float = DEFAULT_BM25_K1
    b: float = DEFAULT_BM25_B

    # `{chunk_id: 该 chunk 的 token 集合}`。用于算"查询词覆盖率" —— 准入裁定
    # 需要的第二个信号（见 `__init__.py` 的 DEFAULT_LEXICAL_MIN_COVERAGE）。
    #
    # 存成集合而不是复用 `CorpusChunk.tokens`：覆盖率判定每次检索都要对一个
    # chunk 查 N 个词，用 list 是 O(N·M)，用 set 是 O(N)。索引是只读的，
    # 这点内存（65 个集合）换掉查询期的线性扫描是划算的。
    token_sets: dict[str, frozenset[str]] = field(default_factory=dict)

    @property
    def n_docs(self) -> int:
        return len(self.chunk_ids)


def build_index(
    chunks: list[CorpusChunk],
    k1: float = DEFAULT_BM25_K1,
    b: float = DEFAULT_BM25_B,
) -> LexicalIndex:
    """用启动期拉回的语料建倒排索引。

    **空语料 MUST NOT 抛异常**：返回一个 `avgdl=1.0` 的空索引，关键词路查它
    得到空列表（Edge Case：库为空是正常状态，不是错误）。

    `chunks` 的 `tokens` / `tokens_len` 若已填好（`store.fetch_corpus` 会填），
    直接复用；未填则在此处补算 —— 两条路径共用 `tokenize()`，不会有第二套口径。
    """

    index = LexicalIndex(k1=k1, b=b)

    for doc_index, chunk in enumerate(chunks):
        tokens = chunk.tokens or tokenize(chunk.text)

        # 回填到语料条目上。查询期只用到 `doc_len`（已在索引里），但 `corpus`
        # 子命令要报告平均文档长度与高频词，那时再分词一遍就是重复劳动。
        chunk.tokens = tokens
        chunk.tokens_len = len(tokens)

        index.token_sets[chunk.chunk_id] = frozenset(tokens)
        index.chunk_ids.append(chunk.chunk_id)
        index.doc_len.append(len(tokens))

        # 先在本地累计 tf，再一次性追加进 postings —— 逐 token 追加会让
        # 同一个词的多个位置产生多个长度 1 的列表项，后面还要再去合并。
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        for token, count in tf.items():
            index.postings.setdefault(token, []).append((doc_index, count))

    index.df = {term: len(items) for term, items in index.postings.items()}

    # avgdl 的除零保护。空语料时取 1.0 而非 0.0：0.0 会让 BM25 的长度归一化项
    # 在后续任何一次打分中变成 0/0。空索引本就不会被打分，但让一个"用不上的
    # 值"是安全的，比让它是 NaN 更省心。
    total = sum(index.doc_len)
    index.avgdl = (total / len(index.doc_len)) if index.doc_len else 1.0

    return index


def _idf(df: int, n_docs: int) -> float:
    """`ln(1 + (N − df + 0.5)/(df + 0.5))` —— **恒正**（research R1 锁定的公式）。

    加 1 的作用是把 IDF 压到正数域：`df = N` 时（词出现在每个文档里）
    括号内为 `1 + 0.5/(N+0.5) > 1`，取对数仍为正。

    为什么这件事重要：有些实现用不带 `1 +` 的版本，`df > N/2` 时 IDF 变负 ——
    于是"文档包含查询词"反而**降低**分数。那类实现在语料小、查询词常见时
    （正是本项目当前的状态：65 chunks）会给出反直觉的排序。
    """

    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def coverage(index: LexicalIndex, chunk_id: str, terms: set[str]) -> float:
    """该 chunk 命中的查询词占比（0.0–1.0）。

    **这是关键词路准入的第二个信号**（2026-09-28 裁决）。BM25 名次单独不可用：
    它对任何查询都必然返回前 N 名，包括"今天晚饭吃什么好"这种与知识库完全
    无关的提问 —— 而 constitution 原则 II 要求这种情况必须拒答。

    `terms` 为空时返回 0.0（而不是 1.0 或除零）：空查询不对应任何"命中"，
    让它拿到 0.0 会让准入判为不通过，这是保守且正确的方向。

    实测（语料 65 chunks）：
        氢氯噻嗪 → 1.00   今天晚饭吃什么好 → 0.20
    """

    if not terms:
        return 0.0
    return len(terms & index.token_sets.get(chunk_id, frozenset())) / len(terms)


def search(index: LexicalIndex, question: str, limit: int) -> list[tuple[str, float]]:
    """BM25 检索，返回 `[(chunk_id, 分数)]` 按分数降序。

    只对**查询词命中过的文档**打分（借倒排表，复杂度与命中数而非语料规模相关）。

    排序键带 `chunk_id` 末位，理由与 `fuse.sort_key` 完全相同：分数相同的两条
    必须有确定顺序，否则同一问题两次检索可能给出不同结果（SC-006）。
    """

    terms = set(tokenize(question))
    if not terms or not index.postings:
        return []

    scores: dict[int, float] = {}

    for term in terms:
        postings = index.postings.get(term)
        if not postings:
            continue

        idf = _idf(index.df[term], index.n_docs)

        for doc_index, tf in postings:
            norm = 1.0 - index.b + index.b * (index.doc_len[doc_index] / index.avgdl)
            scores[doc_index] = scores.get(doc_index, 0.0) + idf * (
                tf * (index.k1 + 1.0)
            ) / (tf + index.k1 * norm)

    # 负分数 + chunk_id：把降序写成升序，并带上确定性的末位键。
    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], index.chunk_ids[item[0]]),
    )

    return [(index.chunk_ids[doc_index], score) for doc_index, score in ranked[:limit]]
