# -*- coding: utf-8 -*-
"""混合检索：向量召回 + BM25 关键词召回，两路融合后交给 Reranker 精排。

BM25 用 chunk 的 text 字段自建索引（纯标准库实现，中文按「单字 + 相邻双字」切分），
对应设计文档里的「混合检索：向量检索 + BM25」。

**t68（健壮性与可观测）**：关键词通道在真实规模（14 万 chunk）下曾把整库行（含 1024 维
向量）与全量 ``Counter`` 一次性攒进内存 ⇒ 真实首问 ``MemoryError``，日志里静默降级成
纯向量、一次请求 171.5s（t26 残留 R1 的实测）。现在的做法：

1. **不吃向量**：走 ``store.iter_all_chunks(with_vector=False)``（逐页流式，页级内存）；
2. **压缩索引**：倒排表用 ``array`` 存（同一条打分公式、同样的查询项顺序 ⇒ 分数与
   改前**逐位一致**），不再为每个文档留一个 ``Counter``；
3. **不阻塞请求**：索引在**后台线程**里建；未就绪时关键词通道返回空并**可见降级**
   （日志 WARNING + 计数器 ``keyword_recall_degraded_total{reason}``）；
4. **降级原因可查**：取块窗口被 Milvus 截断 / 建索引异常 / 索引在建 / 未启用 —— 各自
   一条日志 + 一条计数（口径登记在 :data:`METRIC_CATALOG`）。
"""
from __future__ import annotations

import logging
import math
import os
import pickle
import re
import threading
import time
from array import array
from collections import Counter
from hashlib import sha1
from pathlib import Path
from typing import Iterable, Iterator

from .. import metrics as M
from ..config import RetrievalConfig
from ..embedding.base import Embedder
from ..embedding.offline import OfflineEmbedder
from ..schemas import Chunk, SearchHit
from ..store.base import VectorStore
# P0.5：函数级耗时观测（L1 直方图 + L2 慢调用清单），见 docs/REFACTOR-PLAN.md §3.7
from ..observability import timed
from .law_scope import LawScope
from .rerank import Reranker
from .term_map import expand_query_logged
from .version_filter import VersionFilter

logger = logging.getLogger(__name__)

# 复用离线向量化的切分规则，保证向量侧与关键词侧的词元一致
_tokenize = OfflineEmbedder._terms


def _content_terms(text: str) -> set[str]:
    """``text`` 里的**实义词元** = 长度 >= 2 的词元（双字词 / 英文词）。

    单字词元（的、是、不、一……）在中文法律语料里几乎处处命中，**不算证据**：
    真实链路实测非法律问题「推荐几部好看的科幻电影」正是靠单字命中的关键词通道
    把融合分抬到 1.8977（只比法律题下限 1.8544 低一点点），所以闸门除了看分数，
    还必须看有没有「长度 >= 2 的词元」真的落在碎片正文里。
    """
    return {term for term in _tokenize(text) if len(term) >= 2}


#: 「词面证据」的门槛 / 下限：**同一条碎片正文**里要落进这么多个实义词元才算有依据。
#: 取 3 是 t109+t115 两轮真链路标定的结果（逐题读数表见 ``.pytmp/t115/diag-*.json``、
#: ``.pytmp/t109/overlap-console.log``）：
#:   * 法律题 5~12 个（民间/借贷/利率/司法/保护/上限…；劳动/争议/仲裁/时效…）；
#:   * 非法律题 0~2 个（「推荐几部好看的科幻电影」与《电影产业促进法》共享
#:     "推荐/电影" 恰好 2 个，融合分却被抬到 1.8977 ⇒ 只共享 2 个的一律不算证据）；
#:   * 边界题「房东不退押金」只共享 1 个（押金）⇒ 走引导补充，不硬答。
#: 这个门槛**尺度无关**（不看分数），所以关键词通道还在建索引、分数不可比时照样能执行
#: —— t115 用它堵住启动窗口的伪依据。
_MIN_CONTENT_TERMS = 3

#: 预热/降级窗口的**用户可见**提示（t115 F2）。关键词通道（BM25 索引）在后台建时
#: （真实语料 11.9 万块约 30~75s），本次召回只有向量通道 —— 用户必须看得见"资料可能不全"，
#: 而不是拿到一个看起来完整的答案（COPY-STANDARD 第 C 条：解释系统在忙什么）。
#: 文案只给人看：不出现 BM25/索引/dim/集合名这类内部词，也不吓人、不甩锅。
WARMUP_NOTE = ("[提示] 法规资料还在预热（大约 1 分钟内完成），这次用到的资料可能不全，"
               "答案未必完整；稍后再问一次会更准。")

#: **尚未生效的新版**被用上时必须说的那句话（用户 2026-09-23 拍板的口径：
#: 回答现行问题用**现行有效版**；一旦用到了尚未施行的新版条文，必须在答案里提示，
#: 否则律师可能把"还没生效的条文"当成现行法用）。
PENDING_VERSION_NOTE = ("[提示] 这次用到的资料里含**尚未施行**的新版条文（{sources}），"
                        "请以现行有效版本为准；如需按新版分析，请另行说明。")


def _display_source(name: str) -> str:
    """给用户看的来源名：去掉 ``__<hash>`` 复制件后缀与扩展名（与引用展示同一口径）。"""
    stem = Path(str(name or "")).name
    if stem.endswith(".md"):
        stem = stem[:-3]
    return re.sub(r"__[0-9a-f]{6,}$", "", stem, flags=re.IGNORECASE)

#: 新指标口径（t68）。登记时机 = 本模块 import 时。
#:
#: 为什么写在这里而不是 `metrics.py`：本任务的 inScope 是 `legal_rag/retrieve` 与
#: `legal_rag/store`，`metrics.py` 不在其中；而项目的硬约定是"计数器必须登记进
#: `METRIC_CATALOG`"（否则 `/metrics` 渲染不出 TYPE/HELP）。`setdefault` 保证不覆盖
#: 既有条目、可重复 import，且口径就写在与发射点同一个文件里。
_NEW_METRICS = {
    "keyword_index_build_seconds": ("histogram", "BM25 索引全量建索引耗时", "seconds"),
    "keyword_index_docs": ("histogram", "BM25 索引收录的文档（chunk）条数", "items"),
    "keyword_recall_degraded_total": (
        "counter",
        "关键词(BM25)通道降级次数（标签 reason=index_failed/feed_failed/building/"
        "disabled/empty）；>0 表示本次召回**只有向量通道**，且日志里有同一原因的 WARNING",
        "count",
    ),
}
for _name, _spec in _NEW_METRICS.items():
    M.METRIC_CATALOG.setdefault(_name, _spec)

#: 小语料阈值：``count() <= 它`` 时 BM25 索引**同步**建（毫秒级，行为与改前一致）；
#: 超过则转后台线程，避免"一次请求等百秒"。环境变量 ``BM25_SYNC_MAX_DOCS`` 可调。
_DEFAULT_SYNC_MAX_DOCS = 5000


def _sync_build_max_docs() -> int:
    try:
        value = int(str(os.environ.get("BM25_SYNC_MAX_DOCS", "")).strip()
                    or _DEFAULT_SYNC_MAX_DOCS)
    except ValueError:
        return _DEFAULT_SYNC_MAX_DOCS
    return value if value >= 0 else _DEFAULT_SYNC_MAX_DOCS


class BM25Index:
    """标准 BM25（Okapi），纯标准库实现。

    **打分公式与改前逐字一致**（同样的 ``k1/b``、同样的平滑 IDF、同样的查询项顺序），
    差别只在存储：术语倒排表用 ``array('i')``（doc 下标）+ ``array('H')``（词频，
    上限 65535）压着存，不逐文档保存 ``Counter``。
    """

    def __init__(self, chunks: Iterable[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self._doc_len: array = array("I")
        self._df: dict[str, int] = {}
        self._postings: dict[str, tuple[array, array]] = {}

        for chunk in chunks:
            self.add(chunk)

    @property
    def _n(self) -> int:
        """文档数 = 已收录条数（**动态取**，因为索引是流式增量建的）。"""
        return len(self.chunks)

    @property
    def _avgdl(self) -> float:
        """平均文档长度（同样动态取；`_doc_len` 是紧凑数组，求和很便宜）。"""
        return (sum(self._doc_len) / self._n) if self._n else 0.0

    def add(self, chunk: Chunk) -> int:
        """增量收录一篇文档（**流式建索引**：调用方可以边读边喂，不必先攒全量）。"""
        index = len(self.chunks)
        self.chunks.append(chunk)
        tokens = _tokenize(chunk.text)
        self._doc_len.append(len(tokens))
        counts = Counter(tokens)
        for term, tf in counts.items():
            postings = self._postings.get(term)
            if postings is None:
                self._postings[term] = (array("i", [index]), array("H", [min(tf, 65535)]))
                self._df[term] = 1
            else:
                postings[0].append(index)
                postings[1].append(min(tf, 65535))
                self._df[term] = self._df.get(term, 0) + 1
        return index

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        # BM25 的平滑 IDF，df=0 时给一个很小的正值
        return math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))

    # P0.5：关键词路（BM25 倒排打分）。纯内存计算，阈值 200ms 已经算"异常慢"。
    @timed(operation="retrieve.bm25_search", slow_ms=200.0)
    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        if not self._n:
            return []
        terms = [t for t in _tokenize(query) if t in self._df]
        if not terms:
            return []

        # 经典倒排打分：**查询项外层、命中文档内层**，对同一文档按查询项顺序累加 ⇒
        # 与改前"逐文档按查询项顺序累加"的浮点结果**逐位一致**；未命中的文档分数恒为 0，
        # 本来就不会入选，因此不用扫全库。
        avgdl = self._avgdl or 1.0
        scored: dict[int, float] = {}
        for term in terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            docs, tfs = postings
            for position, doc_index in enumerate(docs):
                tf = int(tfs[position])
                doc_len = self._doc_len[doc_index] or 1
                norm = 1.0 - self.b + self.b * (doc_len / avgdl)
                contribution = idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
                scored[doc_index] = scored.get(doc_index, 0.0) + contribution

        # 与改前同口径：分数降序；同分按下标升序（等价于稳定排序）
        ranked = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
        return [(self.chunks[i], s) for i, s in ranked[:top_k]]

    def search_within(self, query: str, *, sources: set[str],
                      top_k: int = 10) -> list[tuple[Chunk, float]]:
        """**只在指定来源的文档内**按 BM25 打分（法内检索用，见 P8）。

        与 :meth:`search` 同一套打分公式与并列规则，只把候选限制在 ``sources`` 里的文档上。
        为什么需要它：全局 top-k 的竞争里，"那一条"经常被别的文档挤掉；而**在该法内部**，
        它的全部条文都是候选 —— 这正是 P8"召回问题"的直接对策。
        """
        if not self._n or not sources:
            return []
        terms = [t for t in _tokenize(query) if t in self._df]
        if not terms:
            return []
        avgdl = self._avgdl or 1.0
        scored: dict[int, float] = {}
        for term in terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            docs, tfs = postings
            for position, doc_index in enumerate(docs):
                chunk = self.chunks[doc_index]
                if chunk.source not in sources:
                    continue
                tf = int(tfs[position])
                doc_len = self._doc_len[doc_index] or 1
                norm = 1.0 - self.b + self.b * (doc_len / avgdl)
                contribution = idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
                scored[doc_index] = scored.get(doc_index, 0.0) + contribution
        ranked = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
        return [(self.chunks[i], s) for i, s in ranked[:top_k]]


#: BM25 落盘缓存的格式版本（改结构就 +1，旧缓存自动失效）
_BM25_CACHE_VERSION = 1


def bm25_cache_path(cache_dir: str | Path | None, signature: tuple,
                    *, enabled: bool = True) -> Path | None:
    """缓存文件路径：按"签名"（语料条数 + 过滤范围）分文件，签名变了自然换文件。

    ``enabled=False`` 或没给目录 ⇒ ``None``（调用方据此跳过整套缓存逻辑）。
    """
    if not enabled or not cache_dir:
        return None
    digest = sha1(repr(signature).encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / "bm25_cache" / f"bm25_v{_BM25_CACHE_VERSION}_{digest}.pkl"


def load_bm25_cache(path: str | Path, signature: tuple) -> BM25Index | None:
    """读缓存；**任何**不可信/不匹配的情况一律返回 ``None``（宁可重建，绝不用错索引）。

    这是本项目自己写的文件（``index/bm25_cache/*.pkl``），来源可信；即便如此也逐项校验
    版本、签名、类型与条数 —— 缓存错一位就是全库召回错位，代价远大于重建 160 秒。
    """
    target = Path(path)
    try:
        payload = pickle.loads(target.read_bytes())
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError,
            IndexError, ValueError, TypeError) as exc:
        logger.warning("BM25 缓存读取失败（将重建）：path=%s 原因=%s: %s",
                       target, type(exc).__name__, exc)
        return None
    if not isinstance(payload, dict) or payload.get("version") != _BM25_CACHE_VERSION:
        logger.warning("BM25 缓存版本不符（将重建）：path=%s", target)
        return None
    if payload.get("signature") != signature:
        logger.warning("BM25 缓存签名不符（将重建）：path=%s", target)
        return None
    index = payload.get("index")
    if not isinstance(index, BM25Index):
        logger.warning("BM25 缓存内容不是索引（将重建）：path=%s", target)
        return None
    expected = signature[0] if signature else None
    # ⚠️ 这里**不能**要求 ``len(index.chunks) == expected``：签名里的条数是**库里的总条数**
    # （父块 + 子块，实测 131,596），而 BM25 只喂**子块**（``is_parent == False``，119,586）
    # —— 两者本来就不相等。第一版就是这么写的，结果是缓存**永远命中不了**、每次启动照样
    # 重建 160s（真机被预热的 ``bm25_chunks=None`` 抓出来）。过期判定由**签名相等**负责，
    # 这里只做两种"文件坏了"的合理性校验：
    #   1) pickle 里自报的条数与索引实际条数一致（防截断/半截文件）；
    #   2) 索引条数不超过库里总条数（防张冠李戴读到别的语料的缓存）。
    cached_chunks = payload.get("chunks")
    if isinstance(cached_chunks, int) and cached_chunks != len(index.chunks):
        logger.warning("BM25 缓存自报条数与索引不符（将重建）：缓存=%s 索引=%d path=%s",
                       cached_chunks, len(index.chunks), target)
        return None
    # **空索引绝不接受**（真机留痕：`index/bm25_cache/` 里出现过一个 290 字节的缓存，
    # 里面是 0 条 chunk 的索引）。空索引一旦被载入，关键词通道就**永久静默失效**——
    # 而且它不会报错，只会让召回悄悄变成纯向量，指标悄悄变差。库里非空 ⇒ 缓存也不该为空。
    if not index.chunks and (not isinstance(expected, int) or expected > 0):
        logger.warning("BM25 缓存里是**空索引**（将重建）：path=%s 库里条数=%s",
                       target, expected)
        return None
    if isinstance(expected, int) and len(index.chunks) > expected:
        logger.warning("BM25 缓存条数多于库里总条数（将重建）：索引=%d 库=%d path=%s",
                       len(index.chunks), expected, target)
        return None
    return index


def dump_bm25_cache(path: str | Path, index: BM25Index, signature: tuple) -> bool:
    """写缓存（失败只记日志，绝不影响检索）。先写 ``.tmp`` 再原子替换，避免半截文件。

    **空索引不落盘**：把 0 条的索引写进去，下次启动就会"秒载"一个空索引，关键词通道
    从此静默失效（不报错、只变差）。宁可下次重建，也不要留一个错的缓存。
    """
    target = Path(path)
    if not getattr(index, "chunks", None):
        logger.warning("BM25 索引为空，**不写缓存**（避免下次载入空索引让关键词通道静默失效）："
                       "path=%s signature=%r", target, signature)
        return False
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(
            {"version": _BM25_CACHE_VERSION, "signature": signature, "index": index,
             "chunks": len(index.chunks)},
            protocol=pickle.HIGHEST_PROTOCOL))
        tmp.replace(target)
        return True
    except (OSError, pickle.PicklingError, TypeError) as exc:
        logger.warning("BM25 缓存写入失败（下次仍会重建，不影响本次检索）：path=%s 原因=%s: %s",
                       target, type(exc).__name__, exc)
        return False


def apply_source_quota(hits: list[SearchHit], *, limit: int,
                       per_source: int) -> tuple[list[SearchHit], int]:
    """同一来源文件最多留 ``per_source`` 条，返回 ``(截到 limit 的命中, 被挤掉的条数)``。

    真机踩过（L23「怎样判断是否侵犯专利权」）：top-3 **全是同一份批复的 3 个拷贝**，
    名额被重复文档吃光、专利法本体挤不进来。做法是先按原顺序满足配额，再用被挤掉的
    候选按原顺序补齐到 ``limit``（不够就不补，不硬凑）。
    """
    picked: list[SearchHit] = []
    overflow: list[SearchHit] = []
    counts: dict[str, int] = {}
    for hit in hits:
        source = getattr(getattr(hit, "chunk", None), "source", "") or ""
        if counts.get(source, 0) < per_source:
            counts[source] = counts.get(source, 0) + 1
            picked.append(hit)
        else:
            overflow.append(hit)
    return (picked + overflow)[:limit], len(overflow)


def merge_hits_unique(primary: list[SearchHit], extra: list[SearchHit]) -> list[SearchHit]:
    """把 ``extra`` 里 ``primary`` 没有的块追加进来（按 chunk.id 去重，保持原顺序）。

    定向召回的候选**只加不删**：宁可多给重排几条，也不硬过滤既有召回
    —— 一问多法时硬过滤会丢召回（见 :mod:`legal_rag.retrieve.law_scope`）。
    """
    seen = {(hit.chunk.id or hit.chunk.text[:64]) for hit in primary}
    merged = list(primary)
    for hit in extra:
        key = hit.chunk.id or hit.chunk.text[:64]
        if key in seen:
            continue
        seen.add(key)
        merged.append(hit)
    return merged


class HybridRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore,
                 config: RetrievalConfig | None = None, reranker: Reranker | None = None,
                 law_scope: LawScope | None = None,
                 cache_dir: str | Path | None = None,
                 version_filter: VersionFilter | None = None,
                 selector: object | None = None) -> None:
        self.embedder = embedder
        self.store = store
        self.config = config or RetrievalConfig()
        self.reranker = reranker
        #: BM25 索引落盘缓存的目录（``None`` = 不缓存；见 ``bm25_cache_path``）
        self.cache_dir = cache_dir
        self._cache_enabled = bool(getattr(self.config, "bm25_cache_enabled", True))
        #: 版本过滤（``None`` 或开关关闭 = 不过滤；见 ``retrieve/version_filter.py``）
        self.version_filter = (
            version_filter if getattr(self.config, "version_filter_enabled", True) else None
        )
        #: 法名定向召回索引（可为空 = 该路不生效，行为与改动前一致）
        self.law_scope = law_scope or LawScope({})
        #: LLM 列表式选择器（``None`` = 该路不生效；见 ``retrieve/selector.py``）。
        #: 开关在 config（``law_selector_enabled``），未实测的杠杆默认不进链路。
        self.selector = (
            selector if getattr(self.config, "law_selector_enabled", False) else None
        )
        self._bm25: BM25Index | None = None
        self._bm25_signature: tuple | None = None
        #: 后台建索引的状态（t68）：none / building / ready / failed(+原因)
        self._build_thread: threading.Thread | None = None
        self._build_lock = threading.Lock()
        self._build_state = "none"
        self._build_reason = ""
        #: 降级日志去重（同一状态只吼一次 WARNING，计数照常累计）
        self._logged_reasons: set[str] = set()
        #: 本轮召回的**用户可见提示**（t115 F2）：处于预热/降级窗口时非空，见 WARMUP_NOTE。
        #: 每次召回开头重置；引擎负责把它贴到答案里（并计 retrieval_warmup_notice_total）。
        self.last_retrieval_note: str = ""

    # ---------- 可观测 ----------
    def _degrade(self, reason: str, message: str, *args: object) -> None:
        """降级**必须可见**：一条计数器 + 一条 WARNING（同原因只打一次，计数不丢）。"""
        M.counter("keyword_recall_degraded_total").inc(
            reason=reason, store=getattr(self.store, "name", "unknown"))
        if reason not in self._logged_reasons:
            self._logged_reasons.add(reason)
            logger.warning("[关键词通道降级] reason=%s " + message, reason, *args)

    def _feed_chunks(self, where: dict | None) -> Iterator[Chunk]:
        """取块：优先流式（不带向量、逐页），退回 ``all_chunks``（兼容非 Milvus 后端）。"""
        stream = getattr(self.store, "iter_all_chunks", None)
        if callable(stream):
            for page in stream(where, page=1000, with_vector=False):
                for chunk in page:
                    yield chunk
            return
        for chunk in self.store.all_chunks(where):
            yield chunk

    def _build_index(self, where: dict | None) -> None:
        """后台线程里的建索引：流式喂入 + 计时 + 失败可见。"""
        started = time.perf_counter()
        index = BM25Index([])
        try:
            for chunk in self._feed_chunks(where):
                index.add(chunk)
        except MemoryError as exc:                 # noqa: PERF203 - 真实规模的入口
            self._build_state, self._build_reason = "failed", f"MemoryError: {exc}"
            self._degrade("index_failed", "建 BM25 索引时内存不足（已降级为纯向量召回）："
                          "where=%r 已收录=%d 原因=%s", where, len(index.chunks), exc)
            return
        except Exception as exc:  # noqa: BLE001 - 任何建索引失败都不能拖垮检索
            self._build_state, self._build_reason = "failed", f"{type(exc).__name__}: {exc}"
            self._degrade("index_failed", "建 BM25 索引失败（已降级为纯向量召回）："
                          "where=%r 已收录=%d 原因=%s: %s",
                          where, len(index.chunks), type(exc).__name__, exc)
            return
        elapsed = time.perf_counter() - started
        signature = self._signature(where)
        # **空索引先重试一次再说**（真机踩到，2026-09-24）：进程刚起来时库可能还没就绪
        # （Milvus Lite 的 LOCK 刚被上个进程释放、或连接抖动）⇒ `all_chunks` 返回 0 条，
        # 于是建出一个 0 条的索引，**整轮关键词通道静默失效**（日志只有 `bm25_chunks=0`）。
        # 但"0 条"也可能是**正常**的：角色隔离下别的角色本来就没有资料（医生角色问律师知识）。
        # 从接口上看不出这两种情况（`count()` 不支持按 where 过滤）⇒ 再喂一遍：
        #   * 拿到了 ⇒ 是抖动（真问题），用这份索引；
        #   * 仍是 0 ⇒ 接受空索引（该 scope 本来就没有），只记 INFO、**不报降级**，
        #     免得给正常的多角色场景刷一堆假警告。
        total = signature[0] if signature else None
        if not index.chunks and isinstance(total, int) and total > 0:
            for chunk in self._feed_chunks(where):
                index.add(chunk)
            if not index.chunks:
                logger.info("本次 BM25 索引为空：该 scope 在库内没有块（库内共 %s 条，where=%r）"
                            "——按正常空索引采用（角色隔离下这是预期结果）", total, where)
        with self._build_lock:
            self._bm25, self._bm25_signature = index, signature
        self._build_state, self._build_reason = "ready", ""
        # 落盘缓存：下次进程启动直接载入，不再花 ~160s 重建（这期间关键词通道是空的）
        self._write_cache(signature, index)
        M.observe("keyword_index_build_seconds", elapsed)
        M.observe("keyword_index_docs", float(len(index.chunks)))
        logger.info("重建 BM25 索引完成：%d 个 chunk，耗时 %.1fs（流式、不含向量）",
                    len(index.chunks), elapsed)

    def _signature(self, where: dict | None) -> tuple:
        """索引签名 = (语料条数, 过滤范围)；取不到条数时按 ``None`` 处理（走同步路径）。"""
        try:
            total: int | None = self.store.count()
        except Exception as exc:  # noqa: BLE001 - 统计失败不该让检索整单失败
            logger.warning("取语料条数失败（%s: %s）——按小语料同步建索引处理",
                           type(exc).__name__, exc)
            total = None
        return (total, repr(sorted((where or {}).items(), key=str)))

    # ---------- BM25 缓存 ----------
    def _bm25_for(self, where: dict | None) -> BM25Index | None:
        """取（必要时建）索引；**未就绪一律返回 None 并留可见降级**。

        建索引策略（t68，按语料规模二选一）：

        * **小语料**（``count() <= BM25_SYNC_MAX_DOCS``，默认 5000）→ **同步建**：
          毫秒级完成，既有语义（"向量挂了还有关键词兜底"）完全不变；
        * **大语料**（真实库 13 万 chunk）→ **后台线程建**：一次请求绝不为它等百秒
          （t26 R1 实测的 98.9s/135.9s/171.5s 就是同步建全量索引的后果），建好前的
          请求只走向量通道，并且有**可见降级**（日志 + 计数器）。
        """
        signature = self._signature(where)
        if self._bm25 is not None and self._bm25_signature == signature:
            return self._bm25

        # 落盘缓存：命中就秒载（这条路径专治"每次重启都要重建 11.9 万块、约 160s"）。
        cached = self._read_cache(signature)
        if cached is not None:
            with self._build_lock:
                self._bm25, self._bm25_signature = cached, signature
            self._build_state, self._build_reason = "ready", ""
            return self._bm25

        # 小语料（或语料条数取不到）：同步建（保证既有行为与延迟特征不变）
        if signature[0] is None or signature[0] <= _sync_build_max_docs():
            self._build_index(where)
            # ⚠️ **必须按签名核对后再返回**（真机踩到，2026-09-24，被既有隔离测试抓住）：
            # `_build_index` 可能**拒绝**本次索引（例如"库非空而本 scope 为空"是正常的空索引
            # —— 角色隔离下别的角色知识本来就不该进来），此时 `self._bm25` 还是**上一个 scope
            # 的索引**；无脑 `return self._bm25` 会把律师知识漏给医生角色（角色隔离被击穿）。
            if self._bm25 is not None and self._bm25_signature == signature:
                return self._bm25
            return None

        with self._build_lock:
            stale = self._bm25 is None or self._bm25_signature != signature
            if not stale:
                return self._bm25
            running = self._build_thread is not None and self._build_thread.is_alive()
            if not running:
                self._build_state, self._build_reason = "building", ""
                self._build_thread = threading.Thread(
                    target=self._build_index, args=(where,),
                    name="bm25-index-build", daemon=True)
                self._build_thread.start()
                logger.info("BM25 索引重建已转入**后台线程**（不阻塞本次请求）：where=%r "
                            "count=%s，本次关键词通道先降级为纯向量",
                            where, signature[0])
        if self._bm25 is not None and self._bm25_signature == signature:
            return self._bm25
        if self._build_state == "failed":
            self._degrade("feed_failed", "关键词通道不可用（索引构建失败，原因=%s）",
                          self._build_reason)
        else:
            self._degrade("building", "BM25 索引正在后台构建，本次只有向量通道结果")
        return None

    # ---------- 融合 ----------
    @staticmethod
    def _normalize(hits: list[SearchHit]) -> None:
        if not hits:
            return
        top = max(h.score for h in hits) or 1.0
        for hit in hits:
            hit.score = hit.score / top

    def _fuse(self, vector_hits: list[SearchHit], keyword_hits: list[SearchHit]) -> list[SearchHit]:
        cfg = self.config
        merged: dict[str, SearchHit] = {}

        if cfg.fusion == "rrf":
            k = 60.0
            for rank, hit in enumerate(vector_hits, start=1):
                target = merged.setdefault(hit.chunk.id, SearchHit(chunk=hit.chunk))
                target.vector_score = hit.vector_score
                target.score += cfg.vector_weight / (k + rank)
            for rank, hit in enumerate(keyword_hits, start=1):
                target = merged.setdefault(hit.chunk.id, SearchHit(chunk=hit.chunk))
                target.keyword_score = hit.score
                target.score += cfg.keyword_weight / (k + rank)
            return list(merged.values())

        # weighted：各自归一化后加权求和
        self._normalize(vector_hits)
        self._normalize(keyword_hits)
        for hit in vector_hits:
            target = merged.setdefault(hit.chunk.id, SearchHit(chunk=hit.chunk))
            target.vector_score = hit.vector_score
            target.score += cfg.vector_weight * hit.score
        for hit in keyword_hits:
            target = merged.setdefault(hit.chunk.id, SearchHit(chunk=hit.chunk))
            target.keyword_score = hit.score
            target.score += cfg.keyword_weight * hit.score
        return list(merged.values())

    # ---------- 相关性闸门（t109 落地 / t115 重标定）----------
    @property
    def _gate_threshold(self) -> float:
        """融合分阈值（``<=0`` 表示关闭自动闸门）。口径见 ``RetrievalConfig.relevance_gate``。"""
        return float(getattr(self.config, "relevance_gate", 0.0) or 0.0)

    def _gate_mode(self, *, vector_ok: bool, keyword_ok: bool) -> tuple[str, str]:
        """本轮用**哪套判据**：(mode, 原因)。每次判定都要能回答"这轮靠什么判的"。

        * ``score_and_lexical`` —— 分数尺度可用（已标定后端 + 两通道都健康）：词面证据是
          **准入门槛**，融合分只用于在门槛内**补充**高分碎片；
        * ``lexical_only``     —— **尺度不可比**（关键词通道还在后台建索引/失败，或向量通道
          降级）：只执行**尺度无关**的词面证据判据 —— t115 修的就是这里：以前整条跳过，
          结果启动窗口内「你好」拿回 3 条伪依据；
        * ``skip``             —— 阈值关闭（``disabled``）或后端未标定（``uncalibrated_backend``，
          内存/离线后端、Milvus 降级兜底）：该链路分数尺度与标定无关，保持既有行为，
          只计数不判定（离线回归依赖这条）。
        """
        if self._gate_threshold <= 0:
            return "skip", "disabled"
        backends = tuple(getattr(self.config, "relevance_gate_backends", ()) or ())
        store_name = str(getattr(self.store, "name", "") or "")
        if backends and store_name not in backends:
            return "skip", "uncalibrated_backend"
        if not vector_ok or not keyword_ok:
            return "lexical_only", "channel_degraded"
        return "score_and_lexical", "ok"

    def _apply_relevance_gate(self, query: str, hits: list[SearchHit], *,
                              vector_ok: bool, keyword_ok: bool,
                              scope: dict | None = None) -> list[SearchHit]:
        """在**召回阶段出口**判"有没有依据"：没依据的碎片不给上层、不进上下文、不进引用。

        判据（t115 重标定，两条判据分工明确）：

        1. **词面证据 = 准入门槛（尺度无关，任何情况都执行）**：问题的实义词元（长度 >= 2 的
           双字/英文词，见 :func:`_content_terms`）至少有 ``_MIN_CONTENT_TERMS`` 个落在**同一条
           碎片正文**里。问题级先判：一条都达不到 ⇒ 整批无依据（``no_lexical_evidence``）。
        2. **融合分 = 门槛内的补充保留**（仅 ``score_and_lexical`` 模式）：分数 >= 阈值也保留，
           免得分数达标、只是词面改写没到门槛的**依据被误杀**…… 但门槛必须先在①通过 ——
           实测反例：「推荐几部好看的科幻电影」融合分 1.8977（关键词通道靠单字命中），
           词面证据只有 2 个，纯看分数就会把法条当依据。

        为什么门槛放在尺度无关的那条上：``劳动争议仲裁的时效是多久？`` 的候选融合分只有
        0.9577~1.0000（关键词通道没命中，只剩向量通道归一化后的 1.0），但词面证据有 6~8 个
        —— 分数判据在这里必然误杀，只有词面证据救得回来；反过来，"你好""水的沸点"这类
        问题的词面证据恒为 0~1。两边的实测分布见 t115 标定表。
        """
        store_name = str(getattr(self.store, "name", "unknown"))
        mode, reason = self._gate_mode(vector_ok=vector_ok, keyword_ok=keyword_ok)
        M.counter("retrieval_gate_mode_total").inc(mode=mode, reason=reason, store=store_name)
        # 预热/降级窗口（t115 F2）：尺度不可比 ⇒ 本轮只有向量通道的候选，**答案可能不全**，
        # 这句话必须走到用户眼前（引擎负责贴，检索层只负责判定）。每次召回都重置，防止上一轮
        # 的提示粘到下一轮。
        self.last_retrieval_note = WARMUP_NOTE if mode == "lexical_only" else ""
        if mode == "skip":
            if reason == "disabled":
                return hits
            M.counter("retrieval_gate_skipped_total").inc(reason=reason, store=store_name)
            logger.info("[RELEVANCE-GATE] 本轮**不执行**闸门（判据=skip, reason=%s）：store=%s "
                        "query=%r —— 该后端没有标定过的分数尺度，保持既有行为",
                        reason, store_name, query)
            return hits

        gate = self._gate_threshold
        query_terms = _content_terms(query)
        readings = [(hit, float(hit.score),
                     sum(1 for term in query_terms if term in (hit.chunk.text or "")))
                    for hit in hits]
        # LLM 列表式选择器挑中的条文**豁免词面证据闸门**：这一路的根因恰恰是"正确条文与问题
        # 没有词面重合"（"商标侵权" ↔ "侵犯注册商标专用权"），再用词面去闸它等于把它杀回来。
        # 豁免是**显式**的：单独计数 + 单独日志（不静默放宽）。
        exempt = [hit for hit in hits if getattr(hit, "selected", False)]
        exempt_ids = {id(hit) for hit in exempt}
        max_overlap = max((overlap for hit, _, overlap in readings if id(hit) not in exempt_ids),
                          default=0)
        M.observe("retrieval_gate_max_overlap", float(max_overlap), store=store_name)
        if exempt:
            M.counter("law_selector_gate_exempt_total").inc(len(exempt), store=store_name)
            logger.info("[RELEVANCE-GATE] 列表式选择器挑中的 %d 条豁免词面证据闸门（其条号=%s，"
                        "本批其余候选最高重合 %d）：query=%r",
                        len(exempt), [h.chunk.source for h in exempt][:3], max_overlap, query)
        reading_text = " ".join("#%d(score=%.4f,重合=%d%s)"
                                % (index + 1, score, overlap, "" if keep else " -> 丢")
                                for index, (hit, score, overlap) in enumerate(readings)
                                for keep in [getattr(hit, "selected", False)
                                             or overlap >= _MIN_CONTENT_TERMS
                                             or (mode == "score_and_lexical" and score >= gate)])

        if max_overlap < _MIN_CONTENT_TERMS and not exempt:
            # 问题级门槛不通过：这批候选**没有任何一条**与问题有实质词面关系
            if hits:
                M.counter("retrieval_gate_dropped_total").inc(
                    len(hits), reason="no_lexical_evidence", store=store_name)
                logger.info("[RELEVANCE-GATE] 判据=%s 结论=整批无依据（实义词元下限 %d，本批最高重合 %d）"
                            "：query=%r scope=%r store=%s 候选 %d 条 %s —— 一条都不给上层",
                            mode, _MIN_CONTENT_TERMS, max_overlap, query, scope, store_name,
                            len(hits), reading_text)
            return []

        kept: list[SearchHit] = []
        for hit, score, overlap in readings:
            if (getattr(hit, "selected", False) or overlap >= _MIN_CONTENT_TERMS
                    or (mode == "score_and_lexical" and score >= gate)):
                kept.append(hit)
        dropped = len(hits) - len(kept)
        if dropped:
            M.counter("retrieval_gate_dropped_total").inc(
                dropped, reason="below_evidence", store=store_name)
        logger.info("[RELEVANCE-GATE] 判据=%s（词面证据>=%d 为准入；融合分>=%.4f 为补充）reason=%s "
                    "结论=保留 %d/%d 条：query=%r scope=%r store=%s 候选 %s",
                    mode, _MIN_CONTENT_TERMS, gate, reason, len(kept), len(hits), query, scope,
                    store_name, reading_text)
        if not kept:
            logger.warning("[RELEVANCE-GATE] 门槛内一条都没留下（候选 %d 条均无词面证据且分数不达标）："
                           "query=%r store=%s —— 上层按三分类路由处理（带法律信号走引导补充/"
                           "换说法，否则走通用对话），两条路都**不会**拿这些碎片当依据",
                           len(hits), query, store_name)
        logger.info("出口 retrieve：%d 条候选 -> 闸门后 %d 条（判据=%s）",
                    len(hits), len(kept), mode)
        return kept

    # ---------- 法名定向召回 ----------
    def _law_scoped_hits(self, query: str, scope: dict,
                         query_vector: list[float] | None) -> list[SearchHit]:
        """问题点到法名时，去该法的文件里再捞一遍候选（**只加不删**）。

        返回 ``[]`` 表示这一路没启用 / 没识别到法名 / 后端不支持 ``source`` 过滤 ——
        都不影响既有召回。失败**不静默**：打 WARNING 说清原因（按项目规则，
        降级必须看得见），但绝不让它把整轮召回带崩。
        """
        cfg = self.config
        if not getattr(cfg, "law_scope_enabled", False) or query_vector is None:
            return []
        laws = self.law_scope.laws_for(query)
        if not laws:
            return []
        groups = self.law_scope.groups_for(query)
        if not groups:
            return []
        collected: list[SearchHit] = []
        for label, sources in groups:
            # 法典本体与司法解释**分开取** top_k：合在一起时司法解释会把名额挤掉
            # （真机踩过：商标法/公司法两题的 source 列表第一项都是司法解释，
            #  结果法典本体只挤进一两条，而讲"什么行为构成商标侵权"的正是商标法第五十七条）。
            per_group = (cfg.law_scope_top_k if label == "canon"
                         else getattr(cfg, "law_scope_related_top_k", 0))
            if per_group <= 0:
                continue
            where = {**(scope or {}), "source": sources}
            try:
                hits = self.store.search(query_vector, top_k=per_group, where=where)
            except Exception as exc:  # noqa: BLE001 - 定向召回失败不影响既有召回
                logger.warning("[法名定向召回] group=%s 失败（不影响既有召回）：laws=%s "
                               "文件=%d 原因=%s: %s", label, laws, len(sources),
                               type(exc).__name__, exc)
                hits = []
            # **法内 BM25**：同一个法名集合内再按关键词打一遍分。全局 top-k 里那条常被别的
            # 文档挤掉，而在该法内部它的全部条文都是候选（P8"召回问题"的直接对策）。
            keyword_hits: list[SearchHit] = []
            try:
                index = self._bm25_for(scope)
                if index is not None:
                    for chunk, score in index.search_within(self._bm25_query(query),
                                                        sources=set(sources),
                                                            top_k=per_group):
                        keyword_hits.append(SearchHit(chunk=chunk, score=score,
                                                      keyword_score=score))
            except Exception as exc:  # noqa: BLE001
                logger.warning("[法内 BM25] group=%s 失败（不影响既有召回）：原因=%s: %s",
                               label, type(exc).__name__, exc)
            collected = merge_hits_unique(collected, self._fuse(hits, keyword_hits))
            logger.info("[法名定向召回] laws=%s group=%s 文件 %d 个 -> 向量 %d + 关键词 %d "
                        "条（累计 %d）", laws, label, len(sources), len(hits),
                        len(keyword_hits), len(collected))
        if not collected:
            return []
        M.counter("law_scope_recall_total").inc(law=",".join(laws))
        M.observe("law_scope_hits", float(len(collected)), law=laws[0])
        return collected

    # ---------- BM25 落盘缓存 ----------
    def _cache_path(self, signature: tuple) -> Path | None:
        return bm25_cache_path(self.cache_dir, signature, enabled=self._cache_enabled)

    def _read_cache(self, signature: tuple) -> BM25Index | None:
        """尝试从磁盘载入索引；没缓存/不匹配都返回 ``None``（调用方照常重建）。"""
        path = self._cache_path(signature)
        if path is None:
            return None
        if not path.is_file():
            M.counter("keyword_index_cache_total").inc(result="miss")
            return None
        index = load_bm25_cache(path, signature)
        if index is None:
            M.counter("keyword_index_cache_total").inc(result="corrupt")
            return None
        M.counter("keyword_index_cache_total").inc(result="hit")
        logger.info("BM25 索引**从磁盘缓存载入**（省掉全量重建）：%d 个 chunk，缓存=%s",
                    len(index.chunks), path.name)
        return index

    def _write_cache(self, signature: tuple, index: BM25Index) -> None:
        path = self._cache_path(signature)
        if path is None:
            return
        if dump_bm25_cache(path, index, signature):
            M.counter("keyword_index_cache_total").inc(result="written")
            logger.info("BM25 索引已落盘缓存（下次启动秒载）：%d 个 chunk，缓存=%s",
                        len(index.chunks), path.name)
        else:
            M.counter("keyword_index_cache_total").inc(result="write_failed")

    # ---------- 预热 ----------
    # P0.5：预热本身可能要几十秒（真实语料 11.9 万块建 BM25 索引约 30~75s），
    # 阈值 5000ms；它的耗时是"首问为什么慢"的直接证据。
    @timed(operation="retrieve.warm_up", slow_ms=5000.0)
    def warm_up(self, where: dict | None = None) -> dict:
        """预热：**嵌一次** + 建（或从磁盘载入）BM25 索引。

        为什么要它（真机实测）：首查要等 bge-m3 冷加载（7.6–9.3s），而 BM25 是**懒触发**的
        （第一次检索才开始建，11.9 万块约 160s）—— 这两个成本本该在**进程启动的后台**付掉，
        用户的第一问才不会撞上「资料还在预热」的降级窗口。有磁盘缓存时第二次启动是秒级。

        **绝不抛异常**（预热失败只是"没预热"，功能照旧）：失败打 WARNING 并记进返回值。
        """
        result: dict = {"embed": False, "bm25_chunks": None, "errors": []}
        try:
            self.embedder.embed_query("预热")
            result["embed"] = True
        except Exception as exc:  # noqa: BLE001 - 预热失败不能影响服务启动
            message = f"embed: {type(exc).__name__}: {exc}"
            result["errors"].append(message)
            logger.warning("预热嵌入失败（不影响功能，只是首查仍会冷加载）：%s", message)
        try:
            index = self._bm25_for(where)
            # None = 索引**未就绪**（建失败/仍在后台建）；0 才是"就绪但语料为空"——
            # 两者语义不同，别把"没建成"糊成"0 条"。
            result["bm25_chunks"] = None if index is None else len(index.chunks)
        except Exception as exc:  # noqa: BLE001
            message = f"bm25: {type(exc).__name__}: {exc}"
            result["errors"].append(message)
            logger.warning("预热 BM25 失败（不影响功能，只是首查仍要等它建完）：%s", message)
        logger.info("预热完成：embed=%s bm25_chunks=%s errors=%d",
                    result["embed"], result["bm25_chunks"], len(result["errors"]))
        return result

    # ---------- 候选池（供检索侧评测/调试；不含重排与闸门）----------
    # P0.5：召回 + 融合两路（**不含**重排与闸门），用于把"慢在召回还是慢在精排"分开。
    @timed(operation="retrieve.candidates", slow_ms=500.0)
    def candidates(self, query: str, where: dict | None = None) -> list[SearchHit]:
        """返回**重排前**的候选池（召回 + 融合 + 版本过滤 + 法名定向召回）。

        为什么要暴露它：比较不同重排器时必须让它们打**同一批候选**（apples-to-apples），
        否则差异里混着召回波动；也方便排查"对的条文到底有没有进池子"（P8 的诊断）。
        **不执行**重排与相关性闸门 —— 那是 ``retrieve()`` 的职责。
        """
        cfg = self.config
        scope = {"is_parent": False} if where is None else where
        vector_hits: list[SearchHit] = []
        query_vector: list[float] | None = None
        try:
            query_vector = self.embedder.embed_query(query)
            vector_hits = self.store.search(query_vector, top_k=cfg.vector_top_k, where=scope)
        except Exception as exc:  # noqa: BLE001 - 与 retrieve() 同口径：降级要可见
            logger.warning("[候选池] 向量召回失败（本次只有关键词通道）：query=%r 原因=%s: %s",
                           query, type(exc).__name__, exc)
        keyword_hits: list[SearchHit] = []
        try:
            index = self._bm25_for(scope)
            if index is not None:
                for chunk, score in index.search(self._bm25_query(query), cfg.keyword_top_k):
                    keyword_hits.append(SearchHit(chunk=chunk, score=score, keyword_score=score))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[候选池] 关键词召回失败：query=%r 原因=%s: %s",
                           query, type(exc).__name__, exc)
        vector_hits = merge_hits_unique(vector_hits,
                                        self._law_scoped_hits(query, scope, query_vector))
        pooled = self._fuse(vector_hits, keyword_hits)
        if self.version_filter is not None and pooled:
            pooled, dropped = self.version_filter.filter_hits(pooled)
            if dropped:
                logger.info("[候选池] 版本过滤剔除 %d 条非现行版本", dropped)
        return pooled

    # ---------- 关键词查询（术语映射） ----------
    def _bm25_query(self, query: str) -> str:
        """BM25 通道用的查询：按开关追加"口语→立法术语"映射（只加词、不引模型）。

        实测依据（P8）：问"出资不实"，法里写"未履行出资义务"（问句词 0 命中）；
        问"商标侵权"，《商标法》第五十七条写"侵犯注册商标专用权"（问句词在整部法有 19 次、
        却不在那一条里）。词面钩子补上，BM25 才可能把对的条文捞出来。
        """
        if not getattr(self.config, "term_map_enabled", False):
            return query
        limit = int(getattr(self.config, "term_map_limit", 6) or 6)
        return expand_query_logged(query, limit=limit)

    # ---------- 法内全量候选（P8 的关键一步）----------
    def law_internal_candidates(self, query: str, scope: dict | None = None) -> list[SearchHit]:
        """公开入口：该法的**法内宽候选**（自己算查询向量，失败只降级不抛）。

        给"法内重排 / 名额保留 / LLM 列表式选择器"和诊断脚本用；没识别到法名或通道不可用时
        返回 ``[]``（原因都在日志里，不静默）。
        """
        where = {"is_parent": False} if scope is None else scope
        try:
            vector = self.embedder.embed_query(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[法内检索] 查询向量失败，法内候选为空：原因=%s: %s",
                           type(exc).__name__, exc)
            return []
        return self._law_internal_candidates(query, where, vector)

    def _law_internal_candidates(self, query: str, scope: dict,
                                query_vector: list[float] | None) -> list[SearchHit]:
        """在**该法文件集合内**取较宽的候选（默认 200），供"法内重排 + 名额保留"用。

        为什么不能只用 ``law_scope_top_k``：真机实测（2026-09-22）——法内候选只有 10+5 条时，
        期望条号在法内同样进不来，"名额保留"于是只能换成该法**别的**条文，甚至把原本对的那条
        挤掉（cosine 的条号级 Recall@5 从 0.359 掉到 **0.282**）。一部法几十到几百条，
        所以法内候选要**尽量全**（200 条足够覆盖绝大多数法典），再在法内重排取前 k。
        """
        cfg = self.config
        width = int(getattr(cfg, "law_scope_internal_top_k", 0) or 0)
        if width <= 0 or query_vector is None or not getattr(cfg, "law_scope_enabled", False):
            return []
        canon = [sources for label, sources in self.law_scope.groups_for(query) if label == "canon"]
        if not canon:
            return []
        sources = canon[0]
        where = {**(scope or {}), "source": sources}
        vector_hits: list[SearchHit] = []
        try:
            vector_hits = self.store.search(query_vector, top_k=width, where=where)
        except Exception as exc:  # noqa: BLE001 - 法内检索失败不影响既有召回
            logger.warning("[法内检索] 向量通道失败：文件=%d 原因=%s: %s",
                           len(sources), type(exc).__name__, exc)
        keyword_hits: list[SearchHit] = []
        try:
            index = self._bm25_for(scope)
            if index is not None:
                for chunk, score in index.search_within(self._bm25_query(query),
                                                        sources=set(sources),
                                                        top_k=width):
                    keyword_hits.append(SearchHit(chunk=chunk, score=score, keyword_score=score))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[法内检索] 关键词通道失败：原因=%s: %s", type(exc).__name__, exc)
        pooled = self._fuse(vector_hits, keyword_hits)
        # 版本过滤**必须也覆盖这一路**（真机踩到）：语料里同一部法有多个版本，而且**条号会漂**
        # —— 实测《商标法》"有下列行为之一的，均属侵犯注册商标专用权"在
        #   2013 版/2019 版里是**第五十七条**，在 2026 公布、2027 施行的新版里是**第七十二条**。
        # 不过滤时，选择器会在"混着过期版与未生效版"的候选里挑条，可能挑中已修改版本的条号
        # 并把它当依据给用户 —— 对法律产品这是**实体错误**，不只是指标问题。
        if self.version_filter is not None and pooled:
            pooled, dropped = self.version_filter.filter_hits(pooled)
            if dropped:
                logger.info("[法内检索] 版本过滤剔除 %d 条非现行版本（避免按过期条号作答）", dropped)
        logger.info("[法内检索] 文件 %d 个 -> 向量 %d + 关键词 %d -> 法内候选 %d 条",
                    len(sources), len(vector_hits), len(keyword_hits), len(pooled))
        return pooled

    # ---------- 列表式选择器：该不该叫 ----------
    def _selector_version_labels(self, hits: list[SearchHit]) -> dict[str, str]:
        """给候选打**版本标记**（``{basename: 现行有效 | 尚未施行(YYYY-MM-DD)}``）。

        为什么（2026-09-24 实测）：语料里同一部法有多个版本，同一条内容**条号还会变** ——
        《商标法》"均属侵犯注册商标专用权"在现行有效版是第五十七条、在 2026 公布 / 2027 施行的
        新版是第七十二条。选择器（v1）曾挑中**未生效新版**的条号：内容等价，但"引用尚未施行的
        条文"对法律产品是错的 ⇒ 候选要带版本状态，并要求**优先现行有效版**。
        """
        if self.version_filter is None:
            return {}
        labels: dict[str, str] = {}
        for hit in hits:
            source = str(getattr(getattr(hit, "chunk", None), "source", "") or "")
            name = Path(source.replace("\\", "/")).name
            if not name or name in labels:
                continue
            if self.version_filter.is_pending(name):
                date = self.version_filter.pending_dates.get(name, "")
                labels[name] = f"尚未施行({date})" if date else "尚未施行"
            elif not self.version_filter.is_stale(name):
                labels[name] = "现行有效"
        return labels

    def selector_decision(self, hits: list[SearchHit], law_sources: set[str],
                          limit: int) -> tuple[bool, str]:
        """要不要叫大模型挑条，返回 ``(是否调用, 原因)``。

        用户 2026-09-23 的要求：**速度与质量取中间值**（"速度也不能太慢"）。
        一次调用 ≈5.3 s（60 候选 × 120 字），如果每题都调就把在线链路拖慢一倍以上，
        所以只在**便宜的信号显示"没把握"**时才调。两个判据都**尺度无关**（不看重排分数，
        换重排器/换后端都不用重标定）：

        * ``no_law_hit``   —— 用户点名的这部法，在最终 top-k 里**几乎没有它的条文**
          （少于 ``law_selector_min_law_hits`` 条）⇒ 大概率没找对条；
        * ``thin_context`` —— 过闸门后剩下的条数**少于** ``law_selector_min_hits`` ⇒ 召回本来就薄。

        配置 ``law_selector_trigger=always`` 可退回"每题都调"（做对照实验用）。
        """
        cfg = self.config
        mode = str(getattr(cfg, "law_selector_trigger", "always") or "always").strip().lower()
        if mode != "low_confidence":
            return True, "always"
        min_law = int(getattr(cfg, "law_selector_min_law_hits", 2) or 0)
        min_hits = int(getattr(cfg, "law_selector_min_hits", 5) or 0)
        in_law = sum(1 for hit in hits if (hit.chunk.source or "") in (law_sources or set()))
        if min_law > 0 and in_law < min_law:
            return True, f"no_law_hit({in_law}<{min_law})"
        if min_hits > 0 and len(hits) < min_hits:
            return True, f"thin_context({len(hits)}<{min_hits})"
        return False, f"confident(law_hits={in_law},hits={len(hits)})"

    # ---------- 法条名额保留 ----------
    def _reserve_law_slots(self, query: str, hits: list[SearchHit],
                           law_candidates: list[SearchHit], *, limit: int,
                           protected: set[str] | None = None) -> list[SearchHit]:
        """法名命中时，保证最终上下文里至少 ``law_scope_reserve`` 条来自**该法家族**。

        做法：把该法候选**单独**重排一次（法内最优），再从末尾往前**替换**非该法命中 ——
        只换不增，总条数不变（不动既有行为）；没有该法候选时完全不动。

        ``protected`` 是"不许被挤掉"的来源集合（默认取候选自身的来源）。**必须传该法家族的全部
        文件**（法典本体 + 配套 + 司法解释）：只看法典本体的话，会误把司法解释里**那条对的**
        挤掉 —— 真机实测 cosine 的条号级 Recall@5 因此从 0.359 掉到 0.282。
        """
        reserve = int(getattr(self.config, "law_scope_reserve", 0) or 0)
        if reserve <= 0 or not law_candidates or not hits:
            return hits
        sources = set(protected or ())
        if not sources:
            sources = {hit.chunk.source for hit in law_candidates if hit.chunk.source}
        if not sources:
            return hits

        def _in_law(hit: SearchHit) -> bool:
            return (hit.chunk.source or "") in sources

        have = sum(1 for hit in hits if _in_law(hit))
        if have >= reserve:
            return hits
        if self.reranker is not None:
            law_ranked = self.reranker.rerank(query, law_candidates, top_k=reserve)
        else:
            law_ranked = sorted(law_candidates, key=lambda h: h.score, reverse=True)[:reserve]
        existing = {hit.chunk.id for hit in hits}
        extras = [hit for hit in law_ranked if hit.chunk.id not in existing][: reserve - have]
        if not extras:
            return hits
        chosen = list(hits)
        for candidate in extras:
            for index in range(len(chosen) - 1, -1, -1):
                if not _in_law(chosen[index]):
                    chosen[index] = candidate
                    break
        M.counter("law_scope_reserved_total").inc()
        logger.info("[法条名额] 该法候选单独重排后替换入 %d 条（保证 %d 条来自该法）：sources=%d",
                    len(extras), reserve, len(sources))
        return chosen

    # ---------- 主入口 ----------
    # P0.5：混合检索是单请求最重的一段（向量路 + BM25 建索引 + 融合 + 重排），
    # 阈值给到 1000ms —— 低于它的正常检索不必刷日志，但耗时一律进 func_seconds。
    @timed(operation="retrieve.retrieve", slow_ms=1000.0)
    def retrieve(self, query: str, top_k: int | None = None,
                 where: dict | None = None) -> list[SearchHit]:
        cfg = self.config
        scope = {"is_parent": False} if where is None else where
        started = time.perf_counter()
        # 每轮召回开头先清掉上一轮的用户可见提示（t115 F2），只有本轮真的落在
        # 预热/降级窗口里，闸门才会重新写上（见 _apply_relevance_gate）。
        self.last_retrieval_note = ""

        vector_hits: list[SearchHit] = []
        vector_error = ""
        query_vector: list[float] | None = None
        try:
            vector = self.embedder.embed_query(query)
            query_vector = vector
            vector_hits = self.store.search(vector, top_k=cfg.vector_top_k, where=scope)
        except Exception as exc:  # noqa: BLE001 - 向量侧失败仍可用关键词侧
            # ⚠️ 这条降级**必须看得见**（t8 诊断留下的要求）：只掉一条 ERROR 不够，
            # 还要有可计数的指标 + 一句话能看懂的上下文（哪条查询、哪个角色范围、
            # 什么原因），否则「向量召回失败 -> 已降级」会被误读成「本来就没命中」。
            vector_error = f"{type(exc).__name__}: {exc}"
            M.counter("vector_recall_degraded_total").inc(
                store=getattr(self.store, "name", "unknown"),
                embedder=self.embedder.name)
            logger.exception(
                "【已降级】向量召回失败 -> 本次**只用关键词(BM25)召回**，"
                "召回质量下降：query=%r where=%r top_k=%s store=%s 原因=%s",
                query, scope, cfg.vector_top_k,
                getattr(self.store, "name", "unknown"), vector_error)
        M.observe("vector_hits", float(len(vector_hits)), embedder=self.embedder.name)

        keyword_hits: list[SearchHit] = []
        # 关键词通道是否**健康**：None（索引未就绪/建失败）时本轮只有单通道结果，
        # 融合分尺度残缺 —— 闸门退到**尺度无关的词面证据判据**（见 _gate_mode）。
        keyword_ok = False
        try:
            index = self._bm25_for(scope)
            keyword_ok = index is not None
            if index is not None:
                for chunk, score in index.search(self._bm25_query(query), cfg.keyword_top_k):
                    keyword_hits.append(SearchHit(chunk=chunk, score=score, keyword_score=score))
        except Exception as exc:  # noqa: BLE001
            self._degrade("feed_failed", "关键词召回失败（已降级为纯向量召回）："
                          "query=%r where=%r 原因=%s: %s",
                          query, scope, type(exc).__name__, exc)
            logger.exception("关键词召回失败，仅使用向量召回")
        M.observe("bm25_hits", float(len(keyword_hits)))

        # 法名定向召回（评测 v1/v2 失分主因的针对性修法）：问题点了哪部法，就额外去
        # 那部法的文件里捞一遍候选并**并入**向量候选（只加不删）。没点法名时这一路
        # 完全不动，行为与改动前一致。
        scoped_hits = self._law_scoped_hits(query, scope, query_vector)
        if scoped_hits:
            before = len(vector_hits)
            vector_hits = merge_hits_unique(vector_hits, scoped_hits)
            logger.info("[法名定向召回] 向量候选 %d -> %d 条（并入 %d 条该法条文）",
                        before, len(vector_hits), len(vector_hits) - before)

        logger.info("召回：向量 %d 条%s，关键词 %d 条", len(vector_hits),
                    "（⚠️ 已降级：向量通道失败，本次只有关键词结果）" if vector_error else "",
                    len(keyword_hits))

        fused = self._fuse(vector_hits, keyword_hits)
        # 版本过滤：同一份文件有多个版本时只留现行版（旧版仍可能被召回引用，
        # 对法律 RAG 是高危 —— 见 retrieve/version_filter.py 与附录 P7）。
        if self.version_filter is not None and fused:
            fused, dropped = self.version_filter.filter_hits(fused)
            if dropped:
                M.counter("version_filter_dropped_total").inc(
                    store=getattr(self.store, "name", "unknown"))
                logger.info("[版本过滤] 剔除 %d 条非现行版本候选（剩余 %d 条）",
                            dropped, len(fused))
        M.observe("fused_hits", float(len(fused)))
        if not fused:
            return []

        limit = top_k or cfg.final_top_k
        per_source = int(getattr(cfg, "max_per_source", 0) or 0)
        # 有配额时先多取一截：直接按 limit 截断，重复文档会把名额吃光
        # （真机踩过：L23 的 top-3 是同一份批复的 3 个拷贝）。
        pool = max(limit * 3, 12) if per_source > 0 else limit
        if self.reranker is not None:
            rerank_started = time.perf_counter()
            ranked = self.reranker.rerank(query, fused, top_k=pool)
            M.observe("rerank_seconds", time.perf_counter() - rerank_started,
                      reranker=self.reranker.name)
        else:
            ranked = sorted(fused, key=lambda h: h.score, reverse=True)[:pool]

        if per_source > 0:
            ranked, overflow = apply_source_quota(ranked, limit=limit, per_source=per_source)
            if overflow:
                M.counter("retrieval_source_quota_total").inc(
                    store=getattr(self.store, "name", "unknown"))
                logger.info("[来源配额] 同一来源最多 %d 条：挤掉 %d 条重复文档候选",
                            per_source, overflow)
        hits = ranked[:limit]
        # 法名命中时，保证最终上下文里至少 `law_scope_reserve` 条来自**该法**。
        # 用的是**法内较宽候选**（`law_scope_internal_top_k`）在法内**单独重排**后的前几名 ——
        # 这是 P8 的直接对策：该法条文在全局竞争里常被别的文档挤掉，而用户问的正是这部法。
        # 放在闸门**之前**：只影响"给闸门看哪些"，最终能不能用仍由闸门的词面证据说了算。
        reserve = int(getattr(self.config, "law_scope_reserve", 0) or 0)
        internal = (self._law_internal_candidates(query, scope, query_vector)
                    if (reserve > 0 or self.selector is not None) else [])
        protected = {name for _label, group in self.law_scope.groups_for(query) for name in group}
        hits = self._reserve_law_slots(query, hits, internal or scoped_hits, limit=limit,
                                       protected=protected or None)

        # LLM 列表式选择器（P8 第五条杠杆）：让模型在"该法的宽候选"里挑最能回答问题的条。
        # 候选刻意用 `internal`（法内重排前的一批），而不是已经进最终结果的 hits ——
        # 根因就是"正确的那条压根没进 top-k"，从 hits 里挑等于让模型在错的池子里选。
        selected_hits: list[SearchHit] = []
        if self.selector is not None and internal:
            ask, why = self.selector_decision(hits, protected, limit)
            if not ask:
                # 「有把握就不叫」：这一步是速度/质量的折中点，跳过必须看得见（单独计数 + 日志）
                M.counter("law_selector_total").inc(
                    result="skipped_confident", store=getattr(self.store, "name", "unknown"))
                logger.info("[列表式选择器] 本轮**不调用**（%s）：query=%r 最终 %d 条",
                            why, query, len(hits))
            else:
                # 版本标记：**默认关**（`LAW_SELECTOR_VERSION_LABELS`）。实测开了略差
                # （m3 0.5385→0.5256、cosine 0.5513→0.5385，各 −1 题）⇒ 不进默认链路；
                # 开关留给"新旧版条号差异更大"的语料。打开时**每轮现算**标记。
                if getattr(self.config, "law_selector_version_labels", False):
                    labels = self._selector_version_labels(internal)
                    if labels and hasattr(self.selector, "version_labels"):
                        self.selector.version_labels = labels
                        pending_labels = [name for name, label in labels.items()
                                          if label.startswith("尚未施行")]
                        if pending_labels:
                            logger.info("[列表式选择器] 候选含尚未施行的版本（已在提示词里标记）：%s",
                                        pending_labels[:3])
                outcome = self.selector.select(query, internal)
                selected_hits = list(outcome.get("picked") or [])
                M.counter("law_selector_total").inc(
                    result="picked" if selected_hits
                    else ("failed" if outcome.get("error") else "empty"),
                    trigger=why.split("(")[0],
                    store=getattr(self.store, "name", "unknown"))
                for hit in selected_hits:
                    hit.selected = True
                if selected_hits:
                    chosen = {h.chunk.id for h in selected_hits}
                    hits = selected_hits + [h for h in hits if h.chunk.id not in chosen]
                    hits = hits[:limit]
                    logger.info("[列表式选择器] 触发=%s 选中 %d 条置顶：%s", why, len(selected_hits),
                                [h.chunk.source for h in selected_hits])

        # 命中子块时补回父块上下文
        for hit in hits:
            if hit.chunk.parent_id:
                parent = self.store.get(hit.chunk.parent_id)
                if parent is not None:
                    hit.chunk.summary = parent.summary or hit.chunk.summary

        if hits:
            M.observe("top1_score", hits[0].score, reranker=getattr(self.reranker, "name", "none"))
        # 闸门放在**召回阶段出口**（不是 engine.prepare）：低分碎片根本不成为候选，
        # 上层拿到的就是"要么有依据、要么空"两种明确状态。
        hits = self._apply_relevance_gate(query, hits, vector_ok=not vector_error,
                                         keyword_ok=keyword_ok, scope=scope)
        # 用到了**尚未生效**的新版条文必须提示（用户口径：现行有效版为准，涉新规要提示）。
        # 与预热提示同样走 `last_retrieval_note`（引擎负责贴到答案上），两条可以并存。
        if self.version_filter is not None and hits:
            pending = self.version_filter.pending_sources(hits)
            if pending:
                M.counter("retrieval_pending_version_total").inc(
                    len(pending), store=getattr(self.store, "name", "unknown"))
                labels = "、".join(_display_source(name) for name in pending[:3])
                note = PENDING_VERSION_NOTE.format(sources=labels)
                self.last_retrieval_note = (self.last_retrieval_note + " " + note).strip()
                logger.info("[版本提示] 本轮命中含尚未施行的新版：%s（已在答案前加提示）", pending)
        logger.info("出口 retrieve：%d 条候选 -> 返回 %d 条（耗时 %.1fms）",
                    len(fused), len(hits), (time.perf_counter() - started) * 1000.0)
        return hits
