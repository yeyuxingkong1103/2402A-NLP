"""检索索引的构建与生命周期。**启动期关注的东西全在这里。**

    build_index(...) —— 连库、拉语料、建 BM25 索引
    set_index(...)   —— 装进常驻单例（服务端启动期）
    get_index()      —— 请求期取用
    reset_index()    —— 自检与验收脚本用

与 `service.py` 的分界：本模块管**"索引怎么来、活多久"**，`service.py` 管
**"给一条问题怎么检索"**。前者的失败时机是启动期（Milvus 不可达、语料为空），
后者的失败时机是请求期（维度不符、库中途挂了）—— 两类失败的处置完全不同，
因此分在两处。

---

## 为什么索引常驻内存

`build_index()` 在**启动期**调用一次，`get_index()` 在请求期复用 ——
与 S8 的 `backend/query/service.py` 的 `get_encoder()` 完全同一模式。

不做持久化（research R4）：当前 65 chunks，倒排索引是几 KB 量级，构建在毫秒级。
持久化会引入"索引文件与库内容是否一致"这个新的一致性维度 —— 为一个不存在的
性能问题引入一个真实的一致性风险，是净负收益。

**何时需要重新考虑**：语料达到十万条量级、启动构建超过 5 秒时。届时的正解是
把 BM25 迁到 Milvus 原生稀疏向量，而不是给内存索引加持久化。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from . import (
    DEFAULT_ADMIT_RANK,
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    DEFAULT_CANDIDATES,
    DEFAULT_COLLECTION,
    DEFAULT_LEXICAL_MIN_COVERAGE,
    DEFAULT_RRF_K,
    DEFAULT_URI,
    EXIT_DATA,
    RetrievalError,
    STATE_NO_CANDIDATES,
)
from . import lexical, store
from .lexical import LexicalIndex
from .models import Candidate, CorpusChunk, RetrievalResult

logger = logging.getLogger(__name__)

__all__ = [
    "IndexBundle",
    "SearchTrace",
    "build_index",
    "get_index",
    "set_index",
    "reset_index",
]

# 常驻单例。**启动期构建，请求期绝不重建。**
#
# 用模块级变量而不是容器/类：本进程只有一个索引，"全局唯一"这件事用最简单的
# 形式表达即可。引入依赖注入容器去管理一个永远不会变的东西，是为不存在的需求
# 付复杂度 —— 与 `backend/query/service.py` 的 `_encoder` 同一取向。
_index: IndexBundle | None = None


@dataclass
class IndexBundle:
    """启动期固定下来的全部检索状态。**索引侧参数在进程内不变。**"""

    client: object
    collection: str
    chunks: list[CorpusChunk]
    chunks_by_id: dict[str, CorpusChunk]
    lexical: LexicalIndex
    candidates: int = DEFAULT_CANDIDATES
    rrf_k: int = DEFAULT_RRF_K
    admit_rank: int = DEFAULT_ADMIT_RANK
    min_coverage: float = DEFAULT_LEXICAL_MIN_COVERAGE
    manifest_total: int | None = None

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def matches_manifest(self) -> bool | None:
        """与 `index_manifest.json` 的 `total_chunks` 是否一致。

        `None` 表示**无法比对**（清单缺失或格式不符）—— 调用方 MUST 如实报告
        "无法比对"，MUST NOT 把它当成"一致"。
        """

        if self.manifest_total is None:
            return None
        return self.size == self.manifest_total


@dataclass
class SearchTrace:
    """一次检索的全部内部细节。**仅供 CLI 与日志使用，MUST NOT 进响应体。**

    `result` 是对外契约；其余字段存在的意义是让融合规则的有效性能被验证 ——
    只显示最终排名，CLI 就退化成一个"看起来对不对"的工具，标定不了任何参数。
    """

    result: RetrievalResult
    fused: list[Candidate] = field(default_factory=list)
    semantic_n: int = 0
    lexical_n: int = 0
    selected: list[Candidate] = field(default_factory=list)
    state: str = STATE_NO_CANDIDATES
    elapsed_ms: int = 0


def build_index(
    *,
    uri: str = DEFAULT_URI,
    collection: str = DEFAULT_COLLECTION,
    candidates: int = DEFAULT_CANDIDATES,
    rrf_k: int = DEFAULT_RRF_K,
    admit_rank: int = DEFAULT_ADMIT_RANK,
    min_coverage: float = DEFAULT_LEXICAL_MIN_COVERAGE,
    bm25_k1: float = DEFAULT_BM25_K1,
    bm25_b: float = DEFAULT_BM25_B,
    client=None,
) -> IndexBundle:
    """连库、拉语料、建 BM25 索引。**在服务启动期调用一次。**

    ⚠️ 这一步让 Milvus 从"可选"变成"启动的硬前提"。语料拉不到即抛
    `RetrievalError`，调用方（`backend/serve.py`）以非 0 状态退出 ——
    MUST NOT 以"能启动但关键词路是空的"状态继续：那正是 spec 的 Edge Case
    明令禁止的静默降级。

    `client` 可由调用方注入（CLI 与验收脚本复用同一条连接）。
    """

    if client is None:
        client = store.connect(uri)

    chunks = store.fetch_corpus(client, collection)

    if not chunks:
        raise RetrievalError(
            EXIT_DATA,
            "语料为空（collection=%s）。关键词索引无从建立。\n"
            "  多半是向量库被重建过，或 collection 名写错了。请先运行 S6 入库：\n"
            "    D:/zg6_Project/9/med_rag/rag/python.exe -m backend.index_milvus"
            % collection,
        )

    # 先建 BM25 索引（顺带把 tokens 回填到语料上），再建查表 ——
    # 两步都只依赖 `chunks`，顺序无实质影响，但先建索引能让失败暴露得更早。
    lex = lexical.build_index(chunks, k1=bm25_k1, b=bm25_b)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}

    bundle = IndexBundle(
        client=client,
        collection=collection,
        chunks=chunks,
        chunks_by_id=by_id,
        lexical=lex,
        candidates=candidates,
        rrf_k=rrf_k,
        admit_rank=admit_rank,
        min_coverage=min_coverage,
        manifest_total=store.manifest_total_chunks(),
    )

    # 与清单比对：**不一致只告警，不阻止启动**。理由是库可能刚被重跑过而
    # manifest 是旧的 —— 那种情况下"照常启动 + 明确告警"比"拒绝启动"有用。
    # 但 MUST NOT 静默：一份被截断的语料看起来完全正常。
    matched = bundle.matches_manifest
    if matched is None:
        logger.warning("语料条数无法与索引清单比对（清单缺失或格式不符）")
    elif not matched:
        logger.warning(
            "语料条数与索引清单不一致：库中 %d 条，清单记 %s 条。"
            "若刚重新入库，属正常；否则说明清单已过时。",
            bundle.size,
            bundle.manifest_total,
        )
    else:
        logger.info("语料 %d 条，与索引清单一致", bundle.size)

    return bundle


def set_index(bundle: IndexBundle) -> None:
    """把构建好的索引装进常驻单例。**服务端启动期调用。**

    为什么不让 `build_index()` 自己装：构建是纯函数式的（给参数、得索引），
    装进全局是**进程生命周期**的决定。混在一起会让 CLI 也污染全局 ——
    而 CLI 与最终用户的服务是不同进程，它们不该共享这份状态。
    """

    global _index
    _index = bundle


def get_index() -> IndexBundle:
    """返回常驻索引。**未构建时抛错而不是惰性构建。**

    惰性构建（首次调用时连库）会让第一次用户提问承担启动成本，且把
    "Milvus 不可达"从启动期推迟到请求期 —— 而启动期的失败是可见的、可修的，
    请求期的失败只是用户看到一句拒答。
    """

    if _index is None:
        raise RetrievalError(
            EXIT_DATA,
            "检索索引尚未构建。这是**编程错误**而非数据问题："
            "服务端 MUST 在启动期调用 retrieve.bundle.build_index() 与 set_index()"
            "（见 backend/serve.py），CLI MUST 先调一次。",
        )
    return _index


def reset_index() -> None:
    """丢弃常驻索引并关闭连接。**仅供自检与验收脚本使用。**"""

    global _index
    if _index is not None and _index.client is not None:
        try:
            _index.client.close()
        except Exception:  # noqa: BLE001 —— 关闭失败不影响本函数的目的
            logger.debug("关闭 Milvus 连接失败", exc_info=True)
    _index = None
