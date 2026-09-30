# -*- coding: utf-8 -*-
"""向量库接口与工厂。

接口刻意对齐 Milvus：
  * upsert / search / delete / count / get / all_chunks 是基础能力；
  * all_chunks 用于构建 BM25 关键词索引；
  * delete_by_role / replace_role 表达「按 Role_ID 分区、先删后插」的原子更新语义。

三个实现：
  * :class:`~legal_rag.store.milvus_store.MilvusVectorStore` —— **真实后端**（gRPC）；
  * :class:`~legal_rag.store.chroma_store.ChromaVectorStore` —— 本地轻量替代；
  * :class:`~legal_rag.store.memory_store.MemoryVectorStore` —— 零依赖兜底/降级目标。
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from ..schemas import Chunk, SearchHit

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """向量库的**统一接口**（三个实现：milvus / chroma / memory）。

    调用方（引擎、入库管线、BM25 索引）只依赖这个接口，不依赖具体后端：

    * ``upsert`` / ``search`` / ``get`` / ``delete`` / ``delete_by_role`` / ``count``
      是检索与入库的最小集合；
    * ``all_chunks`` 供**关键词（BM25）通道**一次性建索引；
    * ``replace_role`` 表达「按 role_id 分区、先删后插」的原子更新语义。

    ⚠️ ``count()`` 的口径是**活条数**（Milvus 侧为 ``query(count(*))``）——
    **不要**换成 ``num_entities`` / ``row_count``：后者含 upsert 留下的软删除墓碑，
    实测虚报过 5 倍（见 `docs/API.md` §6.3）。
    """

    #: 后端名（``memory`` / ``chroma`` / ``milvus``）；降级时的真实身份看这个字段。
    name: str = "base"

    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> int:
        """写入或覆盖（按 id）。返回写入条数。"""

    @abstractmethod
    def search(self, vector: list[float], top_k: int = 10,
               where: dict | None = None) -> list[SearchHit]:
        """向量检索 top-k。"""

    @abstractmethod
    def get(self, chunk_id: str) -> Chunk | None:
        """按 ``chunk_id`` 取单条；不存在返回 ``None``（**不抛异常**）。"""

    @abstractmethod
    def all_chunks(self, where: dict | None = None) -> list[Chunk]:
        """取出全部 chunk（供 BM25 建索引）。"""

    @abstractmethod
    def delete(self, ids: list[str]) -> int:
        """按 id 批量删除；返回实际删除条数。"""

    @abstractmethod
    def delete_by_role(self, role_id: str) -> int:
        """删掉某个 role 分区的**全部** chunk（先删后插的第一步）。"""

    @abstractmethod
    def count(self) -> int:
        """当前**活条数**（口径见类 docstring：绝不用 num_entities/row_count）。"""

    def replace_role(self, role_id: str, chunks: list[Chunk]) -> int:
        """先删后插：按 role_id 原子替换该分区的知识。"""
        self.delete_by_role(role_id)
        return self.upsert(chunks)

    def persist(self) -> None:
        """持久化（内存实现落盘，Chroma 自身已落盘）。"""


#: Milvus 连不上/维度不匹配时，是否降级到内存向量库（默认 True：保证服务不崩）。
#:
#: **降级的代价**：内存库数据不持久，进程重启即失、多进程之间不可见；
#: 只适用于「本机没有 Milvus 的本地调试」。
#: 若某个部署确实不想降级（希望连不上就直接失败、别让服务「假装健康」），
#: 设 ``MILVUS_FALLBACK_TO_MEMORY=0``，或调用方显式传 ``build_store(..., fallback=False)``，
#: 或用配置对象 ``RagConfig(…).milvus.fallback_to_memory = False``。
#: 优先级链见 :func:`resolve_milvus_fallback`；
#: 本常量是它的**最后一层兜底**，与 ``legal_rag.config.DEFAULT_MILVUS_FALLBACK_TO_MEMORY``
#: 保持同值（store 层不反向依赖 config，故各自持有一份；已有测试断言两者一致）。
DEFAULT_MILVUS_FALLBACK_TO_MEMORY = True


def _env_fallback_to_memory_or_none() -> bool | None:
    """读 ``MILVUS_FALLBACK_TO_MEMORY``；**未设置/空白 → None**（表示交给下一层决定）。

    与 :data:`DEFAULT_MILVUS_FALLBACK_TO_MEMORY` 的分工：本函数只回答「环境变量说了什么」，
    「没配时用什么」由优先级链的常量兜底层负责 —— 这样链条每一层都不互相掩盖。
    """
    import os

    raw = os.environ.get("MILVUS_FALLBACK_TO_MEMORY")
    if raw is None or raw.strip() == "":
        return None
    text = raw.strip().lower()
    if text in {"1", "true", "yes", "on", "y", "t"}:
        return True
    if text in {"0", "false", "no", "off", "n", "f"}:
        return False
    return None


def resolve_milvus_fallback(fallback: bool | None, config=None) -> bool:
    """决定 Milvus 构建失败时是否降级为内存库。

    优先级（**四层，逐层短路，任一层都不会被跳过**）：

    1. 显式 ``fallback`` 参数（调用方直接指定，最优先）；
    2. 环境变量 ``MILVUS_FALLBACK_TO_MEMORY``（显式配置优先于对象默认值）；
    3. ``config.milvus.fallback_to_memory``（``RagConfig`` 的一等字段）；
    4. :data:`DEFAULT_MILVUS_FALLBACK_TO_MEMORY`（常量兜底）。

    为什么把「环境变量」排在「配置字段」之前：``RagConfig`` 可以不经 ``from_env()``
    直接构造（如 ``RagConfig()``），此时字段取的是**类默认值**而非环境值；
    若让该字段压过环境变量，运维用 ``MILVUS_FALLBACK_TO_MEMORY=0`` 关闭降级就会失效。
    ``from_env()`` 本身已把环境变量解析进该字段，所以第 2 层命中时两者取值一致，
    只有「手工构造的 config + 环境变量」这一种组合会体现出分层差异 —— 而那种组合下
    尊重环境变量才是符合直觉的行为。
    """
    if fallback is not None:
        return bool(fallback)
    env_value = _env_fallback_to_memory_or_none()
    if env_value is not None:
        return env_value
    if config is not None:
        milvus = getattr(config, "milvus", None)
        configured = getattr(milvus, "fallback_to_memory", None) if milvus is not None else None
        if configured is None:
            # 兼容更早的顶层字段写法（历史遗留；config 现已改为 config.milvus.fallback_to_memory）
            configured = getattr(config, "milvus_fallback_to_memory", None)
        if configured is not None:
            return bool(configured)
    return DEFAULT_MILVUS_FALLBACK_TO_MEMORY


def build_store(provider: str, persist_dir=None, collection: str = "legal_kb",
                *, config=None, dim: int = 0, embedder=None,
                fallback: bool | None = None) -> VectorStore:
    """按名字构建向量库后端。

    ==========  ====================================================================
    provider    说明
    ==========  ====================================================================
    memory      内存兜底（零依赖，可选 pickle 落盘）
    chroma      本地 ChromaDB（轻量，替代 Milvus）
    milvus      **Milvus 真实后端（gRPC）**，走 ``legal_rag.store.milvus_store``
    ==========  ====================================================================

    Milvus 需要知道向量维度：优先用传入的 ``dim``；没传就向 ``embedder`` 实测
    （``ensure_dim()``）；再没有才用 ``config.embedding_dim`` 作为**兜底提示**
    （⚠️ 这个值在离线模式下是 256，与 bge-m3 的 1024 维不符，会触发维度不匹配）。

    因此**调用方（engine/脚本）必须把 ``config`` 与实测过维度的 ``embedder`` 一起传进来**，
    否则会退化成「用 256 维的配置提示去连 1024 维的 collection」——这正是 t7 F-t2-01
    那个「服务报 ok 但从未连过真实 Milvus」的根因。

    连接失败（或维度不匹配）时的行为由 :func:`resolve_milvus_fallback` 决定：
    默认降级到 :class:`~legal_rag.store.memory_store.MemoryVectorStore` 并打 WARNING，
    同时在返回对象上留下面三个标记，供调用方与 ``/health`` 识别真实后端：

    * ``primary``        —— 原本要用的后端名（``"milvus"``）；
    * ``fallback_reason``—— 降级原因（异常类型 + 消息，含修复建议）；
    * ``collection``     —— 原本要写的 collection 名。
    """
    key = (provider or "memory").strip().lower()
    if key in ("memory", "mem", "inmemory", "fallback"):
        from .memory_store import MemoryVectorStore

        return MemoryVectorStore(persist_dir=persist_dir)
    if key in ("chroma", "chromadb"):
        from .chroma_store import ChromaVectorStore

        return ChromaVectorStore(persist_dir=persist_dir, collection=collection)
    if key in ("milvus", "milvus_grpc", "zilliz"):
        return _build_milvus_store(persist_dir=persist_dir, collection=collection,
                                   config=config, dim=dim, embedder=embedder,
                                   fallback=fallback)
    raise ValueError(
        f"未知的 vector_store: {provider!r}（可选 memory | chroma | milvus）")


def _resolve_milvus_dim(store_dim: int, embedder, config) -> int:
    """确定 Milvus 的向量维度：显式 dim > embedder 实测 > config 兜底提示。

    **维度不从代码里硬编码**——真实部署一定要走 embedder 实测（bge-m3 = 1024）。
    最后那条 ``config.embedding_dim`` 只是「拿不到 embedder 时的兜底提示」，
    在离线模式下它是 256，与 bge-m3 的 1024 不符，会触发维度不匹配 —— 因此
    走到这一步会打 WARNING，避免它像 t7 F-t2-01 那样静默生效。
    """
    if store_dim and int(store_dim) > 0:
        return int(store_dim)
    if embedder is not None:
        ensure = getattr(embedder, "ensure_dim", None)
        if callable(ensure):
            return int(ensure())              # 实测：向 Ollama 真发一次请求
        current = int(getattr(embedder, "dim", 0) or 0)
        if current > 0:
            return current
        try:  # 没有 ensure_dim 的实现：用一次真实嵌入探出维度
            return len(embedder.embed_texts(["维度探测 probe"])[0])
        except Exception:  # noqa: BLE001 - 探不动就退回配置提示
            logging.getLogger(__name__).warning(
                "无法从 embedder 实测维度，退回配置提示值", exc_info=True)
    else:
        # 调用方只给了 config、没给 embedder（t7 F-t2-01 的原始姿势）。
        # 与其静默使用 config.embedding_dim（离线模式=256，与 1024 维 collection 冲突），
        # 不如按 config 现场造一个「一次性探测 embedder」真测一次维度 —— 这样即使别的
        # 调用点忘了传 embedder，也仍然遵守「维度必须实测」这条原则。
        probed = _probe_dim_from_config(config)
        if probed > 0:
            return probed
    hint = int(getattr(config, "embedding_dim", 0) or 0)
    logging.getLogger(__name__).warning(
        "Milvus 构建时既没有显式 dim、也没有可实测维度的 embedder，"
        "退回 config.embedding_dim=%s 作为兜底提示。\n"
        "该值若与目标 collection 的维度不一致会直接导致降级内存库 —— "
        "调用方请传 embedder（如 OllamaEmbedder.ensure_dim()）或显式 dim。", hint)
    return hint


def _probe_dim_from_config(config) -> int:
    """按 config 现场造一个一次性 embedder，实测维度后立刻关闭。

    只在调用方没传 embedder/dim 时兜底（见 :func:`_resolve_milvus_dim`）。
    任何失败都返回 0，由调用方决定是否退回 ``config.embedding_dim``。
    """
    log = logging.getLogger(__name__)
    provider = str(getattr(config, "embedding_provider", "") or "").strip()
    if not provider or provider.lower() in ("offline", "hash", "fallback"):
        # 离线兜底没有「真实模型」可言，其维度就等于 config.embedding_dim，无需探测
        return 0
    probe = None
    try:
        from ..embedding.base import build_embedder

        probe = build_embedder(provider, str(getattr(config, "embedding_model", "") or ""),
                              int(getattr(config, "embedding_dim", 0) or 0), config=config)
        measured = int(probe.ensure_dim())
        log.info("调用方未传 embedder，已按 config 现场实测维度：provider=%s -> dim=%d",
                 provider, measured)
        return measured
    except Exception as exc:  # noqa: BLE001 - 探测失败不阻断，交给兜底提示
        log.warning(
            "调用方未传 embedder，且按 config 现场实测维度失败（provider=%s）：%s: %s；"
            "将退回 config.embedding_dim 作为兜底提示", provider, type(exc).__name__, exc)
        return 0
    finally:
        closer = getattr(probe, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:  # noqa: BLE001
                log.debug("关闭探测 embedder 失败", exc_info=True)


def _build_milvus_store(*, persist_dir, collection: str, config, dim: int,
                        embedder=None, fallback: bool | None = None) -> VectorStore:
    """构建 Milvus 后端；连不上（或维度不匹配）时按 :func:`resolve_milvus_fallback` 降级。"""
    from ..config import RagConfig
    from .memory_store import MemoryVectorStore
    from .milvus_store import (MANIFEST_NAME, MilvusVectorStore,  # noqa: F401
                               MilvusError, MilvusUnavailableError,
                               VectorDimMismatchError)

    log = logging.getLogger(__name__)
    config = config if config is not None else RagConfig.from_env()

    resolved_dim = _resolve_milvus_dim(dim, embedder, config)
    collection_name = collection or ""
    default_collection = str(getattr(getattr(config, "milvus", None), "collection", ""))
    if not collection_name or collection_name in ("legal_kb", "legal_kb_default"):
        # 旧的通用默认名没有表达「模型 + 维度」，换成本项目的默认名（含 bge_m3_1024）
        collection_name = default_collection or "legal_rag_chunks_bge_m3_1024"

    do_fallback = resolve_milvus_fallback(fallback, config)

    if resolved_dim <= 0:
        raise ValueError(
            "Milvus 向量库需要明确的向量维度，但没能确定（dim=0）。\n"
            "请传入 dim，或传入可实测维度的 embedder（OllamaEmbedder.ensure_dim()），\n"
            "或设置 EMBEDDING_DIM 环境变量。"
        )

    store = MilvusVectorStore(
        config, dim=resolved_dim, collection=collection_name,
        manifest_dir=persist_dir,
        embedding_model=str(getattr(getattr(config, "ollama", None),
                                    "embedding_model", "") or ""),
        # t72：**生产工厂**是交付清单的合法拥有者（引擎/入库要刷新它）。
        # 直接 new MilvusVectorStore(...)（探针/测试）不带这个开关 ⇒ 默认 fail-closed，
        # 不会静默改写 index/milvus_manifest.json。
        allow_delivery_manifest=True,
    )
    try:
        store.connect()
        log.info("向量库使用 Milvus：%s（collection=%s，dim=%d，metric=%s）",
                 store.uri, store.collection, store.dim_for_log(), store.metric_type)
        return store
    except (MilvusUnavailableError, MilvusError, VectorDimMismatchError) as exc:
        if not do_fallback:
            raise
        log.warning(
            "Milvus 不可用（%s: %s），向量库降级为内存实现 MemoryVectorStore。\n"
            "注意：降级后重启进程索引就没了（只适合跑通链路/单测），"
            "真实使用请先修好 Milvus（curl http://%s:9091/healthz）",
            type(exc).__name__, exc, store.host)
        fallback_store = MemoryVectorStore(persist_dir=persist_dir)
        fallback_store.name = "memory"        # 保持接口语义清晰
        fallback_store.fallback_reason = f"{type(exc).__name__}: {exc}"  # type: ignore[attr-defined]
        fallback_store.primary = "milvus"     # type: ignore[attr-defined]
        fallback_store.collection = collection_name  # type: ignore[attr-defined]
        return fallback_store
