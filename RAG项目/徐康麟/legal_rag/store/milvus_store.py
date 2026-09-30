# -*- coding: utf-8 -*-
"""Milvus 向量库真实实现（gRPC，免 token）。

为什么用 pymilvus 的 ``MilvusClient``
------------------------------------
Milvus 的对外协议就是 gRPC（19530），``MilvusClient`` 是官方推荐的「高层 + gRPC」
客户端；REST（19530 的 ``/v2/vectordb/*``）只用来排障（例如
``POST /v2/vectordb/collections/describe``），不进入业务链路。

维度治理（用户点名「注意维度差异」）
-----------------------------------
队长的只读探测已经证明：既有 3 个 collection 的命名与维度是**反的**——

    ==========================  ======  =======
    collection                  dim     metric
    ==========================  ======  =======
    user_long_term_memory       1024    COSINE
    user_long_term_memory_bge   512     COSINE
    ragas_test_collection       512     COSINE
    ==========================  ======  =======

而 Milvus 的 ``FLOAT_VECTOR`` 维度**建好之后不可修改**。因此本实现：

1. **绝不复用**上述任何 collection；新 collection 名取自 config
   （默认 ``legal_rag_chunks_bge_m3_1024``，名字自带模型与维度）；
2. **维度不写死**：由 t2 的 ``OllamaEmbedder`` 实测（bge-m3 = 1024）后传入，
   建 collection / 首次写入前与已存在的 collection 比对；
3. **不匹配不许静默**：抛 :class:`VectorDimMismatchError`，消息里带
   期望维度 / 实际维度 / 模型名 / collection 名 + 两条出路（换名 /
   ``MILVUS_RECREATE_ON_DIM_MISMATCH=true`` 重建），并明确警告重建会丢数据；
4. **模型与维度写进元数据**：collection ``description`` 里存
   ``legal_rag.chunk_schema=1; embedding.model=...; embedding.dim=...; metric_type=COSINE``，
   同时落到本地 manifest ``index/milvus_manifest.json``，``/health``（t4）可直接报出。

并发策略（t4 会压）
-------------------
``MilvusClient`` 内部持有一条 gRPC channel，多线程共享时要走它自己的锁，压测下会成为
串行点；而且不同线程并发切换 ``db_name`` / 变更 collection 时行为容易出乎意料。
本实现采用 **thread-local client**：每个线程各持一个 ``MilvusClient``，用
``threading.local()`` 管理与复用，并对已完成的请求做 ``flush``。理由与依据：

* gRPC channel 建立成本不高（同一进程内长连接复用），但 **共享 channel 的写放大**
  与 pymilvus 内部状态让并发写更容易踩坑；thread-local 把「一个 client 只被一个线程
  使用」这条前提写死在代码里，可预测性最好；
* 线程数由 ``ConcurrencyConfig.thread_pool_size`` / ``ingest_workers`` 约束（默认 4~8），
  因此最多 4~8 条 channel，不会打爆 Milvus；
* 每个线程的 client 在使用前都会 ``_ensure_ready`` 检查 collection 是否存在并
  ``load_collection``，避免「上个线程刚建好、这个线程还没加载」的竞态。

连不上 Milvus
-------------
:meth:`MilvusVectorStore.connect` 抛 :class:`MilvusUnavailableError`（带 host/port、
底层异常、修复建议）；工厂 :func:`legal_rag.store.base.build_store` 捕获后按配置
降级到 :class:`~legal_rag.store.memory_store.MemoryVectorStore` 并打 WARNING
（服务不崩，但日志明确说明「当前不是真实 Milvus」）。
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .. import metrics as M
from ..observability import log_io, timed, traced, wrap_errors
from ..schemas import Chunk, SearchHit
from .base import VectorStore

from ..config import release_local_milvus_uri_env

# pymilvus 的 ORM 单例在 import ``pymilvus.orm.collection`` 时就会解析环境变量
# ``MILVUS_URI``（只认 http(s)://）；本地文件式的配置必须先摘掉，否则本模块在
# import MilvusClient / DataType 的那一刻就抛 ConnectionConfigException，
# 表现为「Milvus 初始化失败」并被降级成内存库。详见 legal_rag/config.py。
release_local_milvus_uri_env()


logger = logging.getLogger(__name__)

#: 本实现的 schema 版本（写进 collection description，便于以后兼容升级）
CHUNK_SCHEMA_VERSION = 1
#: 本地 manifest 文件名（记录「模型 + 维度 + collection」）
MANIFEST_NAME = "milvus_manifest.json"

VARCHAR_MAX_LENGTH = {
    "id": 64,
    "text": 8192,
    "source": 512,
    "doc_id": 256,
    "parent_id": 64,
    "role_id": 128,
    "summary": 2048,
}
#: 标量字段（除向量外全部要落进 Milvus schema）
SCALAR_FIELDS = (
    "text", "source", "doc_id", "chunk_index", "parent_id", "is_parent",
    "role_id", "summary", "created_at", "updated_at",
)
OUTPUT_FIELDS = ("id", "text", "vector", *[f for f in SCALAR_FIELDS if f != "text"])

#: **不含向量**的取字段（t68：BM25 / 关键词通道专用）。
#:
#: 为什么必须单独一套：``all_chunks`` 默认把 ``vector`` 一起搬回来，1024 维 × 14 万行
#: ≈ GB 级内存（Python float 对象更放大数倍）；而 BM25 只用 ``text`` + 元数据。
#: 关键词通道全量建索引必须走这一套，否则真实规模下会 ``MemoryError``
#: （t26 残留 R1 的实测：真实首问 MemoryError → 静默降级纯向量 → 171.5s）。
LEAN_OUTPUT_FIELDS = tuple(f for f in OUTPUT_FIELDS if f != "vector")

#: Milvus ``query`` 的**查询窗口上限**（实测 v2.5.1：``1 <= offset+limit <= 16384``）。
#: 超过它会被服务端直接拒绝 ⇒ 任何走 ``limit/offset`` 的全量读取都必须把它算进去，
#: 并在窗口用尽时**明确告警**（t68：关键词通道的 1.6 万行截断就是踩了这个）。
QUERY_WINDOW_LIMIT = 16384

#: 交付清单护栏（t72）的指标口径登记。⚠️ 集中登记的位置应是 `legal_rag/metrics.py`；
#: 它不在本任务 inScope，故按 t68 的先例在**发射模块** import 时登记，待 t86 收编。
M.METRIC_CATALOG.setdefault(
    "manifest_write_rejected_total",
    ("counter",
     "拒绝改写交付清单 index/milvus_manifest.json 的次数（reason=delivery_manifest；"
     "默认 manifest_dir 指向交付目录时发生，属保护性拒绝，不是业务失败）",
     "count"))


def delivery_manifest_path() -> Path:
    """工作区**交付清单**的绝对路径（``<repo>/index/milvus_manifest.json``）。"""
    return Path(__file__).resolve().parents[2] / "index" / MANIFEST_NAME


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in ("1", "true", "yes", "on")


def _manifest_guard_mode() -> str:
    """``LEGAL_RAG_MANIFEST_GUARD=raise`` 时**硬失败**（拒绝启动）；默认 ``skip``（拒绝写入）。"""
    mode = str(os.environ.get("LEGAL_RAG_MANIFEST_GUARD", "")).strip().lower()
    return "raise" if mode in ("raise", "strict", "fail") else "skip"

#: ``search`` 专用的 output_fields：**刻意不含 ``vector``**。
#:
#: 依据（2026-09-16 实测，Milvus v2.5.1 + pymilvus 3.0.1，collection
#: ``legal_rag_chunks_bge_m3_1024``）：search 的「字段回查」在**结果 ≥3 条**且
#: output_fields 含 ``vector`` 时返回不全，SDK 报
#: ``code=2200 incomplete query result, missing id ...: inconsistent requery result``
#: （100% 复现：串行 12/12、并发 2×5 全失败、冷启动/刚 load 后同样失败；
#: 换成纯标量 + ``text`` 字段从不触发；结果 ≤2 条也不触发）。
#: 检索链路**不需要**命中块的原始向量（rerank / 引用 / 上下文只用 text 与标量），
#: 真要向量用 ``get(chunk_id)`` 单条取（单条不触发该问题）。
SEARCH_OUTPUT_FIELDS = tuple(f for f in OUTPUT_FIELDS if f != "vector")

#: 只读盘点（generation inventory）用的标量字段：**刻意不含 ``text`` 与 ``vector``**。
#: 盘点只关心"哪一代、属于哪个文件、父块还是子块、角色是谁"，把 1.6 万条正文搬回来
#: 只会让一次只读盘点变慢、变重（旧索引里旧策略文本很长）。
INVENTORY_FIELDS = ("id", "source", "doc_id", "created_at", "is_parent",
                    "role_id", "chunk_index")

#: 检索侧「最新代」过滤（t52）：启用后 search 的**候选倍数**（过滤在排序之后，
#: 只取 top_k 再过滤会让旧代块把名次让空 ⇒ 多取候选、过滤后截回 top_k）。
GENERATION_FILTER_OVERFETCH = 4
#: 候选上限（避免 top_k 很大时把一次检索放大成几百条）。
GENERATION_FILTER_MAX_OVERFETCH = 50


class MilvusError(RuntimeError):
    """Milvus 相关错误的基类。"""


class MilvusUnavailableError(MilvusError):
    """连不上 Milvus（gRPC 不通 / 超时 / 认证失败），消息里带修复建议。"""


class VectorDimMismatchError(MilvusError):
    """向量维度与 collection 现有维度不一致（**不静默**、**不自动删数据**）。"""

    def __init__(self, message: str, *, expected: int = 0, actual: int = 0,
                 collection: str = "", model: str = "") -> None:
        super().__init__(message)
        self.expected = expected
        self.actual = actual
        self.collection = collection
        self.model = model

    def as_dict(self) -> dict[str, Any]:
        return {
            "expected_dim": self.expected,
            "actual_dim": self.actual,
            "collection": self.collection,
            "embedding_model": self.model,
        }


def build_dim_mismatch_message(*, expected: int, actual: int, collection: str,
                               model: str, recreate_enabled: bool) -> str:
    """构造（并集中维护）维度不匹配的明确报错文案。"""
    return (
        f"Milvus collection 「{collection}」的向量维度是 {actual}，"
        f"但当前嵌入模型「{model}」实测输出 {expected} 维 —— 维度不匹配，拒绝写入。\n"
        f"原因：Milvus 的 FLOAT_VECTOR 维度建好后不可修改，写进去只会报底层错误。\n"
        f"两条出路（任选其一）：\n"
        f"  ① 换一个名字匹配的 collection（不动老数据，最安全）：\n"
        f"       set MILVUS_COLLECTION=legal_rag_chunks_{model}_${expected}\n"
        f"     （Windows PowerShell：$env:MILVUS_COLLECTION=\"legal_rag_chunks_bge_m3_{expected}\"）\n"
        f"  ② 允许重建该 collection（⚠️ 会先 DROP 再建，**该 collection 内的向量数据全部丢失**，"
        f"不可恢复；\n"
        f"     仅当确认这个 collection 是本项目专用、且可以重新入库时才用）：\n"
        f"       set MILVUS_RECREATE_ON_DIM_MISMATCH=true\n"
        f"  当前 MILVUS_RECREATE_ON_DIM_MISMATCH={str(recreate_enabled).lower()}"
        f"（false = 只报错，不删任何数据）。\n"
        f"注意：绝不触碰既有 user_long_term_memory / user_long_term_memory_bge / "
        f"ragas_test_collection。"
    )


class MilvusVectorStore(VectorStore):
    """Milvus 向量库（gRPC ``MilvusClient``）。

    典型用法::

        store = MilvusVectorStore(config, dim=embedder.ensure_dim())
        store.connect()             # 幂等；失败抛 MilvusUnavailableError
        store.upsert(chunks)
        hits = store.search(vector, top_k=5, where={"is_parent": False})
    """

    name = "milvus"

    def __init__(
        self,
        config: Any = None,
        *,
        dim: int = 0,
        collection: str = "",
        host: str = "",
        port: int = 0,
        uri: str = "",
        token: str = "",
        db_name: str = "",
        timeout: float | None = None,
        connect_timeout: float | None = None,
        metric_type: str = "",
        index_type: str = "",
        recreate_on_dim_mismatch: bool | None = None,
        consistency_level: str = "",
        manifest_dir: str | Path | None = None,
        embedding_model: str = "",
        fallback: VectorStore | None = None,
        allow_delivery_manifest: bool = False,
    ) -> None:
        cfg = getattr(config, "milvus", None)
        get = (lambda name, default: getattr(cfg, name, default)) if cfg is not None else (
            lambda name, default: default)

        self.config = config
        self.host = host or str(get("host", "127.0.0.1"))
        self.port = int(port or get("port", 19530))
        self._uri = uri or str(get("uri", "") or "")
        self.token = token if token else str(get("token", "") or "")
        self.user = str(get("user", "") or "")
        self.password = str(get("password", "") or "")
        self.db_name = db_name or str(get("db_name", "default") or "default")
        self.timeout = float(timeout if timeout is not None
                             else get("timeout", 30.0))
        self.connect_timeout = float(connect_timeout if connect_timeout is not None
                                     else get("connect_timeout", 10.0))
        self.metric_type = (metric_type or str(get("metric_type", "COSINE"))).upper()
        self.index_type = index_type or str(get("index_type", "HNSW"))
        self.index_m = int(get("index_m", 16))
        self.index_ef_construction = int(get("index_ef_construction", 200))
        self.index_ef_search = int(get("index_ef_search", 64))
        if recreate_on_dim_mismatch is None:
            recreate_on_dim_mismatch = bool(get("recreate_on_dim_mismatch", False))
        self.recreate_on_dim_mismatch = bool(recreate_on_dim_mismatch)

        #: 一致性级别：默认 Session（同一客户端「写完立刻能读到」），
        #: 避免入库后马上检索出现「刚写的查不到」的假阴性。
        #: Milvus 支持 Strong / Session / Bounded / Eventually，其中 Session 是
        #: 「本客户端读自己写」的语义，对「入库 -> 立即检索自检」这条链路最合适，
        #: 又不像 Strong 那样每次都强制等待全局时间戳。
        self.consistency_level = consistency_level or str(
            get("consistency_level", "Session") or "Session")

        if not collection:
            top_collection = getattr(config, "collection", "") or ""
            collection = top_collection or str(get("collection", "legal_rag_chunks_bge_m3_1024"))
        self.collection = collection

        ollama = getattr(config, "ollama", None)
        self.embedding_model = embedding_model or (
            str(getattr(ollama, "embedding_model", "")) if ollama is not None else "")

        #: 维度由调用方从 embedder 实测后传入（不写死）；首次写入前必须已确定
        self._dim = int(dim)
        self._dim_lock = threading.Lock()

        base_dir = manifest_dir if manifest_dir is not None else getattr(
            config, "index_dir", None)
        self.manifest_path: Path | None = (
            Path(base_dir) / MANIFEST_NAME if base_dir is not None else None)
        #: 交付清单护栏（t72）：默认 manifest_dir 指向工作区交付目录时**不得改写交付物**。
        #: 只有生产工厂 ``build_store()``（或显式 env 放行）才允许刷新它。
        self._delivery_manifest = delivery_manifest_path()
        self._allow_delivery_manifest = bool(
            allow_delivery_manifest or _env_flag("LEGAL_RAG_ALLOW_DELIVERY_MANIFEST"))
        #: 上一次写交付清单是否被护栏拒了（探针/测试可直接读，便于取证）
        self.manifest_write_blocked = False

        #: 降级目标（连不上 Milvus 时由工厂设置并向调用方暴露）
        self.fallback = fallback

        self._local = threading.local()
        self._clients: list[Any] = []
        self._clients_lock = threading.Lock()
        self._ready = False
        self._created = False
        self._remote_dim = 0
        self._description = ""
        self._manifest: dict[str, Any] = {}
        self._log = logger
        #: 最近一次 :meth:`iter_all_chunks` 的只读统计（read/delivered/filtered_out，
        #: 供探针/运维核对"到底喂了多少、有没有被截断"）。
        self.last_feed_stats: dict[str, Any] = {}

        #: 检索侧「最新代」过滤（t52）——**默认开启**（`RAG_GENERATION_FILTER=0` 关闭），
        #: 开启后每个文件只返回最新一代的块。为什么在检索侧做：t40 实测 Milvus 的删除
        #: 对 query/search 路径不生效（报删 15,116 条、`get()` 取不到，但 13,760 行仍可被
        #: 召回），所以「只保留最新一代」不能依赖删除语义（见 handoff/t40-CONVERGENCE-r4.md）。
        #: 索引文件（`index/latest_generations.json`）缺失时 **fail-open**：跳过过滤 + 告警
        #: 一次，召回范围不变 —— 绝不让"索引没建好"变成检索不可用。
        gen = None
        try:
            from legal_rag.retrieve import generation_filter as gen  # 延迟导入，避免环
        except Exception:  # noqa: BLE001 - 工具环境缺少该模块时保持既有行为
            logger.debug("未被代数过滤模块可用（跳过检索侧最新代过滤）", exc_info=True)
        self._gen_filter_mod = gen
        index_dir = base_dir if base_dir is not None else getattr(config, "index_dir", None)
        self._gen_index_path = (
            gen.index_path(index_dir) if (gen is not None and index_dir is not None) else None)
        self._gen_filter_enabled = bool(gen.filter_enabled()) if gen is not None else False
        self._gen_filter_unavailable_logged = False
        self._gen_index = None            # LatestGenerationIndex | None（懒加载）
        self._gen_index_loaded = False
        self._gen_kept_total = 0
        self._gen_dropped_total = 0

    # ------------------------------------------------------------------
    # 连接 / 建表
    # ------------------------------------------------------------------
    @property
    def uri(self) -> str:
        return self._uri or f"http://{self.host}:{self.port}"

    # ------------------------------------------------------------------
    # 检索侧「最新代」过滤（t52）
    # ------------------------------------------------------------------
    @property
    def generation_filter_enabled(self) -> bool:
        """总开关（**默认开启**；`RAG_GENERATION_FILTER=0` 关闭 = 回到改前行为）。"""
        return self._gen_filter_enabled

    @property
    def generation_index_path(self) -> Path | None:
        return self._gen_index_path

    def generation_index(self):
        """懒加载「最新代」索引（`index/latest_generations.json`）。

        * 文件不存在 / 解析失败 ⇒ 返回 ``None`` 并**告警一次**（不改变召回）；
        * 可用 `RAG_GENERATION_FILTER_REQUIRE_INDEX=1` 把它变成硬失败（fail-closed）。
        """
        if self._gen_index_loaded:
            return self._gen_index
        self._gen_index_loaded = True
        gen = self._gen_filter_mod
        if gen is None or not self._gen_filter_enabled:
            return None
        index = gen.LatestGenerationIndex.load(self._gen_index_path)
        if index is None:
            raise_gen = gen.require_index()
            logger.warning(
                "[GEN-FILTER] 已开启最新代过滤（RAG_GENERATION_FILTER=1），但索引文件不可用：%s"
                " —— %s；重建命令：python -m legal_rag.retrieve.generation_filter --rebuild",
                self._gen_index_path,
                "按配置 requirement 拒绝继续（fail-closed）" if raise_gen
                else "本次**跳过**过滤（fail-open，召回范围不变）")
            self._gen_filter_unavailable_logged = True
            if raise_gen:
                raise FileNotFoundError(
                    f"最新代索引不可用：{self._gen_index_path}（RAG_GENERATION_FILTER_REQUIRE_INDEX=1）")
            return None
        self._gen_index = index
        logger.info("[GEN-FILTER] 最新代过滤已启用：key_field=%s 索引 %d 个文件（built_at=%s）",
                    index.field, index.size, index.built_at or "未知")
        return self._gen_index

    def is_latest_generation(self, row: Any) -> bool:
        """一条记录是否来自其文件的**最新代**；未开启过滤时恒为 True。"""
        if not self._gen_filter_enabled:
            return True
        index = self.generation_index()
        if index is None:
            return True
        return index.is_latest(row)

    def _generation_filter_rows(self, rows: Sequence[Any]) -> list[Any]:
        """按「最新代」过滤行，并累计计数（**`_filter_generation_rows` 的薄壳**）。"""
        keep, _ = self._filter_generation_rows(rows)
        return keep

    def generation_filter_stats(self) -> dict:
        index = self._gen_index
        if index is None and self._gen_filter_enabled:
            try:
                index = self.generation_index()   # 触发懒加载，否则统计永远是"不可用"
            except Exception:                     # noqa: BLE001 - fail-closed 配置下也别让统计抛
                index = None
        return {
            "enabled": self._gen_filter_enabled,
            "index_available": index is not None,
            "index_path": str(self._gen_index_path) if self._gen_index_path else "",
            "index_size": index.size if index is not None else 0,
            "key_field": index.field if index is not None else "",
            "dropped_total": self._gen_dropped_total,
            "kept_total": self._gen_kept_total,
        }

    @property
    def dim(self) -> int:
        """当前已知的向量维度（0 表示还没实测确定）。"""
        return self._dim

    def dim_for_log(self) -> int:
        return self._dim

    def describe(self) -> dict:
        """一行摘要（日志 / health / 报错都用它）。"""
        return {
            "store": self.name,
            "uri": self.uri,
            "db_name": self.db_name,
            "collection": self.collection,
            "dim": self._dim,
            "remote_dim": self._remote_dim,
            "metric_type": self.metric_type,
            "index_type": self.index_type,
            "embedding_model": self.embedding_model,
            "token": "***" if self.token else "(empty)",
            "ready": self._ready,
            "created": self._created,
        }

    def _import_client(self):
        try:
            from pymilvus import MilvusClient  # type: ignore
        except ImportError as exc:
            raise MilvusUnavailableError(
                "未安装 pymilvus，无法使用 Milvus 向量库。\n"
                "    .venv\\Scripts\\python.exe -m pip install -r requirements.txt\n"
                "或改用兜底向量库：VECTOR_STORE=memory"
            ) from exc
        return MilvusClient

    def _new_client(self):
        MilvusClient = self._import_client()
        kwargs: dict[str, Any] = {
            "uri": self.uri,
            "db_name": self.db_name,
            "timeout": self.connect_timeout,
        }
        if self.token:
            kwargs["token"] = self.token
        elif self.user:
            kwargs["user"] = self.user
            kwargs["password"] = self.password
        try:
            client = MilvusClient(**kwargs)
        except Exception as exc:  # noqa: BLE001 - 统一转成可读错误
            raise MilvusUnavailableError(
                f"连接 Milvus 失败：{self.uri}（db={self.db_name}，"
                f"超时 {self.connect_timeout:g}s）：{type(exc).__name__}: {exc}\n"
                f"排查建议：\n"
                f"  1) 确认 Milvus 在跑：curl http://{self.host}:9091/healthz\n"
                f"  2) 确认 gRPC 端口（默认 19530）可连通（19530 不是 REST，别用浏览器打开）\n"
                f"  3) 确认 MILVUS_HOST 指向正确的 Milvus（默认已是本机 127.0.0.1）：\n"
                f"     - 本机 / WSL 里用 Docker 起了 Milvus：用 MILVUS_HOST=127.0.0.1\n"
                f"     - 连远端（另一台机器 / 旧部署）：必须显式设 MILVUS_HOST=<那台机器>\n"
                f"  4) 免 token 部署请把 MILVUS_TOKEN 留空\n"
                f"  5) 需要先跑通链路可临时降级：VECTOR_STORE=memory"
            ) from exc
        with self._clients_lock:
            self._clients.append(client)
        return client

    def _client(self):
        """当前线程的 MilvusClient（thread-local 复用，见模块 docstring 的并发策略）。

        Collection 的 ``load`` 状态由 Milvus 服务端按 collection 维护，但**新客户端
        首次查询前仍要先 load 一次**（否则可能拿到「未加载」错误），因此这里在
        创建线程专属 client 后补一次幂等 load。
        """
        client = getattr(self._local, "client", None)
        if client is None:
            client = self._new_client()
            self._local.client = client
            if self._ready:
                self._load_collection(client)
        return client

    def connect(self, *, create: bool = True) -> "MilvusVectorStore":
        """幂等连接：建/校验 collection 与索引。

        维度不匹配时抛 :class:`VectorDimMismatchError`（或按配置重建后再建）。
        """
        if self._ready:
            return self
        with self._dim_lock:
            if self._ready:
                return self
            started = time.perf_counter()
            self._log.info("[BEFORE] Milvus connect uri=%s db=%s collection=%s dim=%s",
                           self.uri, self.db_name, self.collection, self._dim or "(待实测)")
            try:
                if self._dim <= 0:
                    raise MilvusError(
                        f"未确定向量维度就连接 Milvus（collection={self.collection}）。\n"
                        f"请先用 embedder 实测维度再构造 MilvusVectorStore：\n"
                        f"    dim = embedder.ensure_dim()   # bge-m3 经 Ollama = 1024\n"
                        f"    store = MilvusVectorStore(config, dim=dim)"
                    )
                self._ensure_collection(create=create)
                # 载入 Collection 必须在索引建好之后（无索引时 load 会报 code=700）
                self._load_collection(self._client())
                self._ready = True
            except (MilvusUnavailableError, VectorDimMismatchError, MilvusError):
                raise
            except Exception as exc:  # noqa: BLE001 - 带上下文继续抛
                raise MilvusError(
                    f"Milvus 初始化失败（uri={self.uri}, collection={self.collection}, "
                    f"dim={self._dim}）：{type(exc).__name__}: {exc}") from exc
            elapsed = (time.perf_counter() - started) * 1000.0
            self._log.info("[AFTER] Milvus connect 完成，耗时 %.1fms：%s",
                           elapsed, json.dumps(self.describe(), ensure_ascii=False))
            return self

    def _ensure_collection(self, *, create: bool = True) -> None:
        from pymilvus import DataType  # type: ignore

        client = self._client()
        exists = False
        try:
            exists = bool(client.has_collection(self.collection, timeout=self.timeout))
        except Exception as exc:  # noqa: BLE001
            raise MilvusUnavailableError(
                f"查询 collection 是否存在失败（uri={self.uri}, "
                f"collection={self.collection}）：{type(exc).__name__}: {exc}\n"
                f"提示：先确认 Milvus 可达：curl http://{self.host}:9091/healthz"
            ) from exc

        if exists:
            self._remote_dim = self._read_remote_dim(client)
            if self._remote_dim and self._remote_dim != self._dim:
                M.counter("dim_mismatch_total").inc(collection=self.collection)
                self._log.error(
                    "[DIM-MISMATCH] collection=%s 现有 dim=%d，当前嵌入模型 %s 实测 dim=%d",
                    self.collection, self._remote_dim, self.embedding_model or "(未指定)",
                    self._dim)
                if not self.recreate_on_dim_mismatch:
                    raise VectorDimMismatchError(
                        build_dim_mismatch_message(
                            expected=self._dim, actual=self._remote_dim,
                            collection=self.collection,
                            model=self.embedding_model or "unknown-embedding-model",
                            recreate_enabled=self.recreate_on_dim_mismatch),
                        expected=self._dim, actual=self._remote_dim,
                        collection=self.collection, model=self.embedding_model)
                self._log.warning(
                    "MILVUS_RECREATE_ON_DIM_MISMATCH=true：即将 DROP 并用 dim=%d 重建 "
                    "collection %s —— 该 collection 内原有向量将永久丢失（仅限本项目专用 "
                    "collection 才可接受）", self._dim, self.collection)
                client.drop_collection(self.collection, timeout=self.timeout)
                exists = False
            else:
                self._description = self._read_description(client, self.collection)
                self._write_manifest(created=False)
                self._log.info(
                    "复用已存在的 collection %s（dim=%d metric=%s description=%s）",
                    self.collection, self._remote_dim, self._index_metric(client),
                    self._description or "(空)")
                return

        if not exists:
            if not create:
                raise MilvusError(f"collection {self.collection} 不存在且 create=False")
            schema = self._build_schema(DataType)
            self._description = self._build_description()
            try:
                client.create_collection(
                    collection_name=self.collection,
                    schema=schema,
                    timeout=self.timeout,
                    consistency_level=self.consistency_level,
                )
                self._created = True
                self._remote_dim = self._dim
                self._log.info("已创建 collection %s（dim=%d，schema=%s）",
                               self.collection, self._dim, [f.name for f in schema.fields])
            except Exception as exc:  # noqa: BLE001
                raise MilvusError(
                    f"创建 collection 失败（uri={self.uri}, collection={self.collection}, "
                    f"dim={self._dim}）：{type(exc).__name__}: {exc}\n"
                    f"提示：若报「collection already exists」，可能是并发建表，重试一次即可；\n"
                    f"      若报权限/配额错误，请确认 Milvus 后端可用磁盘空间。"
                ) from exc
            self._write_manifest(created=True)
        self._ensure_index()

    def _build_schema(self, DataType):
        """构造显式 schema（不用 dynamic field，字段全部落成真列，便于 where 过滤）。"""
        from pymilvus import CollectionSchema, FieldSchema

        fields = [
            FieldSchema(
                name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False,
                max_length=VARCHAR_MAX_LENGTH["id"],
                description="分块主键（由 doc_id/父块/序号/正文派生，天然幂等）",
            ),
            FieldSchema(
                name="vector", dtype=DataType.FLOAT_VECTOR, dim=self._dim,
                description=f"{self.embedding_model or 'embedding'} 输出的 {self._dim} 维向量",
            ),
        ]
        for name in SCALAR_FIELDS:
            if name == "chunk_index":
                fields.append(FieldSchema(name="chunk_index", dtype=DataType.INT64,
                                          description="块在该父块内的序号"))
                continue
            if name == "is_parent":
                fields.append(FieldSchema(name="is_parent", dtype=DataType.BOOL,
                                          description="是否父块（保上下文用）"))
                continue
            if name in ("created_at", "updated_at"):
                fields.append(FieldSchema(name=name, dtype=DataType.DOUBLE,
                                          description="Unix 时间戳（秒）"))
                continue
            fields.append(FieldSchema(
                name=name, dtype=DataType.VARCHAR,
                max_length=VARCHAR_MAX_LENGTH.get(name, 1024),
                description="",
            ))
        return CollectionSchema(
            fields=fields,
            description=self._build_description(),
            enable_dynamic_field=False,
        )

    def _build_description(self) -> str:
        """把「模型 + 维度 + 度量」写进 collection description（元数据可查）。"""
        return (
            f"legal_rag.chunk_schema={CHUNK_SCHEMA_VERSION}; "
            f"embedding.model={self.embedding_model or 'unknown'}; "
            f"embedding.dim={self._dim}; "
            f"metric_type={self.metric_type}; "
            f"created_by=legal_rag.t2"
        )

    # ------------------------------------------------------------------
    # 远端元数据读取
    # ------------------------------------------------------------------
    def _read_remote_dim(self, client) -> int:
        try:
            info = client.describe_collection(self.collection, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001
            raise MilvusError(
                f"describe_collection 失败（collection={self.collection}）："
                f"{type(exc).__name__}: {exc}") from exc
        for field in info.get("fields", []):
            if field.get("name") == "vector":
                try:
                    return int((field.get("params") or {}).get("dim", 0))
                except (TypeError, ValueError):
                    return 0
        return 0

    @staticmethod
    def _read_description(client_or_info, collection: str = "") -> str:
        """读取 collection description（client 必须同时给 collection 名）。"""
        try:
            if hasattr(client_or_info, "describe_collection"):
                if not collection:
                    return ""
                info = client_or_info.describe_collection(collection)
            else:
                info = client_or_info
        except Exception:  # noqa: BLE001
            return ""
        return str(info.get("description") or "")

    def _index_metric(self, client) -> str:
        try:
            names = client.list_indexes(self.collection, timeout=self.timeout)
            if not names:
                return ""
            described = client.describe_index(self.collection, names[0], timeout=self.timeout)
            return str(described.get("metric_type") or "")
        except Exception:  # noqa: BLE001 - 仅用于日志
            return ""

    def _ensure_index(self) -> None:
        from pymilvus.milvus_client.index import IndexParams  # type: ignore

        client = self._client()
        try:
            existing = client.list_indexes(self.collection, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001
            raise MilvusError(
                f"list_indexes 失败（collection={self.collection}）："
                f"{type(exc).__name__}: {exc}") from exc
        if existing:
            self._log.info("collection %s 已有索引 %s（metric=%s），跳过创建",
                           self.collection, existing, self._index_metric(client))
            return

        params = IndexParams()
        extra: dict[str, Any] = {"metric_type": self.metric_type}
        if self.index_type.upper() == "HNSW":
            extra.update({"M": self.index_m, "efConstruction": self.index_ef_construction})
        elif self.index_type.upper() in ("IVF_FLAT", "IVF_SQ8"):
            extra.update({"nlist": 1024})
        params.add_index(
            field_name="vector",
            index_type=self.index_type,
            index_name="vector",
            **extra,
        )
        try:
            client.create_index(self.collection, params, timeout=self.timeout)
            self._log.info("已创建索引 %s（field=vector metric=%s params=%s）",
                           self.index_type, self.metric_type, extra)
        except Exception as exc:  # noqa: BLE001
            raise MilvusError(
                f"创建索引失败（collection={self.collection}, index_type={self.index_type}, "
                f"metric={self.metric_type}, params={extra}）：{type(exc).__name__}: {exc}"
            ) from exc
        self._load_collection(client)

    def _load_collection(self, client) -> None:
        """把 collection 载入内存（查询/检索前必须 load；幂等）。"""
        try:
            client.load_collection(self.collection, timeout=self.timeout)
        except TypeError:  # pragma: no cover - 老版本签名差异
            client.load_collection(self.collection)
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if "already loaded" in message.lower():
                return
            raise MilvusError(
                f"加载 collection 失败（collection={self.collection}）："
                f"{type(exc).__name__}: {exc}") from exc

    # ------------------------------------------------------------------
    # manifest（模型 + 维度本地留痕，/health 可读）
    # ------------------------------------------------------------------
    def _manifest_payload(self, *, created: bool) -> dict[str, Any]:
        return {
            "store": self.name,
            "uri": self.uri,
            "db_name": self.db_name,
            "collection": self.collection,
            "embedding_model": self.embedding_model,
            "dim": self._dim,
            "metric_type": self.metric_type,
            "index_type": self.index_type,
            "chunk_schema_version": CHUNK_SCHEMA_VERSION,
            "description": self._description,
            "schema_fields": ["id", "vector", *SCALAR_FIELDS],
            "created": created,
            "updated_at": time.time(),
        }

    def _write_manifest(self, *, created: bool) -> None:
        payload = self._manifest_payload(created=created)
        self._manifest = payload
        if self.manifest_path is None:
            return
        # ---- t72 护栏：默认 manifest_dir 指向交付目录时**拒绝改写交付物** ----
        if self._is_delivery_manifest() and not self._allow_delivery_manifest:
            M.counter("manifest_write_rejected_total").inc(reason="delivery_manifest",
                                                           collection=self.collection)
            self.manifest_write_blocked = True
            message = (
                f"[MANIFEST-GUARD] 拒绝把 Milvus manifest 写到**交付文件** {self.manifest_path}"
                f"（默认 manifest_dir 指向工作区交付目录时不得改写交付物；本次 collection="
                f"{self.collection} dim={self._dim} created={created}）。"
                f" 正确做法：探针/测试显式传 manifest_dir=tmp_path（或把 cfg.index_dir 指到"
                f"临时目录）；生产路径请用 build_store()（内部显式放行）")
            if _manifest_guard_mode() == "raise":
                self._log.error(message)
                raise RuntimeError(message)
            self._log.error("%s —— 本次**未写入**，交付文件保持原样（LEGAL_RAG_MANIFEST_GUARD=raise"
                            " 可改为硬失败）", message)
            return
        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            same = False
            if self.manifest_path.is_file():
                try:
                    same = json.loads(self.manifest_path.read_text(encoding="utf-8")) == payload
                except (OSError, json.JSONDecodeError):
                    same = False
            if same:
                return
            self.manifest_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self._log.info("Milvus 元数据 manifest 已写入 %s（model=%s dim=%d collection=%s）",
                           self.manifest_path, self.embedding_model, self._dim, self.collection)
        except OSError as exc:  # 只读环境不阻塞业务
            self._log.warning("写入 Milvus manifest 失败（不影响业务）：%s", exc)

    def _is_delivery_manifest(self) -> bool:
        """本次要写的 manifest 是不是工作区**交付清单**（按解析后的真实路径比）。"""
        if self.manifest_path is None:
            return False
        try:
            return Path(self.manifest_path).resolve() == self._delivery_manifest.resolve()
        except OSError:  # pragma: no cover - 路径不可解析时按"不是交付件"处理
            return False

    def manifest(self) -> dict:
        """当前元数据（内存优先；启动早于连接时读本地文件）。"""
        if self._manifest:
            return dict(self._manifest)
        if self.manifest_path and self.manifest_path.is_file():
            try:
                return json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                self._log.warning("读取 Milvus manifest 失败：%s", exc)
        return {}

    # ------------------------------------------------------------------
    # 维度校验
    # ------------------------------------------------------------------
    def set_dim(self, dim: int, *, model: str = "") -> int:
        """由调用方（或测试）设置实测维度；与远端不一致时立即报错。"""
        dim = int(dim)
        if dim <= 0:
            raise ValueError(f"向量维度必须为正整数，收到 {dim}")
        if self._ready and self._remote_dim and self._remote_dim != dim:
            M.counter("dim_mismatch_total").inc(collection=self.collection)
            raise VectorDimMismatchError(
                build_dim_mismatch_message(
                    expected=dim, actual=self._remote_dim, collection=self.collection,
                    model=model or self.embedding_model or "unknown",
                    recreate_enabled=self.recreate_on_dim_mismatch),
                expected=dim, actual=self._remote_dim, collection=self.collection,
                model=model or self.embedding_model)
        self._dim = dim
        if model:
            self.embedding_model = model
        return self._dim

    def _validate_vectors(self, chunks: Sequence[Chunk]) -> int:
        """写入前校验维度：任何一条不等就抛（不静默、不截断）。"""
        for index, chunk in enumerate(chunks):
            vector = chunk.vector
            if not vector:
                raise ValueError(
                    f"chunk {chunk.id}（第 {index} 条）没有向量，拒绝写入 Milvus。\n"
                    f"提示：入库前必须先用 embedder 生成向量（pipeline 已自动完成）。")
            if len(vector) != self._dim:
                M.counter("dim_mismatch_total").inc(collection=self.collection)
                raise VectorDimMismatchError(
                    f"第 {index} 条 chunk（id={chunk.id}）向量维度 {len(vector)} "
                    f"与 collection 期望的 {self._dim} 不一致（model={self.embedding_model}，"
                    f"collection={self.collection}）。\n"
                    f"说明：同一次入库的所有向量必须来自同一个嵌入模型。",
                    expected=self._dim, actual=len(vector),
                    collection=self.collection, model=self.embedding_model)
        return self._dim

    # ------------------------------------------------------------------
    # where / filter 表达式
    # ------------------------------------------------------------------
    @staticmethod
    def _quote(value: str) -> str:
        return str(value).replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @classmethod
    def build_filter(cls, where: dict | None) -> str:
        """把 ``{"is_parent": False, "role_id": ["a", "b"]}`` 转成 Milvus 表达式。

        * 标量：``==``；布尔写成 ``true``/``false``；
        * 容器（list/tuple/set）：``in [...]`` —— 与内存实现的「命中即通过」语义一致；
        * ``None``：``is null``；
        * 空 dict / None：不过滤。
        """
        if not where:
            return ""
        clauses: list[str] = []
        for key, expected in where.items():
            if expected is None:
                clauses.append(f'{key} is null')
                continue
            if isinstance(expected, (list, tuple, set, frozenset)):
                values = list(expected)
                if not values:
                    clauses.append("false")           # 空集合 => 永不匹配
                    continue
                if all(cls._is_number(v) for v in values):
                    rendered = ", ".join(str(v) for v in values)
                elif all(isinstance(v, bool) for v in values):
                    rendered = ", ".join("true" if v else "false" for v in values)
                else:
                    rendered = ", ".join(f'"{cls._quote(v)}"' for v in values)
                clauses.append(f"{key} in [{rendered}]")
                continue
            if isinstance(expected, bool):
                clauses.append(f"{key} == {'true' if expected else 'false'}")
            elif cls._is_number(expected):
                clauses.append(f"{key} == {expected}")
            else:
                clauses.append(f'{key} == "{cls._quote(expected)}"')
        return " and ".join(clauses)

    # ------------------------------------------------------------------
    # gRPC 调用包装（指标 + 日志 + 异常上下文）
    # ------------------------------------------------------------------
    def _call(self, op: str, func, *args, **kwargs):
        started = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - 带上下文继续抛，不吞
            self._log.error(
                "[ERR] milvus.%s 失败（uri=%s, collection=%s, dim=%s）：%s: %s",
                op, self.uri, self.collection, self._dim, type(exc).__name__, exc)
            raise MilvusError(
                f"milvus.{op} 失败（uri={self.uri}, collection={self.collection}, "
                f"dim={self._dim}）：{type(exc).__name__}: {exc}") from exc
        finally:
            M.observe("milvus_seconds", time.perf_counter() - started, op=op)
        return result

    # ------------------------------------------------------------------
    # VectorStore 接口
    # ------------------------------------------------------------------
    @traced(operation="milvus.upsert", level=logging.DEBUG, log_result=False)
    def upsert(self, chunks: list[Chunk]) -> int:
        """幂等写入：主键是内容派生的稳定 id，重复 upsert 是覆盖语义。"""
        items = list(chunks or [])
        if not items:
            return 0
        self.connect()
        dim = self._validate_vectors(items)
        rows = [self._row(chunk) for chunk in items]
        logger.info("[BEFORE] Milvus upsert collection=%s 条数=%d dim=%d",
                    self.collection, len(rows), dim)
        with wrap_errors("milvus.upsert", collection=self.collection,
                         uri=self.uri, count=len(rows), dim=dim):
            with log_io(f"chunks[{len(rows)}]", f"milvus:{self.collection}",
                        operation="upsert", count=len(rows),
                        meta={"dim": dim, "collection": self.collection}) as io:
                client = self._client()
                self._call("upsert", client.upsert, self.collection, rows,
                           timeout=self.timeout)
                io["meta"]["ids"] = len(rows)
        logger.info("[AFTER] Milvus upsert 完成 collection=%s 条数=%d",
                    self.collection, len(rows))
        return len(rows)

    @staticmethod
    def _clip_varchar(field: str, value: str) -> str:
        """按 Milvus 的 varchar 上限裁剪 —— **上限是按 UTF-8 字节算的**。

        实测依据（2026-09-17）：``text`` 上限 8192 **字节**，一个 3,394 汉字的块被报
        ``length: 10182``（=3394×3）并导致**整篇文档 upsert 失败**；``doc_id`` 同理（256 字节）。
        这里做兜底，是为了让"一个超长字段"不再升级成"一篇文档不入库"；裁剪会打 WARNING
        并计数（``milvus_varchar_clipped_total``），**不静默**。
        """
        limit = VARCHAR_MAX_LENGTH.get(field)
        if limit is None:
            return value
        raw = value.encode("utf-8")
        if len(raw) <= limit:
            return value
        M.counter("milvus_varchar_clipped_total").inc(field=field)
        logger.warning("Milvus VARCHAR 上限裁剪：字段=%s 原字节=%d 上限=%d（上限按 UTF-8 字节算）",
                       field, len(raw), limit)
        return raw[:limit].decode("utf-8", errors="ignore")

    def _row(self, chunk: Chunk) -> dict[str, Any]:
        return {
            "id": self._clip_varchar("id", str(chunk.id)),
            "vector": [float(v) for v in (chunk.vector or [])],
            "text": self._clip_varchar("text", str(chunk.text or "")),
            "source": self._clip_varchar("source", str(chunk.source or "")),
            "doc_id": self._clip_varchar("doc_id", str(chunk.doc_id or "")),
            "chunk_index": int(chunk.chunk_index or 0),
            "parent_id": self._clip_varchar("parent_id", str(chunk.parent_id or "")),
            "is_parent": bool(chunk.is_parent),
            "role_id": self._clip_varchar("role_id", str(chunk.role_id or "")),
            "summary": self._clip_varchar("summary", str(chunk.summary or "")),
            "created_at": float(chunk.created_at or 0.0),
            "updated_at": float(chunk.updated_at or 0.0),
        }

    @staticmethod
    def _to_chunk(row: dict[str, Any], *, with_vector: bool = True) -> Chunk:
        parent = row.get("parent_id") or None
        return Chunk(
            id=str(row.get("id", "")),
            text=str(row.get("text", "") or ""),
            source=str(row.get("source", "") or ""),
            doc_id=str(row.get("doc_id", "") or ""),
            chunk_index=int(row.get("chunk_index", 0) or 0),
            parent_id=parent,
            is_parent=bool(row.get("is_parent", False)),
            role_id=str(row.get("role_id", "") or ""),
            summary=str(row.get("summary", "") or ""),
            created_at=float(row.get("created_at", 0.0) or 0.0),
            updated_at=float(row.get("updated_at", 0.0) or 0.0),
            vector=([float(v) for v in row["vector"]] if with_vector and row.get("vector")
                    else None),
        )

    @staticmethod
    def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
        """把 search 结果里的 ``{'id':.., 'distance':.., 'entity': {...}}`` 摊平。

        不同 pymilvus 版本的 search 返回结构不一样：旧版把标量字段直接挂在顶层，
        新版塞进 ``entity``。这里两种都兼容，避免「明明有数据却读出空 chunk」。
        """
        entity = row.get("entity")
        if isinstance(entity, dict):
            merged = dict(entity)
            for key, value in row.items():
                if key != "entity":
                    merged.setdefault(key, value)
            return merged
        return row

    @staticmethod
    def _is_requery_inconsistency(exc: BaseException) -> bool:
        """是不是 Milvus「字段回查结果不全」这一类服务端不一致（``code=2200``）。

        判定看异常链上的 ``code`` 与原始文本（``inconsistent requery result`` /
        ``incomplete query result``），**不看具体 id** —— 每次查询命中的 id 都不同，
        按 id 匹配等于把判定绑死在某次实验上。
        """
        current: BaseException | None = exc
        for _ in range(8):
            if current is None:
                return False
            if getattr(current, "code", None) == 2200:
                return True
            text = str(current)
            if "inconsistent requery result" in text or "incomplete query result" in text:
                return True
            current = current.__cause__ or current.__context__
        return False

    # P0.5：向量检索是每次问答都要走的 gRPC 往返，阈值 300ms。
    @timed(operation="milvus.search", slow_ms=300.0)
    def search(self, vector: list[float], top_k: int = 10,
               where: dict | None = None) -> list[SearchHit]:
        """向量检索 top-k；``where`` 支持标量等值与 ``in`` 列表过滤。

        ⚠️ 已知服务端问题与这里的处置（t8 诊断，2026-09-16；证据见
        :data:`SEARCH_OUTPUT_FIELDS`）：Milvus 的「字段回查」在结果 ≥3 条且请求
        ``vector`` 字段时会返回不全并报 ``code=2200 inconsistent requery result``。
        因此这里：

        ① 不请求 ``vector``（检索链路不需要命中块的向量）；
        ② 万一仍撞上该类不一致，退到「两步取回」（先只要 id -> 再按 id 取字段），
           个别按 id 也读不回来的命中**记 WARNING + 计数后丢弃** ——
           宁可少两条候选，也不让整条向量召回通道静默死掉。
        """
        if not vector:
            raise ValueError("search 需要非空查询向量")
        self.connect()
        if len(vector) != self._dim:
            M.counter("dim_mismatch_total").inc(collection=self.collection)
            raise VectorDimMismatchError(
                f"查询向量维度 {len(vector)} 与 collection「{self.collection}」期望的 "
                f"{self._dim} 维不一致（model={self.embedding_model}）。\n"
                f"提示：查询向量必须由「同一个嵌入模型」生成。",
                expected=self._dim, actual=len(vector), collection=self.collection,
                model=self.embedding_model)

        expression = self.build_filter(where)
        search_params = {"metric_type": self.metric_type,
                         "params": {"ef": self.index_ef_search}}
        # 检索侧「最新代」过滤（t52）：过滤发生在**排序之后**，若只取 top_k 条再过滤，
        # 被丢掉的旧代块会把名次让空（召回变少）。所以这里按倍数**多取候选**，
        # 过滤后再截回 top_k —— 最新代块因此不会被旧代块挤掉。
        fetch_k = int(top_k)
        if self._gen_filter_enabled:
            fetch_k = max(int(top_k), min(int(top_k) * GENERATION_FILTER_OVERFETCH,
                                          GENERATION_FILTER_MAX_OVERFETCH))
        logger.debug("[BEFORE] Milvus search collection=%s top_k=%d(fetch=%d) filter=%s",
                     self.collection, top_k, fetch_k, expression or "(无)")
        started = time.perf_counter()
        try:
            raw = self._search_raw(vector, top_k=fetch_k, expression=expression,
                                   search_params=search_params)
        except Exception as exc:  # noqa: BLE001 - 只接「回查不一致」，其余原样抛
            if not self._is_requery_inconsistency(exc):
                raise
            M.counter("milvus_requery_fallback_total").inc(collection=self.collection)
            logger.warning(
                "[FALLBACK] milvus.search 的字段回查不一致（%s）；改用两步取回"
                "（search 只要 id -> 按 id 取字段）：collection=%s top_k=%d filter=%s",
                exc, self.collection, fetch_k, expression or "(无)")
            raw = self._search_two_step(vector, top_k=fetch_k, expression=expression,
                                        search_params=search_params)
        elapsed = time.perf_counter() - started
        rows = (raw[0] if raw else []) or []
        hits: list[SearchHit] = []
        for row in rows:
            flat = self._flatten_row(row)
            score = float(flat.get("distance", flat.get("score", 0.0)) or 0.0)
            hits.append(SearchHit(chunk=self._to_chunk(flat), score=score,
                                  vector_score=score))
        before_filter = len(hits)
        if self._gen_filter_enabled:
            hits = [hit for hit in hits if self.is_latest_generation(hit.chunk)]
            if len(hits) != before_filter:
                self._gen_dropped_total += before_filter - len(hits)
                M.counter("milvus_generation_filter_dropped_total").inc(
                    before_filter - len(hits), collection=self.collection)
                logger.info(
                    "[GEN-FILTER] search 过滤掉 %d 条非最新代命中（%d → %d，"
                    "collection=%s key_field=%s）",
                    before_filter - len(hits), before_filter, len(hits), self.collection,
                    self._gen_filter_mod.key_field() if self._gen_filter_mod else "")
            hits = hits[:int(top_k)]
        M.observe("vector_hits", float(len(hits)), collection=self.collection)
        if hits:
            M.observe("top1_score", hits[0].score, collection=self.collection)
        logger.info("[AFTER] Milvus search 命中 %d 条，耗时 %.1fms，top1=%.4f，filter=%s",
                    len(hits), elapsed * 1000.0, hits[0].score if hits else 0.0,
                    expression or "(无)")
        return hits

    def _search_raw(self, vector: list[float], *, top_k: int, expression: str,
                    search_params: dict):
        """常规路径：一次 search，字段回查由服务端完成。"""
        with wrap_errors("milvus.search", collection=self.collection, top_k=top_k,
                         where=expression, dim=len(vector)):
            client = self._client()
            return self._call(
                "search", client.search,
                collection_name=self.collection,
                data=[list(vector)],
                filter=expression,
                limit=int(top_k),
                output_fields=list(SEARCH_OUTPUT_FIELDS),
                search_params=search_params,
                timeout=self.timeout,
            )

    def _search_two_step(self, vector: list[float], *, top_k: int, expression: str,
                         search_params: dict):
        """兜底路径：先只取 id（不回查字段），再按 id 把字段取回来。

        返回结构与 ``client.search`` 对齐（``[[{...}]]``），好让主流程复用同一段
        建 :class:`SearchHit` 的代码。
        """
        client = self._client()
        raw = self._call("search", client.search, collection_name=self.collection,
                         data=[list(vector)], filter=expression, limit=int(top_k),
                         output_fields=["id"], search_params=search_params,
                         timeout=self.timeout)
        ranked = [self._flatten_row(row) for row in ((raw[0] if raw else []) or [])]
        ids = [str(row.get("id")) for row in ranked if row.get("id")]
        fetched = self._fetch_rows_by_ids(ids)
        out: list[dict] = []
        for row in ranked:
            chunk_id = str(row.get("id") or "")
            entity = fetched.get(chunk_id)
            if entity is None:
                continue
            merged = dict(entity)
            merged["id"] = chunk_id
            merged["distance"] = row.get("distance", row.get("score", 0.0))
            out.append(merged)
        logger.warning(
            "[FALLBACK] 两步取回完成：search 拿到 %d 个 id -> 返回 %d 条命中"
            "（丢弃 %d 条）；结论按「已降级为两步取回」读",
            len(ids), len(out), len(ids) - len(out))
        return [out]

    def _fetch_rows_by_ids(self, ids: list[str]) -> dict[str, dict]:
        """按 id 取字段：先批量（``id in [...]``），批量里没有的逐个补。

        仍读不回来的命中被**丢弃**，并留下 WARNING + ``milvus_requery_missing_total``
        计数 —— 「服务端回查不全」在日志与 /metrics 里都可定位、可告警。
        """
        out: dict[str, dict] = {}
        if not ids:
            return out
        client = self._client()
        expression = "id in [" + ", ".join(f'"{self._quote(i)}"' for i in ids) + "]"
        try:
            rows = self._call("query", client.query, self.collection, filter=expression,
                              output_fields=list(SEARCH_OUTPUT_FIELDS), limit=len(ids),
                              timeout=self.timeout)
            for row in rows or []:
                out[str(row.get("id") or "")] = dict(row)
        except Exception as exc:  # noqa: BLE001 - 批量失败就全靠逐个补
            logger.warning("两步取回：批量按 id 取字段失败（%s），改为逐个补", exc)
        for chunk_id in [i for i in ids if i not in out]:
            try:
                got = self._call("query", client.query, self.collection,
                                 filter=f'id == "{self._quote(chunk_id)}"',
                                 output_fields=list(SEARCH_OUTPUT_FIELDS), limit=1,
                                 timeout=self.timeout)
            except Exception as exc:  # noqa: BLE001 - 单条失败只丢这一条
                logger.warning("两步取回：单条取回失败 id=%s（%s）", chunk_id[:12], exc)
                continue
            if got:
                out[str(got[0].get("id") or chunk_id)] = dict(got[0])
        still_missing = [i for i in ids if i not in out]
        if still_missing:
            M.counter("milvus_requery_missing_total").inc(
                len(still_missing), collection=self.collection)
            logger.warning(
                "两步取回：%d/%d 条命中按 id 也读不回来，已丢弃（前 3 个 id：%s）"
                "—— 这是服务端分片/回查不一致的痕迹，不是本项目数据缺失",
                len(still_missing), len(ids), [i[:12] for i in still_missing[:3]])
        return out

    def get(self, chunk_id: str) -> Chunk | None:
        """按 id 取单条 chunk。

        ⚠️ 真栈实测（t8，2026-09-16，Milvus v2.5.1）：``client.get(ids=[...])`` 走的是
        「按主键批量取」这条路，和 search 的字段回查同源 —— 在同一批数据上，**约 17%
        的 parent id 用 get() 取不到，而 ``query(id == X)`` 能取到**（30 次里 5 次）。
        父块上下文回填就挂在这条路径上，所以这里补一层兜底：get() 空 -> 退回
        ``query(id == X)``；仍然取不到才如实返回 None（并留 DEBUG 痕迹，不静默）。
        """
        self.connect()
        client = self._client()
        wanted = str(chunk_id)
        try:
            rows = self._call("get", client.get, self.collection,
                              ids=[wanted], output_fields=list(OUTPUT_FIELDS),
                              timeout=self.timeout)
        except MilvusError as exc:
            logger.error("读取 chunk 失败 id=%s：%s", chunk_id, exc)
            raise
        if not rows:
            rows = self._query_by_id(client, wanted)
            if rows:
                # 兜底命中：计数 + WARNING（默认级别可见），不再只用 DEBUG
                M.counter("milvus_get_fallback_total").inc(collection=self.collection)
                logger.warning(
                    "get() 没取到 id=%s，已用 query(id ==) 兜底取到（服务端「按主键取行」"
                    "会间歇漏行，本次已补偿）", wanted)
        if not rows:
            # 最终取不到：这是调用方真的拿不到数据，必须默认可见 + 计数
            M.counter("milvus_get_missing_total").inc(collection=self.collection)
            logger.warning(
                "id=%s 在 collection %s 里**确实取不到**（get 与 query(id ==) 都为空）——"
                "调用方将拿到 None；若这是父块回填，该条命中会少掉父块上下文",
                wanted, self.collection)
            return None
        return self._to_chunk(rows[0])

    def _query_by_id(self, client, chunk_id: str) -> list[dict]:
        """``query(id == X)`` 兜底（get 与 search 回查都可能漏行，这条不漏）。"""
        try:
            return list(self._call(
                "query", client.query, self.collection,
                filter=f'id == "{self._quote(chunk_id)}"',
                output_fields=list(OUTPUT_FIELDS), limit=1, timeout=self.timeout) or [])
        except Exception as exc:  # noqa: BLE001 - 兜底失败就当取不到，别把上层带崩
            logger.warning("query(id == %s) 兜底也失败：%s", chunk_id[:12], exc)
            return []

    def scan_rows(self, *, fields: Sequence[str] | None = None, expression: str = "",
                  page: int = 1000, max_rows: int = 0,
                  dedupe: bool = True) -> list[dict]:
        """**只读**扫描标量字段，返回原始行 dict（给运维/盘点用；**不做任何写入**）。

        为什么单独开一个方法而不复用 :meth:`all_chunks`：盘点只要
        ``id/source/doc_id/created_at/is_parent/role_id``（见 :data:`INVENTORY_FIELDS`），
        不需要正文与向量 —— ``all_chunks`` 会把每条 ``text``（旧策略产物动辄数千字）
        一起搬回来，一次全库盘点会白白重好几倍。

        * ``expression`` 为空表示全表；``max_rows > 0`` 时达到上限即停（抽样用）；
        * ``dedupe=True``（默认）：同一 ``id`` 重复出现只留一行 —— 盘点口径下
          "重复行"是服务端少行/重复返回的痕迹，**去重后的条数会与 ``count()``
          一起打印**，调用方据此对账（绝不拿去重后的数字冒充总量）。
        * 只读保证：本方法只调 ``query``，**没有** delete/upsert/compact/flush。

        ⚠️ 分页必须优先走 ``query_iterator``：Milvus 的 ``query`` 有**查询窗口上限**
        （实测 v2.5.1：``offset+limit`` 必须落在 ``[1, 16384]``，而本库活行数已超过它，
        用 limit/offset 翻到第 17 页会被服务端直接拒绝）。回退路径也把窗口算进去，
        并在窗口用尽时 **WARNING 说明只拿到部分行**（绝不假装拿全了）。
        """
        self.connect()
        client = self._client()
        wanted = list(fields or INVENTORY_FIELDS)
        rows: list[dict] = []
        seen: set[str] = set()

        def absorb(batch) -> int:
            added = 0
            for row in batch:
                key = str(row.get("id") or f"__no_id_{len(rows)}_{added}")
                if dedupe and key in seen:
                    continue
                seen.add(key)
                rows.append(dict(row))
                added += 1
            return added

        iterator_factory = getattr(client, "query_iterator", None)
        if callable(iterator_factory):
            try:
                iterator = self._call("query", iterator_factory, self.collection,
                                      batch_size=page, limit=-1, filter=expression,
                                      output_fields=wanted)
                stall = 0
                while True:
                    try:
                        batch = iterator.next()
                    except StopIteration:
                        break
                    if not batch:
                        break
                    before = len(rows)
                    absorb(batch)
                    stall = stall + 1 if len(rows) == before else 0
                    if stall >= 3:
                        logger.warning(
                            "scan_rows：query_iterator 连续 3 批没有新行（已取 %d 条，"
                            "filter=%s），停止迭代", len(rows), expression or "(无)")
                        break
                    if max_rows and len(rows) >= max_rows:
                        return rows[:max_rows]
                return rows
            except MilvusError:
                logger.warning("scan_rows：query_iterator 不可用，回退 limit/offset 分页"
                               "（受查询窗口上限约束，可能拿不全）")

        # 回退：limit/offset 分页，且**把窗口上限算进去**（offset+limit <= 16384）
        window = QUERY_WINDOW_LIMIT
        offset = 0
        while True:
            limit = min(page, window - offset)
            if limit <= 0:
                logger.warning(
                    "scan_rows：limit/offset 触及 Milvus 查询窗口上限（%d），已取 %d 条，"
                    "可能少于全量（filter=%s）—— 请优先修好 query_iterator 路径",
                    window, len(rows), expression or "(无)")
                break
            batch = self._call("query", client.query, self.collection,
                               filter=expression, output_fields=wanted,
                               limit=limit, offset=offset, timeout=self.timeout) or []
            if not batch:
                break
            absorb(batch)
            if max_rows and len(rows) >= max_rows:
                return rows[:max_rows]
            if len(batch) < limit:
                break
            offset += len(batch)
        return rows

    def all_chunks(self, where: dict | None = None, *,
                   generation_filter: bool | None = None,
                   with_vector: bool = True) -> list[Chunk]:
        """取出全部 chunk（供 BM25 建索引）。

        分页策略：优先用 ``query_iterator``（Milvus 2.4+ 原生游标，不受
        ``query`` 单次返回上限约束）；老版本回退到 ``limit/offset`` 分页。

        ⚠️ **完整性对账（t10 关闭 t9 §9 的缺口 A）**：读完后拿**权威值**
        （同 filter 口径的 ``count(*)``，见 :meth:`_authoritative_count`，
        **不写死任何数字**）与实际取到的条数对账。少于权威值时：

        1. **不再静默终止分页** —— 改走 ``limit/offset`` 分页重读补齐；
        2. 仍不足则 **WARNING（默认日志级别可见）+ ``milvus_all_chunks_short_total``
           计数**，并写明「这是服务端少行的痕迹，不是本项目数据缺失」。

        「短批」（``len(rows) < batch_size``）**不再被当作"到底了"**：继续
        ``next()`` 直到空批；只有连续 3 批拿不到任何**新行**时才跳出（防死循环）。

        **语义**：正常路径（足量）与改前**逐行 append 完全一致**（不排序、不去重、
        不截断）；只有"不足后补齐"那一趟才会按 id 去重（offset 重读可能重叠）。

        ``generation_filter``（t52，关键字专用）：``None`` = 跟随全局开关
        （``RAG_GENERATION_FILTER``，默认开）；``False`` = **本次调用不过滤**，
        取回该 filter 下的**全部行（含更早代）**。

        对账口径（t61）：与 ``count(*)`` 比对的是**过滤前**真实读到的**不同**行数
        （``read_keys``）—— 权威 count(*) 与它同属"未过滤"口径，可直接比。所以
        「最新代过滤」**不会**掩盖缺口：服务端少给行照旧走 [SHORT] 告警 + 计数 +
        分页补齐；被过滤掉的条数只作为日志里的"预期内差额"（``filtered_out``）。

        ⚠️ 需要「看到全部行（含更早代）」的维护/清点调用方，请显式传
        ``generation_filter=False``。现状核查（t52）：`legal_rag/api/documents.py` 的
        「删除文档」在 Milvus 上走 ``delete_by_doc``（**按 filter 删**，不经本方法）
        ⇒ **不受过滤影响**；`retrieve/hybrid.py`（BM25）用本方法取块，那**正是要被过滤**
        的召回通道。其它未来调用方见 `handoff/t52-LATEST-GENERATION-FILTER.md` §7。
        """
        self.connect()
        expression = self.build_filter(where)
        client = self._client()
        fields = OUTPUT_FIELDS if with_vector else LEAN_OUTPUT_FIELDS
        expected = self._authoritative_count(expression)
        gen_filter = (self._gen_filter_enabled if generation_filter is None
                      else bool(generation_filter))
        logger.info("[BEFORE] Milvus all_chunks collection=%s filter=%s（权威 count=%s）%s",
                    self.collection, expression or "(无)",
                    "未知" if expected is None else expected,
                    "，检索侧最新代过滤=开" if gen_filter else "")
        started = time.perf_counter()
        chunks: list[Chunk] = []
        seen: set[str] = set()
        # 与权威 count(*) 对账的基准 = **过滤前**真实读到的不同行数（``read_keys``）：
        # 「最新代」过滤只决定"这行要不要留给召回"，不改变"服务端有没有把行给我们"
        # ⇒ 过滤开启时**照常**对账，缺口告警不得被过滤路径短路（t61 修复）。
        read_keys: set[str] = set()
        #: 过滤**后**留下的不同行（用来给出"到底丢了几条"，按 id 去重 ⇒ 分页补齐不会重复计数）
        kept_keys: set[str] = set()

        iterator_factory = getattr(client, "query_iterator", None)
        if callable(iterator_factory):
            iterator = None
            try:
                iterator = self._call(
                    "query", iterator_factory, self.collection, batch_size=1000,
                    limit=-1, filter=expression, output_fields=list(fields))
                # QueryIterator 不是可迭代对象，要显式 next() 直到拿到空批
                stall = 0
                while True:
                    rows = iterator.next()
                    if not rows:
                        break
                    self._note_read_rows(rows, read_keys)   # 过滤前先记账（对账基准）
                    if gen_filter:
                        rows, _ = self._filter_generation_rows(rows)
                    self._note_read_rows(rows, kept_keys)   # 过滤后留下的（去重后）
                    before = len(seen)
                    self._append_all(chunks, seen, rows, with_vector=with_vector)
                    # 进展 = 出现**新 id**（或新行）。⚠️ 不能用 len(chunks)。
                    # 开启代数过滤后"整批都被过滤掉"是正常进展，若拿 chunks 长度
                    # 当进展会被误判成"没有进展"而提前停下（= 悄悄少给行）；
                    # 主路径允许同一行重复出现，故同时看 seen 与"本批非空"两者。
                    progress = len(seen) > before or bool(rows)
                    stall = 0 if progress else stall + 1
                    if stall >= 3:
                        logger.warning(
                            "query_iterator 连续 3 批没有新行（已取 %d 条，filter=%s），停止迭代",
                            len(chunks), expression or "(无)")
                        break
            except MilvusError:
                logger.warning("query_iterator 不可用，回退到 limit/offset 分页",
                               exc_info=True)
                chunks, seen = [], set()
                iterator_factory = None
            except Exception:  # noqa: BLE001 - 迭代器协议差异也回退
                logger.warning("query_iterator 迭代异常，回退到 limit/offset 分页",
                               exc_info=True)
                chunks, seen = [], set()
                iterator_factory = None
            finally:
                if iterator is not None and hasattr(iterator, "close"):
                    try:
                        iterator.close()
                    except Exception:  # noqa: BLE001 - 关闭失败无伤大雅
                        logger.debug("关闭 query_iterator 失败", exc_info=True)
        used_iterator = callable(iterator_factory)
        if not used_iterator:
            self._paged_all_chunks(client, expression, chunks, seen, read_keys=read_keys,
                                   kept_keys=kept_keys, fields=fields)

        # 缺口 = 权威 count(*) − **过滤前**读到的不同行数（与过滤开关无关 ⇒ t10 门禁不被掩盖）
        read_total = len(read_keys)
        # 丢弃数按 **id 去重**口径给（分页补齐会重读同一批行 ⇒ 不能累加原始计数）
        filtered_out = max(0, len(read_keys) - len(kept_keys))
        short = (expected - read_total) if (expected is not None and read_total < expected) else 0
        if used_iterator and short > 0:
            logger.warning(
                "[SHORT] all_chunks 经 query_iterator 只取到 %d 条（权威 count(*)=%d，filter=%s%s）"
                "—— **不静默终止**：改用 limit/offset 分页重读补齐",
                read_total, expected, expression or "(无)",
                "，其中已按「最新代」丢弃 %d 条" % filtered_out if filtered_out else "")
            self._paged_all_chunks(client, expression, chunks, seen, dedupe=True,
                                   read_keys=read_keys, kept_keys=kept_keys, fields=fields)
            read_total = len(read_keys)
            filtered_out = max(0, len(read_keys) - len(kept_keys))
            short = (expected - read_total) if read_total < expected else 0
        if gen_filter and expected is not None and short == 0:
            logger.info(
                "[GEN-FILTER] all_chunks 权威 count(*)=%d，按「最新代」过滤掉 %d 条"
                "（key_field=%s），实取 %d 条 —— 差额属**预期内**，不按服务端少行处理",
                expected, filtered_out,
                self._gen_filter_mod.key_field() if self._gen_filter_mod else "",
                len(chunks))

        got = len(chunks)
        if short > 0:
            M.counter("milvus_all_chunks_short_total").inc(short, collection=self.collection)
            logger.warning(
                "[SHORT] Milvus all_chunks 只取到 %d 条（过滤前实际读到 %d 条不同行，其中按"
                "「最新代」丢弃 %d 条），权威 count(*) 是 %d 条（filter=%s），差 %d 条；已尝试"
                "分页补齐仍不足 —— 这是服务端少行的痕迹，不是本项目数据缺失",
                got, read_total, filtered_out, expected, expression or "(无)", short)
        logger.info("[AFTER] Milvus all_chunks 取出 %d 条（filter=%s，权威 %s%s），耗时 %.1fms",
                    got, expression or "(无)",
                    "未知" if expected is None else expected,
                    "，已按最新代过滤 %d 条" % filtered_out if gen_filter else "",
                    (time.perf_counter() - started) * 1000)
        return chunks

    def _filter_generation_rows(self, rows: Sequence[Any]) -> tuple[list[Any], int]:
        """读取路径上的最新代过滤：返回 ``(保留行, 丢弃条数)``（只读，不改存储）。"""
        index = self.generation_index() if self._gen_filter_enabled else None
        if index is None:
            return list(rows), 0
        keep: list[Any] = []
        dropped = 0
        for row in rows:
            if index.is_latest(row):
                keep.append(row)
            else:
                dropped += 1
        if dropped:
            self._gen_dropped_total += dropped
        return keep, dropped

    @staticmethod
    def _append_all(chunks: list[Chunk], seen: set[str], rows,
                    *, with_vector: bool = True) -> int:
        """把 ``rows`` 全部转成 chunk 追加（**与改前逐行 append 完全一致**）。

        返回**追加条数**（正常等于 ``len(rows)``）。⚠️ 刻意**不**返回"新 id 数"
        （F-C 定根因时发现的坑）：主读取路径允许同一行重复出现（历史数据 / 后端语义
        差异），拿"新 id 数"当进展判断会把"翻到新的一页、但整页都是重复 id"
        误判成"没有进展"，从而提前 return、**悄悄少给行**。

        ``with_vector=False``（t68）：行里本来就没有向量，构造 chunk 时不要去找 ——
        关键词通道全量建索引必须走这条路，否则真实规模下 ``MemoryError``。
        """
        added = 0
        for row in rows:
            chunk = MilvusVectorStore._to_chunk(row, with_vector=with_vector)
            key = chunk.id or f"__no_id_{len(seen)}_{len(chunks)}"
            seen.add(key)
            chunks.append(chunk)
            added += 1
        return added

    @staticmethod
    def _extend_unique(chunks: list[Chunk], seen: set[str], rows,
                       *, with_vector: bool = True) -> int:
        """把 ``rows`` 里**未见过的** id 追加进 ``chunks``，返回追加条数。

        只用于「不足后的补齐重读」：offset 重读可能与首次读取重叠，按 id 去重
        才不会把同一行算两遍。正常路径不走这里，因此不改变既有行为。
        """
        added = 0
        for row in rows:
            chunk = MilvusVectorStore._to_chunk(row, with_vector=with_vector)
            key = chunk.id or f"__no_id_{len(seen)}_{added}"
            if key in seen:
                continue
            seen.add(key)
            chunks.append(chunk)
            added += 1
        return added

    @staticmethod
    def _note_read_rows(rows: Sequence[Any], keys: set[str]) -> None:
        """把读到的行标识记进 ``keys``（**按 id 去重**；供与权威 ``count(*)`` 对账）。

        调用点有两处口径：**过滤前**（``read_keys``，对账基准）与**过滤后**
        （``kept_keys``，用来算"丢了几条"）。

        键口径与 :meth:`_to_chunk` 的 ``id`` 一致；无 ``id`` 的异常行用**当前集合大小**
        兜底（每次 add 后集合即变大 ⇒ 键唯一，不会两行并成一行）。
        """
        for row in rows:
            try:
                key = str(row.get("id", ""))
            except AttributeError:          # 非 dict 的行实现：按无 id 处理
                key = ""
            keys.add(key or f"__no_id_{len(keys)}")

    def _paged_all_chunks(self, client, expression: str, chunks: list[Chunk],
                          seen: set[str], *, page: int = 1000,
                          dedupe: bool = False,
                          read_keys: set[str] | None = None,
                          kept_keys: set[str] | None = None,
                          fields: Sequence[str] | None = None) -> None:
        """``limit/offset`` 分页读取。

        * ``dedupe=False``（默认）：iterator 不可用时的**主路径**，逐行 append，
          与改前行为一致；
        * ``dedupe=True``：不足后的**补齐**路径，按 id 去重（避免重叠重复）。

        开启「最新代」过滤时，每页先过滤再追加，并把丢弃条数累加进
        ``self._paged_filtered_out``（原始的逐页计数，**未按 id 去重**；对账日志用的是
        ``all_chunks`` 里按 id 去重的口径）。

        ``read_keys`` / ``kept_keys``（t61）：传进来则分别记录**过滤前**与**过滤后**
        读到的行标识 —— 前者是 ``all_chunks`` 与权威 ``count(*)`` 对账的基准
        （**不受过滤影响**），后者用来算出"到底丢了几条（去重后）"。

        ``fields``（t68）：取字段清单，默认含 ``vector``；关键词通道传
        :data:`LEAN_OUTPUT_FIELDS`（不含向量）以避开 GB 级内存。
        """
        with_vector = "vector" in (fields or OUTPUT_FIELDS)
        offset = 0
        gen_filter = self._gen_filter_enabled
        self._paged_filtered_out = 0
        while True:
            try:
                rows = self._call(
                    "query", client.query, self.collection,
                    filter=expression, output_fields=list(fields or OUTPUT_FIELDS),
                    limit=page, offset=offset, timeout=self.timeout)
            except MilvusError:
                if offset == 0:
                    raise
                logger.warning("all_chunks 分页在第 %d 条中断，返回已取到的 %d 条",
                               offset, len(chunks))
                return
            if not rows:
                return
            if read_keys is not None:
                self._note_read_rows(rows, read_keys)
            if gen_filter:
                rows, dropped = self._filter_generation_rows(rows)
                self._paged_filtered_out += dropped
            if kept_keys is not None:
                self._note_read_rows(rows, kept_keys)
            if not rows:
                # 整页都被"最新代"过滤掉了：仍要翻页（否则会提前收工少给行）
                offset += page
                continue
            added = (self._extend_unique(chunks, seen, rows, with_vector=with_vector) if dedupe
                     else self._append_all(chunks, seen, rows, with_vector=with_vector))
            # 主路径的 added 恒等于本页行数（>0），因此只有"空页"会走到 return；
            # 补齐路径 added 可能为 0（整页都已见过）→ 那时停手是对的（再翻也是重复）。
            if len(rows) < page or added == 0:
                return
            offset += len(rows)

    def iter_all_chunks(self, where: dict | None = None, *, page: int = 1000,
                        with_vector: bool = False,
                        generation_filter: bool | None = None) -> Iterator[list[Chunk]]:
        """**流式**分页读取：每次 ``yield`` 一页 ``Chunk``（t68，关键词通道专用）。

        与 :meth:`all_chunks` 的三点差别（都是为了真实规模下不 ``MemoryError``
        也不会百秒级停顿）：

        1. **不把全量行攒在内存里**：调用方逐页消费（BM25 建索引即"边读边建"）；
        2. **默认不带向量**（``with_vector=False``）：1024 维 × 14 万行 = GB 级，
           而关键词通道只需要 ``text``；
        3. **分页优先走 ``query_iterator``**，只有它不可用时才退回 ``limit/offset`` ——
           Milvus 的 ``limit/offset`` 受**查询窗口上限**（实测 v2.5.1：``offset+limit``
           必须落在 ``[1, 16384]``）约束，本库活行数已超过它 ⇒ 翻到第 17 页会被服务端
           拒绝、**只能拿到约 1.6 万行**（t26 R1 实测 98.9s/135.9s/159.1s 就是这条路径）。
           窗口用尽时**明确 WARNING + 计数**（绝不假装拿全）。

        只读：只调 ``query``/``query_iterator``/``count``，**没有**任何写入。
        """
        self.connect()
        expression = self.build_filter(where)
        client = self._client()
        fields = OUTPUT_FIELDS if with_vector else LEAN_OUTPUT_FIELDS
        expected = self._authoritative_count(expression)
        gen_filter = (self._gen_filter_enabled if generation_filter is None
                      else bool(generation_filter))
        read_keys: set[str] = set()
        delivered = 0
        started = time.perf_counter()
        logger.info("[BM25-FEED] 流式取块开始：filter=%s 权威 count=%s 带向量=%s 页大小=%d",
                    expression or "(无)", "未知" if expected is None else expected,
                    with_vector, page)

        iterator_factory = getattr(client, "query_iterator", None)
        if callable(iterator_factory):
            iterator = None
            try:
                iterator = self._call(
                    "query", iterator_factory, self.collection, batch_size=page,
                    limit=-1, filter=expression, output_fields=list(fields))
                while True:
                    rows = iterator.next()
                    if not rows:
                        break
                    self._note_read_rows(rows, read_keys)
                    if gen_filter:
                        rows, _ = self._filter_generation_rows(rows)
                    page_chunks = [self._to_chunk(row, with_vector=with_vector) for row in rows]
                    if page_chunks:
                        delivered += len(page_chunks)
                        yield page_chunks
            except MilvusError:
                logger.warning("[BM25-FEED] query_iterator 不可用，回退 limit/offset 分页"
                               "（⚠️ 受查询窗口上限约束，可能只拿到约 1.6 万行）",
                               exc_info=True)
                iterator_factory = None
            except Exception:  # noqa: BLE001 - 迭代器协议差异也回退
                logger.warning("[BM25-FEED] query_iterator 迭代异常，回退 limit/offset 分页",
                               exc_info=True)
                iterator_factory = None
            finally:
                if iterator is not None and hasattr(iterator, "close"):
                    try:
                        iterator.close()
                    except Exception:  # noqa: BLE001
                        logger.debug("关闭 query_iterator 失败", exc_info=True)

        if not callable(iterator_factory):
            window = QUERY_WINDOW_LIMIT
            offset = 0
            truncated = ""                      # ""=拿全；query_window / paged_failed=没拿全
            while offset < window:
                limit = min(page, window - offset)
                try:
                    rows = self._call(
                        "query", client.query, self.collection, filter=expression,
                        output_fields=list(fields), limit=limit, offset=offset,
                        timeout=self.timeout)
                except MilvusError as exc:
                    truncated = "query_window" if "window" in str(exc).lower() else "paged_failed"
                    logger.warning("[BM25-FEED] limit/offset 分页在第 %d 条中断（已交付 %d 条，"
                                   "归类=%s）：%s", offset, delivered, truncated, exc)
                    break
                if not rows:
                    break
                self._note_read_rows(rows, read_keys)
                if gen_filter:
                    rows, _ = self._filter_generation_rows(rows)
                page_chunks = [self._to_chunk(row, with_vector=with_vector) for row in rows]
                if page_chunks:
                    delivered += len(page_chunks)
                    yield page_chunks
                offset += len(rows)
                if len(rows) < limit:
                    break
            else:
                truncated = "query_window"      # 窗口用尽仍有更多行
            if truncated:
                M.counter("milvus_all_chunks_short_total").inc(
                    reason=truncated, collection=self.collection)
                if truncated == "query_window":
                    logger.warning(
                        "[BM25-FEED] ⚠️ limit/offset 触及 Milvus 查询窗口上限"
                        "（offset+limit ≤ %d），已交付 %d 条：**关键词语料可能不完整**"
                        "（filter=%s）—— 正确修法是让 query_iterator 可用",
                        window, delivered, expression or "(无)")
                else:
                    logger.warning(
                        "[BM25-FEED] ⚠️ limit/offset 分页异常中断，已交付 %d 条："
                        "**关键词语料可能不完整**（filter=%s）",
                        delivered, expression or "(无)")

        read_total = len(read_keys)
        short = (expected - read_total) if (expected is not None and read_total < expected) else 0
        self.last_feed_stats = {              # 供探针/运维核对"到底喂了多少"（只读统计）
            "filter": expression or "", "read": read_total, "delivered": delivered,
            "authoritative": expected,
            "filtered_out": max(0, read_total - delivered),
            "truncated": bool(short > 0),
        }
        if short > 0:
            M.counter("milvus_all_chunks_short_total").inc(short, collection=self.collection)
            logger.warning(
                "[SHORT] 流式取块只读到 %d 条不同行，权威 count(*) 是 %d 条（filter=%s），"
                "差 %d 条 —— 关键词语料不完整（这是服务端少行的痕迹，不是本项目数据缺失）",
                read_total, expected, expression or "(无)", short)
        logger.info("[BM25-FEED] 流式取块结束：交付 %d 条 / 权威 %s / 读字段=%s 耗时 %.1fms",
                    delivered, "未知" if expected is None else expected,
                    "含向量" if with_vector else "不含向量",
                    (time.perf_counter() - started) * 1000.0)

    def _authoritative_count(self, expression: str) -> int | None:
        """filter 口径的权威条数（``count(*)``）。

        取不到时返回 ``None`` 并记 WARNING —— 调用方据此**跳过**对账，
        绝不拿一个猜的数字去比对（否则会把"查不到基准"误报成"少行"）。
        """
        try:
            rows = self._call("count", self._client().query, self.collection,
                              filter=expression, output_fields=["count(*)"],
                              timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001 - 对账基准拿不到就如实说
            logger.warning("取权威 count(*) 失败（filter=%s，%s: %s）——本次跳过完整性对账",
                           expression or "(无)", type(exc).__name__, exc)
            return None
        return int(self._count_of(rows))

    def delete(self, ids: list[str]) -> int:
        targets = [str(chunk_id) for chunk_id in (ids or []) if str(chunk_id)]
        if not targets:
            return 0
        self.connect()
        client = self._client()
        removed = 0
        with log_io(f"ids[{len(targets)}]", f"milvus:{self.collection}",
                    operation="delete", count=len(targets)):
            for start in range(0, len(targets), 500):
                batch = targets[start:start + 500]
                result = self._call("delete", client.delete, self.collection, ids=batch,
                                    timeout=self.timeout)
                removed += int((result or {}).get("delete_count", 0) or 0)
        logger.info("[AFTER] Milvus delete collection=%s 请求 %d 条，实际删除 %d 条",
                    self.collection, len(targets), removed)
        return removed

    def delete_by_doc(self, doc_id: str) -> int:
        """按文档删除（供 t4 的「删除文档」接口用）。"""
        if not doc_id:
            return 0
        self.connect()
        expression = f'doc_id == "{self._quote(doc_id)}"'
        return self._delete_by_filter(expression, operation=f"delete_by_doc({doc_id})")

    def delete_by_role(self, role_id: str) -> int:
        if role_id is None:
            return 0
        self.connect()
        expression = f'role_id == "{self._quote(role_id)}"'
        return self._delete_by_filter(expression, operation=f"delete_by_role({role_id})")

    def delete_by_source(self, source: str) -> int:
        """按来源文件名删除（增量重入常用）。"""
        if not source:
            return 0
        self.connect()
        expression = f'source == "{self._quote(source)}"'
        return self._delete_by_filter(expression, operation=f"delete_by_source({source})")

    def _delete_by_filter(self, expression: str, *, operation: str) -> int:
        client = self._client()
        matched = self._count_filtered(expression)
        logger.info("[BEFORE] Milvus %s collection=%s 匹配 %d 条",
                    operation, self.collection, matched)
        # log_io 的第 1/2 个位置参数是「来源 / 去向」，操作名必须走 operation=，
        # 否则日志会打成「delete_by_doc(x) -> milvus:... 开始」这种把操作名当来源的错位。
        with log_io(f"filter:{expression}", f"milvus:{self.collection}",
                    operation=operation, count=matched,
                    meta={"filter": expression}) as io:
            result = self._call("delete", client.delete, self.collection,
                                filter=expression, timeout=self.timeout)
            removed = int((result or {}).get("delete_count", 0) or 0)
            io["count"] = removed
        if matched != removed:
            # 「应删 N / 实删 M」对账（t10 关闭 t9 §9 缺口 C）：不一致必须默认可见
            short = matched - removed
            M.counter("milvus_delete_mismatch_total").inc(abs(short),
                                                          collection=self.collection)
            logger.warning(
                "[MISMATCH] Milvus %s：匹配 %d 条、实际删 %d 条（差 %+d），表达式=%s —— "
                "少删会留下残留分块（「删了还能检索到」），请查服务端删除/compaction 状态",
                operation, matched, removed, short, expression)
        logger.info("[AFTER] Milvus %s 删除 %d 条（表达式：%s）",
                    operation, removed, expression)
        return removed

    def _count_filtered(self, expression: str) -> int:
        client = self._client()
        try:
            rows = self._call("count", client.query, self.collection, filter=expression,
                              output_fields=["count(*)"], timeout=self.timeout)
        except MilvusError:
            return 0
        return int(self._count_of(rows))

    def count(self) -> int:
        """当前 collection 的实体总数。"""
        self.connect()
        client = self._client()
        rows = self._call("count", client.query, self.collection, filter="",
                          output_fields=["count(*)"], timeout=self.timeout)
        return int(self._count_of(rows))

    @staticmethod
    def _count_of(rows: Any) -> int:
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            for key, value in rows[0].items():
                if "count" in str(key).lower():
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return 0
        return 0

    def persist(self) -> None:
        """落盘：flush 让已写入数据持久化（Milvus 自身的持久化语义）。"""
        if not self._ready:
            return
        client = self._client()
        self._call("flush", client.flush, self.collection, timeout=self.timeout)
        logger.info("Milvus collection %s 已 flush", self.collection)

    def replace_role(self, role_id: str, chunks: list[Chunk]) -> int:
        """先删后插：按 role_id 替换该分区（沿用基类的原子更新语义）。"""
        removed = self.delete_by_role(role_id)
        logger.info("replace_role(%s)：先删 %d 条，再写入 %d 条",
                    role_id, removed, len(chunks or []))
        return self.upsert(chunks)

    # ------------------------------------------------------------------
    # 运维
    # ------------------------------------------------------------------
    def health(self, *, deep: bool = False) -> dict:
        """健康信息（t4 的 /health 直接把它并进响应）。"""
        info = self.describe()
        info["manifest"] = self.manifest()
        try:
            client = self._client()
            info["ping"] = bool(client.has_collection(self.collection, timeout=5.0))
            info["collections"] = sorted(client.list_collections())
            if deep:
                info["count"] = self.count()
        except Exception as exc:  # noqa: BLE001 - 健康检查本身不抛
            info["ping"] = False
            info["error"] = f"{type(exc).__name__}: {exc}"
        return info

    def close(self) -> None:
        with self._clients_lock:
            clients, self._clients = self._clients, []
        for client in clients:
            try:
                client.close()
            except Exception:  # pragma: no cover
                logger.debug("关闭 Milvus client 失败", exc_info=True)
        self._local = threading.local()
        self._ready = False


def build_milvus_store(config: Any, *, dim: int, collection: str = "",
                       embedding_model: str = "", **kwargs) -> MilvusVectorStore:
    """便捷工厂：按 config 造一个已连接（幂等建表）的 Milvus 向量库。

    维度必须由调用方实测后传入（例如 ``OllamaEmbedder.ensure_dim()``）。
    """
    store = MilvusVectorStore(
        config, dim=dim, collection=collection,
        embedding_model=embedding_model or
        str(getattr(getattr(config, "ollama", None), "embedding_model", "") or ""),
        **kwargs)
    store.connect()
    return store
