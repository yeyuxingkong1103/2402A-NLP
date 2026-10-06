"""Milvus 向量库封装。

MVP 使用 Milvus Lite（本地文件、免服务），生产将 MILVUS_DB_URI 换成
`http://host:19530` 即可无缝切换 Milvus 服务器。

Collection 设计（role_knowledge）：
    id(主键) / role_id / text(原文 chunk) / vector(稠密向量) /
    title / source(文档来源) / chunk_index / summary / created_at / updated_at
检索按 role_id 过滤，实现「多角色共享库、按角色隔离召回」。
"""
from __future__ import annotations

import re
import threading
import time
import uuid

from pymilvus import DataType, MilvusClient

from ..config import Settings
from ..logging_config import get_logger

log = get_logger("milvus")

# 角色 id 是标识符，只允许这些字符。它会被拼进 Milvus 过滤表达式，
# 必须白名单校验：像 `x" or role_id != "` 这样的输入可以改写过滤条件、绕过角色隔离。
ROLE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


def role_filter(role_id: str) -> str:
    """构造 role_id 过滤表达式（带白名单校验）。"""
    if not ROLE_ID_PATTERN.match(role_id or ""):
        raise ValueError(f"非法角色 id: {role_id!r}（只允许字母/数字/下划线/点/连字符）")
    return f'role_id == "{role_id}"'


def quote_literal(value: str) -> str:
    """把字符串安全地嵌进 Milvus 表达式字面量。

    这里不做转义而是直接拒绝引号/反斜杠：表达式里的这两种字符只会被用来
    闭合字符串、改写过滤条件，正常来源（角色 id、文件路径）用不到。
    """
    if '"' in value or "\\" in value:
        raise ValueError(f"表达式字面量含非法字符（引号或反斜杠）: {value!r}")
    return f'"{value}"'


class MilvusStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client: MilvusClient | None = None
        self._connected = False
        self._loaded = False
        # 连接是**懒**的：第一次用到才建，而 Lite 模式的首次连接要起那个嵌入式服务、
        # 独占数据目录的锁。同步路由跑在线程池里，两个请求同时首次访问就都会看到
        # _connected=False，各自去起一个服务，输的那个抛 DataDirLockedError——
        # 日志里那条「Milvus 不可达」+ 一大坨 traceback 就是这么来的，而它其实无害：
        # 赢的那个已经把服务起来了。串行化首次连接，别让这种事再发生。
        self._connect_lock = threading.Lock()
        # 建集合（DDL）的串行化锁，见 create_collection：connect 的锁盖不住
        # 「查存在性 -> 建」这段复合操作。
        self._ddl_lock = threading.Lock()

    # ---- 生命周期 ----
    def connect(self) -> None:
        if self._connected:
            return
        with self._connect_lock:
            if self._connected:  # 等锁期间可能已经被别的线程连上了
                return
            self.client = MilvusClient(uri=self.settings.milvus_uri)
            self._connected = True

    def ping(self) -> bool:
        try:
            self.connect()
            self.client.list_collections()
            return True
        except Exception as exc:  # noqa: BLE001
            # 本地 Lite 模式连不上，最常见的原因是数据目录的锁被别的进程占着
            # （另一个 app、ingest 或 eval）。抛出来的 message 只有一句
            # 「Open local milvus failed」，看不出这层，所以在这里点明。
            log.warning(
                "Milvus 不可达（%s）: %s —— 本地 Lite 模式多半是数据目录被另一个进程"
                "占着（app / ingest / eval 只能有一个）。跑 CLI 前先停服务。",
                self.settings.milvus_uri,
                exc,
            )
            return False

    def _ensure_loaded(self) -> None:
        """检索/查询前确保 collection 已加载到内存。

        Milvus 集合插入后需显式 load 才能 search/query；入库脚本与 app 是
        不同进程、各自起 Milvus Lite，加载状态不会跨进程保持，故在检索侧兜底。
        """
        self.connect()
        if not self.client.has_collection(self.settings.milvus_collection):
            return
        if self._loaded:
            return
        self.client.load_collection(self.settings.milvus_collection)
        self._loaded = True

    def create_collection(self) -> None:
        self.connect()
        if self.client.has_collection(self.settings.milvus_collection):
            return
        # 无锁的 check-then-act：同步路由跑在线程池里，全新库上两个请求同时首次上传会
        # 双双看到 has_collection=False，一起去建 → 输的那个收
        # `MilvusException: File exists: collections/role_knowledge` → 该请求 500、
        # chunk 一条没落库（embedding 白花）。实测 2 线程并发首次上传必挂 1 个。
        # 建集合是 DDL，只在首次发生，串行化代价可忽略。
        with self._ddl_lock:
            if self.client.has_collection(self.settings.milvus_collection):  # 等锁期间可能已建好
                return
            self._create_collection_locked()

    def _create_collection_locked(self) -> None:
        # auto_id=False：主键由应用侧生成 uuid4（见 insert），调用方必须自己拿到 id，
        # 否则插完对不上号。enable_dynamic_field=False：schema 之外的多余字段直接报错，
        # 而不是悄悄塞进动态列——入参拼错字段名时宁可在写入期炸，也不要检索期才发现是空。
        # 下面每个 VARCHAR 的 max_length 都是**硬上限**，写超长 Milvus 报错、不截断；
        # text 给到 65535 是因为 chunk_size 可配、没有上界（分块越大越容易顶到这里）。
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, max_length=64, is_primary=True)
        schema.add_field("role_id", DataType.VARCHAR, max_length=64)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("title", DataType.VARCHAR, max_length=512)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("summary", DataType.VARCHAR, max_length=2048)
        schema.add_field("created_at", DataType.INT64)
        schema.add_field("updated_at", DataType.INT64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.settings.embed_dim)

        # FLAT + IP：小规模数据免训练，最稳；生产可换 HNSW/IVF
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="FLAT", metric_type="IP")

        self.client.create_collection(
            collection_name=self.settings.milvus_collection,
            schema=schema,
            index_params=index_params,
        )
        log.info("已创建 Milvus collection: %s", self.settings.milvus_collection)

    # ---- 写入 ----
    def insert(self, role_id: str, chunks: list[dict]) -> list[str]:
        """chunks: [{text, title, source, chunk_index, summary, vector}, ...]

        返回本批新生成的向量 id（顺序与 chunks 一致）。insert 是纯追加：同一份文档
        重灌不会覆盖旧行，去重得靠调用方先 delete_by_source——漏了就会出现同一条资料
        在检索结果里占多个坑（RRF 还会因为多路命中而给它更高的融合分）。
        """
        self.connect()
        # 入库脚本可能没建过集合就直连（app 与 CLI 是两个进程、各自起 Milvus Lite），
        # 这里兜底：create_collection 内部先查存在性，已有集合时是空操作。
        self.create_collection()
        now = int(time.time())  # created_at/updated_at 都是 unix 秒（INT64），不是 datetime
        rows = []
        ids = []
        for c in chunks:
            rid = str(uuid.uuid4())
            ids.append(rid)
            rows.append(
                {
                    "id": rid,
                    "role_id": role_id,
                    "text": c["text"],
                    "title": c.get("title", ""),
                    "source": c.get("source", ""),
                    "chunk_index": int(c.get("chunk_index", 0)),
                    "summary": c.get("summary", ""),
                    "created_at": now,
                    "updated_at": now,
                    "vector": c["vector"],
                }
            )
        self.client.insert(self.settings.milvus_collection, rows)
        self._loaded = False  # 新增数据后需重新 load 才能被检索到
        return ids

    def delete_by_source(self, role_id: str, source: str) -> int:
        """删除**某角色下**某 source 的全部向量，返回删除前匹配到的条数。

        两个都必修：
        - **必须带 role_id**：多角色共享同一张向量表，只按 source 删会把别的角色
          灌过的同一份文档一起删掉（实测过：一份文件灌给 A 和 B，对 A 重灌后
          B 的向量全没了，而 B 的关系库登记还在 → 列表显示已入库、检索却全空）。
          所以这里不留"不传角色"的默认值，强制调用方说清删谁。
        - **不能用 Milvus 返回的 delete_count**：实测（Milvus Lite，2026-09-21）
          过滤删除明明生效，delete_count 却恒为 0——插入 3 条后 count() 38 → 删 → 35，
          而返回值和第二次删除一样都是 0。后果是调用方日志永远在说谎
          （「先删除旧数据 0 条」），而「日志和实际对不上」正是本项目反复踩过的坑。
        """
        self.connect()
        if not self.client.has_collection(self.settings.milvus_collection):
            return 0
        self._ensure_loaded()
        flt = f"{role_filter(role_id)} and source == {quote_literal(source)}"
        matched = 0
        try:
            res = self.client.query(
                collection_name=self.settings.milvus_collection,
                filter=flt,
                output_fields=["count(*)"],
            )
            if res:
                matched = int(res[0].get("count(*)", 0))
        except Exception as exc:  # noqa: BLE001
            # 数不出来不影响删除本身：退化成「删了但不知道几条」，返回 0
            log.warning("统计待删除条数失败（照常继续删除）: %s", exc)
        self.client.delete(self.settings.milvus_collection, filter=flt)
        return matched

    # ---- 查询 ----
    def search(self, role_id: str, query_vector: list[float], top_k: int) -> list[dict]:
        """角色内的 ANN 检索，返回按相似度降序的 top_k 条。

        `filter=role_filter(role_id)` 是多角色共享一张表时唯一的隔离手段——**不能省**，
        少了它就是把别人的角色资料喂给当前角色。相似度阈值不在这里做：本函数只管
        "取最近邻"，要不要按 score_threshold 滤掉是 HybridRetriever 的事（那边还要
        和 BM25 那一路一起看），在这里过滤会让两路的召回数不可预期。
        """
        self.connect()
        if not self.client.has_collection(self.settings.milvus_collection):
            # 静默返回空会让整条链路以为"库里就是没资料"，模型在没有资料时照常识编
            # （build_messages 只在 contexts 非空时才要求"只依据资料"）。至少要留痕。
            log.warning(
                "向量集合 %s 不存在，本次检索返回空（是还没入库，还是 MILVUS_DB_URI 指错了库？）",
                self.settings.milvus_collection,
            )
            return []
        self._ensure_loaded()
        results = self.client.search(
            collection_name=self.settings.milvus_collection,
            data=[query_vector],
            filter=role_filter(role_id),
            limit=top_k,
            output_fields=["id", "text", "title", "source", "chunk_index"],
        )
        hits = []
        for r in results[0]:
            ent = r.get("entity", {})
            hits.append(
                {
                    "id": r["id"],
                    "score": r["distance"],  # IP 度量 + 归一化向量 = 余弦相似度
                    "text": ent.get("text", ""),
                    "title": ent.get("title", ""),
                    "source": ent.get("source", ""),
                    "chunk_index": ent.get("chunk_index", 0),
                }
            )
        return hits

    # limit 默认 10000：语料是要整份拉进进程内存建 BM25 索引的，单角色知识库远小于
    # 这个量级；真超过就该把 BM25 迁进 Milvus 稀疏向量（见模块 docstring），
    # 而不是把默认值往上调——那样只是把内存和建索引耗时一起放大。
    def list_texts(self, role_id: str, limit: int = 10000) -> list[dict]:
        """拉取某角色全部 chunk 文本，供 Python 侧 BM25 建索引。

        Milvus 的 query 不保证顺序，超过 limit 时拿到的是**任意子集**：BM25 那一路
        会静默只覆盖部分语料，而稠密路不受影响，两路召回悄悄不对齐。所以撞到上限要留痕。
        """
        self.connect()
        if not self.client.has_collection(self.settings.milvus_collection):
            log.warning("向量集合 %s 不存在，BM25 语料为空（关键词一路没有召回）", self.settings.milvus_collection)
            return []
        self._ensure_loaded()
        rows = self.client.query(
            collection_name=self.settings.milvus_collection,
            filter=role_filter(role_id),
            output_fields=["id", "text", "title", "source", "chunk_index"],
            limit=limit,
        )
        if len(rows) >= limit:
            log.warning(
                "角色 %s 的 chunk 数达到 list_texts 上限 %d：BM25 只索引到其中任意 %d 条，"
                "关键词召回会漏（稠密路不受影响，两路不再对齐）",
                role_id, limit, limit,
            )
        return rows

    def count(self) -> int:
        self.connect()
        if not self.client.has_collection(self.settings.milvus_collection):
            return 0
        stats = self.client.get_collection_stats(self.settings.milvus_collection)
        return int(stats.get("row_count", 0))

    def drop_collection(self) -> None:
        """删掉整个集合——**是全部角色**，不是某一个角色的数据（reset 用，见 scripts/seed.py）。

        没有"按角色清空"的对应方法：要删单个角色得走 delete_by_source 或自行 query+delete。
        """
        self.connect()
        if self.client.has_collection(self.settings.milvus_collection):
            self.client.drop_collection(self.settings.milvus_collection)
