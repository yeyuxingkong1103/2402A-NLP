# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 工单01 - 基于PDF文档的问答系统
# 工单02 - 基于PDF文档的问答系统的优化
"""
向量库：Milvus Standalone。

【为什么是 Standalone 而不是 Milvus Lite】
1. Lite 是**嵌入式**的（uri 是本地文件路径），**不监听任何 TCP 端口**，
   因此 Attu 连不上 —— 而课程要求用 Milvus 可视化，选 Lite 等于当场违背需求。
2. Lite 是单进程 + 对数据目录加文件锁，FastAPI 起多 worker 会直接冲突，
   过不了工单01「高并发环境下稳定运行」这条验收。
Standalone 走 Docker，Attu 可连，多进程安全。

【为什么建表时就把 sparse 字段留出来】
Milvus 的 collection schema **建表即定死，事后不能加字段**
（要加只能新建 collection 再迁移）。工单06 要做混合检索，需要稀疏向量，
所以工单01 建表时就把 `sparse` 建好 —— 哪怕现在只写空值。
这样工单06 不必重新嵌入 1100 个 chunk。

【稀疏向量的权重怎么来】
不用 Milvus 原生 BM25 Function（它必须在建表时定义，且内置中文 analyzer
会把 `5,520.00`、`1-1-42` 这类数字页码切碎 —— 而招股书里这些恰恰是答案）。
改为**应用层用 jieba 分词后计算 TF 权重**写入 sparse 字段。
工单06 再把 IDF 补上并接 WeightedRanker 融合。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import jieba
from pymilvus import DataType, MilvusClient

from app.config import settings

# 数字/百分比/页码这类 token 必须整体保留 —— 招股书的核心答案大量是数字
_NUM_TOKEN = re.compile(r"\d[\d,.\-]*%?")
_CN_NUM = re.compile(r"1-1-\d+")


class VectorStoreError(RuntimeError):
    """Milvus 不可用或操作失败。"""


@dataclass
class SearchHit:
    chunk_id: int
    score: float
    content: str
    page_no: int
    page_label: str
    chunk_type: str
    section_path: str
    doc_name: str
    chunk_index: int
    # 工单02 新增。doc_id 是**邻块查询的安全前提**：不按它限定，一旦库里
    # 有第二份文档（工单03 就要加《招股说明书2.pdf》），chunk_index±1 会跨文档串块。
    doc_id: str = ""
    # 工单02 新增：该条是不是「邻块扩展」补进来的。
    # 它不参与重排、不计入检索精确率的分母（否则新机制会机械压低自己的成绩）。
    is_neighbor: bool = False
    # 补进来的邻块是在给哪个命中块做补充（记 chunk_index，便于前端标注）
    serves: int | None = None


# ----------------------------------------------------------------------
# 稀疏向量（词法权重）
# ----------------------------------------------------------------------
def lexical_sparse(text: str, max_terms: int = 256) -> dict[int, float]:
    """
    用 jieba 分词 + 词频构造稀疏向量。

    【数字保护】先把数字和页码形态的 token 用正则**整体抠出来**单独计数，
    剩下的中文再走 jieba。否则 `5,520.00` 会被切成 `5` `,` `520` `.` `00`，
    检索「注册资本是多少」时这个 token 就废了。
    索引用 32 位稳定的字符串哈希，保证同一 token 每次映射到同一维。
    """
    tokens: list[str] = []
    # 1) 先抠号码
    for m in _CN_NUM.finditer(text):
        tokens.append(m.group())
    for m in _NUM_TOKEN.finditer(text):
        tok = m.group().rstrip(".,")
        if len(tok) >= 2:
            tokens.append(tok)
    # 2) 剩余中文走 jieba（数字已抠走的位置留下标点，jieba 会自行过滤）
    for tok in jieba.cut(text):
        tok = tok.strip()
        if len(tok) >= 2 or tok.isdigit():
            tokens.append(tok)

    if not tokens:
        return {}

    counts = Counter(tokens)
    total = sum(counts.values())
    # 归一化 TF，避免长 chunk 因词数多而在 IP 内积下天然占优
    weighted = {t: c / total for t, c in counts.items()}
    top = sorted(weighted.items(), key=lambda kv: -kv[1])[:max_terms]
    return {abs(hash(t)) % (2**31 - 1): float(w) for t, w in top}


# ----------------------------------------------------------------------
class VectorStore:
    def __init__(self, client: MilvusClient | None = None,
                 collection: str | None = None) -> None:
        self.collection = collection or settings.milvus_collection
        self._client = client

    # ------------------------------------------------------------------
    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
            try:
                self._client = MilvusClient(uri=uri, timeout=10)
            except Exception as e:  # noqa: BLE001
                raise VectorStoreError(
                    f"连不上 Milvus ({uri})：{e}\n"
                    f"请先启动：docker compose -f docker-compose.milvus.yml up -d"
                ) from e
        return self._client

    def health(self) -> tuple[bool, str]:
        try:
            cols = self.client.list_collections()
            return True, f"已连接，现有 collection：{cols or '(空)'}"
        except Exception as e:  # noqa: BLE001
            return False, str(e)[:200]

    # ------------------------------------------------------------------
    def build_schema(self):
        """建表 schema。字段与工单01 验收及后续工单的扩展点一一对应。"""
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)

        schema.add_field("id", DataType.INT64, is_primary=True)
        # ---- 多租户扩展点（工单01 只用一个默认值）----
        schema.add_field("kb_id", DataType.VARCHAR, max_length=64)
        # ---- 文档溯源 ----
        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
        schema.add_field("doc_name", DataType.VARCHAR, max_length=512)
        # page_no = PDF position index（程序用）；page_label = 印刷页码（给人看）
        schema.add_field("page_no", DataType.INT32)
        schema.add_field("page_label", DataType.VARCHAR, max_length=32)
        schema.add_field("chunk_index", DataType.INT32)
        # ---- 工单02 small-to-big 的父块指针 ----
        schema.add_field("parent_id", DataType.INT64)
        # ---- 工单03/04 区分 text / table / image ----
        schema.add_field("chunk_type", DataType.VARCHAR, max_length=16)
        # ---- 层级路径「第X节 > 一、 > （一）」----
        schema.add_field("section_path", DataType.VARCHAR, max_length=512)
        # ---- 去重与版本 ----
        schema.add_field("content_hash", DataType.VARCHAR, max_length=64)
        # ---- 工单11 微调嵌入模型后，用它区分新旧向量 ----
        schema.add_field("embed_model", DataType.VARCHAR, max_length=64)

        schema.add_field("content", DataType.VARCHAR, max_length=8192)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=settings.embed_dim)
        # 工单06 混合检索用；工单01 允许写空值
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

        return schema

    def build_index_params(self):
        ip = self.client.prepare_index_params()
        ip.add_index(
            field_name="dense",
            index_type="HNSW",              # 内存索引，召回与延迟优于 IVF 系列
            metric_type="COSINE",           # bge-m3 的输出适合余弦
            params={"M": 16, "efConstruction": 200},
        )
        ip.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"drop_ratio_build": 0.2},
        )
        return ip

    # ------------------------------------------------------------------
    def create_collection(self, *, drop_existing: bool = False) -> str:
        if drop_existing and self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
        if self.client.has_collection(self.collection):
            return "exists"
        self.client.create_collection(
            collection_name=self.collection,
            schema=self.build_schema(),
            index_params=self.build_index_params(),
        )
        return "created"

    def drop_collection(self) -> None:
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)

    def count(self) -> int:
        try:
            st = self.client.get_collection_stats(self.collection)
            return int(st.get("row_count", 0))
        except Exception:  # noqa: BLE001
            return 0

    # ------------------------------------------------------------------
    def insert_chunks(self, chunks: Sequence, vectors: Sequence[Sequence[float]],
                      doc_id: str, doc_name: str,
                      kb_id: str = "default") -> int:
        """批量写入。chunk 与 vector 一一对应。"""
        if len(chunks) != len(vectors):
            raise ValueError(f"chunk 数 {len(chunks)} 与向量数 {len(vectors)} 不一致")

        from app.core.dedup import content_hash

        rows: list[dict[str, Any]] = []
        for c, v in zip(chunks, vectors):
            rows.append({
                "kb_id": kb_id,
                "doc_id": doc_id,
                "doc_name": doc_name,
                "page_no": int(c.page_no),
                "page_label": c.page_label,
                "chunk_index": int(c.chunk_index),
                "parent_id": int(getattr(c, "parent_id", 0) or 0),
                "chunk_type": c.chunk_type,
                "section_path": (c.section_path or "")[:500],
                "content_hash": content_hash(c.content),
                "embed_model": settings.embed_model,
                "content": c.content[:8000],
                "dense": list(v),
                "sparse": lexical_sparse(c.content),
            })

        try:
            res = self.client.insert(collection_name=self.collection, data=rows)
        except Exception as e:  # noqa: BLE001
            raise VectorStoreError(f"写入 Milvus 失败：{e}") from e
        return int(res.get("insert_count", len(rows)))

    def flush(self) -> None:
        try:
            self.client.flush(self.collection)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    def search(self, vector: Sequence[float], top_k: int | None = None,
               *, expr: str | None = None) -> list[SearchHit]:
        """稠密向量检索。"""
        k = top_k or settings.retrieve_top_k
        try:
            res = self.client.search(
                collection_name=self.collection,
                data=[list(vector)],
                limit=k,
                output_fields=["content", "page_no", "page_label",
                               "chunk_type", "section_path", "doc_name",
                               "chunk_index", "doc_id"],
                # 表里有 dense 和 sparse 两个向量字段，Milvus 无法自行判断用哪个，
                # 必须显式指定 anns_field，否则报 "multiple anns_fields exist"。
                # 注意：MilvusClient.search 的 anns_field 是**顶层参数**，
                # 放进 search_params 不生效（实测仍报同一个错）。
                # 工单06 做混合检索时，这里会再加上 sparse 的一路。
                search_params={"metric_type": "COSINE", "params": {"ef": 64}},
                anns_field="dense",
                filter=expr or "",
            )
        except Exception as e:  # noqa: BLE001
            raise VectorStoreError(f"检索失败：{e}") from e

        hits: list[SearchHit] = []
        for group in res:
            for h in group:
                hits.append(self._hit_from_row(
                    h.get("entity", {}),
                    chunk_id=int(h.get("id", 0)),
                    score=float(h.get("distance", 0.0)),
                ))
        return hits

    # ------------------------------------------------------------------
    @staticmethod
    def _hit_from_row(ent: dict, *, chunk_id: int = 0, score: float = 0.0,
                      is_neighbor: bool = False) -> SearchHit:
        """把 Milvus 返回的一行（entity）转成 SearchHit。

        search() 与 fetch_by_chunk_index() 共用 —— Milvus 的 `query()` 不返回
        distance，所以邻块只能拿 score=0.0，且必须打上 is_neighbor 标记，
        否则它会被误当成检索结果参与重排、污染精确率统计。
        """
        return SearchHit(
            chunk_id=chunk_id,
            score=score,
            content=ent.get("content", ""),
            page_no=int(ent.get("page_no", -1)),
            page_label=ent.get("page_label", ""),
            chunk_type=ent.get("chunk_type", "text"),
            section_path=ent.get("section_path", ""),
            doc_name=ent.get("doc_name", ""),
            chunk_index=int(ent.get("chunk_index", 0)),
            doc_id=ent.get("doc_id", ""),
            is_neighbor=is_neighbor,
        )

    def fetch_by_chunk_index(self, indices: Sequence[int], doc_id: str,
                             ) -> list[SearchHit]:
        """按 chunk_index 取块（工单02 邻块扩展用）。

        【为什么用 `in [...]` 而不是范围表达式】
        入库去重（dedup.py）只丢弃重复块，**不重新编号**，所以 chunk_index 有空洞
        （实测 1123 块去重后保留 1116，缺 7 个号）。用 `chunk_index >= a and <= b`
        会把空洞当正常情况返回乱七八糟的块；用 `in [...]` 精确列举则天然容忍空洞
        —— 查不到就是查不到，不会误伤。

        【为什么要按 doc_id 限定】
        chunk_index 只在**文档内**唯一。库里一旦有第二份文档（工单03 就要加
        《招股说明书2.pdf》），不限定 doc_id 就会跨文档串块 —— 而且是静默串，
        答案会张冠李戴。所以 doc_id 是必填参数，不允许省。
        """
        idxs = sorted({int(i) for i in indices if i is not None})
        if not idxs:
            return []
        expr = f"chunk_index in {idxs}"
        if doc_id:
            expr = f'doc_id == "{doc_id}" and {expr}'
        try:
            rows = self.client.query(
                collection_name=self.collection,
                filter=expr,
                output_fields=["content", "page_no", "page_label", "chunk_type",
                               "section_path", "doc_name", "chunk_index", "doc_id"],
            )
        except Exception as e:  # noqa: BLE001
            # 邻块是锦上添花，取不到不该让整个问答挂掉 —— 退化成"没有邻块"。
            raise VectorStoreError(f"邻块查询失败：{e}") from e

        hits = [self._hit_from_row(r, chunk_id=int(r.get("id", 0) or 0),
                                   score=0.0, is_neighbor=True)
                for r in rows]
        return sorted(hits, key=lambda h: h.chunk_index)
