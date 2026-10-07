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

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from pymilvus import DataType, MilvusClient

from app.config import settings
from app.core.text_analysis import term_id, tokenize


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
    # ---------- 工单06 新增 ----------
    # 【字段只能追加在最后】单测里有按位置参数构造 SearchHit 的地方（tests/test_pipeline.py），
    # 插在中间会让它们静默错位。
    #
    # score_kind：这个 score 是哪把尺子量出来的。**必须有**，因为重排第 3 步的
    # `(score+1)/2` 只在 COSINE（∈[-1,1]）下成立；BM25 无上界、RRF 只有 0.008~0.05，
    # 直接套那个公式会把它们压成近乎常数 —— 排序悄悄退化成"只按另一个信号排"，
    # 而且**不报错**。
    score_kind: str = "cosine"      # cosine | bm25 | rank | fused
    # retrieval_src：这条命中来自哪一路（排查混合检索"谁召回的"用）
    retrieval_src: str = "dense"    # dense | keyword | fused | neighbor
    fuzzy: bool = False             # 是否由模糊匹配命中（分数已打折）
    # 【为什么要把"融合前的分"单独留一份】融合后的 `score` 是打分器（RRF/加权）
    # 给的，尺度已经不是 COSINE 了；但**重排做归一化时必须知道它原本是什么尺度** ——
    # 实测拿排名尺度的分去做词法融合，页召回会从 60.4% 掉到 38.5%：
    # COSINE 分数密集在 0.75~0.83 的窄带里，融合时约九成权重落在"查询词覆盖率"上；
    # 换成铺满 0~1 的排名分之后，这个信号被盖掉，排序就变了。
    base_score: float | None = None   # 融合前的原始检索分
    base_kind: str = ""               # 融合前的分数尺度（cosine / bm25 / none）


# ----------------------------------------------------------------------
# 稀疏向量（词法权重）
# ----------------------------------------------------------------------
# 切词与「token → 维度」的映射**收拢在 app/core/text_analysis.py**（工单06）。
# 这里只负责"词频 → 稀疏向量"这一步。两条词法路（Milvus 稀疏向量 / 应用层倒排索引）
# 必须共用同一套切词，否则它们的命中无法互相印证。
def lexical_sparse(text: str, max_terms: int = 256,
                   idf: dict[int, float] | None = None) -> dict[int, float]:
    """用分词结果构造稀疏向量（工单06 起支持 IDF）。

    【为什么原先只有 TF】入库是**逐文档**写的，算不出全局 DF（IDF 需要全库统计）。
    工单01 建表时就把 sparse 字段留好了；工单06 用 `scripts/rebuild_sparse.py`
    统计全局 DF 后重算一遍再回写 —— 这就是那条注释里说的"工单06 再把 IDF 补上"。

    【工单06 同时修掉的一个静默 bug】旧实现的维度映射用的是内置 `hash()`，
    而它对 str **逐进程随机化**（实测两个进程里 `hash("万元")` 完全不同）。
    入库与查询是两个进程 → 维度对不上 → Milvus 稀疏检索会**静默失效**
    （不报错、召回率≈0）。已换成 `text_analysis.term_id`（crc32，内容稳定）。
    """
    tokens = tokenize(text)
    if not tokens:
        return {}

    counts = Counter(tokens)
    total = sum(counts.values())
    # 归一化 TF，避免长 chunk 因词数多而在 IP 内积下天然占优
    weighted = {t: c / total for t, c in counts.items()}
    if idf:
        weighted = {t: w * idf.get(term_id(t), 1.0) for t, w in weighted.items()}
    top = sorted(weighted.items(), key=lambda kv: -kv[1])[:max_terms]
    return {term_id(t): float(w) for t, w in top}



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
    def build_schema(self, dim: int | None = None):
        """建表 schema。字段与工单01 验收及后续工单的扩展点一一对应。

        【工单06：dim 可传】支持多种嵌入模型。**维度建表即定死**，所以换一个
        维度不同的模型（如 nomic-embed-text 的 768）必须用**独立 collection** ——
        不能往同一个集合里混（见 docs/工单06-混合检索.md 的多模型一节）。
        """
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
        schema.add_field("dense", DataType.FLOAT_VECTOR,
                         dim=dim or settings.embed_dim)
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
    def create_collection(self, *, drop_existing: bool = False,
                          dim: int | None = None) -> str:
        if drop_existing and self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
        if self.client.has_collection(self.collection):
            return "exists"
        self.client.create_collection(
            collection_name=self.collection,
            schema=self.build_schema(dim=dim),
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
                      kb_id: str = "default",
                      embed_model: str | None = None) -> int:
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
                "embed_model": embed_model or settings.embed_model,
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
    # 工单06：稀疏路与原生混合检索
    # ------------------------------------------------------------------
    _OUT_FIELDS = ["content", "page_no", "page_label", "chunk_type",
                   "section_path", "doc_name", "chunk_index", "doc_id"]

    def search_sparse(self, sparse_vec: dict[int, float], top_k: int | None = None,
                      *, expr: str | None = None) -> list[SearchHit]:
        """稀疏向量检索（关键词路）。

        【注意 query 侧的稀疏向量必须用 `text_analysis.term_id` 构造】
        旧实现用的是内置 `hash()`，它对 str **逐进程随机化** —— 入库进程与查询
        进程算出的维度对不上，检索会**静默失效**（不报错、召回率≈0）。
        见 `app/core/text_analysis.term_id` 的说明。
        """
        k = top_k or settings.retrieve_top_k
        try:
            res = self.client.search(
                collection_name=self.collection, data=[sparse_vec], limit=k,
                output_fields=self._OUT_FIELDS,
                search_params={"metric_type": "IP"},
                anns_field="sparse", filter=expr or "",
            )
        except Exception as e:  # noqa: BLE001
            raise VectorStoreError(f"稀疏检索失败：{e}") from e
        return [self._hit_from_row(h.get("entity", {}),
                                   chunk_id=int(h.get("id", 0)),
                                   score=float(h.get("distance", 0.0)),
                                   src="keyword")
                for group in res for h in group]

    def hybrid_search(self, dense_vec: Sequence[float], sparse_vec: dict[int, float],
                      top_k: int | None = None, *, expr: str | None = None,
                      ranker: str = "rrf", w_keyword: float = 0.5) -> list[SearchHit]:
        """向量库**原生**混合检索：`hybrid_search` + Ranker。

        `ranker="rrf"` → RRFRanker（投票/RRF）；`"weighted"` → WeightedRanker（加权平均）。
        两个 Ranker 都由 Milvus 提供，正好对应工单点名的两种融合算法。
        """
        from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

        k = top_k or settings.retrieve_top_k
        reqs = [
            AnnSearchRequest(data=[list(dense_vec)], anns_field="dense",
                             param={"metric_type": "COSINE", "params": {"ef": 64}},
                             limit=k, expr=expr or None),
            AnnSearchRequest(data=[sparse_vec], anns_field="sparse",
                             param={"metric_type": "IP"}, limit=k, expr=expr or None),
        ]
        rk = RRFRanker(60) if ranker == "rrf" else WeightedRanker(1.0 - w_keyword, w_keyword)
        try:
            res = self.client.hybrid_search(
                collection_name=self.collection, reqs=reqs, ranker=rk,
                limit=k, output_fields=self._OUT_FIELDS)
        except Exception as e:  # noqa: BLE001
            raise VectorStoreError(f"原生混合检索失败：{e}") from e
        return [self._hit_from_row(h.get("entity", {}),
                                   chunk_id=int(h.get("id", 0)),
                                   score=float(h.get("distance", 0.0)),
                                   src="fused", kind="fused")
                for group in res for h in group]

    def fetch_all(self, *, limit: int = 16384, with_vectors: bool = False) -> list[dict]:
        """拉全量行（倒排索引构建 / sparse 重建用）。"""
        fields = ["id", "content", "page_no", "page_label", "chunk_type",
                  "section_path", "doc_name", "chunk_index", "doc_id",
                  "content_hash", "embed_model", "kb_id", "parent_id"]
        if with_vectors:
            fields += ["dense", "sparse"]
        return self.client.query(collection_name=self.collection, filter="id >= 0",
                                 output_fields=fields, limit=limit)

    # ------------------------------------------------------------------
    @staticmethod
    def _hit_from_row(ent: dict, *, chunk_id: int = 0, score: float = 0.0,
                      is_neighbor: bool = False,
                      src: str | None = None, kind: str | None = None) -> SearchHit:
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
            score_kind=(kind or ("cosine" if not is_neighbor else "none")),
            retrieval_src=(src or ("neighbor" if is_neighbor else "dense")),
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
