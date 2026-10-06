"""src/offline/milvus_store.py —— kb_chunks 的 Milvus 2.5 存储和 dense+sparse 召回。

在链路中的位置：
    写入：src/offline/pipeline.py → 【本文件】 → Milvus 集合 kb_chunks
    读取：src/online/retriever.py → 【本文件】 → Milvus

与 backend/vector_store.py 的关系（两个模块都叫"向量存储"，但服务于两条不同的主线）：
    backend/vector_store.py 服务 backend 那条单角色主线，集合是 rag_docs_v6，
        走"动态字段 + AUTOINDEX + 单路稠密向量"
    本文件服务 src/ 这条多角色多租户主线，集合是 kb_chunks，
        走"固定 schema + HNSW 稠密索引 + SPARSE_INVERTED_INDEX 稀疏索引"，
        并且在存储层就做了 dense + sparse 的 RRF 融合

本文件的核心特点：**把两路召回下沉到了存储层**。
    hybrid_search 一次调用就完成"稠密召回 + 稀疏召回 + RRF 融合"，
    调用方拿到的已经是融合排序后的结果，不需要自己再做融合。
"""
from __future__ import annotations

from typing import Any

from configs.settings import get_settings


class MilvusStore:
    """kb_chunks 的 Milvus 2.5 存储和 dense+sparse 召回。"""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client: Any = None
        self._loaded: set[str] = set()  # 记录已 load 的集合，避免重复 load（load 是重操作）

    def connect(self) -> Any:
        """建立并复用 Milvus 连接（惰性连接 + 单例）。

        返回：
            MilvusClient 实例。

        两个细节：
            pymilvus 在函数内 import —— 没装它时本模块仍可被导入
            token 只在配置了才传 —— 本地无鉴权部署不该被一个空 token 干扰
        """
        if self.client is None:
            from pymilvus import MilvusClient

            kwargs = {"uri": self.settings.milvus_uri}
            if self.settings.milvus_token:
                kwargs["token"] = self.settings.milvus_token
            self.client = MilvusClient(**kwargs)
        return self.client

    def ensure_collection(self, name: str | None = None) -> str:
        """确保集合、索引、加载状态就绪（幂等）。

        参数：
            name: 集合名，默认取配置里的 milvus_collection
        返回：
            实际使用的集合名。

        字段设计要点（与 backend 那条主线的最大区别在这里）：
            id            自增主键（auto_id=True）—— 由 Milvus 生成，不需要调用方提供
            dense_vector  FLOAT_VECTOR(1024)        稠密语义向量
            sparse_vector SPARSE_FLOAT_VECTOR       稀疏关键词向量
            content       VARCHAR(8192) + enable_analyzer=True
                          enable_analyzer 打开的是 Milvus 内置文本分析器，
                          它是稀疏检索（BM25 类）能工作的前提
            summary/parent_id/doc_id/role_id/tenant_id/doc_source/page/
            create_time/update_time/user_id
                          业务元数据。role_id / tenant_id / user_id 三个隔离字段，
                          支撑多角色、多租户、多用户三层数据隔离

        注意 enable_dynamic_field=False（与 backend 那个集合相反）：
            这里字段全部预先声明，换来的是更严格的类型校验和更小的存储开销。
            代价是加字段要迁移集合 —— 这是新架构对"结构稳定性"的取舍。

        索引选择：
            dense  HNSW + COSINE，M=16 / efConstruction=200
                   —— HNSW 是图索引，检索快且召回高。M 控制每个节点的连边数
                   （越大越准越费内存），efConstruction 控制建索引时的搜索广度
                   （越大索引质量越好、建得越慢）。16/200 是常见的高质量配置。
            sparse SPARSE_INVERTED_INDEX + IP（内积）
                   —— 倒排索引是稀疏向量的标准选择；稀疏向量用内积衡量相似度

        为什么每次都要检查 _loaded：
            Milvus 的集合必须 load 进内存才能检索。进程重启或集合被释放后
            load 状态会丢失，所以要在每次使用前确认（用集合缓存避免重复 load）。
        """
        name = name or self.settings.milvus_collection
        client = self.connect()
        if not client.has_collection(collection_name=name):
            from pymilvus import DataType

            schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
            schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
            schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
            schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field("content", DataType.VARCHAR, max_length=8192, enable_analyzer=True)
            schema.add_field("summary", DataType.VARCHAR, max_length=1024)
            schema.add_field("parent_id", DataType.INT64)
            schema.add_field("doc_id", DataType.INT64)
            schema.add_field("role_id", DataType.VARCHAR, max_length=64)
            schema.add_field("tenant_id", DataType.VARCHAR, max_length=64)
            schema.add_field("doc_source", DataType.VARCHAR, max_length=512)
            schema.add_field("page", DataType.INT64)
            schema.add_field("create_time", DataType.INT64)
            schema.add_field("update_time", DataType.INT64)
            schema.add_field("user_id", DataType.VARCHAR, max_length=64)
            index_params = client.prepare_index_params()
            index_params.add_index(field_name="dense_vector", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 200})
            index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
            client.create_collection(collection_name=name, schema=schema, index_params=index_params, consistency_level="Strong")
        if name not in self._loaded:
            client.load_collection(collection_name=name)
            self._loaded.add(name)
        return name

    def health(self) -> tuple[bool, str]:
        """探测 Milvus 可用性。

        返回：
            (是否连通, 说明)。异常类型和信息一并带回，便于接口层直接展示原因。
        """
        try:
            self.ensure_collection()
            return True, "connected"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def insert_chunks(self, records: list[dict[str, Any]], collection: str | None = None) -> int:
        """批量写入向量记录。

        参数：
            records: 记录列表（字段需与集合 schema 对应）
            collection: 集合名，None 用默认集合
        返回：
            写入条数；records 为空时返回 0。

        注意这里没有 upsert 的幂等语义：
            主键由 Milvus 自增生成，重复插入会产生重复数据。
            所以重建必须显式先调 delete_document（pipeline 里的 rebuild 分支就是干这个的）。
        """
        if not records:
            return 0
        collection = self.ensure_collection(collection)
        self.connect().insert(collection_name=collection, data=records)
        return len(records)

    def delete_document(self, doc_id: int, role_id: str, tenant_id: str, collection: str | None = None) -> None:
        """按（文档 id, 角色, 租户）三元组删除该文档的全部向量。

        参数：
            doc_id: 元数据库里的文档 id
            role_id / tenant_id: 隔离维度
            collection: 集合名

        三元组缺一不可：
            只用 doc_id 删，可能删到别的租户的文档（不同租户的 doc_id 可能重号）；
            加上 role_id 和 tenant_id 才能精确定位到"这一份"。
            这不是可选的严谨，而是多租户场景下的数据安全问题。
        """
        collection = self.ensure_collection(collection)
        expression = f'doc_id == {int(doc_id)} and role_id == "{_escape(role_id)}" and tenant_id == "{_escape(tenant_id)}"'
        self.connect().delete(collection_name=collection, filter=expression)

    def _search(self, collection: str, field: str, vector: Any, limit: int, expression: str = "") -> list[dict[str, Any]]:
        """单路向量检索的内部实现（稠密/稀疏共用）。

        参数：
            collection: 集合名
            field: 检索字段，"dense_vector" 或 "sparse_vector"
            vector: 查询向量（稠密是浮点数组，稀疏是 {词id: 权重}）
            limit: 返回条数
            expression: 可选的元数据过滤
        返回：
            命中的原始记录列表（hits[0]，因为批量查询返回的是按查询向量分组的嵌套列表）。

        两路用不同的检索参数：
            稠密 HNSW 用 COSINE，ef = max(64, limit*4)
                 ef 是检索时的搜索广度，越大越准越慢。跟 limit 挂钩是因为
                 要返回 limit 条，搜索范围至少得比它宽几倍才有得挑
            稀疏 倒排索引用 IP（内积），参数为空（倒排索引不需要调参）
        """
        params = {"metric_type": "COSINE", "params": {"ef": max(64, limit * 4)}} if field == "dense_vector" else {"metric_type": "IP", "params": {}}
        kwargs: dict[str, Any] = {"collection_name": collection, "data": [vector], "anns_field": field, "limit": limit, "output_fields": ["*"] , "search_params": params}
        if expression:
            kwargs["filter"] = expression
        hits = self.connect().search(**kwargs)
        return hits[0] if hits else []

    def hybrid_search(self, dense: list[float], sparse: dict[int, float], limit: int = 12, expression: str = "", collection: str | None = None) -> list[dict[str, Any]]:
        """稠密 + 稀疏双路召回并做加权 RRF 融合。

        参数：
            dense: 查询的稠密向量
            sparse: 查询的稀疏向量
            limit: 最终返回条数
            expression: 元数据过滤（如限定 role_id / tenant_id）
            collection: 集合名
        返回：
            融合后按 rrf_score 降序的记录列表，每项带 rrf_score 字段。

        与 backend/retrieval.py 的 rrf_merge 是同一套思路的存储层实现：
            两路分数量纲不同（COSINE 在 0~1，IP 无上界），不能直接相加，
            所以用排名倒数融合：score = 权重 / (60 + 排名)
            60 是 RRF 的常用平滑常数，作用同 backend 那边。

        权重 1.5（稀疏路）vs 1.0（稠密路）：
            稀疏路给更高权重，因为它对标准编号、专业术语这类精确字面匹配更可靠。
            注意这个 1.5 与 backend 那条主线的 2.5 不同 —— 两套实现是独立调参的，
            且本文件的稀疏路来自 Milvus 倒排索引（而非 jieba BM25），特性不同。

        merged.setdefault(...) 的写法保证了：
            两路都命中的记录只保留一份（首次出现时以稠密路的 hit 数据为基底），
            rrf_score 累加两路的贡献 —— 所以"双路都命中"的片段会明显排到前面。
        """
        collection = self.ensure_collection(collection)
        dense_hits = self._search(collection, "dense_vector", dense, limit, expression)
        sparse_hits = self._search(collection, "sparse_vector", sparse, limit, expression)
        merged: dict[Any, dict[str, Any]] = {}
        for rank, hit in enumerate(dense_hits, 1):
            merged.setdefault(hit.get("id"), {**hit, "rrf_score": 0.0})["rrf_score"] += 1 / (60 + rank)
        for rank, hit in enumerate(sparse_hits, 1):
            # setdefault 保证稀疏路先命中而稠密路未命中时，也能以它自己的数据建条目
            item = merged.setdefault(hit.get("id"), {**hit, "rrf_score": 0.0})
            item["rrf_score"] += 1.5 / (60 + rank)
        return sorted(merged.values(), key=lambda item: -item["rrf_score"])[:limit]

    def query(self, expression: str = "", limit: int = 1000, collection: str | None = None) -> list[dict[str, Any]]:
        """按条件查询记录（不做向量检索）。

        参数：
            expression: 过滤表达式
            limit: 返回条数上限
            collection: 集合名

        用途：
            知识库管理页查看某角色/某租户下的 chunk 明细。
        """
        collection = self.ensure_collection(collection)
        return self.connect().query(collection_name=collection, filter=expression, output_fields=["*"], limit=limit)


def _escape(value: str) -> str:
    """转义过滤表达式里的字符串字面量。

    参数：
        value: 待转义的值
    返回：
        反斜杠与双引号已转义的值。

    先转义反斜杠、再转义双引号，顺序不可颠倒 ——
    反过来的话，第二次转义会把第一次加上的反斜杠再转一遍，得到错误结果。
    不转义则 role_id 里若含引号会提前闭合字面量，导致删除/过滤条件被改写。
    """
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


# 模块级单例：连接、集合状态都缓存在实例里，重复 new 会丢掉这些缓存
store = MilvusStore()
