from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from backend.app.core.config import get_settings


settings = get_settings()


class MilvusStore:
    """负责把文档向量和长期记忆保存到 Milvus。"""

    def __init__(self) -> None:
        # 建立 Milvus 连接，alias 是 pymilvus 的连接名称。
        connections.connect(alias="default", host=settings.milvus_host, port=settings.milvus_port)
        # 确保文档向量集合存在。
        self.document_collection = self._ensure_document_collection()
        # 确保长期记忆集合存在。
        self.memory_collection = self._ensure_memory_collection()

    def _ensure_document_collection(self) -> Collection:
        # 如果集合已经存在，就检查它是否符合 Attu 中的项目字段结构。
        if utility.has_collection(settings.milvus_document_collection):
            # 获取 Attu 中已经创建的集合。
            collection = Collection(settings.milvus_document_collection)
            # 校验字段和向量维度，避免写入时出现难懂的错误。
            self._validate_document_schema(collection)
            # 加载集合到 Milvus 内存。
            collection.load()
            # 返回现有集合。
            return collection
        # 如果集合不存在，就按 Attu 使用的字段结构创建集合。
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="chapter", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="tenant_id", dtype=DataType.INT64),
            FieldSchema(name="knowledge_base_id", dtype=DataType.INT64),
            FieldSchema(name="document_id", dtype=DataType.INT64),
            FieldSchema(name="document_version_id", dtype=DataType.INT64),
            FieldSchema(name="document_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="access_scope", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="is_active", dtype=DataType.BOOL),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.milvus_embedding_dim),
        ]
        # 创建和 Attu 一致的集合结构。
        schema = CollectionSchema(fields=fields, description="法律文档分块向量集合")
        # 在 Milvus 中创建集合。
        collection = Collection(settings.milvus_document_collection, schema=schema)
        # 给向量字段创建 HNSW 索引，提升向量检索速度。
        collection.create_index("embedding", {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}})
        # 加载集合到内存。
        collection.load()
        # 返回新集合。
        return collection

    def _validate_document_schema(self, collection: Collection) -> None:
        # 读取 Attu 中已经创建的字段名称。
        actual_fields = {field.name for field in collection.schema.fields}
        # 这些字段是当前导入和检索必须使用的字段。
        required_fields = {
            "id",
            "source_file",
            "chapter",
            "text",
            "tenant_id",
            "knowledge_base_id",
            "document_id",
            "document_version_id",
            "document_type",
            "access_scope",
            "is_active",
            "embedding",
        }
        # 找出缺少的字段。
        missing_fields = required_fields - actual_fields
        # 如果字段不完整，给出实际字段和缺失字段，方便在 Attu 中修正。
        if missing_fields:
            raise RuntimeError(
                f"Milvus 集合 {settings.milvus_document_collection} 缺少字段：{sorted(missing_fields)}；"
                f"当前字段：{sorted(actual_fields)}。"
            )
        # 找到向量字段，检查向量维度。
        embedding_field = next(field for field in collection.schema.fields if field.name == "embedding")
        # 读取向量字段的 dim 参数。
        actual_dim = embedding_field.params.get("dim")
        # BGE-M3 的向量维度必须和配置一致。
        if int(actual_dim) != settings.milvus_embedding_dim:
            raise RuntimeError(
                f"Milvus 向量维度不匹配：当前是 {actual_dim}，项目配置是 {settings.milvus_embedding_dim}。"
            )

    def _ensure_memory_collection(self) -> Collection:
        # 如果长期记忆集合已存在，直接加载并返回。
        if utility.has_collection(settings.milvus_memory_collection):
            collection = Collection(settings.milvus_memory_collection)
            collection.load()
            return collection
        # 定义长期记忆集合字段。
        fields = [
            FieldSchema(name="memory_id", dtype=DataType.VARCHAR, is_primary=True, max_length=80),
            FieldSchema(name="user_id", dtype=DataType.INT64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.milvus_embedding_dim),
        ]
        # 创建长期记忆集合结构。
        schema = CollectionSchema(fields=fields, description="用户长期记忆向量集合")
        # 真正创建集合。
        collection = Collection(settings.milvus_memory_collection, schema=schema)
        # 给记忆向量创建索引。
        collection.create_index("embedding", {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}})
        # 加载集合。
        collection.load()
        # 返回集合对象。
        return collection

    def insert_chunks(self, chunks: list[dict], vectors: list[list[float]], source_file: str) -> None:
        # 如果没有分块，就不需要写入 Milvus。
        if not chunks:
            return
        # 按 Attu 集合字段顺序组织数据。
        rows = [
            [chunk["chunk_uid"] for chunk in chunks],
            [source_file[:256] for _ in chunks],
            [chunk["title_path"][:512] for chunk in chunks],
            [chunk["content"][:8192] for chunk in chunks],
            [chunk["user_id"] for chunk in chunks],
            [chunk["knowledge_base_id"] for chunk in chunks],
            [chunk["document_id"] for chunk in chunks],
            [chunk["document_id"] for chunk in chunks],
            ["law" for _ in chunks],
            ["private" for _ in chunks],
            [True for _ in chunks],
            vectors,
        ]
        # 插入数据。
        self.document_collection.insert(rows)
        # flush 保证数据落盘并可被检索。
        self.document_collection.flush()

    def search_chunks(self, query_vector: list[float], user_id: int, knowledge_base_id: int, top_k: int) -> list[dict]:
        # 使用 tenant_id 过滤当前用户，knowledge_base_id 过滤当前知识库，并只检索有效数据。
        expr = f"tenant_id == {user_id} and knowledge_base_id == {knowledge_base_id} and is_active == true"
        # 执行向量检索。
        results = self.document_collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            expr=expr,
            output_fields=["id", "document_id", "text", "chapter", "source_file"],
        )
        # 准备返回普通 Python 字典，方便后续融合。
        items: list[dict] = []
        # Milvus 返回的是二维结果，这里只查一个问题，所以取第一组。
        for hit in results[0]:
            # entity 里包含 output_fields 中指定的字段。
            entity = hit.entity
            # 把 Attu 字段转换成项目内部统一字段，方便后续 RRF 和 Rerank 使用。
            items.append(
                {
                    "chunk_uid": entity.get("id"),
                    "document_id": entity.get("document_id"),
                    "content": entity.get("text"),
                    "title_path": entity.get("chapter"),
                    "page_number": 0,
                    "filename": entity.get("source_file") or "未知文档",
                    "score": float(hit.score),
                    "source": "vector",
                }
            )
        # 返回向量候选。
        return items

    def insert_memory(self, memory_id: str, user_id: int, content: str, vector: list[float]) -> None:
        # 将用户长期记忆写入 Milvus。
        self.memory_collection.insert([[memory_id], [user_id], [content[:4096]], [vector]])
        # flush 后可以被后续问题检索到。
        self.memory_collection.flush()

    def search_memory(self, query_vector: list[float], user_id: int, top_k: int = 3) -> list[str]:
        # 长期记忆只按当前用户检索，避免不同用户相互影响。
        expr = f"user_id == {user_id}"
        # 执行长期记忆向量检索。
        results = self.memory_collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            expr=expr,
            output_fields=["content"],
        )
        # 返回记忆文本列表。
        return [hit.entity.get("content") for hit in results[0]]
