from pathlib import Path
from typing import Any

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility


# 这个类专门负责和 Milvus 向量数据库交互。
# 简单说：embedding.py 负责“生成向量”，这个文件负责“把向量存进去”。
class MilvusKnowledgeStore:
    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        dimension: int,
        batch_size: int = 64,
    ) -> None:
        # collection_name 是 Milvus 里的集合名，可以理解成一张“向量表”。
        self.collection_name = collection_name
        # dimension 是向量维度，必须和 embedding 模型输出维度一致。
        self.dimension = dimension
        # 写入时分批插入，避免一次插太多数据。
        self.batch_size = batch_size
        # 连接 Milvus 服务。
        connections.connect(alias="default", host=host, port=port, timeout=10)
        # 如果 collection 已经存在就复用；不存在就创建一个新的。
        self.collection = self._get_or_create_collection()
        # load 之后 collection 才能被查询/检索。
        self.collection.load()

    def insert_chunks(self, rows: list[dict[str, Any]]) -> int:
        # rows 里每一项就是一个准备写入 Milvus 的 chunk。
        # 它包含 chunk_id、页码、正文 text、向量 vector 等字段。
        if not rows:
            return 0

        inserted = 0
        # 按 batch_size 分批写入，防止一次插入太多导致性能或内存问题。
        for start in range(0, len(rows), self.batch_size):
            batch = rows[start : start + self.batch_size]
            # Milvus 这里是“按列插入”，所以每个字段都要整理成一个列表。
            # 比如所有 chunk_id 放一个列表，所有 vector 放一个列表。
            self.collection.insert(
                [
                    [row["chunk_id"] for row in batch],
                    [row["document_id"] for row in batch],
                    [row["source_file"] for row in batch],
                    [row["page_start"] for row in batch],
                    [row["page_end"] for row in batch],
                    [row["chunk_index"] for row in batch],
                    [row["chunk_type"] for row in batch],
                    [row["text"] for row in batch],
                    [row["vector"] for row in batch],
                ]
            )
            inserted += len(batch)
        # flush 表示把刚才插入的数据真正刷到 Milvus，保证后面能查到。
        self.collection.flush()
        return inserted

    def has_chunk_ids(self, chunk_ids: list[str]) -> set[str]:
        # 这个方法用来查哪些 chunk 已经在 Milvus 里了。
        # 作用是避免重复入库，同一个 chunk 不要插两遍。
        if not chunk_ids:
            return set()
        # chunk_id 里如果有双引号，要先转义，不然后面拼查询表达式会出错。
        escaped = [chunk_id.replace('"', '\\"') for chunk_id in chunk_ids]
        # 拼出 Milvus 查询条件：chunk_id in ["id1", "id2", ...]。
        expression = "chunk_id in [" + ",".join(f'"{chunk_id}"' for chunk_id in escaped) + "]"
        # 只查 chunk_id 字段就够了，不需要把整段文本和向量都查回来。
        rows = self.collection.query(expression, output_fields=["chunk_id"])
        return {str(row["chunk_id"]) for row in rows}

    def close(self) -> None:
        # 用完后断开 Milvus 连接。
        connections.disconnect("default")

    def _get_or_create_collection(self) -> Collection:
        # 如果 Milvus 里已经有这个 collection，就直接拿来用。
        if utility.has_collection(self.collection_name):
            collection = Collection(self.collection_name)
            # 但复用前要检查向量维度，防止模型维度和库里的维度不一致。
            self._validate_dimension(collection)
            return collection

        # 如果 collection 不存在，就定义它的字段结构。
        fields = [
            # id 是 Milvus 自动生成的主键。
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            # chunk_id 是业务上的唯一片段编号。
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256),
            # document_id 表示这个 chunk 属于哪篇文档。
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256),
            # source_file 用来记录来源文件名，方便后面展示出处。
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=512),
            # page_start/page_end 用来记录页码范围。
            FieldSchema(name="page_start", dtype=DataType.INT64),
            FieldSchema(name="page_end", dtype=DataType.INT64),
            # chunk_index 表示 chunk 在文档里的顺序。
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            # chunk_type 目前主要是 text，预留给以后区分表格、图片等类型。
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=32),
            # text 是 chunk 正文，检索出来后要给大模型参考。
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=12000),
            # vector 是最关键的向量字段，dim 必须等于 embedding 模型的输出维度。
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
        ]
        # schema 就是 collection 的表结构说明。
        schema = CollectionSchema(fields, description="Mental health knowledge chunks from cleaned MinerU output")
        # 创建 collection。
        collection = Collection(self.collection_name, schema=schema)
        # 给 vector 字段建索引。
        # HNSW 是常用的近似向量检索索引，COSINE 表示用余弦相似度。
        collection.create_index(
            field_name="vector",
            index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}},
        )
        return collection

    def _validate_dimension(self, collection: Collection) -> None:
        # 找到 collection 里的 vector 字段。
        vector_field = next(field for field in collection.schema.fields if field.name == "vector")
        # 取出 Milvus 里已经存在的向量维度。
        actual = int(vector_field.params["dim"])
        # 如果已有 collection 的维度和当前模型维度不一致，就不能继续写入。
        # 否则 Milvus 会报错，或者检索结果不可靠。
        if actual != self.dimension:
            raise ValueError(
                f"Milvus vector dimension mismatch: collection={actual}, model={self.dimension}"
            )
