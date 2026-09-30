"""使用 BGE-M3 向量化，并管理 Milvus 稠密向量与 BM25 索引。"""

from functools import lru_cache
from threading import Lock

from pymilvus import DataType, Function, FunctionType, MilvusClient
from sentence_transformers import SentenceTransformer

from app.config import get_settings

settings = get_settings()
_model_lock = Lock()


@lru_cache(maxsize=1)
def embedding_model() -> SentenceTransformer:
    """延迟加载本地 BGE-M3，并在当前 Python 进程内只保留一个模型实例。"""
    # 多个请求同时首次调用时由锁保护，避免重复加载约 2 GB 模型造成内存峰值。
    with _model_lock:
        # local_files_only=True 禁止运行时联网，模型必须已存在于配置的本地缓存目录。
        return SentenceTransformer(
            settings.embedding_model,
            device=settings.model_device,
            local_files_only=True,
            cache_folder=str(settings.huggingface_cache_dir),
        )


def encode_documents(texts: list[str]) -> list[list[float]]:
    """完成第 3 步中的向量化：批量编码知识库文本块。"""
    # 空列表直接返回，避免 SentenceTransformer 对无输入执行无意义推理。
    if not texts:
        return []
    # batch_size=8 在 CPU 内存和吞吐之间折中；过大会增加内存占用。
    vectors = embedding_model().encode(
        texts, batch_size=8, normalize_embeddings=True, show_progress_bar=False
    )
    # 归一化后可直接使用 COSINE 比较方向相似度；NumPy 数组转成 Milvus 接受的列表。
    return vectors.tolist()


@lru_cache(maxsize=1)
def milvus_client() -> MilvusClient:
    """创建并缓存 Milvus 客户端，避免每次检索重复建立连接对象。"""
    # uri 通常是 Docker 暴露到本机的 http://127.0.0.1:19530。
    options = {"uri": settings.milvus_uri}
    # 本地无鉴权部署可留空；连接受保护的 Milvus 时再追加 token。
    if settings.milvus_token:
        options["token"] = settings.milvus_token
    return MilvusClient(**options)


def ensure_collection() -> None:
    """按需创建同时支持稠密向量与 BM25 的 Milvus collection。"""
    client = milvus_client()
    name = settings.milvus_collection
    # collection 已存在时不能重复建表和索引，直接复用其中数据。
    if client.has_collection(name):
        return
    # 关闭 auto_id：本项目自己生成可重复计算的主键，视觉增强时才能 upsert 同一块。
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    # id：每个文本块的唯一主键，使用 UUID 字符串。
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    # document_id：MySQL 文档 ID，用于删除或重建某一文档的全部向量。
    schema.add_field("document_id", DataType.INT64)
    # owner_id：上传者用户 ID；role_id：该知识库对应的角色 ID。
    schema.add_field("owner_id", DataType.INT64)
    schema.add_field("role_id", DataType.INT64)
    # 公开文档所有用户可查，私有文档只允许 owner_id 对应的用户查询。
    schema.add_field("is_public", DataType.BOOL)
    # source 用于回答时显示引用来源，summary 用于管理页快速预览。
    schema.add_field("source", DataType.VARCHAR, max_length=1000)
    schema.add_field("summary", DataType.VARCHAR, max_length=2000)
    # 时间戳使用整数，便于排序、过滤，也避免不同数据库日期格式不一致。
    schema.add_field("created_at", DataType.INT64)
    schema.add_field("updated_at", DataType.INT64)
    # text 保存原文；启用中文分析器后，Milvus 才能为它自动生成 BM25 稀疏向量。
    schema.add_field(
        "text", DataType.VARCHAR, max_length=65535, enable_analyzer=True,
        analyzer_params={"type": "chinese"},
    )
    # dense_vector 存 BGE-M3 的稠密语义向量，维数必须与模型输出完全一致。
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=settings.embedding_dimension)
    # sparse_vector 无需 Python 手工赋值，由下方 BM25 Function 自动计算。
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    # 写入 text 时，Milvus 对中文分词并把 BM25 权重输出到 sparse_vector。
    schema.add_function(Function(
        name="text_bm25", input_field_names=["text"], output_field_names=["sparse_vector"],
        function_type=FunctionType.BM25,
    ))
    # 为语义向量建立 HNSW 近似最近邻索引：M 越高召回更好，但更占内存。
    indexes = client.prepare_index_params()
    indexes.add_index("dense_vector", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 200})
    # 为 BM25 建稀疏倒排索引；DAAT_MAXSCORE 用于高效跳过低价值候选。
    indexes.add_index("sparse_vector", index_type="SPARSE_INVERTED_INDEX",
                      metric_type="BM25", params={"inverted_index_algo": "DAAT_MAXSCORE"})
    client.create_collection(name, schema=schema, index_params=indexes)


def delete_document(document_id: int) -> None:
    """只删除指定 MySQL 文档 ID 对应的全部 Milvus 文本块。"""
    # collection 不存在时先创建空集合，保证 delete 调用目标始终有效。
    ensure_collection()
    # int 强制转换避免把任意字符串拼入 Milvus 过滤表达式。
    milvus_client().delete(settings.milvus_collection, filter=f"document_id == {int(document_id)}")

