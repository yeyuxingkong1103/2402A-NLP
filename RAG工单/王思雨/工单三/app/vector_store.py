# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 文件说明：向量层。bge-m3 本地编码 + Milvus 建集合/写入/检索，字段含表格定位信息。
from . import config as C                                                          # 复用集中配置

# 集合字段定义：(字段名, Milvus 类型, 附加参数)，与 docs/01_design.md §3 的表格一一对应
_FIELDS = [("doc", "VARCHAR", {"max_length": 128}),                                # 文档名，两篇招股说明书靠它区分
           ("chunk_id", "VARCHAR", {"max_length": 64}),                            # 全局唯一块号
           ("text", "VARCHAR", {"max_length": 65535}),                             # 块文本（表格行块已带表头语境）
           ("page_no", "INT16", {}), ("page_end", "INT16", {}),                    # 起止页，1 起，与 PDF 页码一致
           ("block_type", "VARCHAR", {"max_length": 16}),                          # title / text / table_row / table
           ("table_id", "VARCHAR", {"max_length": 128}),                           # 表格定位：<doc>#p<页码>#t<该页表序>
           ("table_caption", "VARCHAR", {"max_length": 512}),                      # 表题
           ("row_index", "INT16", {}),                                             # 行号，1 起；0 表示整表块
           ("locator", "VARCHAR", {"max_length": 256})]                            # 可读定位串，前端直接展示

_EMBED = None                                                                      # 向量模型单例
_RERANK = None                                                                     # 重排模型单例


def embed_model():                                                                 # 懒加载 bge-m3
    """首次调用加载并常驻显存，避免每次请求重复加载几百 MB 权重。"""
    global _EMBED                                                                  # 声明单例
    if _EMBED is None:                                                             # 尚未加载
        from sentence_transformers import SentenceTransformer                      # 延迟导入，缩短服务启动时间
        _EMBED = SentenceTransformer(C.EMBED_MODEL_PATH, device=C.DEVICE)           # 从本地路径加载，不联网
    return _EMBED                                                                  # 返回单例


def embed(texts: list):                                                            # 批量向量化
    """L2 归一化后内积等价于余弦相似度，可直接用 Milvus 的 IP 度量。"""
    vecs = embed_model().encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False)  # 归一化编码
    import numpy as np                                                             # 延迟导入
    return np.asarray(vecs, dtype=np.float32)                                      # 统一 float32，Milvus 要求


def milvus_client():                                                               # Milvus 客户端
    """连接 Milvus；连接失败由上层捕获并给出友好提示。"""
    from pymilvus import MilvusClient                                              # 延迟导入
    return MilvusClient(uri=C.MILVUS_URI, timeout=10)                              # 10 秒超时，避免 Milvus 挂了把接口拖死


def recreate_collection(client, name: str = None) -> None:                          # 重建集合
    """删旧建新，保证重建索引时是干净状态；显式声明字段便于按表格定位过滤与展示。"""
    from pymilvus import DataType                                                  # 延迟导入字段类型
    name = name or C.MILVUS_COLLECTION                                             # 默认使用配置的集合名
    if client.has_collection(name): client.drop_collection(name)                    # 已存在则先删除
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)         # 主键自增，字段固定
    schema.add_field("id", DataType.INT64, is_primary=True)                        # 自增主键
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=C.EMBED_DIM)             # bge-m3 向量
    for fname, ftype, kw in _FIELDS: schema.add_field(fname, getattr(DataType, ftype), **kw)  # 业务字段逐一声明
    index = client.prepare_index_params()                                          # 索引参数
    index.add_index(field_name="vector", index_type="IVF_FLAT", metric_type="IP", params={"nlist": 1024})  # 内积度量 + IVF_FLAT
    client.create_collection(collection_name=name, schema=schema, index_params=index)  # 建集合与索引


def insert_chunks(client, chunks: list, name: str = None, batch: int = 512) -> None:  # 向量入库
    """分批编码并写入，避免一次性把全部文本读进显存；块字段与列定义严格对应。"""
    name = name or C.MILVUS_COLLECTION                                             # 默认集合
    for i in range(0, len(chunks), batch):                                         # 分批遍历
        part = chunks[i:i + batch]                                                 # 取一批
        vecs = embed([c["text"] for c in part])                                    # 批量编码
        rows = [dict(vector=v.tolist(), doc=c.get("doc", ""), chunk_id=c["chunk_id"], text=c["text"],  # 组装行数据
                     page_no=int(c["page_no"]), page_end=int(c.get("page_end", c["page_no"])),           # 页码
                     block_type=c.get("block_type", "text"), table_id=c.get("table_id", ""),             # 类型与表格标识
                     table_caption=c.get("caption", ""), row_index=int(c.get("row_index", 0)),           # 表题与行号
                     locator=c.get("locator", "")) for v, c in zip(vecs, part)]                          # 定位串
        client.insert(collection_name=name, data=rows)                              # 写入集合
    # 说明：pymilvus 的 MilvusClient 没有 flush 方法，刚写入即可查靠检索时的 Strong 一致性保证。


def vector_search(client, question: str, top_k: int, name: str = None) -> list:      # 向量检索
    """问题编码后取最相近的 top_k 个块；Milvus 不可用时打印原因并返回空，由上层降级。"""
    name = name or C.MILVUS_COLLECTION                                             # 默认集合
    if client is None: return []                                                   # Milvus 未连接（不可用），向量通道直接空手而归，由 BM25 撑住
    # 向量化与向量检索分开兜底：二者的故障源完全不同，混在一个 except 里会把
    # "向量模型路径错"报成"Milvus 不可用"——实测踩过，排障时会被日志带偏。
    try: qvec = embed([question])[0].tolist()                                      # 问题编码
    except Exception as exc:                                                       # 模型路径错、显存不足等
        print(f"[检索降级] 向量模型不可用：{type(exc).__name__}: {exc}"); return []  # 明确指向模型
    try:
        hits = client.search(collection_name=name, data=[qvec], limit=top_k,        # 单条查询向量
                             output_fields=[f for f, _, _ in _FIELDS], consistency_level="Strong")  # 强一致读，规避刚写入读不到
    except Exception as exc:                                                       # 连接失败或超时
        print(f"[检索降级] Milvus 检索失败：{type(exc).__name__}: {exc}"); return []  # 明确指向 Milvus
    return [dict(h["entity"], vec_score=float(h["distance"])) for h in (hits[0] if hits else [])]  # 拍平实体并附向量得分
