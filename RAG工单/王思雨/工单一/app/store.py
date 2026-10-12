# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 本模块职责：向量库（Milvus）与关键词库（BM25）的读写、文档管理，即「知识库存储层」。
# 拆出本模块是为了满足「每个 .py 不超过 300 行」的工程约束；解析与分块见 app/rag.py。

import pickle                         # 持久化 BM25 语料
from pathlib import Path              # 路径处理

from . import config as C             # 全局配置
from . import rag                     # 复用向量化与分词能力

_MILVUS = None                        # Milvus 客户端全局单例
_BM25_CACHE = None                    # BM25 对象的内存缓存


def get_milvus():                                                   # Milvus 客户端单例（含降级）
    """优先连 Docker 版 Milvus；连不上自动降级到 milvus-lite 本地文件库。"""
    global _MILVUS                                                  # 使用模块级单例
    if _MILVUS is None:                                             # 首次调用
        from pymilvus import MilvusClient                            # 延迟导入
        try:                                                        # 先试服务端
            client = MilvusClient(uri=C.MILVUS_URI)                 # 连接 http://localhost:19530
            client.list_collections()                               # 发一次请求确认真的通
            _MILVUS = client                                        # 连接成功，缓存单例
        except Exception as exc:                                    # 服务端不可用
            print(f"[Milvus 降级] 服务端不可用({type(exc).__name__})，改用 milvus-lite 本地库")
            _MILVUS = MilvusClient(uri=C.MILVUS_LITE_PATH)          # 降级为本地文件库
    return _MILVUS                                                  # 返回可用客户端


def ensure_collection(drop: bool = False):                          # 建集合与索引
    """按设计文档的 schema 建 collection（HNSW + COSINE），已存在则复用。"""
    from pymilvus import DataType                                   # Milvus 数据类型枚举
    client = get_milvus()                                           # 拿到客户端
    name = C.MILVUS_COLLECTION                                      # 集合名
    if drop and client.has_collection(name):                        # 需要重建时先删
        client.drop_collection(name)                                # 删除旧集合
    if client.has_collection(name):                                 # 已存在就直接返回
        return client                                               # 复用
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)  # 显式 schema，不开动态字段
    schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)    # 主键自增
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=C.EMBED_DIM)       # 稠密向量 1024 维
    schema.add_field("doc_id", DataType.VARCHAR, max_length=64)              # 文档标识
    schema.add_field("doc_name", DataType.VARCHAR, max_length=256)           # 原文件名
    schema.add_field("chunk_index", DataType.INT64)                          # 分块序号
    schema.add_field("page_no", DataType.INT64)                              # 起始页
    schema.add_field("page_end", DataType.INT64)                             # 结束页
    schema.add_field("chunk_type", DataType.VARCHAR, max_length=16)          # text / table
    schema.add_field("content", DataType.VARCHAR, max_length=8192)           # 分块正文
    idx = client.prepare_index_params()                             # 准备索引参数
    idx.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE",  # 向量索引
                  params={"M": 16, "efConstruction": 200})          # HNSW 参数
    idx.add_index(field_name="doc_id", index_type="INVERTED")       # 标量倒排索引，支持按文档过滤
    client.create_collection(name, schema=schema, index_params=idx,
                             consistency_level="Strong")   # Strong 一致性：写入/删除后立即可见，避免刚上传就检索不到

    return client                                                   # 返回客户端


def index_chunks(chunks: list[dict], doc_id: str, doc_name: str) -> int:  # 写入 Milvus + BM25
    """把分块向量化后写入 Milvus，并把 BM25 语料落盘；返回写入条数。

    空语料一律拒绝，且**在删除旧数据之前**就拒绝。
    动机：扫描件（无文本层）解析出 0 个块时，原实现会先删掉该 doc_id 的旧数据、
    再"成功"写入 0 条并返回 0 —— 用户得到一个空知识库却显示上传成功，比显式报错危险得多。
    这个断言放在写入层而不是上传接口，是为了让所有入口（网页上传、直接调 API、批量导入脚本）
    都走同一道闸门，新加入口不会绕过。
    """
    if not chunks:                                                  # 空语料
        raise ValueError("分块结果为空，拒绝写入索引：该文档可能无文本层（扫描件）或内容全被过滤。"
                         "如需清空该文档，请调用删除接口。")            # 显式失败，不静默成功
    client = ensure_collection()                                    # 确保集合存在
    client.delete(collection_name=C.MILVUS_COLLECTION, filter=f'doc_id == "{doc_id}"')  # 幂等：先删旧数据
    rows = []                                                       # 待插入的行
    order = sorted(range(len(chunks)), key=lambda i: len(chunks[i]["content"]))  # 全局按长度排序
    # 排序后分批：同批长度接近，避免一个长表格块把 64 条一起 padding 到超长序列（实测可快 8 倍）
    for i in range(0, len(order), 64):                              # 分批向量化，控制显存峰值
        idxs = order[i:i + 64]                                      # 当前批的原始下标
        vecs = rag.embed_texts([chunks[j]["content"] for j in idxs])   # 批量编码
        for j, v in zip(idxs, vecs):                                # 逐条组装
            c = chunks[j]                                           # 取回对应分块
            rows.append({"vector": v.tolist(), "doc_id": doc_id, "doc_name": doc_name,
                         "chunk_index": int(c["chunk_index"]), "page_no": int(c["page_no"]),
                         "page_end": int(c["page_end"]), "chunk_type": c["type"],
                         "content": c["content"][:8000]})           # content 按 schema 上限截断
    if rows:                                                        # 有数据才插入
        client.insert(collection_name=C.MILVUS_COLLECTION, data=rows)  # 写入向量库
    _save_bm25_corpus(chunks, doc_id, doc_name)                     # 同步保存 BM25 语料
    return len(rows)                                                # 返回写入条数


def _corpus_path() -> Path:                                         # BM25 语料文件路径
    return C.INDEX_DIR / "bm25.pkl"                                 # 固定文件名，单文件全量存储


def _save_bm25_corpus(chunks: list[dict], doc_id: str, doc_name: str) -> None:  # 保存/追加 BM25 语料
    """把分块追加进 BM25 语料文件；同 doc_id 的旧语料先剔除，保证重复建索引不重复。"""
    global _BM25_CACHE                                              # 使内存缓存失效
    data = _load_bm25_corpus()                                      # 读已有语料（可能为空）
    keep = [c for c in data["chunks"] if c["doc_id"] != doc_id]     # 剔除同文档旧数据
    for c in chunks:                                                # 追加新分块
        keep.append({"doc_id": doc_id, "doc_name": doc_name, "chunk_index": c["chunk_index"],
                     "page_no": c["page_no"], "page_end": c["page_end"],
                     "type": c["type"], "content": c["content"]})   # 只存检索需要的字段
    tokens = [rag.tokenize(c["content"]) for c in keep]             # 预先分词，加速 BM25 构建
    _corpus_path().write_bytes(pickle.dumps({"chunks": keep, "tokens": tokens}))  # 落盘
    _BM25_CACHE = None                                              # 下次检索时重建


def _load_bm25_corpus() -> dict:                                    # 读取 BM25 语料
    p = _corpus_path()                                              # 语料路径
    if not p.exists():                                              # 文件不存在
        return {"chunks": [], "tokens": []}                         # 返回空结构
    try:                                                            # 防御：文件损坏时不让服务崩
        return pickle.loads(p.read_bytes())                         # 反序列化
    except Exception:                                               # 解析失败
        return {"chunks": [], "tokens": []}                         # 退回空结构


def get_bm25():                                                     # 获取 BM25 检索器
    """返回 (BM25 对象, 语料列表)；语料未变化时复用缓存，避免每次重建。"""
    global _BM25_CACHE                                              # 使用模块级缓存
    data = _load_bm25_corpus()                                      # 读取语料
    if not data["tokens"]:                                          # 没有语料
        return None, []                                             # 返回空，调用方降级为纯向量检索
    if _BM25_CACHE is None:                                         # 缓存未命中
        from rank_bm25 import BM25Okapi                             # 延迟导入
        _BM25_CACHE = BM25Okapi(data["tokens"])                     # 构建 BM25 索引（内存中）
    return _BM25_CACHE, data["chunks"]                              # 返回检索器与语料


def list_documents() -> list[dict]:                                 # 知识库文档列表
    """从 Milvus 统计每个 doc_id 的分块数与页数，供前端知识库页展示。"""
    client = get_milvus()                                           # 客户端
    if not client.has_collection(C.MILVUS_COLLECTION):              # 集合不存在
        return []                                                   # 空知识库
    rows = client.query(collection_name=C.MILVUS_COLLECTION,        # 查询全部标量字段（不取向量）
                        filter="pk >= 0", output_fields=["doc_id", "doc_name", "page_no", "chunk_type"],
                        limit=16384, consistency_level="Strong")    # 强一致读取，避免刚入库看不到
    agg = {}                                                        # 按 doc_id 聚合
    for r in rows:                                                  # 遍历每条记录
        d = agg.setdefault(r["doc_id"], {"doc_id": r["doc_id"], "doc_name": r["doc_name"],
                                         "chunks": 0, "pages": 0, "tables": 0})  # 初始化聚合项
        d["chunks"] += 1                                            # 累加分块数
        d["pages"] = max(d["pages"], int(r["page_no"]))             # 记录最大页码
        d["tables"] += 1 if r["chunk_type"] == "table" else 0       # 统计表格块数
    return sorted(agg.values(), key=lambda x: x["doc_name"])        # 按文件名排序返回


def get_chunks(doc_id: str, page: int = 1, size: int = 20, kind: str = "") -> dict:  # 查看分块内容
    """分页返回指定文档的分块，支持按 text/table 过滤；用于知识库管理页。"""
    client = get_milvus()                                           # 客户端
    if not client.has_collection(C.MILVUS_COLLECTION):              # 集合不存在
        return {"total": 0, "items": []}                            # 空结果
    expr = f'doc_id == "{doc_id}"'                                  # 基础过滤条件
    if kind in ("text", "table"):                                   # 类型过滤
        expr += f' and chunk_type == "{kind}"'                      # 追加条件
    rows = client.query(collection_name=C.MILVUS_COLLECTION, filter=expr,   # 查询
                        output_fields=["chunk_index", "page_no", "page_end", "chunk_type", "content"],
                        limit=16384, consistency_level="Strong")    # 强一致读取
    rows.sort(key=lambda r: int(r["chunk_index"]))                  # 按分块序号排序
    start = max(0, (page - 1) * size)                               # 分页起点
    return {"total": len(rows), "items": rows[start:start + size]}  # 返回总数与当页数据


def delete_document(doc_id: str) -> int:                            # 删除文档
    """删除 Milvus 中该文档的全部向量与 BM25 语料，返回删除条数。"""
    global _BM25_CACHE                                              # 让缓存失效
    client = get_milvus()                                           # 客户端
    if not client.has_collection(C.MILVUS_COLLECTION):              # 集合不存在
        return 0                                                    # 无需删除
    client.delete(collection_name=C.MILVUS_COLLECTION, filter=f'doc_id == "{doc_id}"')  # 删向量
    data = _load_bm25_corpus()                                      # 读语料
    keep = [c for c in data["chunks"] if c["doc_id"] != doc_id]     # 剔除目标文档
    removed = len(data["chunks"]) - len(keep)                       # 计算删除条数
    tokens = [rag.tokenize(c["content"]) for c in keep]             # 剩余语料重新分词
    _corpus_path().write_bytes(pickle.dumps({"chunks": keep, "tokens": tokens}))  # 落盘
    _BM25_CACHE = None                                              # 置空
    return removed                                                  # 返回删除条数
