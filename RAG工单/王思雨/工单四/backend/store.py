# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 文件说明：向量层与重排层。bge-m3 本地编码、Milvus 建集合/写入/检索、bge-reranker-v2-m3 精排。
import numpy as np                                                          # 数值与向量运算
import config as C                                                          # 集中配置

# Milvus 集合字段：(字段名, 类型, 附加参数)。block_type/image_type 供前端区分文本块与图像语义块
FIELDS = [("doc", "VARCHAR", {"max_length": 128}),                          # 来源文档名
          ("chunk_id", "VARCHAR", {"max_length": 64}),                      # 全局唯一块号
          ("text", "VARCHAR", {"max_length": 65535}),                       # 块文本（图像块含语义描述+图内文字）
          ("page_no", "INT16", {}), ("page_end", "INT16", {}),              # 起止页（1 起）
          ("block_type", "VARCHAR", {"max_length": 16}),                    # title / text / table_row / image
          ("table_id", "VARCHAR", {"max_length": 128}),                     # 所属表标识：命中任一行即可据此取回同表其余行
          ("row_index", "INT16", {}),                                       # 表内行号，1 起；0 表示非表格行
          ("image_type", "VARCHAR", {"max_length": 32}),                    # ★图像语义类型（CLIP 判定）：组织结构图/饼图/流程图…
          ("image_path", "VARCHAR", {"max_length": 256}),                   # 图像文件相对路径，前端可回显原图
          ("locator", "VARCHAR", {"max_length": 256})]                      # 可读定位串，前端直接展示

_EMBED, _RERANK = None, None                                                # 向量模型、重排模型单例


def embed(texts: list):                                                     # 批量向量化
    """L2 归一化后内积等价余弦相似度，可直接用 Milvus 的 IP 度量。"""
    global _EMBED                                                           # 声明单例
    if _EMBED is None:                                                      # 首次调用才加载，避免拖慢服务启动
        from sentence_transformers import SentenceTransformer               # 延迟导入
        _EMBED = SentenceTransformer(C.EMBED_MODEL_PATH, device=C.DEVICE)   # 从本地路径加载，不联网
    v = _EMBED.encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False)  # 归一化编码
    return np.asarray(v, dtype=np.float32)                                  # Milvus 要求 float32


def milvus_client():                                                        # Milvus 客户端
    from pymilvus import MilvusClient                                       # 延迟导入
    return MilvusClient(uri=C.MILVUS_URI, timeout=10)                       # 10 秒超时，Milvus 挂了不把接口拖死


def recreate_collection(client, name: str = None) -> None:                   # 重建集合
    """删旧建新，保证重建索引时状态干净；显式声明字段便于按图像类型过滤与展示。"""
    from pymilvus import DataType                                           # 字段类型
    name = name or C.MILVUS_COLLECTION                                      # 默认集合名
    if client.has_collection(name): client.drop_collection(name)            # 已存在先删
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)  # 主键自增，字段固定
    schema.add_field("id", DataType.INT64, is_primary=True)                 # 自增主键
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=C.EMBED_DIM)      # bge-m3 向量
    for n, t, kw in FIELDS: schema.add_field(n, getattr(DataType, t), **kw)  # 业务字段
    idx = client.prepare_index_params()                                     # 索引参数
    idx.add_index(field_name="vector", index_type="IVF_FLAT", metric_type="IP", params={"nlist": 1024})  # 内积 + IVF_FLAT
    client.create_collection(collection_name=name, schema=schema, index_params=idx)  # 建集合


def insert_chunks(client, chunks: list, name: str = None, batch: int = 512) -> None:  # 向量入库
    """分批编码写入，避免一次性把全部文本读进显存。"""
    name = name or C.MILVUS_COLLECTION                                      # 默认集合
    for i in range(0, len(chunks), batch):                                  # 分批
        part = chunks[i:i + batch]                                          # 当前批
        vecs = embed([c["text"] for c in part])                             # 批量编码
        rows = [dict(vector=v.tolist(), doc=c.get("doc", ""), chunk_id=c["chunk_id"], text=c["text"],  # 组装行
                     page_no=int(c["page_no"]), page_end=int(c.get("page_end", c["page_no"])),          # 页码
                     block_type=c.get("block_type", "text"), image_type=c.get("image_type", ""),        # 类型
                     table_id=c.get("table_id", ""), row_index=int(c.get("row_index", 0)),              # 表格行定位
                     image_path=c.get("image_path", ""), locator=c.get("locator", ""))                  # 图像路径与定位
                for v, c in zip(vecs, part)]                                # 逐条对应
        client.insert(collection_name=name, data=rows)                      # 写入
    # 说明：pymilvus 的 MilvusClient 没有 flush；刚写入即可查靠检索时的 consistency_level="Strong"


def vector_search(client, question: str, top_k: int, name: str = None) -> list:  # 向量检索
    """问题编码后取最相近的 top_k；Milvus 不可用时返回空，由上层降级到 BM25。"""
    name = name or C.MILVUS_COLLECTION                                      # 默认集合
    if client is None: return []                                            # 未连接（不可用）→ 空手而归，交给 BM25 撑住
    # 编码与检索分开兜底：两者故障源不同，混在一个 except 里会把"模型路径错"报成"Milvus 不可用"，排障会被带偏
    try: q = embed([question])[0].tolist()                                  # 问题编码
    except Exception as e:                                                  # 模型路径错、显存不足
        print(f"[检索降级] 向量模型不可用：{type(e).__name__}: {e}"); return []  # 明确指向模型
    try:                                                                    # 检索阶段
        hits = client.search(collection_name=name, data=[q], limit=top_k,   # 单条查询向量
                             output_fields=[f for f, _, _ in FIELDS], consistency_level="Strong")  # 强一致读，规避刚写入读不到
    except Exception as e:                                                  # 连接失败或超时
        print(f"[检索降级] Milvus 检索失败：{type(e).__name__}: {e}"); return []  # 明确指向 Milvus
    return [dict(h["entity"], vec_score=float(h["distance"])) for h in (hits[0] if hits else [])]  # 附向量得分


def rerank(question: str, docs: list, top_n: int) -> list:                   # 交叉编码重排
    """(问题, 候选块) 成对送入 bge-reranker-v2-m3 打分取 top_n；模型不可用时按融合分兜底，保证仍有可用上下文。"""
    global _RERANK                                                          # 声明单例
    if not docs: return []                                                  # 无候选
    try:                                                                    # 重排失败不应让整个问答挂掉
        if _RERANK is None:                                                 # 首次加载
            from FlagEmbedding import FlagReranker                          # 延迟导入
            _RERANK = FlagReranker(C.RERANK_MODEL_PATH, use_fp16=(C.DEVICE == "cuda"))  # GPU 上启用半精度
        pairs = [[question, d["text"][:C.RERANK_MAX_CHARS]] for d in docs]   # 只截断打分输入；返回给大模型的仍是全文
        # batch_size 必须显式给：默认批大小对长文本极不友好，实测 60 条候选要 2704ms，
        # 显式 batch_size=16 后降到 591ms（4.6 倍），且打分结果不变。
        s = _RERANK.compute_score(pairs, normalize=True, batch_size=C.RERANK_BATCH)  # 批量打分归一化
        s = s if isinstance(s, list) else [s]                               # 单条时保证是列表
        for d, x in zip(docs, s): d["rerank_score"] = float(x)              # 写回分数
    except Exception as e:                                                  # 模型加载或推理失败
        print(f"[重排降级] {type(e).__name__}: {e}，改用融合分排序")            # 打印原因
        for d in docs: d["rerank_score"] = float(d.get("score", 0.0))       # 融合分顶替
    return sorted(docs, key=lambda d: d["rerank_score"], reverse=True)[:top_n]  # 降序取前 N
