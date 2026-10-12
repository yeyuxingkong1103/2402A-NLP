# 工单编号：人工智能NLP-RAG-Query理解优化
# 文件说明：向量层。bge-m3 本地编码 + Milvus 建集合/写入/检索，字段保留来源与页码以便生成引用。
from typing import List, Dict, Optional                                         # 类型标注

from . import config as C                                                       # 复用集中配置

# 集合字段定义：(字段名, Milvus 类型, 附加参数)；与 docs/设计文档.md 的字段表一一对应
_FIELDS = [("doc", "VARCHAR", {"max_length": 128}),                             # 来源文档名，两篇招股说明书靠它区分
           ("chunk_id", "VARCHAR", {"max_length": 64}),                         # 全局唯一块号
           ("text", "VARCHAR", {"max_length": 65535}),                          # 块文本（正文或整表文本）
           ("source", "VARCHAR", {"max_length": 128}),                          # 工单要求字段：来源
           ("section", "VARCHAR", {"max_length": 256}),                         # 所属章节，用于展示与过滤
           ("page_no", "INT16", {}),                                            # 起始页（1 起，与 PDF 页码一致）
           ("page_end", "INT16", {}),                                           # 结束页
           ("block_type", "VARCHAR", {"max_length": 16}),                       # text / table / image / chart
           ("locator", "VARCHAR", {"max_length": 256})]                         # 可读定位串，前端直接展示

_EMBED = None                                                                   # 向量模型单例


def embed_model():                                                              # 懒加载 bge-m3
    """首次调用加载并常驻显存，避免每个请求重复加载数百 MB 权重。"""
    global _EMBED                                                               # 声明单例
    if _EMBED is None:                                                          # 尚未加载
        from sentence_transformers import SentenceTransformer                   # 延迟导入，缩短服务启动时间
        _EMBED = SentenceTransformer(C.EMBED_MODEL_PATH, device=C.DEVICE)       # 从本地路径加载，不联网
    return _EMBED                                                               # 返回单例


def embed(texts: List[str]) -> "object":                                        # 批量向量化
    """L2 归一化后内积等价于余弦相似度，可直接用 Milvus 的 IP 度量。"""
    import numpy as np                                                          # 延迟导入
    vecs = embed_model().encode(texts, normalize_embeddings=True,               # 归一化编码
                                batch_size=C.EMBED_BATCH, show_progress_bar=False)  # 批大小来自配置
    return np.asarray(vecs, dtype=np.float32)                                   # 统一 float32，Milvus 要求


def milvus_client():                                                            # 建 Milvus 客户端
    """连接 Milvus；连接失败由上层捕获并给出友好提示，不让异常冒到接口外。"""
    import logging                                                              # 抑制 pymilvus 的 DEBUG 噪声
    logging.getLogger("pymilvus").setLevel(logging.WARNING)                     # 默认会逐条打印建连接/建集合，刷屏淹没自己的进度日志
    from pymilvus import MilvusClient                                           # 延迟导入
    return MilvusClient(uri=C.MILVUS_URI, timeout=C.MILVUS_TIMEOUT)             # 带超时，Milvus 挂了要快速失败


def recreate_collection(client, name: Optional[str] = None) -> None:            # 重建集合
    """删旧建新，保证重建索引时是干净状态；字段显式声明便于按来源过滤与展示引用。"""
    from pymilvus import DataType                                               # 延迟导入字段类型
    name = name or C.MILVUS_COLLECTION                                          # 默认使用配置的集合名
    if client.has_collection(name):                                             # 集合已存在
        client.drop_collection(name)                                            # 先删除，避免字段不一致
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)     # 主键自增，字段固定
    schema.add_field("id", DataType.INT64, is_primary=True)                     # 自增主键
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=C.EMBED_DIM)          # bge-m3 向量
    for fname, ftype, kw in _FIELDS:                                            # 业务字段逐一声明
        schema.add_field(fname, getattr(DataType, ftype), **kw)                 # 按类型动态添加
    index = client.prepare_index_params()                                       # 索引参数
    index.add_index(field_name="vector", index_type="IVF_FLAT",                 # 倒排文件索引，10 万级块量足够
                    metric_type="IP", params={"nlist": 1024})                   # 内积度量（向量已归一化）
    client.create_collection(collection_name=name, schema=schema, index_params=index)  # 建集合与索引


def insert_chunks(client, chunks: List[Dict], name: Optional[str] = None) -> int:   # 向量入库
    """分批编码并写入，避免一次性把全部文本读进显存；返回写入条数。

    每批打印一次进度：本机 8G 显存下全量入库需数分钟，没有进度输出就无法判断是"慢"还是"卡死"。
    （实测教训：曾因两个进程同时占显卡导致入库从 5 分钟拖到 13 分钟以上，且看不出在动。）
    """
    import time                                                                 # 计时
    name = name or C.MILVUS_COLLECTION                                          # 默认集合
    total = 0                                                                   # 计数
    batch = C.MILVUS_INSERT_BATCH                                               # 批大小
    t0 = time.time()                                                            # 起始时间
    for i in range(0, len(chunks), batch):                                      # 分批遍历
        part = chunks[i:i + batch]                                              # 取一批
        vecs = embed([c["text"] for c in part])                                 # 批量编码
        rows = []                                                               # 行数据
        for v, c in zip(vecs, part):                                            # 向量与块一一对应
            rows.append({"vector": v.tolist(),                                  # 向量
                         "doc": c.get("doc", ""),                               # 文档名
                         "chunk_id": c["chunk_id"],                             # 块号
                         "text": c["text"],                                     # 文本
                         "source": c.get("source", c.get("doc", "")),           # 来源
                         "section": (c.get("section", "") or "")[:250],         # 章节（截断到字段上限内）
                         "page_no": int(c.get("page_no", 0)),                   # 起始页
                         "page_end": int(c.get("page_end", c.get("page_no", 0))),  # 结束页
                         "block_type": c.get("block_type", "text"),             # 块类型
                         "locator": (c.get("locator", "") or "")[:250]})        # 定位串
        client.insert(collection_name=name, data=rows)                          # 写入集合
        total += len(rows)                                                      # 累加计数
        el = time.time() - t0                                                   # 已用时
        print(f"  [Milvus] {total}/{len(chunks)} 条，已用 {el:.0f}s，"
              f"预计剩余 {el / max(total, 1) * (len(chunks) - total):.0f}s", flush=True)  # 进度与预估
    # 坑：pymilvus 的 MilvusClient 没有 flush 方法；刚写入就查会读不到，
    #     所以检索侧一律显式指定 consistency_level="Strong"（见 vector_search）。
    return total                                                                # 返回写入条数


def vector_search(client, question: str, top_k: int,                    # 向量检索
                  name: Optional[str] = None) -> List[Dict]:                    # 返回值带出处
    """问题编码后取最相近的 top_k 个块；任一环节失败都返回空列表，由上层降级到 BM25。"""
    name = name or C.MILVUS_COLLECTION                                          # 默认集合
    if client is None:                                                          # Milvus 未连接
        print("[检索降级] Milvus 客户端不可用，向量通道返回空")                    # 明确报因
        return []                                                               # 让 BM25 撑住
    # 向量化与向量检索分开兜底：二者故障源完全不同，混在一个 except 里会把
    # "向量模型路径错"报成"Milvus 不可用"，排障时会被日志带偏（实测踩过）。
    try:                                                                        # 先做向量化
        qvec = embed([question])[0].tolist()                                    # 问题向量
    except Exception as exc:                                                    # 模型路径错、显存不足等
        print(f"[检索降级] 向量模型不可用：{type(exc).__name__}: {exc}")          # 明确指向模型
        return []                                                               # 返回空
    try:                                                                        # 再做检索
        hits = client.search(collection_name=name, data=[qvec], limit=top_k,    # 单条查询向量
                             output_fields=[f for f, _, _ in _FIELDS],          # 回带全部业务字段
                             consistency_level="Strong")                        # 强一致读，规避"刚写入读不到"
    except Exception as exc:                                                    # 连接失败或超时
        print(f"[检索降级] Milvus 检索失败：{type(exc).__name__}: {exc}")         # 明确指向 Milvus
        return []                                                               # 返回空
    out: List[Dict] = []                                                        # 结果容器
    for h in (hits[0] if hits else []):                                         # 展开第一组结果
        ent = dict(h.get("entity", {}))                                         # 实体字段
        ent["vec_score"] = float(h.get("distance", 0.0))                        # 附上向量得分
        out.append(ent)                                                         # 收集
    return out                                                                  # 返回


def collection_count(client, name: Optional[str] = None) -> int:                # 集合内条数
    """取集合统计行数；Milvus 未落盘时可能返回 0，故仅作展示用，不参与业务判断。"""
    try:                                                                        # 统计可能失败
        name = name or C.MILVUS_COLLECTION                                      # 默认集合
        if client is None or not client.has_collection(name):                   # 客户端缺失或集合不存在
            return 0                                                            # 返回 0
        st = client.get_collection_stats(collection_name=name)                  # 取统计
        return int(st.get("row_count", 0))                                      # 行数
    except Exception as exc:                                                    # 任何异常
        print(f"[Milvus] 统计行数失败：{type(exc).__name__}: {exc}")             # 记录
        return 0                                                                # 返回 0
