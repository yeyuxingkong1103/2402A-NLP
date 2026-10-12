# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""检索模块：bge-m3 向量检索、BM25 关键词检索、加权融合、bge-reranker-v2-m3 交叉重排。"""
import jieba, numpy as np                                                                        # 中文分词、向量运算与排序
from . import config as C                                                                        # 复用集中配置

_MODELS = {}                                                                                     # 模型单例表，避免重复加载


def _model(kind: str):                                                                           # 懒加载本地模型
    """按需加载并缓存 bge-m3 或 bge-reranker-v2-m3，首次加载后常驻显存。"""
    if kind not in _MODELS:                                                                      # 尚未加载才加载
        if kind == "embed":                                                                      # 向量模型
            from sentence_transformers import SentenceTransformer                                 # 延迟导入，缩短启动时间
            _MODELS[kind] = SentenceTransformer(C.EMBED_MODEL_PATH, device=C.DEVICE)              # 从本地路径加载，不联网
        else:                                                                                    # 重排模型
            from FlagEmbedding import FlagReranker                                                # 延迟导入
            _MODELS[kind] = FlagReranker(C.RERANK_MODEL_PATH, use_fp16=(C.DEVICE == "cuda"))       # GPU 上启用半精度加速
    return _MODELS[kind]                                                                         # 返回单例


def embed(texts: list) -> np.ndarray:                                                            # 文本批量向量化
    """批量编码并做 L2 归一化，使内积等价于余弦相似度，可直接用 IP 度量检索。"""
    vecs = _model("embed").encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False)  # 归一化编码并关闭进度条
    return np.asarray(vecs, dtype=np.float32)                                                    # 统一 float32，Milvus 要求


def rerank(query: str, docs: list, top_n: int) -> list:                                           # 交叉编码重排
    """把 (问题, 候选块) 成对送入重排模型打分，返回得分最高的 top_n 个块。"""
    if not docs: return []                                                                        # 没有候选直接返回空
    scores = _model("rerank").compute_score([[query, d["text"]] for d in docs], normalize=True)    # 批量打分并归一化到 0~1
    scores = scores if isinstance(scores, list) else [scores]                                      # 单条结果时保证是列表
    for doc, score in zip(docs, scores): doc["rerank_score"] = float(score)                        # 把重排得分写回候选块
    return sorted(docs, key=lambda d: d["rerank_score"], reverse=True)[:top_n]                     # 按得分降序取前 N 条


def milvus_client():                                                                             # Milvus 客户端
    """连接 Milvus，用于存取 bge-m3 向量。"""
    from pymilvus import MilvusClient                                                             # 延迟导入，加快启动
    return MilvusClient(uri=C.MILVUS_URI)                                                        # 返回客户端


def recreate_collection(client, name: str) -> None:                                              # 重建集合
    """删除同名集合后新建，保证每次重建索引都是干净状态，不残留旧向量。"""
    if client.has_collection(name): client.drop_collection(name)                                  # 集合已存在则先删除
    client.create_collection(collection_name=name, dimension=C.EMBED_DIM, metric_type="IP", auto_id=True, enable_dynamic_field=True)  # 内积度量 + 主键自增 + 允许业务字段


def insert_chunks(client, name: str, chunks: list, batch: int = 512) -> None:                      # 向量入库
    """分批编码并写入 Milvus，避免一次性占用过多显存与内存。"""
    for i in range(0, len(chunks), batch):                                                        # 按批遍历全部块
        part = chunks[i:i + batch]                                                                # 取出一批
        rows = [dict(vector=v.tolist(), text=c["text"], page_no=c["page_no"], chunk_id=c["chunk_id"]) for v, c in zip(embed([c["text"] for c in part]), part)]  # 编码并组装行数据
        client.insert(collection_name=name, data=rows)                                             # 写入集合
    # 说明：pymilvus 2.4.9 的 MilvusClient 没有 flush 方法，即时可查靠检索时的 Strong 一致性保证。


def vector_search(client, name: str, question: str, top_k: int) -> list:                           # 向量检索
    """把问题编码后检索最相近的 top_k 个块，用 Strong 一致读规避"刚写入读不到"的问题。"""
    try:                                                                                          # Milvus 不可用时降级而不抛栈
        hits = client.search(collection_name=name, data=[embed([question])[0].tolist()], limit=top_k, output_fields=["text", "page_no", "chunk_id"], consistency_level="Strong")  # 强一致读
    except Exception as exc:                                                                      # 连接失败或超时
        print(f"[检索降级] Milvus 不可用：{type(exc).__name__}: {exc}"); return []                  # 打印原因并返回空，由上层给友好提示
    return [dict(text=h["entity"]["text"], page_no=h["entity"]["page_no"], chunk_id=h["entity"]["chunk_id"], vec_score=float(h["distance"])) for h in (hits[0] if hits else [])]  # 整理成统一结构


def tokenize(text: str) -> list:                                                                   # BM25 分词器
    """中文用 jieba 分词，英文与数字保留原样，共同构成检索词。"""
    return [t.lower() for t in jieba.lcut(text) if t.strip()]                                      # 分词并统一小写


def bm25_search(bm25, chunks: list, question: str, top_k: int) -> list:                             # 关键词检索
    """按 BM25 得分取 top_k，善于命中数字、比例、专有名词等向量检索易漏的精确串。"""
    if bm25 is None: return []                                                                      # 索引未就绪时返回空
    scores = np.asarray(bm25.get_scores(tokenize(question)))                                         # 计算每个块的 BM25 得分
    return [dict(text=chunks[i]["text"], page_no=chunks[i]["page_no"], chunk_id=chunks[i]["chunk_id"], bm25_score=float(scores[i])) for i in np.argsort(scores)[::-1][:top_k] if scores[i] > 0]  # 降序取前 K 个非零分块


def build_bm25(chunks: list):                                                                       # 构建 BM25 索引
    """用全部块文本建 BM25，索引常驻内存，单次检索耗时在毫秒级。"""
    from rank_bm25 import BM25Okapi                                                                 # 延迟导入
    return BM25Okapi([tokenize(c["text"]) for c in chunks])                                          # 逐块分词后建索引


def _norm(values: list) -> list:                                                                    # 分数归一化
    """把一组分数线性映射到 0~1，使向量分与 BM25 分可以加权相加。"""
    lo, hi = (min(values), max(values)) if len(values) else (0.0, 0.0)                               # 取最小最大值
    return [(v - lo) / (hi - lo) for v in values] if hi > lo else [0.0] * len(values)                # 全部相等时统一给 0


def fuse(vec_hits: list, bm25_hits: list) -> list:                                                  # 两路结果加权融合
    """向量与 BM25 分数各自归一化后按权重相加，同一块命中两路则分数叠加。"""
    merged = {}                                                                                     # chunk_id → 候选块
    for key, hits, weight in (("vec_score", vec_hits, C.VEC_WEIGHT), ("bm25_score", bm25_hits, C.BM25_WEIGHT)):  # 两个通道及各自权重
        for hit, score in zip(hits, _norm([h[key] for h in hits])):                                  # 逐一归一化后累加
            merged.setdefault(hit["chunk_id"], dict(text=hit["text"], page_no=hit["page_no"], score=0.0))["score"] += weight * score  # 命中两路时分值叠加
    return sorted(merged.values(), key=lambda d: d["score"], reverse=True)[:C.FUSION_TOP_K]           # 降序取待重排候选


def hybrid_search(client, collection: str, bm25, chunks: list, question: str) -> list:               # 混合检索总入口
    """向量召回 + BM25 召回 → 加权融合，返回待重排的候选块。"""
    vec_hits = vector_search(client, collection, question, C.VEC_TOP_K)                              # 向量通道召回
    bm_hits = bm25_search(bm25, chunks, question, C.BM25_TOP_K)                                       # 关键词通道召回
    if not vec_hits and not bm_hits: raise RuntimeError("检索服务暂时不可用，请稍后重试")                # 两路都空说明检索不可用
    return fuse(vec_hits, bm_hits)                                                                   # 返回融合结果
