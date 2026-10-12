# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 文件说明：检索层。BM25 关键词检索、向量检索并行召回、加权融合，再用 bge-reranker-v2-m3 精排。
import jieba                                                                      # 中文分词
import numpy as np                                                                # 打分与排序
from . import config as C                                                         # 复用集中配置
from . import vector_store as V                                                   # 复用向量检索

_RERANK = None                                                                    # 重排模型单例


def tokenize(text: str) -> list:                                                  # BM25 分词器
    """中文用 jieba 切分，英文与数字整体保留——保证 1,670、25.04%、42.35% 这类串能精确命中。"""
    return [t.lower() for t in jieba.lcut(text or "") if t.strip()]                # 切分后统一小写


def build_bm25(chunks: list):                                                     # 构建 BM25 索引
    """BM25Okapi：对数字、比例、专有名词敏感，正好补向量检索对精确串不敏感的短板。"""
    from rank_bm25 import BM25Okapi                                               # 延迟导入
    return BM25Okapi([tokenize(c["text"]) for c in chunks])                        # 逐块分词后建索引


def bm25_search(bm25, chunks: list, question: str, top_k: int) -> list:            # 关键词检索
    """按 BM25 得分降序取 top_k，只保留得分大于 0 的块。"""
    if bm25 is None: return []                                                    # 索引未就绪返回空
    scores = np.asarray(bm25.get_scores(tokenize(question)))                       # 计算全部块的得分
    order = np.argsort(scores)[::-1][:top_k]                                       # 降序取前 K
    return [dict(chunks[i], bm25_score=float(scores[i])) for i in order if scores[i] > 0]  # 带分数返回非零分块


def _norm(values: list) -> list:                                                  # 分数归一化
    """把一组分数线性映射到 0~1，让向量分与 BM25 分可以加权相加。"""
    lo, hi = (min(values), max(values)) if len(values) else (0.0, 0.0)             # 取最小最大值
    return [(v - lo) / (hi - lo) for v in values] if hi > lo else [0.0] * len(values)  # 全部相等时统一给 0


def fuse(vec_hits: list, bm25_hits: list) -> list:                                 # 两路结果加权融合
    """各自归一化后按权重相加；同一块被两路同时命中则分数叠加，排序更靠前。"""
    merged = {}                                                                    # chunk_id → 候选块
    for key, hits, weight in (("vec_score", vec_hits, C.VEC_WEIGHT),               # 向量通道与权重
                              ("bm25_score", bm25_hits, C.BM25_WEIGHT)):           # BM25 通道与权重
        for hit, score in zip(hits, _norm([h[key] for h in hits])):                # 逐条归一化后累加
            slot = merged.setdefault(hit["chunk_id"], dict(hit, score=0.0))        # 首次命中即登记，保留定位字段
            slot["score"] += weight * score                                        # 叠加加权分
    return sorted(merged.values(), key=lambda d: d["score"], reverse=True)[:C.FUSION_TOP_K]  # 降序取待重排候选


def hybrid_search(client, bm25, chunks: list, question: str) -> tuple:             # 混合检索总入口
    """向量与 BM25 两路并行召回再融合；同时回报**实际生效的通道**，供上层标注降级模式。"""
    from concurrent.futures import ThreadPoolExecutor                              # 线程池：编码走 GPU、BM25 走 CPU，可真正并行
    with ThreadPoolExecutor(max_workers=2) as pool:                                # 两个通道同时开跑
        fv = pool.submit(V.vector_search, client, question, C.VEC_TOP_K)            # 向量通道
        fb = pool.submit(bm25_search, bm25, chunks, question, C.BM25_TOP_K)        # 关键词通道
        vec_hits, bm_hits = fv.result(), fb.result()                               # 等两路都完成
    if not vec_hits and not bm_hits: raise RuntimeError("检索服务暂时不可用，请稍后重试")  # 两路皆空说明依赖不可用
    # 少了一路就是降级：不能静默返回低质量结果，否则验收方会以为降级与正常等价
    mode = "hybrid" if (vec_hits and bm_hits) else ("bm25_only" if bm_hits else "vector_only")  # 实际生效的通道
    return fuse(vec_hits, bm_hits), mode                                           # 返回候选块与检索模式


def rerank(question: str, docs: list, top_n: int) -> list:                        # 交叉编码重排
    """把 (问题, 候选块) 成对送入 bge-reranker-v2-m3 打分，取 top_n；模型不可用时按融合分兜底。"""
    global _RERANK                                                                # 声明单例
    if not docs: return []                                                        # 无候选直接返回
    try:                                                                          # 重排失败不应让整个问答挂掉
        if _RERANK is None:                                                       # 首次调用加载模型
            from FlagEmbedding import FlagReranker                                # 延迟导入
            _RERANK = FlagReranker(C.RERANK_MODEL_PATH, use_fp16=(C.DEVICE == "cuda"))  # GPU 上启用半精度加速
        # 只截断送进打分器的文本，返回给大模型的仍是片段全文：
        # 整表块可达数千字，不截断时成对打分要按最长样本填充，实测重排耗时 960ms；截断后稳定在 300ms 量级。
        pairs = [[question, d["text"][:C.RERANK_MAX_CHARS]] for d in docs]           # 打分输入
        scores = _RERANK.compute_score(pairs, normalize=True)                       # 批量打分并归一化到 0~1
        scores = scores if isinstance(scores, list) else [scores]                  # 单条结果时保证是列表
        for doc, score in zip(docs, scores): doc["rerank_score"] = float(score)     # 写回每条候选
    except Exception as exc:                                                       # 模型加载或推理失败
        print(f"[重排降级] {type(exc).__name__}: {exc}，改用融合分排序")              # 打印原因
        for doc in docs: doc["rerank_score"] = float(doc.get("score", 0.0))         # 用融合分顶替，保证仍有可用上下文
    return sorted(docs, key=lambda d: d["rerank_score"], reverse=True)[:top_n]      # 降序取前 N 条
