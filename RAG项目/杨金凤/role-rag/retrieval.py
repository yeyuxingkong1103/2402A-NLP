"""检索链路（混合检索）：向量 + BM25 双路召回 → RRF 融合 → BGE-rerank 精排 → 父块替换收尾。

职责
----
对外只暴露 retrieve(query, collection_name) 一个入口，完成「query → 命中片段列表」的完整检索链路，
供上层 RAG 组装 prompt 时调用；其余函数为链路各环节的私有实现。

依赖
----
- ingest：向量模型（load_embedder）、BM25 索引（load_keyword_index）、集合名（COLLECTION / get_collection）
- milvus_store：向量库开关（USE_MILVUS）与 Milvus 检索实现（search）
- sentence_transformers.CrossEncoder：BGE-rerank 精排（懒加载）

输入输出契约
------------
- 输入：query（用户问句，str）、collection_name（集合名，默认 ingest.COLLECTION）
- 输出：list[dict]，每个 dict 至少含 content / page / similarity 三字段；
  similarity 语义见 retrieve 的 docstring（rerank 成功 = sigmoid(logits)，降级 = 1-cosine）。
"""
import logging
import math
import os
import time
from functools import lru_cache

from dotenv import load_dotenv

from ingest import COLLECTION, get_collection, load_embedder, load_keyword_index
from milvus_store import USE_MILVUS, search as milvus_search

load_dotenv()

logger = logging.getLogger(__name__)

# 检索常量：向量召回 RECALL_K 条，再 BGE-rerank 精排取 RERANK_TOP_K 条
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RECALL_K = int(os.getenv("RECALL_K", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "4"))
RRF_K = int(os.getenv("RRF_K", "60"))  # RRF 融合的平滑常数 k
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))  # 余弦相似度过滤阈值


def _sigmoid(x) -> float:
    """把 CrossEncoder 输出的 logits 映射到 (0,1)，作为可解释的相似度分数。

    Args:
        x: reranker 输出的原始 logits（可正可负，也可能是 numpy 标量）。

    Returns:
        sigmoid(x) = 1 / (1 + e^{-x})，恒落在 (0,1)。

    Raises:
        无（纯数学运算）。

    为什么用 sigmoid：reranker 的 logits 没有上界、非 [0,1]，直接展示不直观；
    sigmoid 单调且压缩到 (0,1)，便于统一阈值与展示。显式 float(x) 规避 numpy 标量参与 math.exp 的类型问题。
    """
    return 1.0 / (1.0 + math.exp(-float(x)))


def _tokenize(text: str) -> list[str]:
    """用 jieba 对查询分词，供 BM25 打分。

    Args:
        text: 待分词的字符串（通常是 query）。

    Returns:
        分词后的 token 列表。

    Raises:
        无。

    延迟 import 是为了避免进程启动即加载 jieba 词典，加快启动。
    """
    import jieba

    jieba.setLogLevel(logging.WARNING)  # 压掉 jieba 首次建词典的 DEBUG 噪音
    return jieba.lcut(text)


def _rrf_fuse(vec_hits: list[dict], bm25_hits: list[dict], k: int = RRF_K):
    """用 RRF（Reciprocal Rank Fusion，倒数排名融合）融合向量 / BM25 两路召回。

    Args:
        vec_hits: 向量召回的命中，含 id/content/page/similarity 字段。
        bm25_hits: BM25 召回的命中，含 id/content/page/parent_content 字段。
        k: RRF 平滑常数（默认 RRF_K=60），防止 1/rank 在 rank=1 时过大、放大单路权重。

    Returns:
        (candidates, source)：
        - candidates：融合后按分数降序取 top-RECALL_K，每项含 content/page/parent_content/
          similarity（已归一化到 [0,1]）。
        - source：top1 命中的来源标签（"双路命中"/"仅向量"/"仅BM25"/"无"），用于观测两路贡献。

    Raises:
        无。

    RRF 公式：对每个候选 chunk，score = Σ_{路} 1 / (k + rank_路)。
    某 chunk 在某路排名越靠前（rank 越小），该项贡献越大；两路都命中则相加，天然融合了
    语义（向量）与关键词（BM25）两个维度。
    """
    # 分别记录每个 id 在向量路 / BM25 路的排名（1 起，rank=1 表示该路第 1 名）
    vec_rank = {h["id"]: r for r, h in enumerate(vec_hits, start=1)}
    bm25_rank = {h["id"]: r for r, h in enumerate(bm25_hits, start=1)}
    merged = {}
    for h in vec_hits + bm25_hits:
        # 合并两路命中并去重；parent_content 用于后续父块替换，缺省给空串
        merged.setdefault(h["id"], {
            "content": h["content"], "page": h["page"],
            "parent_content": h.get("parent_content", ""),
        })

    scores = {}
    for cid in merged:
        s = 0.0
        # RRF 累加：仅在该路命中时加上 1/(k+rank)，两路都命中则两项相加
        if cid in vec_rank:
            s += 1.0 / (k + vec_rank[cid])
        if cid in bm25_rank:
            s += 1.0 / (k + bm25_rank[cid])
        scores[cid] = s

    # 按融合分降序，取前 RECALL_K 作为进入 rerank 的候选
    order = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:RECALL_K]
    # 归一化上限：两路都排第 1 的理论最大分 = 2/(k+1)，用它把分数映射到 [0,1] 便于比较与过滤
    max_score = 2.0 / (k + 1)
    candidates = []
    for cid in order:
        c = dict(merged[cid])
        c["similarity"] = round(min(scores[cid] / max_score, 1.0), 4)
        candidates.append(c)

    # 判断 top1 来源：同时命中两路记「双路命中」，否则记命中的那一路
    top1 = order[0] if order else None
    if top1 is None:
        source = "无"
    elif top1 in vec_rank and top1 in bm25_rank:
        source = "双路命中"
    elif top1 in vec_rank:
        source = "仅向量"
    else:
        source = "仅BM25"
    return candidates, source


def _present(sources: list[dict]) -> list[dict]:
    """收尾加工：有 parent_content 的命中用父块替换 content，原子块存 chunk_content 供调试。

    Args:
        sources: rerank 精排后的命中列表（每个含 content/page/similarity/parent_content）。

    Returns:
        格式化后的命中列表：有父块的含 content(父块)/chunk_content(子块)/page/similarity；
        无父块的（Chroma / 表格 / 图片）保持 content/page/similarity 契约不变。

    Raises:
        无。

    替换发生在 rerank 之后：rerank 仍在 400 字子块上打分（子块语义更聚焦），
    但最终喂给 LLM 的是上下文更完整的父块。
    """
    out = []
    for s in sources:
        parent = s.get("parent_content", "")
        if parent:
            out.append({
                "content": parent,
                "chunk_content": s.get("content", ""),
                "page": s["page"],
                "similarity": s["similarity"],
            })
        else:
            out.append({
                "content": s.get("content", ""),
                "page": s["page"],
                "similarity": s["similarity"],
            })
    return out


@lru_cache(maxsize=1)
def load_reranker():
    """懒加载 BGE-reranker（CrossEncoder），进程内单例。

    Returns:
        CrossEncoder(RERANK_MODEL) 实例。

    Raises:
        可能因模型加载失败抛异常（由调用方 retrieve 捕获降级）。

    @lru_cache(maxsize=1)：多次调用只加载一次模型，避免重复加载显存/内存开销。
    延迟 import 保证 HF_ENDPOINT 等环境变量先生效。
    """
    from sentence_transformers import CrossEncoder

    logger.info("加载 reranker 模型：%s ...", RERANK_MODEL)
    return CrossEncoder(RERANK_MODEL)


def retrieve(query: str, collection_name: str = COLLECTION) -> list[dict]:
    """混合检索入口：向量 + BM25 双路召回 → RRF 融合 → BGE-rerank 精排取 top-RERANK_TOP_K。

    Args:
        query: 用户问句。
        collection_name: 目标集合名，默认 ingest.COLLECTION。

    Returns:
        list[dict]，每个元素含 content / page / similarity（可能还有 chunk_content）。
        similarity 双语义（重要）：
        - rerank 成功：sigmoid(logits)，取值 (0,1)；
        - rerank 失败降级：保留 RRF 归一化分数；纯向量降级时为 1-cosine 余弦相似度。

    Raises:
        不向外抛 reranker 相关异常（内部降级），但向量库 / 索引加载异常会向上传播。

    降级路径：
        1) BM25 索引不可用 → 跳过融合，直接用向量召回结果作候选；
        2) reranker 加载或预测失败 → 返回 RRF/向量候选的 top-RERANK_TOP_K（不精排）。
    """
    embedder = load_embedder()
    q_vec = embedder.encode([query], normalize_embeddings=True).tolist()
    # 向量召回 RECALL_K 条：按 USE_MILVUS 切换 Milvus / Chroma 后端
    if USE_MILVUS:
        vec_hits = milvus_search(collection_name, q_vec[0], RECALL_K)
    else:
        # Chroma 分支：query 返回 ids/documents/metadatas/distances 四个并列列表
        res = get_collection(collection_name).query(
            query_embeddings=q_vec,
            n_results=RECALL_K,
            include=["documents", "metadatas", "distances"],  # ids 恒被返回，无需 include
        )
        vec_hits = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            vec_hits.append({
                "id": cid,
                "content": doc,
                "page": int(meta.get("page", 0)),
                "similarity": round(1 - dist, 4),  # 余弦距离 -> 相似度（纯向量降级时用）
            })
    logger.info("向量召回 %d 条", len(vec_hits))

    # 余弦相似度阈值过滤：低于阈值的无关片段不进 RRF/rerank，节省资源、减少稀释
    filtered = [h for h in vec_hits if h["similarity"] >= SIMILARITY_THRESHOLD]
    if filtered:
        logger.info("相似度阈值 %.2f 过滤：%d -> %d 条", SIMILARITY_THRESHOLD, len(vec_hits), len(filtered))
        vec_hits = filtered
    else:
        # 兜底：全部被过滤时保留原始召回，避免无结果
        logger.warning("向量召回 %d 条全部低于相似度阈值 %.2f，保留原始召回兜底", len(vec_hits), SIMILARITY_THRESHOLD)

    # 加载 BM25 索引，可用则双路召回 + RRF 融合
    bm25, chunks = load_keyword_index(collection_name)
    if bm25 is not None and chunks:
        bm_scores = bm25.get_scores(_tokenize(query))
        top_idx = sorted(
            range(len(bm_scores)), key=lambda i: bm_scores[i], reverse=True
        )[:RECALL_K]
        bm25_hits = [
            {
                "id": chunks[i]["id"], "content": chunks[i]["content"], "page": chunks[i]["page"],
                "parent_content": chunks[i].get("parent_content", ""),
            }
            for i in top_idx
        ]
        logger.info("BM25 召回 %d 条", len(bm25_hits))
        candidates, top1_source = _rrf_fuse(vec_hits, bm25_hits)
        logger.info("RRF 融合后 %d 条", len(candidates))
        logger.info("RRF 后 top1 来源：%s", top1_source)
    else:
        # 降级路径 1：BM25 不可用 → 直接用向量召回结果作候选
        candidates = [
            {
                "content": h["content"], "page": h["page"], "similarity": h["similarity"],
                "parent_content": h.get("parent_content", ""),
            }
            for h in vec_hits
        ]

    candidates = candidates[:RECALL_K]

    try:
        start = time.perf_counter()
        reranker = load_reranker()
        # 构造 (query, 候选内容) 对，交给 CrossEncoder 打分
        pairs = [[query, c["content"]] for c in candidates]
        scores = reranker.predict(pairs)
        # 按 rerank 分数降序排序
        ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
        sources = []
        for c, s in ranked[:RERANK_TOP_K]:
            c = dict(c)
            # sigmoid 归一化 logits 到 (0,1)，作为最终 similarity
            c["similarity"] = round(_sigmoid(s), 4)
            sources.append(c)
        elapsed = time.perf_counter() - start
        top1 = sources[0]["similarity"] if sources else 0.0
        logger.info("rerank 耗时 %.2fs，精排后 top1 分数 %s", elapsed, top1)
        return _present(sources)
    except Exception as e:  # noqa: BLE001
        # 降级路径 2：reranker 不可用 → 直接返回候选 top-RERANK_TOP_K，similarity 保留融合/余弦分数
        logger.warning("reranker 不可用，回退混合召回 top-%d: %s", RERANK_TOP_K, e)
        return _present(candidates[:RERANK_TOP_K])
"""
收到检索查询，首先加载向量嵌入模型，把查询文本转为向量，调用向量库执行向量召回，拿到指定数量的候选片段；接着根据预设相似度阈值过滤相关性过低的结果，如果所有召回片段都低于阈值，则保留全部结果作为兜底，避免检索无输出。之后尝试加载 BM25 关键词索引，索引可用时，使用 jieba 对查询分词并执行关键词召回，向量召回结果和 BM25 召回结果两路通过 RRF 倒数排名融合，合并、去重并重新计算融合分数，得到新的候选块集合；如果 BM25 索引加载失败或者索引为空，则走降级逻辑，直接把向量召回的结果作为候选块。得到候选块后，懒加载 BGE-rerank 重排模型，将查询与每个候选文本组成输入对送入模型打分，再通过 sigmoid 函数把模型原始 logits 转换为 0 到 1 之间的相似度，按分数从高到低排序，截取排名靠前的指定数量片段；如果重排模型加载或者推理时报错，则跳过精排，直接从候选块里截取指定条数降级返回。精排完成后执行父块替换处理，用完整父文本替换参与打分的原子子块，同时保存原始子块内容用于调试，最后统一整理输出字段，返回包含文本内容、页码、相似度（调试用子块内容）的检索片段列表。

接收提问 → 急症校验 → 读取会话历史 → 改写查询 →【检索子流程：查询向量化 + 向量召回 → 相似度过滤兜底 → BM25 关键词召回，双路 RRF 融合；BM25 不可用则降级只用向量 → BGE-rerank 精排、sigmoid 转相似度，rerank 失败跳过精排 → 父块替换 → 返回检索片段】→ 拼接人设、资料、历史送入大模型 → 清洗结果 + 追加免责声明 → 返回答案 → 对话存入 Redis → 定期异步抽取长期记忆
"""