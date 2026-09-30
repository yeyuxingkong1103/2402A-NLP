"""检索编排：抽号 → 取条 → 编码 → 粗排 → 回填 → 精排 → 置顶去重。

存在的理由：七个步骤各有自己的测试，但"接起来"本身是一种会出错的东西——
顺序错了（先精排后回填）、或精确块被去重掉，单看每个模块都是绿的。
本模块只做接线，规则尽可能薄，能抽成纯函数的都抽出去（如 merge_blocks）。

本模块是检索层与生成层的分界：不导入任何 langchain 或 generation 的东西，
否则检索的单测会被迫拖上 LLM 依赖。

宽召回 + 窄输出：粗排拿 50 条，精排只留下 5 条给生成层。给模型的原文越多，
3b 模型越容易抓错重点；父块整条本身就不短。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 只导入 hybrid_search：集合名由 milvus/backfill 各自的默认值给出，
# 这里多导入 COLLECTION 是死代码；client 由调用方注入，也不需要 get_client
from app.db.milvus import hybrid_search
from app.ingest.embed import encode_texts
from app.retrieval.article_lookup import fetch_articles
from app.retrieval.backfill import backfill_parents
from app.retrieval.query_parse import extract_article_nos
from app.retrieval.rerank import RERANK_OUTPUT_TOPK, top_blocks

# 技术方案 5.2 的起始值：两路各 top50，融合后取 50
RECALL_TOPK = 50


@dataclass
class RetrievalResult:
    """检索层的全部产出。

    blocks 是给生成层与呈现用的父块（已排序、已去重）；
    chunks 是召回的原始子块，只有校验层的第③关需要——它要靠子块的
    paragraph_no / item_no 判断"第三款"是不是真的存在。
    """
    question: str
    blocks: list[dict] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)
    exact_nos: list[int] = field(default_factory=list)
    # 粗排 + 回填后的父块全集（未精排），供评测算 Recall@k
    recalled_blocks: list[dict] = field(default_factory=list)


def merge_blocks(exact_blocks: list[dict], ranked_blocks: list[dict],
                 top_k: int = RERANK_OUTPUT_TOPK) -> list[dict]:
    """精确块置顶，再拼精排结果，按 article_no 去重（精确块优先）。

    精确块优先保留的理由：它带 MySQL 原文与效力状态，校验层的第②关
    直接读它，不必为同一条再查一次库。

    只读 block["article_no"]：两条路的块键集不对称（精确块 14 键、向量块 12 键），
    精确块独有 law_id/status，向量块的 chunk_id 是真 id 而精确块那份是 None，
    拿这些键判效力或做去重都会在另一条路上对不上号。

    去重按 article_no 而非 chunk_id：精确块没有 Milvus 块 id（chunk_id 为 None），
    按它去会漏掉同一条的向量副本，表现是同一法条在提示词里出现两遍。
    """
    merged: list[dict] = []
    seen: set[int] = set()
    for block in list(exact_blocks) + list(ranked_blocks):
        no = block["article_no"]
        if no in seen:
            continue
        seen.add(no)
        merged.append(block)
    return merged[:top_k]


def retrieve(question: str, *, conn, client, encoder, reranker,
             top_k: int = RERANK_OUTPUT_TOPK) -> RetrievalResult:
    """一条问句走完全部检索步骤。模型与连接都由调用方注入，便于测试与复用。"""
    extracted_nos = extract_article_nos(question)
    exact_blocks = fetch_articles(conn, extracted_nos) if extracted_nos else []
    # 编码一次拿双向量：bge-m3 同一次前向同时出 dense 与 sparse（②期 4.3）
    dense_vec, sparse_vec = encode_texts(encoder, [question])[0]
    chunks = hybrid_search(client, dense_vec, sparse_vec,
                           top_k=RECALL_TOPK, status="现行有效",
                           child_only=True)
    recalled = backfill_parents(client, chunks)
    ranked = top_blocks(question, recalled, reranker.predict)
    # exact_nos 只上报真的取到的条号（fetch_articles 已跳过库里没有的条）。
    # 不用 extract_article_nos 的解析结果：问"第9999条"时库里没有该条，
    # 上报 [9999] 会让下游以为已精确置顶，而 blocks 里根本没有它，
    # 表现为答案与引用对不上号——这正是设计文档 4.1 的边界
    exact_nos = [block["article_no"] for block in exact_blocks]
    return RetrievalResult(question=question,
                           blocks=merge_blocks(exact_blocks, ranked, top_k),
                           chunks=chunks, exact_nos=exact_nos,
                           recalled_blocks=recalled)
