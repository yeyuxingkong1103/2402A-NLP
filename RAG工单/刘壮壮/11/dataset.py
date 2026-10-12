# -*- coding: utf-8 -*-
"""
dataset.py — 微调数据集生成
工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务

从招股说明书生成嵌入微调训练数据：
  格式：正例对 / 三元组（anchor, positive, negative）
  - anchor（查询）：从文本块中抽取的关键句/实体短语
  - positive（正例）：anchor 所在的原文块
  - negative（负例）：BM25 召回的同域相似但非正例的块（hard negative）

无 LLM 时用「实体/关键短语作为查询」；配置 LLM 后可生成更自然的问题（问答对生成）。
"""
from __future__ import annotations

import os
import re
from typing import List, Dict, Tuple

import rag_core


def extract_query_from_chunk(chunk: str) -> str:
    """启发式生成查询：抽取包含实体/数字的关键句作为 anchor。"""
    sentences = [s.strip() for s in re.split(r"[。\n]", chunk) if s.strip()]
    # 优先含数字或公司名/术语的句子
    for s in sentences:
        if re.search(r"\d", s) and len(s) > 10:
            return s[:80]
    for s in sentences:
        if len(s) > 12:
            return s[:80]
    return chunk[:80]


def generate_queries_with_llm(chunk: str, llm) -> str:
    """问答对生成：用 LLM 为文本块生成一个自然查询。"""
    prompt = f"请为下面这段金融文档内容生成一个简短的检索问题（20 字以内）：\n{chunk[:500]}\n问题："
    try:
        q = llm.generate(prompt).strip()
        if q:
            return q
    except Exception:
        pass
    return extract_query_from_chunk(chunk)


def build_dataset(pdf_paths: List[str], config, use_llm: bool = False,
                  per_doc: int = 300) -> List[Dict[str, str]]:
    """生成训练样本。返回 [{"anchor","positive","negative"}...]"""
    from rag_core import BM25Index, split_text, load_pdf_text

    chunks = []
    for p in pdf_paths:
        text = load_pdf_text(p)
        chunks.extend(split_text(text, config.chunk_size, config.chunk_overlap))
    # 每份文档采样，控制规模
    if len(chunks) > per_doc:
        step = len(chunks) // per_doc
        chunks = chunks[::step][:per_doc]

    bm25 = BM25Index(chunks)
    llm = rag_core.make_llm(config) if use_llm else None

    samples = []
    for i, c in enumerate(chunks):
        anchor = generate_queries_with_llm(c, llm) if llm else extract_query_from_chunk(c)
        # 负例：BM25 检索与 anchor 相似但非本块的内容
        negs = [h["index"] for h in bm25.search(anchor, top_k=5) if h["index"] != i]
        if not negs:
            continue
        samples.append({
            "anchor": anchor,
            "positive": c,
            "negative": chunks[negs[0]],
        })
    return samples


def to_triplets(samples: List[Dict[str, str]]) -> List[Tuple[str, str, str]]:
    """转为 (anchor, positive, negative) 三元组。"""
    return [(s["anchor"], s["positive"], s["negative"]) for s in samples]


def to_pairs(samples: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    """转为正例对（用于 MultipleNegativesRankingLoss 需额外打包负例，这里只给正例）。"""
    return [(s["anchor"], s["positive"]) for s in samples]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rag_core import Config
    cfg = Config()
    paths = ["../附件/招股说明书1-无水印.pdf", "../附件/招股说明书2.pdf"]
    data = build_dataset([p for p in paths if os.path.exists(p)], cfg, use_llm=False, per_doc=100)
    print(f"生成 {len(data)} 条训练样本")
    for s in data[:3]:
        print("anchor:", s["anchor"][:50])
        print("positive:", s["positive"][:50])
        print("negative:", s["negative"][:50])
        print("-" * 50)
