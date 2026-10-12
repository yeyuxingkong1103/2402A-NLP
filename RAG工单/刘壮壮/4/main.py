# -*- coding: utf-8 -*-
"""
main.py — PDF 文档的图像内容解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

在工单 01/02/03 基础上，新增《招股说明书2.pdf》的图像内容解析：
  - 提取 PDF 中的图像
  - 用多模态大模型 / CLIP 对图像做语义解析，生成文本描述
  - 图像描述与正文一起入向量库，实现图像类问题（组织结构图、统计图）的检索

运行：
  python main.py                    # 全部问题检索
  python main.py --query "xxx"      # 单问题
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_core import (Config, VectorStore, BM25Index, load_pdf_text, split_text,
                      make_embedder, make_llm, answer, HybridRetriever)  # noqa: E402
from table_parser import parse_tables  # noqa: E402
from image_parser import extract_images, describe_images  # noqa: E402
from questions import QUESTIONS  # noqa: E402

ATTACH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "附件")
PDF2 = os.path.join(ATTACH_DIR, "招股说明书2.pdf")


def build_corpus(pdf_path: str, config: Config) -> list:
    """解析 PDF：正文分块 + 表格块 + 图像描述块 合并为语料。"""
    text = load_pdf_text(pdf_path)
    text_chunks = split_text(text, config.chunk_size, config.chunk_overlap)

    table_chunks = [t["text"] for t in parse_tables(pdf_path)]

    # 图像解析
    images = extract_images(pdf_path)
    print(f"  提取图像 {len(images)} 张，正在生成语义描述……")
    described = describe_images(images, config)
    image_chunks = [f"[图像描述] {d['description']}" for d in described]

    print(f"  正文 {len(text_chunks)} 块，表格 {len(table_chunks)} 块，图像 {len(image_chunks)} 块")
    return text_chunks + table_chunks + image_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()

    config = Config()
    llm = make_llm(config)
    print(f"[解析] {PDF2}")
    corpus = build_corpus(PDF2, config)

    embedder = make_embedder(config.embedding_model, corpus)
    vs = VectorStore(embedder, corpus)
    bm25 = BM25Index(corpus)
    retriever = HybridRetriever(vs, bm25)

    if args.query:
        hits = retriever.retrieve(args.query, top_k=5)
        print(f"\n问题：{args.query}")
        print(f"回答：{answer(args.query, hits, llm)}")
        print("\n相关片段：")
        for h in hits[:4]:
            print(f"  ({h['score']:.3f}) {h['text'][:100]}")
        return

    print("\n" + "=" * 70)
    for q in QUESTIONS:
        hits = retriever.retrieve(q["question"], top_k=5)
        ans = answer(q["question"], hits, llm)
        print(f"\n【问题 {q['id']}】{q['question']}")
        print(f"  [答案] {ans[:120]}")
        for h in hits[:2]:
            print(f"    ↳ ({h['score']:.3f}) {h['text'][:80]}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
