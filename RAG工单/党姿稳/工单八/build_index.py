# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-功能测试及评估
索引预构建：解析语料块后，预先编码并缓存向量（data/vectors.npy），
避免每次评估重复编码；同时构建多字段倒排索引做一次自检。
"""
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from kb import load_chunks
from hybrid_retriever import HybridRetriever


def main():
    chunks = load_chunks()
    print(f"语料块：{len(chunks)}")
    r = HybridRetriever(chunks, reranker="none")
    print(f"向量：{r.vectors.shape}；倒排索引词表：{len(r.fulltext.vocab)}")
    for q in ["平安银行2019年董事长致辞", "不良贷款率", "新业务价值"]:
        res = r.fulltext_search(q, 3)
        print(f"  [{q}] -> {[(c['doc'], c['page']) for c, _ in res]}")


if __name__ == "__main__":
    main()
