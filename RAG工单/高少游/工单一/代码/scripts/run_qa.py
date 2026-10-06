# -*- coding: utf-8 -*-
"""命令行问答测试脚本
用法：
    python scripts/run_qa.py --question "武汉兴图新科注册资本是多少" [--llm-only]
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.qa_engine import QAEngine

logging.basicConfig(level=logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(description="问答测试")
    parser.add_argument("--question", required=True, help="问题文本")
    parser.add_argument("--llm-only", action="store_true", help="仅用LLM（无检索）")
    args = parser.parse_args()

    engine = QAEngine()
    if args.llm_only:
        res = engine.answer_llm_only(args.question)
    else:
        res = engine.answer_rag(args.question)

    print(f"\n[模式] {res.mode}")
    print(f"[耗时] {res.elapsed:.2f}s")
    if res.analysis:
        print(f"[意图] {res.analysis.intent}")
        print(f"[消歧] {res.analysis.disambiguated_query}")
        print(f"[检索子问题] {res.analysis.retrieval_queries}")
    if res.sources:
        print(f"[依据] {len(res.sources)} 条：")
        for s in res.sources:
            print(f"  - 第{s['page']}页 score={s['score']}: {s['text'][:60]}")
    print(f"[回答] {res.answer}\n")


if __name__ == "__main__":
    main()