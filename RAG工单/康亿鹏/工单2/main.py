# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块说明：命令行入口。提供知识库入库、问答、检索、统计与重置等命令，
          便于在无界面环境下运行与调试。

使用示例：
    python main.py ingest                     # 解析默认 PDF 并入库
    python main.py ask "法定代表人是谁？"        # 单轮问答
    python main.py chat                       # 交互式问答
    python main.py search "军用领域收入"        # 仅检索，查看召回结果
    python main.py stats                      # 查看知识库统计
    python main.py reset                      # 清空知识库
"""
import argparse
import json
import sys

import config
from src.knowledge_base import clear_knowledge_base, ingest_pdf, knowledge_base_stats
from src.utils import logger


def cmd_ingest(args):
    stats = ingest_pdf(
        args.pdf,
        drop_old=not args.append,
        enable_tables=not args.no_tables,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_ask(args):
    from src.rag_chain import RAGPipeline

    pipeline = RAGPipeline()
    if args.pure:
        print(pipeline.ask_pure_llm(args.question))
        return
    result = pipeline.ask(args.question, enable_understanding=not args.no_understanding)
    print("=" * 60)
    print(f"问题：{result.question}")
    print(f"意图：{result.analysis.intent} | 规范问题：{result.analysis.rewritten_query}")
    print(f"子问题：{result.analysis.sub_questions}")
    print("-" * 60)
    print(f"回答：{result.answer}")
    print("-" * 60)
    print(f"引用页码：{result.pages}")
    print(f"耗时：总 {result.total_time}s（召回 {result.retrieve_time}s / 重排 "
          f"{result.rerank_time}s / 生成 {result.generate_time}s）")
    print("=" * 60)


def cmd_chat(args):
    from src.rag_chain import RAGPipeline

    pipeline = RAGPipeline()
    print(f"[{config.WORK_ORDER_NO}] 问答已启动，输入 exit / q 退出。")
    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        result = pipeline.ask(question, enable_understanding=not args.no_understanding)
        print(f"\nAI：{result.answer}")
        print(f"[耗时 {result.total_time}s，引用页码 {result.pages}]")


def cmd_search(args):
    from src.embeddings import get_embedding_model
    from src.vector_store import get_vector_store

    vector = get_embedding_model().embed_query(args.query)
    hits = get_vector_store().search(vector, top_k=args.top_k)
    if not hits:
        print("未检索到结果（知识库可能为空，请先执行 ingest）。")
        return
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        print(f"[{i}] score={hit['score']:.4f} 第{meta['page']}页 类型={meta['type']}")
        print(hit["content"][:200].replace("\n", " "))
        print("-" * 60)


def cmd_stats(args):
    print(json.dumps(knowledge_base_stats(), ensure_ascii=False, indent=2))


def cmd_reset(args):
    clear_knowledge_base()
    print("知识库已清空。")


def build_parser():
    parser = argparse.ArgumentParser(description=f"{config.WORK_ORDER_NO} —— 命令行入口")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="解析 PDF 并写入向量库")
    p_ingest.add_argument("--pdf", default=config.DEFAULT_PDF_PATH, help="PDF 文件路径")
    p_ingest.add_argument("--append", action="store_true", help="追加写入（默认清空后重建）")
    p_ingest.add_argument("--no-tables", action="store_true", help="不解析表格")
    p_ingest.add_argument("--chunk-size", type=int, default=None, help="切分窗口大小")
    p_ingest.add_argument("--chunk-overlap", type=int, default=None, help="切分重叠大小")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="单轮问答")
    p_ask.add_argument("question", help="用户问题")
    p_ask.add_argument("--no-understanding", action="store_true", help="关闭 Query 理解")
    p_ask.add_argument("--pure", action="store_true", help="仅使用大模型回答（不检索）")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="交互式问答")
    p_chat.add_argument("--no-understanding", action="store_true", help="关闭 Query 理解")
    p_chat.set_defaults(func=cmd_chat)

    p_search = sub.add_parser("search", help="仅检索，查看召回结果")
    p_search.add_argument("query", help="检索内容")
    p_search.add_argument("--top-k", type=int, default=config.RETRIEVE_TOP_K)
    p_search.set_defaults(func=cmd_search)

    sub.add_parser("stats", help="查看知识库统计").set_defaults(func=cmd_stats)
    sub.add_parser("reset", help="清空知识库").set_defaults(func=cmd_reset)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        logger.error("执行失败：%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
