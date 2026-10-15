# 工单编号：人工智能NLP-RAG-RAG性能瓶颈识别与优化
"""工单编号：人工智能NLP-RAG-基于PDF 文档的问答系统。"""
import argparse
import json
from .core import KnowledgeBase, build_index


def main():
    parser = argparse.ArgumentParser(description="基于 PDF 的问答系统")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="从 data/documents 构建知识库")
    run = sub.add_parser("serve", help="启动网页")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8765)
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--mode", choices=["rag", "llm_only"], default="rag")
    args = parser.parse_args()
    if args.command == "build":
        print(f"构建完成：{build_index()} 个片段")
    elif args.command == "serve":
        from .server import serve
        serve(args.host, args.port)
    else:
        print(json.dumps(KnowledgeBase().ask(args.question, mode=args.mode), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
