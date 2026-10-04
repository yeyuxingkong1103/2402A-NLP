#!/usr/bin/env python
"""命令行入库工具。

用法：
    python scripts/ingest_cli.py --dir laws --domain law --parser auto
    python scripts/ingest_cli.py --file laws/中华人民共和国野生动物保护法.pdf --domain law
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingest.pipeline import ingest_directory, ingest_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 知识库离线入库")
    parser.add_argument("--dir", help="批量入库目录")
    parser.add_argument("--file", help="单个文件路径")
    parser.add_argument("--domain", default="general", help="知识域")
    parser.add_argument("--parser", default="auto",
                        help="解析引擎: auto|pymupdf|pdfplumber|paddleocr|mineru|multimodal")
    args = parser.parse_args()

    if args.file:
        print(json.dumps(ingest_file(args.file, domain=args.domain, parser=args.parser),
                         ensure_ascii=False))
    elif args.dir:
        for result in ingest_directory(args.dir, domain=args.domain, parser=args.parser):
            print(json.dumps(result, ensure_ascii=False))
    else:
        parser.error("需指定 --dir 或 --file")


if __name__ == "__main__":
    main()
