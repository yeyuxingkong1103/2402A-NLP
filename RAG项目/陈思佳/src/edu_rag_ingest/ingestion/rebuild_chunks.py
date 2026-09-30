from __future__ import annotations

"""分块重建命令，用于根据最新清洗和分块配置重新生成 JSONL。"""

import argparse
import logging
import uuid
from pathlib import Path

from .chunker import DocumentChunk, TextChunker, write_chunks_jsonl
from .cleaner import clean_text
from ..config.config import load_config
from .pipeline import build_metadata, setup_logging


def rebuild_chunks(config_path: str, parsed_dir: str) -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config(config_path)
    chunker = TextChunker(config.chunking)
    all_chunks: list[DocumentChunk] = []

    files = [path for path in Path(parsed_dir).glob("*.md") if path.is_file()]
    logger.info("待重建分块 Markdown 文件 %s 个", len(files))

    for file_path in files:
        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(raw_text)
        document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(file_path.resolve())))
        metadata = build_metadata(config.metadata_defaults, str(file_path), file_path, file_path.stem)
        chunks = chunker.split(document_id, cleaned, metadata)
        all_chunks.extend(chunks)
        logger.info("分块完成：%s，分块数：%s", file_path, len(chunks))

    write_chunks_jsonl(all_chunks, config.chunking.output_path)
    logger.info("已输出入库准备文件：%s，chunk 总数：%s", config.chunking.output_path, len(all_chunks))


def main() -> None:
    parser = argparse.ArgumentParser(description="基于已解析 Markdown 重新清洗和分块")
    parser.add_argument("--config", default="configs/crawler.yaml", help="配置文件路径")
    parser.add_argument("--parsed-dir", default="data/parsed", help="已解析 Markdown 目录")
    args = parser.parse_args()
    rebuild_chunks(args.config, args.parsed_dir)


if __name__ == "__main__":
    main()
