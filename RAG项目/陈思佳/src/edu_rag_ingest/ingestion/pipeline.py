from __future__ import annotations

"""网页采集主流程，将网页资源转换为可入库的文档分块。"""

import argparse
import logging
import uuid
from pathlib import Path

from .chunker import DocumentChunk, TextChunker, write_chunks_jsonl
from .cleaner import clean_text
from ..config.config import load_config
from .crawler import EducationCrawler
from .mineru_client import MinerUClient


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def build_metadata(defaults: dict[str, str], source_url: str, source_path: Path, title: str) -> dict[str, str]:
    metadata = dict(defaults)
    metadata.update(
        {
            "source_url": source_url,
            "source_file": str(source_path),
            "title": title or source_path.stem,
            "file_name": source_path.name,
            "file_type": source_path.suffix.lstrip(".").lower(),
        }
    )
    return metadata


def run_pipeline(config_path: str) -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config(config_path)

    crawler = EducationCrawler(config.crawler)
    mineru = MinerUClient(config.mineru)
    chunker = TextChunker(config.chunking)
    all_chunks: list[DocumentChunk] = []

    try:
        resources = crawler.crawl()
    finally:
        crawler.close()

    logger.info("共获取资源 %s 个", len(resources))
    for resource in resources:
        try:
            parsed_text = mineru.parse_or_read(resource.local_path)
            parsed_path = mineru.save_parsed_text(resource.local_path, parsed_text)
            cleaned = clean_text(parsed_text)
            document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, resource.source_url))
            metadata = build_metadata(config.metadata_defaults, resource.source_url, parsed_path, resource.title)
            chunks = chunker.split(document_id, cleaned, metadata)
            all_chunks.extend(chunks)
            logger.info("资源完成：%s，分块数：%s", resource.source_url, len(chunks))
        except Exception as error:
            logger.exception("资源处理失败：%s，原因：%s", resource.source_url, error)

    write_chunks_jsonl(all_chunks, config.chunking.output_path)
    logger.info("已输出入库准备文件：%s，chunk 总数：%s", config.chunking.output_path, len(all_chunks))


def main() -> None:
    parser = argparse.ArgumentParser(description="教育 RAG 公开网页采集与 MinerU 解析流程")
    parser.add_argument("--config", default="configs/crawler.yaml", help="配置文件路径")
    parser.add_argument("--input-dir", default="", help="本地待解析文件目录；提供后跳过网页爬取")
    args = parser.parse_args()
    if args.input_dir:
        from .local_pipeline import run_local_files

        run_local_files(args.config, args.input_dir)
    else:
        run_pipeline(args.config)


if __name__ == "__main__":
    main()
