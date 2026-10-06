from __future__ import annotations

"""本地文件批处理流程，将目录中的教育资料解析并输出分块文件。"""

import argparse
import json
import logging
import uuid
from pathlib import Path

from .chunker import DocumentChunk, TextChunker, write_chunks_jsonl
from .cleaner import clean_text
from ..config.config import load_config
from .mineru_client import MinerUClient
from .pipeline import build_metadata, setup_logging


def run_local_files(config_path: str, input_dir: str, manifest_path: str = "data/source_manifest.json") -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config(config_path)
    mineru = MinerUClient(config.mineru)
    chunker = TextChunker(config.chunking)
    all_chunks: list[DocumentChunk] = []
    manifest = _load_manifest(manifest_path)

    files = [path for path in Path(input_dir).rglob("*") if path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".doc", ".md", ".txt"}]
    logger.info("待处理本地文件 %s 个", len(files))

    for file_path in files:
        manifest_metadata = manifest.get(_manifest_key(file_path), {})
        if manifest_metadata.get("processing_status") == "已下载，待拆分解析":
            logger.info("跳过待拆分解析文件：%s", file_path)
            continue
        try:
            parsed_path = config.mineru.parsed_output_dir / f"{file_path.stem}.md"
            if parsed_path.exists():
                parsed_text = parsed_path.read_text(encoding="utf-8", errors="ignore")
                logger.info("复用已有解析结果：%s", parsed_path)
            else:
                parsed_text = mineru.parse_or_read(file_path)
                parsed_path = mineru.save_parsed_text(file_path, parsed_text)
            cleaned = clean_text(parsed_text)
            document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(file_path.resolve())))
            metadata = build_metadata(config.metadata_defaults, str(file_path), parsed_path, file_path.stem)
            metadata.update(manifest_metadata)
            metadata["source_file"] = str(parsed_path)
            chunks = chunker.split(document_id, cleaned, metadata)
            all_chunks.extend(chunks)
            logger.info("文件完成：%s，分块数：%s", file_path, len(chunks))
        except Exception as error:
            logger.exception("文件处理失败：%s，原因：%s", file_path, error)

    write_chunks_jsonl(all_chunks, config.chunking.output_path)
    logger.info("已输出入库准备文件：%s，chunk 总数：%s", config.chunking.output_path, len(all_chunks))


def _load_manifest(manifest_path: str) -> dict[str, dict[str, str]]:
    path = Path(manifest_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _manifest_key(file_path: Path) -> str:
    return file_path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description="教育 RAG 本地文件 MinerU 解析流程")
    parser.add_argument("--config", default="configs/crawler.yaml", help="配置文件路径")
    parser.add_argument("--input-dir", default="data/raw", help="本地待解析文件目录")
    parser.add_argument("--manifest", default="data/source_manifest.json", help="资料来源元数据清单")
    args = parser.parse_args()
    run_local_files(args.config, args.input_dir, args.manifest)


if __name__ == "__main__":
    main()
