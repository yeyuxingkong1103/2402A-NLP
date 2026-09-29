import argparse
import json
import logging
from pathlib import Path

from app.ingest.offline_ingest import process_raw_directory

logger = logging.getLogger(__name__)

PROJECT_ROOT_DIRECTORY = Path(__file__).resolve().parents[3]
DEFAULT_RAW_ROOT = PROJECT_ROOT_DIRECTORY / "data" / "labor_law_raw"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT_DIRECTORY / "data" / "labor_law_processed"
DEFAULT_MANIFEST = PROJECT_ROOT_DIRECTORY / "data" / "crawl_manifest.json"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将已验证的劳动法真实 HTML 离线解析为 JSONL 数据包。",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="原始 HTML 目录，默认使用 data/labor_law_raw。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="输出 JSONL 数据包目录，默认使用 data/labor_law_processed。",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="采集清单 JSON 文件，包含文件名到元数据的映射（默认：data/crawl_manifest.json）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "显式覆盖已存在的包目录：覆盖前把旧包整体搬到输出目录旁边的"
            "备份目录（不直接删除）；不传该参数时目标目录已存在一律报 failed"
        ),
    )
    return parser


def load_manifest(manifest_path: Path) -> tuple[dict[str, str], dict[str, str], dict[str, str | None]]:
    """从 manifest 文件加载映射表。

    Returns:
        (source_urls, source_ids, document_titles)

    Raises:
        FileNotFoundError: manifest 文件不存在
        ValueError: manifest 格式错误
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"采集清单不存在：{manifest_path}")

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"采集清单 JSON 格式错误：{e}") from e

    if not isinstance(data, dict):
        raise ValueError("采集清单必须是对象（文件名 -> 元数据）")

    source_urls = {}
    source_ids = {}
    document_titles = {}

    for filename, metadata in data.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"文件 {filename} 的元数据必须是对象")

        # 必需字段：source_url, source_id
        if "source_url" not in metadata:
            logger.error(f"文件 {filename} 缺少 source_url，跳过")
            continue
        if "source_id" not in metadata:
            logger.error(f"文件 {filename} 缺少 source_id，跳过")
            continue

        source_urls[filename] = metadata["source_url"]
        source_ids[filename] = metadata["source_id"]
        document_titles[filename] = metadata.get("title")  # title 可以为 None

    if not source_urls:
        raise ValueError("采集清单中没有有效的文件元数据")

    return source_urls, source_ids, document_titles


def main() -> int:
    arguments = build_argument_parser().parse_args()

    # 加载 manifest
    try:
        source_urls, source_ids, document_titles = load_manifest(arguments.manifest)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"加载采集清单失败：{e}")
        return 2

    # 执行离线入库（--force 显式覆盖，默认拒绝覆盖已存在目录）
    result = process_raw_directory(
        arguments.raw_root,
        arguments.output_root,
        source_urls,
        source_ids,
        document_titles,
        force=arguments.force,
    )

    print(f"success={result.success_count}\tfailure={result.failure_count}")
    for error in result.errors:
        print(f"failed\t{error.filename}\t{error.message}")
    return 0 if result.failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
