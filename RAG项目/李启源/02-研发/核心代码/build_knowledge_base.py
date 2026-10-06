"""Offline knowledge-base builder command line entry point.

The reusable workflow lives in :mod:`app.ingestion.offline_builder`; this
module keeps the historical import and script entry points stable.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Clients read configuration during construction, so load .env before use.
load_dotenv()

from app.ingestion.dataset_pipeline import clean_dataset, cleaned_dataset_path
from app.ingestion.offline_builder import OfflineKnowledgeBaseBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    """Define the public CLI contract in one place for reuse and testing."""
    parser = argparse.ArgumentParser(
        description="Offline Knowledge Base Construction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", help="Single file to process")
    input_group.add_argument("--directory", help="Directory to process")

    parser.add_argument("--kb-id", type=int, default=1, help="Knowledge base ID")
    parser.add_argument("--tenant-id", type=int, default=1, help="Tenant ID")
    parser.add_argument("--role-id", type=int, default=0, help="Role ID")
    parser.add_argument(
        "--parser",
        choices=[
            "auto",
            "pymupdf",
            "pdfplumber",
            "paddleocr",
            "mineru",
            "text",
            "jsonl",
        ],
        default="auto",
        help="Document parser",
    )
    parser.add_argument(
        "--strategy",
        choices=["fixed", "sentence", "paragraph", "heading", "semantic"],
        default="paragraph",
        help="Chunking strategy",
    )
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap")
    parser.add_argument("--force", action="store_true", help="Force reindex")
    parser.add_argument(
        "--clean-jsonl",
        action="store_true",
        help="Clean and audit a JSONL input before indexing",
    )
    parser.add_argument(
        "--clean-output-dir",
        default="data/cleaned",
        help="Directory for cleaning artifacts",
    )
    parser.add_argument(
        "--clean-mode",
        choices=["normalized", "curated", "product", "after_sales"],
        default="curated",
        help="Cleaned JSONL artifact to index",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Generate cleaning artifacts without indexing",
    )
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.92,
        help="Similarity threshold for review-only near duplicates",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["*.pdf", "*.txt", "*.md", "*.jsonl"],
        help="File patterns for directory mode",
    )
    return parser


def _prepare_input(args: argparse.Namespace) -> Path | None:
    if not args.clean_jsonl:
        if args.clean_only:
            raise ValueError("--clean-only requires --clean-jsonl")
        return Path(args.file) if args.file else None
    if not args.file or Path(args.file).suffix.lower() != ".jsonl":
        raise ValueError("--clean-jsonl requires a single --file ending in .jsonl")
    report = clean_dataset(
        args.file,
        args.clean_output_dir,
        args.near_duplicate_threshold,
    )
    logger.info("Cleaning report: %s", report)
    if args.clean_only:
        return None
    selected = cleaned_dataset_path(args.clean_output_dir, args.clean_mode)
    if not selected.exists() or selected.stat().st_size == 0:
        raise ValueError(f"cleaned dataset is empty: {selected}")
    return selected


def main() -> None:
    """Initialize configured clients and execute the requested build."""
    args = build_argument_parser().parse_args()
    try:
        prepared_file = _prepare_input(args)
    except (OSError, ValueError) as exc:
        logger.error("Dataset preparation failed: %s", exc)
        raise SystemExit(2) from exc
    if args.clean_only:
        raise SystemExit(0)
    try:
        builder = OfflineKnowledgeBaseBuilder()
    except Exception as exc:
        logger.error("Failed to initialize builder: %s", exc)
        logger.error("Make sure Milvus is running and environment variables are set")
        raise SystemExit(1) from exc

    common = {
        "knowledge_base_id": args.kb_id,
        "tenant_id": args.tenant_id,
        "role_id": args.role_id,
        "parser": args.parser,
        "chunking_strategy": args.strategy,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "force": args.force,
    }
    try:
        if prepared_file is not None:
            result = builder.build_document(str(prepared_file), **common)
        else:
            result = builder.build_directory(
                args.directory,
                patterns=args.patterns,
                **common,
            )
    except KeyboardInterrupt as exc:
        logger.info("Interrupted by user")
        raise SystemExit(130) from exc
    except Exception as exc:
        logger.exception("Pipeline failed")
        raise SystemExit(1) from exc

    raise SystemExit(0 if result["status"] in {"success", "completed"} else 1)


if __name__ == "__main__":
    main()
