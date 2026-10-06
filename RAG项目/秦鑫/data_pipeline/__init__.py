from __future__ import annotations

import sys
from importlib import import_module

from .chunk import chunk, chunk_document, fixed_chunks, make_chunk_id
from .clean import clean_line, clean_public_collections, clean_text, is_placeholder_case, validate_public_records
from .embedding import embed, embed_chunks, embed_public_batch, make_public_batches
from .index import (
    PUBLIC_COLLECTIONS,
    append_public_collections,
    build_public,
    build_public_records,
    file_metadata,
    index,
    metadata,
    ocr_document,
    process,
    process_user_document,
    process_user_file,
    rebuild_public_collections,
    save,
    save_processed_result,
    save_public_records,
)
from .load import (
    ExtractedDocument,
    IMAGE_SUFFIXES,
    SUPPORTED_SUFFIXES,
    TEXT_SUFFIXES,
    extract,
    extract_content,
    load,
    load_docx,
    load_image,
    load_json,
    load_pdf,
    load_text,
    load_xlsx,
)
from .parse import (
    SOURCE_FILES,
    build_citations,
    build_elements,
    build_metadata,
    chinese_number_to_int,
    detect_document_type,
    detect_type,
    extract_articles,
    extract_cases,
    extract_evidence,
    extract_interpretations,
    extract_processes,
    extract_questions,
    extract_sections,
    load_elements,
    parse,
    parse_document,
    related_civil_code_articles,
)


mark_ready = process_user_document
mark_failed = process_user_document
mark_processing = process_user_document

_index_module = import_module(f"{__name__}.index")
_load_module = import_module(f"{__name__}.load")

sys.modules.setdefault(f"{__name__}.law", sys.modules[__name__])
sys.modules.setdefault(f"{__name__}.pipeline", _index_module)
sys.modules.setdefault(f"{__name__}.extract", _load_module)


__all__ = [
    "ExtractedDocument", "IMAGE_SUFFIXES", "PUBLIC_COLLECTIONS", "SOURCE_FILES", "SUPPORTED_SUFFIXES", "TEXT_SUFFIXES",
    "append_public_collections", "build_citations", "build_elements", "build_metadata", "build_public", "build_public_records",
    "chunk", "chunk_document", "clean_line", "clean_public_collections", "clean_text", "chinese_number_to_int",
    "detect_document_type", "detect_type", "embed", "embed_chunks", "embed_public_batch", "extract", "extract_articles",
    "extract_cases", "extract_content", "extract_evidence", "extract_interpretations", "extract_processes", "extract_questions",
    "extract_sections", "file_metadata", "fixed_chunks", "index", "is_placeholder_case", "load", "load_docx", "load_elements",
    "load_image", "load_json", "load_pdf", "load_text", "load_xlsx", "make_chunk_id", "make_public_batches", "metadata",
    "ocr_document", "parse", "parse_document", "process", "process_user_document", "process_user_file", "rebuild_public_collections",
    "related_civil_code_articles", "save", "save_processed_result", "save_public_records", "validate_public_records",
]
