from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.app.config import Settings, get_settings
from backend.app.models import ModelGateway
from backend.app.storage.vector import MilvusStore

from .chunk import chunk_document
from .clean import clean_public_collections, validate_public_records
from .embedding import embed, embedding_text
from .load import extract, extract_content
from .parse import (
    SOURCE_FILES,
    build_citations,
    build_elements,
    extract_articles,
    extract_cases,
    extract_evidence,
    extract_interpretations,
    extract_processes,
    extract_questions,
    load_elements,
    parse_document,
)


PUBLIC_COLLECTIONS = (*SOURCE_FILES.keys(), "civil_citations")


def _vector_store(vector_store: Any | None = None, settings: Settings | None = None):
    return vector_store if vector_store is not None else MilvusStore(settings or get_settings())


def index(rows: list[dict], *, scope: str, collection: str | None = None, user_id: str = "", document_id: str = "", session_id: str = "", vector_store: Any | None = None, settings: Settings | None = None) -> list[dict]:
    if not rows:
        return []
    store = _vector_store(vector_store, settings)
    if scope == "private":
        payload = []
        for position, source in enumerate(rows):
            row = dict(source)
            row.update(
                chunk_id=str(row.get("chunk_id") or f"{document_id or 'document'}_chunk_{position:03d}"),
                user_id=user_id,
                document_id=document_id,
                session_id=session_id,
                chunk_index=int(row.get("chunk_index", position)),
                content=str(row.get("content") or row.get("text") or row.get("embedding_text") or ""),
                file_name=str(row.get("file_name") or ""),
            )
            payload.append(row)
        store.insert_private(payload)
        return payload
    if not collection:
        raise ValueError("public 入库必须提供 collection")
    payload = []
    for source in rows:
        row = dict(source)
        row.setdefault("collection", collection)
        row.setdefault("content", str(row.get("content") or row.get("text") or row.get("embedding_text") or ""))
        payload.append(row)
    store.insert_public(collection, payload)
    return payload


def file_metadata(path: Path) -> dict:
    source = Path(path)
    stat = source.stat()
    return {"file_name": source.name, "suffix": source.suffix.lower(), "size_bytes": stat.st_size}


def save_processed_result(result: dict, output_path: Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


save = save_processed_result
metadata = file_metadata


def process(path: Path, *, scope: str, collection: str | None = None, document_type: str | None = None, user_id: str = "", document_id: str = "", session_id: str = "", chunk_size: int = 1000, model=None, vector_store=None) -> dict:
    source = Path(path)
    settings = get_settings()
    custom_model = model is not None
    gateway = model or ModelGateway(settings)
    store = vector_store if vector_store is not None else (None if custom_model else MilvusStore(settings))
    text = extract(source)
    parsed = parse_document(text, document_type=document_type, collection=collection, source_path=source)
    details = {
        **file_metadata(source),
        "file_id": document_id or source.stem,
        "user_id": user_id,
        "session_id": session_id,
        "scope": scope,
        "collection": collection or "",
    }
    chunks = chunk_document(parsed, details, chunk_size=chunk_size)
    embedded = embed(chunks, gateway, batch_size=settings.embedding_batch_size, max_characters=settings.embedding_batch_max_chars, expected_dimension=None if custom_model else settings.embedding_dim)
    indexed = index(embedded, scope=scope, collection=collection, user_id=user_id, document_id=document_id or source.stem, session_id=session_id, vector_store=store) if store is not None else embedded
    return {
        "metadata": details,
        "extracted": {"text": text, "extraction_method": "text"},
        "parsed": parsed,
        "chunks": chunks,
        "embedded_chunks": embedded,
        "indexed_chunks": indexed,
    }


def process_user_file(path: Path) -> dict:
    source = Path(path)
    details = file_metadata(source)
    chunks = chunk_document(parse_document(extract(source)), details)
    return {"metadata": details, "chunks": [row.get("text", "") for row in chunks]}


def process_user_document(path, file_id: str | None = None, user_id: str = "", session_id: str | None = None, document_type: str | None = None, chunk_size: int = 1000, model=None, vector_store=None):
    return process(path, scope="private", user_id=user_id, document_id=file_id or "", session_id=session_id or "", document_type=document_type, chunk_size=chunk_size, model=model, vector_store=vector_store)


def ocr_document(path: str) -> dict:
    return {"text": extract_content(Path(path)).text}


def _source(data_dir: Path, collection: str) -> Path:
    path = Path(data_dir) / SOURCE_FILES[collection]
    if not path.is_file():
        raise FileNotFoundError(f"{collection} 缺少原始文件：{path}")
    return path


def _read_public_sources(data_dir: Path) -> dict[str, list[dict]]:
    return clean_public_collections({
        "civil_code_articles": extract_articles(_source(data_dir, "civil_code_articles").read_text(encoding="utf-8-sig")),
        "civil_interpretations": extract_interpretations(extract(_source(data_dir, "civil_interpretations"))),
        "civil_cases": extract_cases(_source(data_dir, "civil_cases").read_text(encoding="utf-8-sig")),
        "civil_elements": load_elements(_source(data_dir, "civil_elements")),
        "civil_evidence": extract_evidence(_source(data_dir, "civil_evidence").read_text(encoding="utf-8-sig")),
        "civil_processes": extract_processes(_source(data_dir, "civil_processes").read_text(encoding="utf-8-sig")),
        "civil_questions": extract_questions(_source(data_dir, "civil_questions").read_text(encoding="utf-8-sig")),
    })


def save_public_records(collections: dict[str, list[dict]], output_dir: Path) -> dict[str, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = {}
    for collection, rows in collections.items():
        path = target / f"{collection}.json"
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        paths[collection] = path
    return paths


def build_public_records(data_dir: Path, output_dir: Path) -> dict:
    collections = _read_public_sources(Path(data_dir))
    collections["civil_elements"] = build_elements(collections["civil_code_articles"], collections["civil_interpretations"], collections["civil_cases"])
    collections = clean_public_collections(collections)
    collections["civil_citations"] = build_citations(collections)
    quality = validate_public_records(collections)
    paths = save_public_records(collections, output_dir)
    report_path = Path(output_dir) / "quality_report.json"
    report_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"collections": collections, "quality": quality, "paths": paths, "quality_report": report_path}


build_public = build_public_records


def _public_rows(rows: list[dict], model, dimension: int | None) -> list[dict]:
    vectors = model.embed([embedding_text(row) for row in rows]) if model else [[] for _ in rows]
    if len(vectors) != len(rows):
        raise ValueError("向量数量与公共记录数量不一致")
    output = []
    for row, vector in zip(rows, vectors, strict=True):
        if dimension is not None and vector and len(vector) != dimension:
            raise ValueError(f"向量维度错误：应为 {dimension}，实际为 {len(vector)}")
        output.append({**dict(row), "embedding": vector})
    return output


def _ensure_public_collection(client, collection: str, dimension: int | None) -> None:
    if client.has_collection(collection):
        return
    try:
        from pymilvus import DataType, MilvusClient

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dimension or 1024)
        indexes = client.prepare_index_params()
        indexes.add_index("embedding", metric_type="COSINE", index_type="AUTOINDEX")
        client.create_collection(collection, schema=schema, index_params=indexes)
    except (ImportError, AttributeError, TypeError):
        client.create_collection(collection)


def _milvus_client(settings=None, client=None):
    if client is not None:
        return client
    from pymilvus import MilvusClient

    config = settings or get_settings()
    return MilvusClient(uri=config.milvus_uri, token=config.milvus_token, db_name=config.milvus_database)


def rebuild_public_collections(*, settings=None, collections: dict[str, list[dict]], model=None, client=None, embedding_dimension: int | None = None):
    target = _milvus_client(settings, client)
    result = {}
    for collection, rows in collections.items():
        candidate = f"{collection}__candidate"
        try:
            if target.has_collection(candidate):
                target.drop_collection(candidate)
            _ensure_public_collection(target, candidate, embedding_dimension)
            payload = _public_rows(rows, model, embedding_dimension)
            if payload:
                target.insert(candidate, payload)
            target.flush(candidate)
            count = int((target.get_collection_stats(candidate) or {}).get("row_count", 0))
            if count != len(rows):
                raise RuntimeError(f"{collection} 候选集合行数不一致：应为 {len(rows)}，实际为 {count}")
            target.rename_collection(candidate, collection)
            result[collection] = count
        except Exception:
            if target.has_collection(candidate):
                target.drop_collection(candidate)
            raise
    return result


def append_public_collections(*, settings=None, collections: dict[str, list[dict]], model=None, client=None, embedding_dimension: int | None = None):
    target = _milvus_client(settings, client)
    result = {}
    for collection, rows in collections.items():
        _ensure_public_collection(target, collection, embedding_dimension)
        payload = _public_rows(rows, model, embedding_dimension)
        if payload:
            target.insert(collection, payload)
        target.flush(collection)
        result[collection] = int((target.get_collection_stats(collection) or {}).get("row_count", 0))
    return result


mark_ready = process_user_document
mark_failed = process_user_document
mark_processing = process_user_document


__all__ = [
    "PUBLIC_COLLECTIONS", "append_public_collections", "build_public", "build_public_records",
    "file_metadata", "index", "metadata", "ocr_document", "process", "process_user_document",
    "process_user_file", "rebuild_public_collections", "save", "save_processed_result", "save_public_records",
]
