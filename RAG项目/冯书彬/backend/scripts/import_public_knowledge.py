import argparse
import hashlib
import html
import json
import os
import shutil
import time
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.request import Request, urlopen

import requests
from sqlalchemy import create_engine, text

from backend.app.core.config import settings
from backend.app.database.milvus import create_milvus_store
from backend.app.embeddings.embedding_factory import get_embedding_client
from backend.app.ingestion.structured_chunker import ParentChunk, build_parent_child_chunks

SUPPORTED_SUFFIXES = {".txt", ".json", ".doc", ".docx", ".pdf", ".html", ".htm", ".md", ".csv"}


@dataclass
class ParsedDocument:
    path: Path
    title: str
    markdown: str


class MinerUClient:
    def __init__(self, api_key: str, base_url: str, model_version: str) -> None:
        if not api_key:
            raise RuntimeError("MINERU_API_KEY 未配置")
        self.base_url = base_url.rstrip("/")
        self.model_version = model_version
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def parse_files(self, paths: list[Path], timeout_seconds: int = 900, interval_seconds: int = 5) -> dict[Path, str]:
        if not paths:
            return {}
        files = [{"name": path.name, "data_id": str(index)} for index, path in enumerate(paths)]
        payload = {"files": files, "model_version": self.model_version}
        result = self._request_json("POST", "/api/v4/file-urls/batch", payload)
        if result.get("code") != 0:
            raise RuntimeError(f"MinerU 申请上传 URL 失败：{result.get('msg', 'unknown')}")
        batch_id = result["data"]["batch_id"]
        urls = result["data"]["file_urls"]
        for path, upload_url in zip(paths, urls, strict=True):
            self._upload_file(upload_url, path)
        return self._poll_results(batch_id, paths, timeout_seconds, interval_seconds)

    def _request_json(self, method: str, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(f"{self.base_url}{endpoint}", data=data, method=method, headers=self.headers)
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    def _upload_file(self, upload_url: str, path: Path) -> None:
        with path.open("rb") as file_handle:
            response = requests.put(upload_url, data=file_handle, timeout=180)
        if response.status_code not in {200, 201}:
            raise RuntimeError(f"MinerU 文件上传失败：{path.name}，HTTP {response.status_code}")

    def _poll_results(self, batch_id: str, paths: list[Path], timeout_seconds: int, interval_seconds: int) -> dict[Path, str]:
        deadline = time.time() + timeout_seconds
        by_name = {path.name: path for path in paths}
        parsed: dict[Path, str] = {}
        while time.time() < deadline:
            result = self._request_json("GET", f"/api/v4/extract-results/batch/{batch_id}")
            if result.get("code") != 0:
                raise RuntimeError(f"MinerU 查询结果失败：{result.get('msg', 'unknown')}")
            items = result.get("data", {}).get("extract_result", [])
            for item in items:
                file_name = item.get("file_name")
                path = by_name.get(file_name)
                if not path or path in parsed:
                    continue
                state = item.get("state")
                if state == "failed":
                    raise RuntimeError(f"MinerU 解析失败：{file_name}，{item.get('err_msg', 'unknown')}")
                if state == "done" and item.get("full_zip_url"):
                    parsed[path] = self._download_markdown(item["full_zip_url"])
            print(f"MinerU 解析进度：{len(parsed)}/{len(paths)}")
            if len(parsed) == len(paths):
                return parsed
            time.sleep(interval_seconds)
        raise TimeoutError(f"MinerU 解析超时，batch_id={batch_id}")

    def _download_markdown(self, zip_url: str) -> str:
        with urlopen(zip_url, timeout=180) as response:
            data = response.read()
        with TemporaryDirectory() as temp_dir:
            zip_path = Path(temp_dir) / "mineru.zip"
            zip_path.write_bytes(data)
            with zipfile.ZipFile(zip_path) as archive:
                markdown_names = [name for name in archive.namelist() if name.endswith("full.md") or name.endswith(".md")]
                if not markdown_names:
                    raise RuntimeError("MinerU 结果包中没有 Markdown 文件")
                return archive.read(markdown_names[0]).decode("utf-8", errors="ignore")


def discover_files(source: Path) -> list[Path]:
    return sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)


def _read_text_file(path: Path) -> str:
    # 公共法律数据可能来自 GB18030 或 UTF-8，按顺序尝试避免中文丢失。
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def prepare_mineru_files(paths: list[Path], temp_dir: Path) -> tuple[list[Path], dict[Path, Path]]:
    # MinerU 不接受纯文本和 JSON；这些格式临时包装成 HTML，原始文件保持不变。
    staged_paths: list[Path] = []
    staged_to_original: dict[Path, Path] = {}
    convert_suffixes = {".txt", ".json", ".csv", ".md"}
    for index, original in enumerate(paths):
        if original.suffix.lower() not in convert_suffixes:
            staged = original
        else:
            staged = temp_dir / f"{index:03d}_{original.stem}.html"
            raw_text = _read_text_file(original)
            if original.suffix.lower() == ".json":
                try:
                    raw_text = json.dumps(json.loads(raw_text), ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    pass
            staged.write_text(
                f"<!doctype html><html><head><meta charset=\"utf-8\"><title>{html.escape(original.stem)}</title></head><body><pre>{html.escape(raw_text)}</pre></body></html>",
                encoding="utf-8",
            )
        staged_paths.append(staged)
        staged_to_original[staged] = original
    return staged_paths, staged_to_original


def chunk_text(markdown: str) -> list[str]:
    return [
        child.content
        for parent in build_parent_child_chunks(markdown)
        for child in parent.children
    ]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def insert_mysql(engine, parsed: ParsedDocument, parents: list[ParentChunk]) -> dict[str, str]:
    now = utc_now()
    whitelist_id = str(uuid.uuid4())
    snapshot_id = str(uuid.uuid4())
    material_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())
    source_url = f"public://{parsed.path.as_posix()}"
    raw_hash = content_hash(parsed.markdown)
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT IGNORE INTO source_whitelist_entries
                (id, url, publisher, material_type, active, created_by, created_at, updated_at)
                VALUES (:id, :url, :publisher, :material_type, true, :actor, :created_at, :updated_at)
            """),
            {"id": whitelist_id, "url": source_url, "publisher": "public", "material_type": "law", "actor": "mineru-import", "created_at": now, "updated_at": now},
        )
        conn.execute(
            text("""
                INSERT INTO crawl_snapshots
                (id, source_url, publisher, material_type, raw_content_hash, raw_text, attachments, backup_source_urls, crawled_at, status, searchable, failure_reason)
                VALUES (:id, :source_url, :publisher, :material_type, :raw_content_hash, :raw_text, :attachments, :backup_source_urls, :crawled_at, 'pending_review', false, null)
            """),
            {
                "id": snapshot_id,
                "source_url": source_url,
                "publisher": "public",
                "material_type": "law",
                "raw_content_hash": raw_hash,
                "raw_text": parsed.markdown,
                "attachments": json.dumps([{"name": parsed.path.name, "url": source_url, "content_hash": raw_hash}], ensure_ascii=False),
                "backup_source_urls": json.dumps([], ensure_ascii=False),
                "crawled_at": now,
            },
        )
        conn.execute(
            text("""
                INSERT INTO knowledge_materials
                (id, snapshot_id, source_url, publisher, material_type, raw_text, status, searchable, reviewed_by, published_by, effective_from, created_at, updated_at)
                VALUES (:id, :snapshot_id, :source_url, :publisher, :material_type, :raw_text, 'published', true, :actor, :actor, :effective_from, :created_at, :updated_at)
            """),
            {
                "id": material_id,
                "snapshot_id": snapshot_id,
                "source_url": source_url,
                "publisher": "public",
                "material_type": "law",
                "raw_text": parsed.markdown,
                "actor": "mineru-import",
                "effective_from": now.date().isoformat(),
                "created_at": now,
                "updated_at": now,
            },
        )
        conn.execute(
            text("INSERT INTO material_attachments (id, material_id, name, url, content_hash) VALUES (:id, :material_id, :name, :url, :content_hash)"),
            {"id": str(uuid.uuid4()), "material_id": material_id, "name": parsed.path.name, "url": source_url, "content_hash": raw_hash},
        )
        conn.execute(
            text("INSERT INTO review_records (id, material_id, reviewer_id, decision, reason, reviewed_at) VALUES (:id, :material_id, :actor, 'approved', :reason, :reviewed_at)"),
            {"id": str(uuid.uuid4()), "material_id": material_id, "actor": "mineru-import", "reason": "MinerU 批量解析导入", "reviewed_at": now},
        )
        conn.execute(
            text("INSERT INTO publish_records (id, material_id, publisher_id, published_at) VALUES (:id, :material_id, :actor, :published_at)"),
            {"id": str(uuid.uuid4()), "material_id": material_id, "actor": "mineru-import", "published_at": now},
        )
        conn.execute(
            text("""
                INSERT INTO documents (id, material_id, title, content_hash, status, created_at, updated_at)
                VALUES (:id, :material_id, :title, :content_hash, 'indexed', :created_at, :updated_at)
            """),
            {"id": document_id, "material_id": material_id, "title": parsed.title, "content_hash": raw_hash, "created_at": now, "updated_at": now},
        )
        chunk_index = 0
        for parent in parents:
            for level, chunk_id, content, child_index in _iter_parent_children(parent, document_id):
                conn.execute(
                    text("""
                        INSERT INTO document_chunks
                        (id, document_id, chunk_index, parent_chunk_id, chunk_level, structure_type, content, citation)
                        VALUES (:id, :document_id, :chunk_index, :parent_chunk_id, :chunk_level, :structure_type, :content, :citation)
                    """),
                    {
                        "id": chunk_id,
                        "document_id": document_id,
                        "chunk_index": chunk_index,
                        "parent_chunk_id": f"{document_id}:{parent.id}" if level == "child" else None,
                        "chunk_level": level,
                        "structure_type": parent.structure_type,
                        "content": content,
                        "citation": json.dumps({"source_url": source_url, "title": parsed.title, "chunk_index": chunk_index, "chunk_level": level, "parent_chunk_id": f"{document_id}:{parent.id}" if level == "child" else None}, ensure_ascii=False),
                    },
                )
                chunk_index += 1
    return {"material_id": material_id, "document_id": document_id, "source_url": source_url, "effective_from": now.date().isoformat()}


def _iter_parent_children(parent: ParentChunk, document_id: str):
    parent_id = f"{document_id}:{parent.id}"
    yield "parent", parent_id, parent.content, parent.index
    for child in parent.children:
        child_id = f"{parent_id}:child-{child.index}"
        yield "child", child_id, child.content, child.index


def upsert_milvus(rows: list[dict[str, Any]], batch_size: int = 128) -> int:
    if not rows:
        return 0
    embedding_client = get_embedding_client()
    store = create_milvus_store(settings)
    store.ensure_collection(settings.MILVUS_VECTOR_DIMENSION)
    written = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        embeddings = embedding_client.embed_texts([row["text"] for row in batch])
        for row, vector in zip(batch, embeddings, strict=True):
            row["vector"] = vector
        written += store.upsert(batch)
        print(f"Milvus 写入进度：{min(start + len(batch), len(rows))}/{len(rows)}")
    return written


def build_vector_rows(parsed: ParsedDocument, mysql_ids: dict[str, str], parents: list[ParentChunk]) -> list[dict[str, Any]]:
    rows = []
    for parent in parents:
        for child in parent.children:
            metadata = {
                "source_url": mysql_ids["source_url"],
                "publisher": "public",
                "material_type": "law",
                "status": "published",
                "searchable": True,
                "effective_from": mysql_ids["effective_from"],
                "article": parent.title if parent.structure_type == "article" else None,
                "paragraph": str(child.index + 1),
                "relationship_types": ["general"],
                "text_hash": content_hash(child.content),
                "title": parsed.title,
                "parent_chunk_id": f"{mysql_ids['document_id']}:{parent.id}",
                "parent_text": parent.content[:4000],
                "chunk_level": "child",
                "structure_type": parent.structure_type,
            }
            rows.append(
                {
                    "id": f"{mysql_ids['material_id']}:{parent.id}:{child.id}",
                    "material_id": mysql_ids["material_id"],
                    "version_id": mysql_ids["document_id"],
                    "relationship_type": "general",
                    "text": child.content[:4000],
                    "metadata": metadata,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 MinerU 解析 public 数据并写入 MySQL 与 Milvus")
    parser.add_argument("--source", default=r"C:\Users\bin\Desktop\public")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--interval", type=int, default=5)
    args = parser.parse_args()

    source = Path(args.source)
    files = discover_files(source)
    if not files:
        raise SystemExit(f"没有找到可导入文件：{source}")
    print(f"发现文件：{len(files)} 个")

    mineru = MinerUClient(settings.MINERU_API_KEY, settings.MINERU_BASE_URL, settings.MINERU_MODEL_VERSION)
    with TemporaryDirectory(prefix="mineru-import-") as temp_dir_name:
        staged_files, staged_to_original = prepare_mineru_files(files, Path(temp_dir_name))
        html_files = [path for path in staged_files if path.suffix.lower() in {".html", ".htm"}]
        native_files = [path for path in staged_files if path not in html_files]
        markdown_by_staged_path: dict[Path, str] = {}
        if html_files:
            html_client = MinerUClient(settings.MINERU_API_KEY, settings.MINERU_BASE_URL, "MinerU-HTML")
            markdown_by_staged_path.update(html_client.parse_files(html_files, timeout_seconds=args.timeout, interval_seconds=args.interval))
        if native_files:
            native_client = MinerUClient(settings.MINERU_API_KEY, settings.MINERU_BASE_URL, settings.MINERU_MODEL_VERSION)
            markdown_by_staged_path.update(native_client.parse_files(native_files, timeout_seconds=args.timeout, interval_seconds=args.interval))
    markdown_by_path = {staged_to_original[path]: markdown for path, markdown in markdown_by_staged_path.items()}
    engine = create_engine(settings.DATABASE_URL, future=True)

    all_rows: list[dict[str, Any]] = []
    imported = 0
    chunk_count = 0
    for path in files:
        markdown = markdown_by_path[path]
        parsed = ParsedDocument(path=path, title=path.stem, markdown=markdown)
        parents = build_parent_child_chunks(markdown)
        if not parents or not any(parent.children for parent in parents):
            print(f"跳过空文档：{path.name}")
            continue
        mysql_ids = insert_mysql(engine, parsed, parents)
        all_rows.extend(build_vector_rows(parsed, mysql_ids, parents))
        imported += 1
        chunk_count += sum(len(parent.children) for parent in parents)
        print(f"已写入 MySQL：{path.name}，parents={len(parents)}，children={sum(len(parent.children) for parent in parents)}")

    written = upsert_milvus(all_rows)
    print(f"导入完成：files={imported}, mysql_chunks={chunk_count}, milvus_rows={written}")


if __name__ == "__main__":
    main()
