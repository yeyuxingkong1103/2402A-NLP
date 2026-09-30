"""独立的 PDF -> 分块 -> BGE 向量 -> Milvus 入库脚本。

此文件优先读取项目环境变量，未配置时回落到本机服务。

默认配置：
  PDF 目录:  ./data_set
  Milvus:    http://127.0.0.1:19530
  Embedding: 本地 D:\bge-small-zh-v1.5
  Collection: medical_guideline_chunks_v1

运行：
  python -m src.preprocessing.standalone_pdf_milvus_ingest --dry-run
  python -m src.preprocessing.standalone_pdf_milvus_ingest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TRANSFORMERS_CACHE = PROJECT_ROOT / "data" / "cache" / "huggingface"
_TRANSFORMERS_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_CACHE", str(_TRANSFORMERS_CACHE))
_MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1").strip()
_MILVUS_PORT = os.getenv("MILVUS_PORT", "19530").strip()
MILVUS_URI = os.getenv("MILVUS_URI", f"http://{_MILVUS_HOST}:{_MILVUS_PORT}").strip()
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "").strip().rstrip("/")
EMBEDDING_URL = f"{EMBEDDING_API_URL}/embed" if EMBEDDING_API_URL else ""
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", r"D:\bge-small-zh-v1.5"
).strip()
COLLECTION_NAME = os.getenv(
    "MILVUS_COLLECTION", "medical_guideline_chunks_v1"
).strip()

# bge-small-zh-v1.5 的输入窗口是 512 tokens。中文通常接近一字一 token，
# 正文控制在 360 字，向量化时连同标题和章节上下文控制在 440 字以内。
MAX_CHARS = 360
OVERLAP_CHARS = 48
MIN_CHARS = 24
EMBEDDING_MAX_CHARS = 440
EMBEDDING_BATCH_SIZE = 16

HEADING_RE = re.compile(
    r"^(?:第[一二三四五六七八九十百0-9]+[章节部分]|"
    r"[一二三四五六七八九十]+[、.]|"
    r"\d+(?:\.\d+){0,3}\s+)\S+"
)
SPECIAL_HEADING_RE = re.compile(
    r"^(?:前\s*言|附\s*录\s*[A-Z0-9一二三四五六七八九十]*|参\s*考\s*文\s*献)$"
)
SENTENCE_RE = re.compile(r"(?<=[。！？!?；;])\s*|\n+")


@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    source: str
    page: int
    section: str
    chunk_index: int
    text: str
    chunk_type: str


def normalize_text(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\xa0", " ").replace("　", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def is_heading(value: str) -> bool:
    value = value.strip()
    compact = re.sub(r"\s+", "", value)
    if not 2 <= len(value) <= 70 or value.endswith(("。", "；", ";", "，", ",")):
        return False
    if re.match(
        r"^(?:\d{4}\s*[-年]|\d+\s*(?:g|mg|ml|mmol|kcal|U/L|分钟|小时|天|年|岁|%))",
        value,
        re.IGNORECASE,
    ):
        return False
    return bool(
        HEADING_RE.match(value)
        or SPECIAL_HEADING_RE.match(compact)
        or (value.startswith("问：") and value.endswith(("?", "？")))
    )


def merge_wrapped_question_headings(lines: list[str]) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(lines):
        current = lines[index].strip()
        if re.match(r"^[一二三四五六七八九十]+[、.]", current) and not current.endswith(("?", "？")):
            next_index = index + 1
            while next_index < len(lines) and not lines[next_index].strip():
                next_index += 1
            if next_index < len(lines):
                candidate = current + lines[next_index].strip()
                if len(candidate) <= 70 and candidate.endswith(("?", "？")):
                    merged.append(candidate)
                    index = next_index + 1
                    continue
        merged.append(lines[index])
        index += 1
    return merged


def table_to_markdown(rows: list[list[str | None]]) -> str:
    cleaned = [
        [re.sub(r"\s+", " ", str(cell or "")).strip().replace("|", "\\|") for cell in row]
        for row in rows
        if row and any(str(cell or "").strip() for cell in row)
    ]
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    return "\n".join(
        [
            "| " + " | ".join(cleaned[0]) + " |",
            "| " + " | ".join("---" for _ in range(width)) + " |",
            *("| " + " | ".join(row) + " |" for row in cleaned[1:]),
        ]
    )


def extract_page(page: pymupdf.Page) -> list[tuple[float, str, str]]:
    """返回 (纵坐标, 类型, 内容)，把可识别表格保留为 Markdown。"""
    try:
        detected = [(table, table.extract()) for table in page.find_tables().tables]
    except Exception:
        detected = []
    tables = [
        (table, rows)
        for table, rows in detected
        if rows and max((len(row) for row in rows if row), default=0) <= 12
    ]
    table_boxes = [pymupdf.Rect(table.bbox) for table, _ in tables]

    elements: list[tuple[float, str, str]] = []
    for block in page.get_text("blocks", sort=True):
        box = pymupdf.Rect(block[:4])
        center = ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
        if any(table_box.contains(center) for table_box in table_boxes):
            continue
        text = normalize_text(str(block[4]))
        if not text:
            continue
        one_line = re.sub(r"\s+", " ", text)
        near_margin = box.y0 < page.rect.height * 0.08 or box.y1 > page.rect.height * 0.94
        if near_margin and (
            re.fullmatch(r"(?:WS/T|GBZ)\s*[\d.]+[—-]\d{4}", one_line)
            or re.fullmatch(r"\d+", one_line)
        ):
            continue
        elements.append((box.y0, "prose", text))

    for table, rows in tables:
        markdown = table_to_markdown(rows)
        if markdown:
            elements.append((pymupdf.Rect(table.bbox).y0, "table", markdown))
    return sorted(elements, key=lambda item: item[0])


def sliding_windows(text: str, limit: int, overlap: int) -> list[str]:
    windows: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind("。", start, end), text.rfind("；", start, end))
            if boundary > start + limit // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            windows.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return windows


def split_prose(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    fragments: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= MAX_CHARS:
            fragments.append(paragraph)
        else:
            sentences = [item.strip() for item in SENTENCE_RE.split(paragraph) if item.strip()]
            if len(sentences) <= 1:
                fragments.extend(sliding_windows(paragraph, MAX_CHARS, OVERLAP_CHARS))
            else:
                fragments.extend(sentences)

    chunks: list[str] = []
    current: list[str] = []
    for fragment in fragments:
        if len(fragment) > MAX_CHARS:
            if current:
                chunks.append("\n\n".join(current))
                current = []
            chunks.extend(sliding_windows(fragment, MAX_CHARS, OVERLAP_CHARS))
            continue
        candidate = "\n\n".join([*current, fragment])
        if current and len(candidate) > MAX_CHARS:
            chunks.append("\n\n".join(current))
            carry: list[str] = []
            carry_length = 0
            for part in reversed(current):
                addition = len(part) + (2 if carry else 0)
                if carry_length + addition > OVERLAP_CHARS:
                    break
                carry.insert(0, part)
                carry_length += addition
            current = carry if len("\n\n".join([*carry, fragment])) <= MAX_CHARS else []
        current.append(fragment)
    if current:
        chunks.append("\n\n".join(current))
    return [chunk for chunk in chunks if len(chunk) >= MIN_CHARS]


def split_table(table: str) -> list[str]:
    lines = [line.strip() for line in table.splitlines() if line.strip()]
    if len(table) <= MAX_CHARS or len(lines) <= 3:
        return [table] if len(table) >= MIN_CHARS else []

    header = lines[:2]
    header_text = "\n".join(header)
    budget = max(40, MAX_CHARS - len(header_text) - 1)
    chunks: list[str] = []
    rows: list[str] = []
    for row in lines[2:]:
        if len(row) > budget:
            if rows:
                chunks.append("\n".join([header_text, *rows]))
                rows = []
            flattened = row.strip("| ").replace(" | ", "; ")
            chunks.extend(f"{header_text}\n{window}" for window in sliding_windows(flattened, budget, OVERLAP_CHARS))
            continue
        candidate = "\n".join([header_text, *rows, row])
        if rows and len(candidate) > MAX_CHARS:
            chunks.append("\n".join([header_text, *rows]))
            overlap = rows[-1:]
            rows = overlap if len("\n".join([header_text, *overlap, row])) <= MAX_CHARS else []
        rows.append(row)
    if rows:
        chunks.append("\n".join([header_text, *rows]))
    return [chunk[:MAX_CHARS] for chunk in chunks if len(chunk) >= MIN_CHARS]


def parse_pdf(path: Path) -> list[Chunk]:
    title = path.stem.split("_", 1)[-1]
    document_key = hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:12]
    records: list[tuple[int, str, str, str]] = []
    current_section = title

    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            def process_prose(content: str) -> None:
                nonlocal current_section
                lines = merge_wrapped_question_headings(content.splitlines())
                buffer: list[str] = []

                def flush() -> None:
                    if not buffer:
                        return
                    body = normalize_text("\n".join(buffer))
                    records.extend(
                        (page_number, current_section, "prose", part)
                        for part in split_prose(body)
                    )
                    buffer.clear()

                for line in lines:
                    stripped = line.strip()
                    if re.fullmatch(r"(?:WS/T|GBZ)\s*[\d.]+[—-]\d{4}", stripped):
                        continue
                    if is_heading(stripped):
                        flush()
                        current_section = stripped
                    else:
                        buffer.append(line)
                flush()

            prose_elements: list[str] = []

            def flush_prose_elements() -> None:
                if prose_elements:
                    process_prose("\n\n".join(prose_elements))
                    prose_elements.clear()

            for _, kind, content in extract_page(page):
                if kind == "table":
                    flush_prose_elements()
                    records.extend(
                        (page_number, current_section, "table", part)
                        for part in split_table(content)
                    )
                else:
                    prose_elements.append(content)
            flush_prose_elements()

    chunks: list[Chunk] = []
    for index, (page, section, chunk_type, text) in enumerate(records):
        chunks.append(
            Chunk(
                id=f"{document_key}:{index:05d}",
                title=title,
                source=path.name,
                page=page,
                section=section,
                chunk_index=index,
                text=text,
                chunk_type=chunk_type,
            )
        )
    return chunks


class EmbeddingClient:
    def __init__(self, url: str, model_name: str) -> None:
        self.url = url
        self.model_name = model_name
        self._local_model: Any = None
        probe = self.encode(["向量维度测试"])
        self.dimension = int(probe.shape[1])

    def encode(self, texts: list[str]) -> Any:
        import numpy as np

        if not self.url:
            if self._local_model is None:
                from sentence_transformers import SentenceTransformer

                self._local_model = SentenceTransformer(self.model_name, device="cpu")
            return np.asarray(
                self._local_model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )

        payload = json.dumps({"texts": texts, "normalize_embeddings": True}).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
        embeddings = result.get("embeddings") if isinstance(result, dict) else result
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError(f"Embedding 服务返回格式异常: {type(result).__name__}")
        return np.asarray(embeddings, dtype=np.float32)


def embedding_text(chunk: Chunk) -> str:
    context = f"{chunk.title} > {chunk.section}"[:EMBEDDING_MAX_CHARS]
    remaining = max(0, EMBEDDING_MAX_CHARS - len(context) - 1)
    return f"{context}\n{chunk.text[:remaining]}"


def batched(values: list[Chunk], size: int) -> Iterable[list[Chunk]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def ensure_collection(client: Any, collection: str, dimension: int) -> None:
    from pymilvus import DataType

    if not client.has_collection(collection):
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field("title", DataType.VARCHAR, max_length=512)
        schema.add_field("source", DataType.VARCHAR, max_length=1024)
        schema.add_field("page", DataType.INT64)
        schema.add_field("section", DataType.VARCHAR, max_length=1024)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("chunk_type", DataType.VARCHAR, max_length=16)
        schema.add_field("text", DataType.VARCHAR, max_length=8192)
        client.create_collection(collection_name=collection, schema=schema)

    indexes = client.list_indexes(collection)
    if "vector" not in indexes:
        params = client.prepare_index_params()
        params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        client.create_index(collection_name=collection, index_params=params)


def ingest(
    chunks: list[Chunk],
    milvus_uri: str,
    embedding_url: str,
    embedding_model_name: str,
    collection: str,
) -> None:
    from pymilvus import MilvusClient

    embedder = EmbeddingClient(embedding_url, embedding_model_name)
    milvus = MilvusClient(uri=milvus_uri)
    ensure_collection(milvus, collection, embedder.dimension)

    completed = 0
    for batch in batched(chunks, EMBEDDING_BATCH_SIZE):
        vectors = embedder.encode([embedding_text(chunk) for chunk in batch])
        rows = [
            {
                **asdict(chunk),
                "vector": vector.tolist(),
            }
            for chunk, vector in zip(batch, vectors)
        ]
        milvus.upsert(collection_name=collection, data=rows)
        completed += len(batch)
        print(f"已写入 {completed}/{len(chunks)}", flush=True)
    milvus.flush(collection)
    milvus.load_collection(collection)


def main() -> None:
    parser = argparse.ArgumentParser(description="独立的医疗 PDF 分块与 Milvus 入库工具")
    parser.add_argument("--pdf-dir", type=Path, default=PROJECT_ROOT / "data_set")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "chunks" / "standalone_chunks.jsonl",
    )
    parser.add_argument("--milvus-uri", default=MILVUS_URI)
    parser.add_argument("--embedding-url", default=EMBEDDING_URL)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--dry-run", action="store_true", help="只解析并写 JSONL，不连接内网服务")
    args = parser.parse_args()

    pdf_files = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"目录中没有 PDF: {args.pdf_dir}")

    chunks = [chunk for path in pdf_files for chunk in parse_pdf(path)]
    args.output.write_text(
        "\n".join(json.dumps(asdict(chunk), ensure_ascii=False) for chunk in chunks) + "\n",
        encoding="utf-8",
    )
    lengths = sorted(len(chunk.text) for chunk in chunks)
    print(
        json.dumps(
            {
                "pdfs": len(pdf_files),
                "chunks": len(chunks),
                "min_chars": lengths[0],
                "median_chars": lengths[len(lengths) // 2],
                "max_chars": lengths[-1],
                "jsonl": str(args.output),
                "milvus": args.milvus_uri,
                "embedding": args.embedding_url,
                "collection": args.collection,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not args.dry_run:
        ingest(
            chunks,
            args.milvus_uri,
            args.embedding_url,
            args.embedding_model,
            args.collection,
        )


if __name__ == "__main__":
    main()
