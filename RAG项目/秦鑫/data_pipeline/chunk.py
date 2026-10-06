from __future__ import annotations

import re
import uuid


LAW_COLLECTIONS = {
    "civil_code_articles", "civil_interpretations", "civil_cases", "civil_elements",
    "civil_evidence", "civil_processes", "civil_questions", "civil_citations",
}
SINGLE_RECORD_TYPES = {"law", "case", "evidence"}
SENTENCE_RE = re.compile(r".+?(?:[。！？!?；;]|$)")
BREAK_MARKS = ("。", "！", "？", "!", "?", "；", ";")
LONG_DOCUMENT_LENGTH = 5000
LONG_DOCUMENT_CHUNKS = 4
PARENT_TARGET_LENGTH = 3000


def fixed_chunks(text: str, size: int) -> list[str]:
    if size <= 0:
        raise ValueError("chunk size 必须大于 0")
    value = str(text or "").strip()
    return [value[offset:offset + size] for offset in range(0, len(value), size)] if value else []


def _split_long_unit(text: str, size: int) -> list[str]:
    remaining = str(text or "").strip()
    pieces: list[str] = []
    while len(remaining) > size:
        window = remaining[:size]
        boundary = max((window.rfind(mark) for mark in BREAK_MARKS), default=-1)
        boundary = size if boundary <= 0 else boundary + 1
        pieces.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _sentence_units(text: str) -> list[str]:
    units: list[str] = []
    for paragraph in re.split(r"\n+", str(text or "").strip()):
        value = paragraph.strip()
        if value:
            units.extend(part.strip() for part in SENTENCE_RE.findall(value) if part.strip())
    return units


def _merge_short_tail(chunks: list[str], minimum: int) -> list[str]:
    if len(chunks) < 2 or len(chunks[-1]) >= minimum:
        return chunks
    return [*chunks[:-2], f"{chunks[-2]}\n{chunks[-1]}".strip()]


def _semantic_chunks(text: str, size: int, minimum: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for unit in _sentence_units(text):
        for piece in _split_long_unit(unit, size):
            candidate = f"{current}\n{piece}".strip() if current else piece
            if not current or len(candidate) <= size:
                current = candidate
            else:
                chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return _merge_short_tail(chunks, min(size, max(1, minimum)))


def _group_parents(chunks: list[str], size: int = PARENT_TARGET_LENGTH) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    length = 0
    for chunk in chunks:
        candidate_length = length + len(chunk) + bool(current)
        if current and candidate_length > size:
            groups.append(current)
            current, length = [chunk], len(chunk)
        else:
            current.append(chunk)
            length = int(candidate_length)
    if current:
        groups.append(current)
    return groups


def make_chunk_id(file_id: str, index: int, text: str, role: str = "chunk") -> str:
    base = str(file_id or "file").strip() or "file"
    digest = uuid.uuid5(uuid.NAMESPACE_URL, f"{base}|{role}|{index}|{str(text or '').strip()}").hex[:12]
    return f"{base}_{role}_{index:03d}_{digest}"


def _chunk_row(metadata: dict, parsed: dict, index: int, text: str, chunk_id: str | None = None, **fields) -> dict:
    file_id = str(metadata.get("file_id") or metadata.get("document_id") or metadata.get("file_name") or "file")
    row = {
        "chunk_id": chunk_id or make_chunk_id(file_id, index, text),
        "chunk_index": index,
        "chunk_level": fields.pop("chunk_level", "chunk"),
        "text": text,
        "user_id": metadata.get("user_id", ""),
        "session_id": metadata.get("session_id", ""),
        "file_id": file_id,
        "file_name": metadata.get("file_name", ""),
        "document_type": parsed.get("document_type", "general"),
        "collection": parsed.get("collection", ""),
        "section": "",
        "page_start": 1,
        "page_end": 1,
    }
    row.update({key: value for key, value in fields.items() if value not in (None, "")})
    return row


def _structured_rows(parsed: dict, metadata: dict) -> list[dict]:
    output: list[dict] = []
    file_id = str(metadata.get("file_id") or metadata.get("document_id") or metadata.get("file_name") or "file")
    for index, source in enumerate(parsed.get("records") or parsed.get("rows") or []):
        if not isinstance(source, dict):
            continue
        text = str(source.get("embedding_text") or source.get("content") or source.get("summary") or source.get("case_summary") or "").strip()
        if not text:
            continue
        row = dict(source)
        row.setdefault("content", str(source.get("content") or source.get("summary") or source.get("case_summary") or text))
        row.setdefault("embedding_text", text)
        row.setdefault("collection", parsed.get("collection", ""))
        row.setdefault("document_type", parsed.get("document_type", "general"))
        row.setdefault("text", text)
        row.setdefault("chunk_id", str(source.get("id") or source.get("case_id") or source.get("evidence_id") or source.get("process_id") or source.get("question_id") or source.get("citation_id") or source.get("serial_number") or make_chunk_id(file_id, index, text)))
        row.update(user_id=metadata.get("user_id", ""), session_id=metadata.get("session_id", ""), file_id=metadata.get("file_id", ""), file_name=metadata.get("file_name", ""))
        output.append(row)
    return output


def _parent_child_rows(parsed: dict, metadata: dict, size: int, minimum: int) -> list[dict]:
    children = _semantic_chunks(parsed.get("text", ""), size, minimum)
    file_id = str(metadata.get("file_id") or metadata.get("document_id") or metadata.get("file_name") or "file")
    rows: list[dict] = []
    for parent_index, group in enumerate(_group_parents(children)):
        parent_text = "\n".join(group).strip()
        parent_id = make_chunk_id(file_id, len(rows), parent_text, "parent")
        rows.append(_chunk_row(metadata, parsed, len(rows), parent_text, parent_id, chunk_level="parent", parent_chunk_id=parent_id, parent_index=parent_index, child_count=len(group)))
        for child_index, child_text in enumerate(group):
            rows.append(_chunk_row(metadata, parsed, len(rows), child_text, chunk_level="child", parent_chunk_id=parent_id, parent_text=parent_text, parent_index=parent_index, child_index=child_index))
    return rows


def chunk(parsed: dict, metadata: dict | None = None, size: int = 1000, minimum: int = 120, *, document_type: str | None = None, collection: str | None = None) -> list[dict]:
    if size <= 0:
        raise ValueError("chunk size 必须大于 0")
    base = metadata or {}
    resolved_type = document_type or parsed.get("document_type", "general")
    resolved_collection = collection or parsed.get("collection", "")
    if resolved_collection in LAW_COLLECTIONS or (resolved_collection and resolved_type in SINGLE_RECORD_TYPES):
        return _structured_rows(parsed, base)
    chunks = _semantic_chunks(parsed.get("text", ""), size, minimum)
    if not chunks:
        return []
    if len(str(parsed.get("text", ""))) >= LONG_DOCUMENT_LENGTH or len(chunks) >= LONG_DOCUMENT_CHUNKS:
        return _parent_child_rows(parsed, base, size, minimum)
    return [_chunk_row(base, parsed, index, text) for index, text in enumerate(chunks)]


def chunk_document(parsed: dict, metadata: dict | None = None, chunk_size: int = 1000, minimum: int = 120) -> list[dict]:
    return chunk(parsed, metadata, size=chunk_size, minimum=minimum)


__all__ = ["chunk", "chunk_document", "fixed_chunks", "make_chunk_id"]
