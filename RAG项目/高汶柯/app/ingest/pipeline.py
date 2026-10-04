"""离线入库流水线：解析 → 清洗 → 分块 → 摘要/增强 → 向量化 → 入库。"""
from __future__ import annotations

import os
import uuid

from app.config import settings
from app.core.registry import get_embedder, get_llm
from app.db.milvus_store import milvus_store
from app.db.mongo_store import mongo_store
from app.db.mysql_store import delete_kb_doc, upsert_kb_doc
from app.ingest import clean
from app.ingest.chunker import chunk_parent_child, chunk_text
from app.ingest.parsers import parse_document
from app.logging_conf import log

_SUMMARY_PROMPT = "请用一句话（不超过60字）概括下面内容，只输出摘要本身：\n{text}"
_AUGMENT_PROMPT = (
    "根据下面内容，生成 3 个用户可能提出的、需要该内容才能回答的问题。"
    "每行一个，不要编号：\n{text}"
)


def _summarize(texts: list[str]) -> list[str]:
    """为分块生成摘要；LLM 不可用或超出上限时使用截断兜底。"""
    if not settings.enable_summary:
        return [t[:60] for t in texts]
    llm = None
    out: list[str] = []
    for i, t in enumerate(texts):
        if i < settings.max_summary_chunks:
            try:
                if llm is None:
                    llm = get_llm()
                summary = llm.chat(
                    [{"role": "user", "content": _SUMMARY_PROMPT.format(text=t[:1000])}],
                    temperature=0.2,
                ).strip()
                out.append(summary[:200])
                continue
            except Exception as exc:  # noqa: BLE001
                log.warning("摘要生成失败，改用截断: %s", exc)
                llm = None
        out.append(t[:60])
    return out


def _augment(base_records: list[dict]) -> list[dict]:
    """生成假设问题作为额外知识块，提升召回率。"""
    llm = None
    questions: list[str] = []
    for rec in base_records:
        try:
            if llm is None:
                llm = get_llm()
            raw = llm.chat(
                [{"role": "user", "content": _AUGMENT_PROMPT.format(text=rec["text"][:800])}],
                temperature=0.6,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("数据增强失败: %s", exc)
            break
        for line in raw.splitlines():
            q = line.strip().lstrip("-*0123456789.、) ").strip()
            if len(q) >= 6:
                questions.append(q)
    if not questions:
        return []
    vectors = get_embedder().encode(questions)
    return [
        {
            "text": q,
            "parent": base_records[i % len(base_records)]["text"],
            "summary": "",
            "source": base_records[i % len(base_records)]["source"],
            "domain": base_records[i % len(base_records)]["domain"],
            "doc_id": base_records[i % len(base_records)]["doc_id"],
            "chunk_type": "augment",
            "parser": base_records[i % len(base_records)]["parser"],
            "vector": v,
        }
        for i, (q, v) in enumerate(zip(questions, vectors))
    ]


def ingest_file(
    path: str, domain: str = "general", parser: str = "auto",
    doc_id: str | None = None, strategy: str | None = None,
) -> dict:
    """解析单个文件并写入知识库。"""
    filename = os.path.basename(path)
    doc_id = doc_id or uuid.uuid4().hex[:16]
    parsed = parse_document(path, parser)
    text = clean.normalize(parsed["text"])
    if not text:
        return {"ok": False, "msg": "未解析到文本", "doc_id": doc_id, "chunks": 0, "parser": parsed["parser"]}

    chunks = (
        chunk_parent_child(text, strategy=strategy)
        if settings.parent_child
        else chunk_text(text, strategy=strategy)
    )
    chunks = clean.clean_chunks(chunks)
    if not chunks:
        return {"ok": False, "msg": "分块为空", "doc_id": doc_id, "chunks": 0, "parser": parsed["parser"]}

    texts = [c["text"] for c in chunks]
    summaries = _summarize(texts)
    vectors = get_embedder().encode(texts)

    records: list[dict] = []
    for chunk, summary, vector in zip(chunks, summaries, vectors):
        records.append(
            {
                "text": chunk["text"],
                "parent": chunk.get("parent", ""),
                "summary": summary,
                "source": filename,
                "domain": domain,
                "doc_id": doc_id,
                "chunk_type": chunk.get("chunk_type", "sentence"),
                "parser": parsed["parser"],
                "vector": vector,
            }
        )

    if settings.enable_augment:
        records.extend(_augment(records[: settings.augment_max]))

    inserted = milvus_store.insert(records)
    mongo_store.save_documents(
        [{"doc_id": doc_id, "source": filename, "domain": domain, "text": text}]
    )
    upsert_kb_doc(doc_id, filename, domain, parsed["parser"], inserted)
    return {
        "ok": True,
        "msg": f"已入库 {inserted} 条",
        "doc_id": doc_id,
        "chunks": inserted,
        "parser": parsed["parser"],
    }


def ingest_directory(dir_path: str, domain: str = "general", parser: str = "auto") -> list[dict]:
    """批量入库目录下的 PDF 文件。"""
    results: list[dict] = []
    if not os.path.isdir(dir_path):
        log.warning("目录不存在: %s", dir_path)
        return results
    for fn in sorted(os.listdir(dir_path)):
        if not fn.lower().endswith(".pdf"):
            continue
        try:
            results.append(ingest_file(os.path.join(dir_path, fn), domain=domain, parser=parser))
        except Exception as exc:  # noqa: BLE001
            log.error("入库 %s 失败: %s", fn, exc)
            results.append({"ok": False, "msg": str(exc), "doc_id": "", "chunks": 0, "parser": ""})
    return results


def delete_document(doc_id: str) -> None:
    """删除某文档的全部知识块与元数据。"""
    milvus_store.delete_by_doc(doc_id)
    mongo_store.delete_documents(doc_id)
    delete_kb_doc(doc_id)
