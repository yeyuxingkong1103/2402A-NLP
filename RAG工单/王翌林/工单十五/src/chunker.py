# -*- coding: utf-8 -*-
"""
工单十五：文本分块器
将 OCR 结果按页面 + 语义段落分块，记录图号引用
"""
import re
from typing import List, Dict


def _split_paragraphs(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    paras = []
    for p in raw:
        lines = [l.strip() for l in p.split("\n") if l.strip()]
        paras.append("".join(lines))
    return paras


def chunk_ocr_results(ocr_results: List[Dict], doc_id: str, filename: str,
                      max_chars: int = 500) -> Dict:
    chunks = []
    idx = 0
    for r in ocr_results:
        page = r["page"]
        text = r.get("text", "")
        if not text.strip():
            continue
        paras = _split_paragraphs(text)
        buf = ""
        for para in paras:
            if len(buf) + len(para) > max_chars and buf:
                chunks.append(_make_chunk(idx, doc_id, filename, page, buf))
                idx += 1
                buf = para
            else:
                buf = (buf + "\n" + para).strip() if buf else para
        if buf:
            chunks.append(_make_chunk(idx, doc_id, filename, page, buf))
            idx += 1
    return {"doc_id": doc_id, "filename": filename, "chunks": chunks,
            "chunk_count": len(chunks)}


def _make_chunk(idx, doc_id, filename, page, text):
    return {
        "chunk_id": f"{doc_id}_p{page}_{idx}",
        "doc_id": doc_id,
        "filename": filename,
        "page": page,
        "text": text,
        "source": f"{filename}#page={page}",
        "chunk_index": idx,
        "char_count": len(text),
        "img_refs": _detect_fig_refs(text),
    }


def _detect_fig_refs(text: str) -> List[str]:
    refs = set()
    for m in re.finditer(r"图\s*(\d+)", text):
        refs.add(f"fig_{m.group(1)}")
    for m in re.finditer(r"附图\s*(\d+)", text):
        refs.add(f"fig_{m.group(1)}")
    return sorted(refs)
