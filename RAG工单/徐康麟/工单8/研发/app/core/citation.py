# -*- coding: utf-8 -*-
"""工单4 引用构造与回溯（页码口径：**1-based 物理页码**）。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

设计依据（设计/接口设计.md §3.3 / §4.2 / §5.3）：
    * ``citations[].page`` 必须是 **1-based 物理页码**，``page_basis`` 固定 ``"1-based-physical"``；
    * 图像证据必须能作为引用来源，形如 ``[招股说明书2.pdf: 第39页 组织结构图]``；
    * 「引用页码 == golden 证据页」**不得**当成检索命中（那是 测试/common/golden.py 的判据），
      本模块只负责把**真实取用到的 chunk** 转成可回溯引用，并做页码合法性校验。

页码口径事实（设计/环境事实.md §6）：印刷页码 = 物理页码 − 1；引用一律用物理页码。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .errors import CitationError

WORK_ORDER = "人工智能NLP-RAG-基于Graph RAG 实现金融问答"

PAGE_BASIS = "1-based-physical"

#: 图像模态的中文标注（引用里让人一眼看出「这是图里的证据」）
MODALITY_SUB_LABEL: dict[str, str] = {
    "diagram": "组织结构图",
    "chart": "图表",
    "picture": "插图",
}
MODALITY_LABEL: dict[str, str] = {"text": "正文", "table": "表格", "image": "图像"}


def _lazy_logger(logger: Any, module: str = "citation") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


@dataclass(slots=True)
class Citation:
    """一条可回溯引用。"""

    corpus: str                 # 文件名（如 招股说明书2.pdf）
    page: int                   # 1-based 物理页码
    chunk_id: str
    modality: str               # text | table | image
    modality_sub: str | None = None
    page_basis: str = PAGE_BASIS
    quote: str = ""
    xref: int | None = None
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def display(self) -> str:
        """人类可读引用串：``[招股说明书2.pdf: 第39页 组织结构图]``。"""
        page = _format_page(self.page)
        if self.modality == "image" and self.modality_sub:
            sub = MODALITY_SUB_LABEL.get(self.modality_sub, MODALITY_LABEL.get(self.modality, "图像"))
            return f"[{self.corpus}: 第{page}页 {sub}]"
        if self.modality == "table":
            return f"[{self.corpus}: 第{page}页 表格]"
        return f"[{self.corpus}: 第{page}页]"


def _format_page(page: int) -> str:
    """页码显示：支持跨页（``page`` 为 int；跨页由调用方给出 page_end 时用 ``_page_range``）。"""
    return str(int(page))


def _page_range(start: int, end: int | None) -> str:
    if end is None or int(end) == int(start):
        return str(int(start))
    return f"{int(start)}-{int(end)}"


def _clean_quote(text: str, *, limit: int = 120) -> str:
    """引用摘句：压平空白、截断（只用于展示，不参与判分）。"""
    flat = " ".join(str(text or "").split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


def build_citations(hits: Sequence[Mapping[str, Any]], *, used_chunk_ids: Iterable[str] | None = None,
                    max_citations: int = 5, quote_limit: int = 120, logger: Any = None) -> list[Citation]:
    """把命中的 chunk（dict）转成引用列表。

    * 只保留**真正被答案取用**的 chunk（``used_chunk_ids`` 给出）；不给则全部保留；
    * 按 (corpus, page) 去重，保留首个（更相关的）引用；
    * 页码必须 >= 1，否则抛 ``CitationError``（0-based 页码是缺陷，不是容错项）。
    """
    log = _lazy_logger(logger)
    allowed = set(used_chunk_ids) if used_chunk_ids is not None else None
    with log.enter("build_citations", {"hits": len(hits),
                                       "used": len(allowed) if allowed is not None else None}) as span:
        seen: set[tuple[str, int]] = set()
        out: list[Citation] = []
        for hit in hits:
            cid = str(hit.get("chunk_id") or "")
            if allowed is not None and cid not in allowed:
                continue
            page = hit.get("page")
            try:
                page_int = int(page)
            except (TypeError, ValueError) as exc:
                raise CitationError(f"引用页码非法（必须是整数）：{page!r}（chunk={cid}）",
                                    detail={"chunk_id": cid, "page": page}) from exc
            if page_int < 1:
                raise CitationError(
                    f"引用页码必须是 1-based 物理页码，实际 {page_int}（chunk={cid}）",
                    detail={"chunk_id": cid, "page": page_int, "page_basis": PAGE_BASIS},
                )
            corpus = str(hit.get("file_name") or "")
            key = (corpus, page_int)
            if key in seen:
                continue
            seen.add(key)
            modality = str(hit.get("type") or hit.get("modality") or "text")
            out.append(Citation(
                corpus=corpus, page=page_int, chunk_id=cid, modality=modality,
                modality_sub=(str(hit["modality_sub"]) if hit.get("modality_sub") else None),
                quote=_clean_quote(str(hit.get("content") or hit.get("text") or ""), limit=quote_limit),
                xref=(int(hit["xref"]) if hit.get("xref") is not None else None),
            ))
            if len(out) >= max_citations:
                break
        for cite in out:
            cite.label = cite.display()
        log.log_event("citation.built", count=len(out),
                      pages=[c.page for c in out], corpora=sorted({c.corpus for c in out}))
        span.set_output({"count": len(out), "labels": [c.display() for c in out]})
        return out


def citation_marker(citations: Sequence[Citation]) -> str:
    """答案正文末尾的引用行，如 ``引用：[招股说明书2.pdf: 第39页 组织结构图]``。"""
    if not citations:
        return ""
    return "引用：" + " ".join(c.display() for c in citations)


def cited_pages(citations: Sequence[Citation]) -> list[int]:
    """去重升序的引用页码（供评测/报告与 UI 显示）。"""
    return sorted({int(c.page) for c in citations})


def format_citation(corpus: str, page: int, *, modality: str = "text",
                    modality_sub: str | None = None) -> str:
    """便捷函数：``format_citation("招股说明书2.pdf", 39, modality="image", modality_sub="diagram")``。"""
    if int(page) < 1:
        raise CitationError(f"页码必须是 1-based 物理页码，实际 {page}",
                            detail={"page": int(page), "page_basis": PAGE_BASIS})
    label = ""
    if modality == "image" and modality_sub:
        label = " " + MODALITY_SUB_LABEL.get(modality_sub, "图像")
    elif modality == "table":
        label = " 表格"
    return f"[{corpus}: 第{int(page)}页{label}]"
