# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""PDF 加载与文本块抽取。逐页容错，坏页跳过不中断（硬性要求 7）。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import fitz

from rag04.schema import TextBlock

logger = logging.getLogger("rag04.loader")

# 旋转判定阈值：方向向量与 (1,0) 的夹角余弦小于该值即视为旋转
_ROT_COS_THRESHOLD = 0.9


class PdfOpenError(Exception):
    """PDF 无法打开（不存在、非 PDF、加密且无法解密）。"""


def doc_id_of(pdf_path: Path) -> str:
    """稳定文档 ID：去掉扩展名的文件名。"""
    return Path(pdf_path).stem


def open_pdf(pdf_path: Path) -> fitz.Document:
    """打开 PDF。失败抛 PdfOpenError，绝不返回半开状态。"""
    p = Path(pdf_path)
    if not p.exists():
        raise PdfOpenError(f"文件不存在：{p}")
    try:
        doc = fitz.open(p)
    except Exception as e:  # 损坏或非 PDF
        raise PdfOpenError(f"无法解析 PDF：{p}（{type(e).__name__}: {e}）") from e

    if doc.is_encrypted:
        # 空密码尝试解密；失败则明确报错
        if not doc.authenticate(""):
            doc.close()
            raise PdfOpenError(f"PDF 已加密且无法解密：{p}")
    if doc.page_count == 0:
        doc.close()
        raise PdfOpenError(f"PDF 无有效页面：{p}")
    return doc


def classify_span_dir(span: dict) -> bool:
    """判断文本片段是否为旋转/竖排文本（非水平方向）。

    注意：PyMuPDF 1.25 的 `get_text("dict")` 把 `dir` / `wmode` 放在 **line**
    字典上，span 字典并没有这两个键。因此本函数同时接受 line 与 span 两种
    字典，优先看竖排书写模式 `wmode`，再看方向向量与 (1,0) 的夹角余弦。
    """
    if span.get("wmode"):  # 1=竖排书写模式（CJK 竖排）
        return True
    d = span.get("dir")
    if not d:
        return False
    dx, dy = d
    norm = (dx * dx + dy * dy) ** 0.5
    if norm == 0:
        return False
    return (dx / norm) < _ROT_COS_THRESHOLD


def _page_blocks(page: fitz.Page, doc_id: str, pno: int) -> list[TextBlock]:
    """抽取单页文本块，按块聚合并标记旋转。"""
    raw = page.get_text("dict")
    out: list[TextBlock] = []
    for blk in raw.get("blocks", []):
        if blk.get("type") != 0:  # 0=文本, 1=图像
            continue
        lines = blk.get("lines", [])
        if not lines:
            continue
        parts: list[str] = []
        rotated = False
        for ln in lines:
            # line 字典才带 dir / wmode（PyMuPDF 1.25）
            if classify_span_dir(ln):
                rotated = True
            for sp in ln.get("spans", []):
                t = sp.get("text", "")
                if t:
                    parts.append(t)
                if classify_span_dir(sp):
                    rotated = True
        text = "".join(parts).strip()
        if not text:
            continue
        x0, y0, x1, y1 = blk["bbox"]
        out.append(TextBlock(
            doc_id=doc_id, page=pno, bbox=(x0, y0, x1, y1),
            text=text, has_rotated_text=rotated,
        ))
    return out


def iter_text_blocks(doc: fitz.Document, doc_id: str) -> Iterator[TextBlock]:
    """逐页产出文本块。任一页异常仅记录并跳过，不中断整体解析。"""
    for i in range(doc.page_count):
        pno = i + 1
        try:
            page = doc[i]
            yield from _page_blocks(page, doc_id, pno)
        except Exception as e:
            logger.warning("第 %d 页解析失败，已跳过：%s: %s", pno, type(e).__name__, e)
            continue
