# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
模块：表格分块（工单3 核心新增）

工单要求「不要把表格的 HTML 直接丢进向量库，必须做结构化表示」。本模块把
`table_parser` 产出的结构化表格转成**三类 chunk**：

  1. 表头块（table_header）——「表：X | 表头：列A｜列B｜列C | 页码：N」
     目的：支持"按列名检索"（如问"发行股数"直接命中含该列名的表）。

  2. 行级块（table_row）——「表：X | 列A：值 | 列B：值 | 页码：N」
     目的：检索的主单元。一行一句自然语言，向量与 BM25 都能命中；
     携带 row_index，可精确定位到"哪一行"。

  3. 整表（作为行级块/表头块的 parent_text，并保留 table_html 进 payload）
     目的：① LLM 生成时喂**整表结构**而非孤立一行（表格题的关键）；
           ② HTML 留作溯源与还原。

父子结构复用 01/02 已有的「子块检索 + 父块上下文」机制：
    parent_id   = table_id（如 "招股说明书2.pdf#p1#t0"）
    child.text  = 行级描述（检索单元）
    parent_text = 整表 Markdown（送 LLM 的上下文）
于是 context.expand_parents 无需特判即可把"命中某一行"自动还原为"整张表"。

表头块与行级块同属一个 parent（同一个 table_id），命中任一行都会带出整表。

降级：结构化失败（低质量且 OCR 兜底也拿不到内容）时，退回 01/02 的整表文本块
（block_type="table"），保证链路不中断（工单"容错机制"要求）。
"""
from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

from typing import Sequence

from src import config, table_parser
from src.table_parser import TableStruct

# 工单要求：表格 chunk 必须携带的元数据字段
TABLE_META_FIELDS = ("table_id", "table_html", "table_header", "row_index",
                     "table_caption", "table_quality", "table_n_rows", "from_ocr")


def make_table_id(source: str, page_idx: int, seq_on_page: int) -> str:
    """表格在文档中的唯一编号：{source}#p{page_idx}#t{seq_on_page}。"""
    return f"{source}#p{int(page_idx)}#t{int(seq_on_page)}"


def resolve_caption(item: dict, heading_path: Sequence[str]) -> str:
    """确定表格标题：MinerU 的 table_caption 优先，为空时回退到所在章节标题。

    为什么需要回退（工单3 实测）：招股2 的绝大多数表格 `table_caption` 为空，
    于是行级描述长成「**表：（无标题）** | 关联方名称：赵马克 | 持股比例：42.35%…」——
    "（无标题）"既占字数又稀释语义，表头块的"按表名检索"也失效。
    改用所属章节标题（如"二、关联方及关联交易"）后，表格块自带准确的语境词。
    """
    cap = " ".join(item.get("caption") or []).strip()
    if not cap:
        cap = str(item.get("table_caption") or "").strip()
    if not cap and heading_path:
        cap = str(heading_path[-1]).strip()
    return cap


def _base_meta(struct: TableStruct, table_id: str, html: str,
               n_rows: int) -> dict:
    """所有表格 chunk 共用的元数据（工单硬要求 8 项）。"""
    return {
        "table_id": table_id,
        "table_html": html or "",
        "table_header": "｜".join(c for c in struct.header if c),
        "table_caption": struct.caption,
        "table_quality": struct.quality,
        "table_n_rows": n_rows,
        "from_ocr": bool(struct.from_ocr),
        "row_index": -1,
    }


def build_table_chunks(struct: TableStruct, *, source: str, page_idx: int,
                       heading_path: Sequence[str], table_id: str,
                       chunk_id_start: int, seq_start: int,
                       html: str = "") -> list[dict]:
    """结构化表格 → chunk 列表（表头块 + 行级块）。

    chunk_id 由调用方给定起始序号，保证全库唯一且稳定可复现。
    """
    page_display = int(page_idx) + 1
    parent_text = table_parser.table_to_markdown(struct, config.TABLE_CTX_MAX_CHARS)
    meta = _base_meta(struct, table_id, html, len(struct.rows))
    chunks: list[dict] = []
    next_id = chunk_id_start
    seq = seq_start

    def _chunk(text: str, block_type: str, row_index: int) -> dict:
        nonlocal next_id, seq
        c = {
            "chunk_id": f"c{next_id:06d}",
            "text": text,
            "parent_id": table_id,
            "parent_text": parent_text,
            "parent_page_idx": int(page_idx),
            "page_idx": int(page_idx),
            "heading_path": list(heading_path),
            "source": source,
            "block_type": block_type,
            "seq": seq,
        }
        c.update(meta)
        c["row_index"] = row_index
        next_id += 1
        seq += 1
        return c

    # 1) 表头块（按列名检索）
    chunks.append(_chunk(table_parser.table_heading_text(struct, page_display),
                         "table_header", -1))

    # 2) 行级块（检索主单元）
    for i, row in enumerate(struct.rows):
        desc = table_parser.row_description(struct, row, i, page_display)
        if not desc.strip():
            continue
        chunks.append(_chunk(desc, "table_row", i))

    return chunks


def build_fallback_chunk(text: str, *, source: str, page_idx: int,
                         heading_path: Sequence[str], table_id: str,
                         chunk_id: int, seq: int, html: str = "",
                         caption: str = "", quality: float = 0.0,
                         reason: str = "") -> dict:
    """降级块：表格无法结构化时（低质量 + 无 OCR 结果）退回整表文本块。

    与 01/02 的行为一致（表格整块保留、超长截断），保证「解析失败」不中断链路。
    """
    body = (text or "").strip()
    if len(body) > config.TABLE_MAX_CHARS:
        head = body[: config.TABLE_MAX_CHARS - 200]
        tail = body[-150:]
        body = (f"{head}\n…（表格过长，中间省略 "
                f"{len(text) - len(head) - len(tail)} 字）\n{tail}")
    return {
        "chunk_id": f"c{chunk_id:06d}",
        "text": body,
        "parent_id": table_id,
        "parent_text": body,
        "parent_page_idx": int(page_idx),
        "page_idx": int(page_idx),
        "heading_path": list(heading_path),
        "source": source,
        "block_type": "table",
        "seq": seq,
        "table_id": table_id,
        "table_html": html or "",
        "table_header": "",
        "row_index": -1,
        "table_caption": caption,
        "table_quality": quality,
        "table_n_rows": 0,
        "from_ocr": bool(reason == "ocr"),
        "degrade_reason": reason,
    }


def chunk_table_item(item: dict, *, source: str, heading_path: Sequence[str],
                     table_id: str, chunk_id_start: int, seq_start: int,
                     ocr_text: str = "") -> list[dict]:
    """解析器条目 → 表格 chunk 列表（对外主入口）。

    item 需含：table_html（原始 HTML，工单3 保留）、text（降级用文本）、page_idx。
    ocr_text：该页 PaddleOCR-VL 兜底结果（可为空字符串）。
    """
    page_idx = int(item.get("page_idx", 0))
    caption = resolve_caption(item, heading_path)
    html = item.get("table_html") or ""

    if config.TABLE_ENABLED and html.strip().lower().startswith("<table"):
        struct = table_parser.parse_table_html(html, caption)
        if not struct.is_usable and ocr_text:
            # OCR 兜底：MinerU 的 HTML 解析不出结构，用 PaddleOCR-VL 对该页的识别
            # 结果作为表格内容（保留 table_id/页码等元数据，仍可溯源到该表）。
            return [build_fallback_chunk(ocr_text, source=source, page_idx=page_idx,
                                         heading_path=heading_path, table_id=table_id,
                                         chunk_id=chunk_id_start, seq=seq_start,
                                         html=html, caption=caption, reason="ocr")]
        if struct.is_usable:
            return build_table_chunks(struct, source=source, page_idx=page_idx,
                                      heading_path=heading_path, table_id=table_id,
                                      chunk_id_start=chunk_id_start,
                                      seq_start=seq_start, html=html)
        # 结构化失败且无 OCR：降级为整表文本块
        return [build_fallback_chunk(item.get("text", ""), source=source,
                                     page_idx=page_idx, heading_path=heading_path,
                                     table_id=table_id, chunk_id=chunk_id_start,
                                     seq=seq_start, html=html, caption=caption,
                                     quality=struct.quality, reason="parse_failed")]

    # 无 HTML（MinerU 未产出 table_body）或表格结构化关闭 → 降级
    reason = "disabled" if not config.TABLE_ENABLED else "no_html"
    return [build_fallback_chunk(item.get("text", ""), source=source,
                                 page_idx=page_idx, heading_path=heading_path,
                                 table_id=table_id, chunk_id=chunk_id_start,
                                 seq=seq_start, html=html, caption=caption,
                                 reason=reason)]


def analyze_tables(items: Sequence[dict], source: str,
                   headings: dict[int, list[str]] | None = None
                   ) -> tuple[list[dict], list[int]]:
    """建库前扫描：产出每张表的结构化摘要 + 低质量表所在页列表。

    返回 (表格记录列表, 需要 PaddleOCR-VL 兜底的页码列表)。
    低质量表页码交给 build_index 批量 OCR（按页去重，一次子进程调用处理多页）。

    headings：{条目下标: 该条目的章节标题路径}，用于给无标题表格补标题（见
    resolve_caption）。不传时退化为只读 MinerU 的 table_caption。
    """
    records: list[dict] = []
    low_pages: list[int] = []
    seq_by_page: dict[int, int] = {}
    headings = headings or {}

    for idx, it in enumerate(items):
        if it.get("type") != "table":
            continue
        page_idx = int(it.get("page_idx", 0))
        t_no = seq_by_page.get(page_idx, 0)
        seq_by_page[page_idx] = t_no + 1
        table_id = make_table_id(source, page_idx, t_no)
        caption = resolve_caption(it, headings.get(idx) or [])
        html = it.get("table_html") or ""

        struct = (table_parser.parse_table_html(html, caption)
                  if html.strip().lower().startswith("<table")
                  else TableStruct(caption=caption))
        rec = table_parser.to_json_dict(struct)
        low = table_parser.is_low_quality(struct)
        rec.update({"table_id": table_id, "source": source, "page_idx": page_idx,
                    "page_display": page_idx + 1,
                    "low_quality": low, "has_html": bool(html)})
        records.append(rec)
        # 只有「**有 HTML 但解析不出结构**」才是真正的表格识别失败，值得调
        # PaddleOCR-VL 重识别。完全无 table_body 的块（招股1 实测 113 个）多是把
        # 双栏正文误判成表格，页面文本已由 pdf_parser 的 PyMuPDF 补齐路径找回，
        # 再整页 OCR 一遍既慢又无增益。
        if low and html:
            low_pages.append(page_idx)
    return records, sorted(set(low_pages))


def stats(chunks: Sequence[dict]) -> dict:
    """表格分块统计（建库日志与专项说明用）。"""
    rows = [c for c in chunks if c.get("block_type") == "table_row"]
    heads = [c for c in chunks if c.get("block_type") == "table_header"]
    fallen = [c for c in chunks if c.get("block_type") == "table"
              and c.get("table_id")]
    tables = {c.get("table_id") for c in chunks if c.get("table_id")}
    ocr = [c for c in chunks if c.get("from_ocr")]
    return {
        "tables": len(tables),
        "header_chunks": len(heads),
        "row_chunks": len(rows),
        "fallback_chunks": len(fallen),
        "ocr_chunks": len(ocr),
    }
