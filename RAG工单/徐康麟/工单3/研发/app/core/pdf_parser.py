# -*- coding: utf-8 -*-
"""工单3 PDF 逐页解析与产物落盘（设计/接口设计.md §2.2、§3.5、§5.2/§5.3 冻结）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

硬约束：
    * **页码 1-based 物理页**：``PageText.page = i + 1``，取页一律 ``doc[i]``，越界抛 ChunkError；
    * ``page.find_tables()`` **每页都调用**（表格密集且跨页是常态，禁止抽样）；
    * 跨页续表由 ``table_parser.build_table_blocks`` 按 R7/R8/R9 合并为逻辑表块；
    * 单页失败 → WARN + 空文本并留痕；**整文件失败 → 抛 PdfParseError**；
    * 本机无 ``sqlite_manager``（T7 交付）：此处只做 ``documents/pages/tables`` 的最小幂等落库，
      DDL 与设计 §5.3 完全一致（``CREATE TABLE IF NOT EXISTS``），T7 可直接复用同一库。
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import asdict, dataclass, field, fields as dc_fields
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from . import table_parser
from .config import AppConfig, PdfSource, discover_pdfs, get_config
from .errors import ChunkError, PdfParseError, StorageError, wrap
from .table_parser import TableBlock
from .text_utils import write_jsonl

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

# 设计 §5.3 冻结 DDL（表名/列名不可改；新增列允许）
SQLITE_DDL: str = """
CREATE TABLE IF NOT EXISTS documents (
  file_name TEXT PRIMARY KEY, size_bytes INTEGER NOT NULL, sha256_16 TEXT NOT NULL,
  page_count INTEGER NOT NULL, parsed_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS pages (
  file_name TEXT NOT NULL, page INTEGER NOT NULL, text TEXT NOT NULL,
  char_count INTEGER NOT NULL, table_count INTEGER NOT NULL,
  PRIMARY KEY (file_name, page));
CREATE TABLE IF NOT EXISTS tables (
  table_id TEXT PRIMARY KEY, file_name TEXT NOT NULL, page INTEGER NOT NULL,
  page_start INTEGER NOT NULL, page_end INTEGER NOT NULL,
  n_rows INTEGER NOT NULL, n_cols INTEGER NOT NULL, logical_cols INTEGER NOT NULL,
  markdown TEXT NOT NULL, title TEXT NOT NULL, key_numbers TEXT, degenerate INTEGER NOT NULL DEFAULT 0,
  continued_from TEXT, header_inherited INTEGER NOT NULL DEFAULT 0, row_pages TEXT);
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY, file_name TEXT NOT NULL, page INTEGER NOT NULL,
  page_start INTEGER NOT NULL, page_end INTEGER NOT NULL,
  type TEXT NOT NULL, section TEXT, content TEXT NOT NULL, keywords TEXT,
  table_id TEXT, char_count INTEGER NOT NULL, ord INTEGER NOT NULL, source_block_ids TEXT);
CREATE INDEX IF NOT EXISTS idx_chunks_file_page ON chunks(file_name, page);
CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(type);
CREATE TABLE IF NOT EXISTS conversations (
  session_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  message_count INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL,
  content TEXT NOT NULL, citations_json TEXT, first_token_ms REAL, total_ms REAL,
  created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS retrieval_traces (
  trace_id TEXT PRIMARY KEY, session_id TEXT, question TEXT NOT NULL, rewritten_query TEXT,
  file_names TEXT, top_k INTEGER NOT NULL, chunk_ids_json TEXT NOT NULL, scores_json TEXT,
  stages_json TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, kind TEXT NOT NULL, started_at TEXT NOT NULL,
  finished_at TEXT, stats_json TEXT, ok INTEGER);
"""


# ---------------------------------------------------------------------------
# 数据结构（设计 §2.2）
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class PageText:
    """单页文本（``page`` 为 1-based 物理页）。"""

    file_name: str
    page: int
    text: str
    char_count: int
    table_count: int
    has_table: bool
    parsed_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PageText":
        names = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


@dataclass(slots=True)
class TextBlock:
    """页内文本块（供分块聚合用，保留 bbox）。"""

    block_id: str
    file_name: str
    page: int
    block_index: int
    text: str
    bbox: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = list(self.bbox)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TextBlock":
        names = {f.name for f in dc_fields(cls)}
        payload = {k: v for k, v in data.items() if k in names}
        if "bbox" in payload:
            payload["bbox"] = tuple(payload["bbox"])
        return cls(**payload)


@dataclass(slots=True)
class PdfParseResult:
    """一份 PDF 的完整解析结果。"""

    file_name: str
    page_count: int
    pages: list[PageText]
    tables: list[TableBlock]
    text_blocks: list[TextBlock]
    elapsed_ms: float
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "page_count": self.page_count,
            "pages": len(self.pages),
            "tables": len(self.tables),
            "text_blocks": len(self.text_blocks),
            "elapsed_ms": self.elapsed_ms,
            "warnings": list(self.warnings),
            "stats": dict(self.stats),
        }


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _lazy_logger(logger: Any, module: str) -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _as_source(pdf: PdfSource | Path | str) -> PdfSource:
    """把 ``PdfSource | Path | str`` 统一成 PdfSource（不重新打开文件）。"""
    if isinstance(pdf, PdfSource):
        return pdf
    path = Path(pdf)
    if not path.is_file():
        raise PdfParseError(f"PDF 不存在：{path}", detail={"path": str(path)})
    return PdfSource(
        file_name=path.name, path=path, size_bytes=path.stat().st_size,
        sha256_16="", page_count=None, readable=True,
    )


def _open_doc(pdf: PdfSource | Path | str, *, logger: Any = None) -> Any:
    """打开 PDF（失败抛 PdfParseError）。优先 ``import pymupdf``，避免 fitz 废弃告警。"""
    source = _as_source(pdf)
    log = _lazy_logger(logger, "pdf_parser")
    try:
        import pymupdf
    except ImportError as exc:  # 显式失败，不静默
        raise PdfParseError(f"PyMuPDF 不可用：{exc}", detail={"file_name": source.file_name}) from exc
    try:
        doc = pymupdf.open(str(source.path))
    except Exception as exc:  # noqa: BLE001 —— 打不开就是致命错误
        log.log_event("pdf.open_failed", level="ERROR", file_name=source.file_name,
                      path=str(source.path), error_type=type(exc).__name__, message=str(exc))
        raise PdfParseError(f"PDF 打不开：{source.file_name}（{exc}）",
                            detail={"file_name": source.file_name, "path": str(source.path)}) from exc
    log.log_event("pdf.open", file_name=source.file_name, page_count=int(doc.page_count),
                  path=str(source.path))
    return doc


def _assert_page(page: int, page_count: int) -> int:
    """页码断言（1-based，越界抛 ChunkError，不静默钳制）。"""
    value = int(page)
    if value < 1 or value > int(page_count):
        raise ChunkError(f"页码越界：{value}（合法区间 1~{page_count}）",
                         detail={"page": value, "page_count": int(page_count)})
    return value


# ---------------------------------------------------------------------------
# 页级 API
# ---------------------------------------------------------------------------
def pdf_page_count(pdf: PdfSource | Path | str) -> int:
    """返回 PDF 页数（打开后立即关闭）。"""
    doc = _open_doc(pdf)
    try:
        return int(doc.page_count)
    finally:
        doc.close()


def get_page_text(pdf: PdfSource | Path | str, page: int) -> str:
    """取指定 **1-based 物理页** 的文本（``doc[page - 1]``）。"""
    doc = _open_doc(pdf)
    try:
        index = _assert_page(page, int(doc.page_count)) - 1
        return doc.load_page(index).get_text("text") or ""
    finally:
        doc.close()


def get_page_table_count(pdf: PdfSource | Path | str, page: int) -> int:
    """取指定 **1-based 物理页** 的 ``find_tables()`` 表数。"""
    doc = _open_doc(pdf)
    try:
        index = _assert_page(page, int(doc.page_count)) - 1
        return len(list(doc.load_page(index).find_tables().tables))
    finally:
        doc.close()


def iter_pdf_pages(
    pdf: PdfSource | Path | str,
    *,
    pages: Iterable[int] | None = None,
    scan_tables: bool = True,
    logger: Any = None,
) -> Iterator[PageText]:
    """逐页产出 ``PageText``（可选只跑若干 1-based 页码，用于调试）。"""
    log = _lazy_logger(logger, "pdf_parser")
    source = _as_source(pdf)
    doc = _open_doc(source, logger=log)
    try:
        total = int(doc.page_count)
        wanted = sorted({_assert_page(p, total) for p in pages}) if pages is not None else list(range(1, total + 1))
        for page_no in wanted:
            text = ""
            table_count = 0
            try:
                page = doc.load_page(page_no - 1)
                text = page.get_text("text") or ""
                if scan_tables:
                    table_count = len(list(page.find_tables().tables))
            except Exception as exc:  # noqa: BLE001 —— 单页失败显式降级，不静默
                log.log_event("pdf.page_error", level="ERROR", file_name=source.file_name, page=page_no,
                              error_type=type(exc).__name__, message=str(exc))
            yield PageText(
                file_name=source.file_name, page=page_no, text=text, char_count=len(text),
                table_count=table_count, has_table=table_count > 0, parsed_at=_now_iso(),
            )
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# 文件级解析
# ---------------------------------------------------------------------------
def group_continued_tables(
    raw_tables: Sequence[TableBlock],
    page_texts: Mapping[int, str],
    *,
    logger: Any = None,
) -> tuple[list[TableBlock], dict[str, int]]:
    """按 R7/R8/R9 把逐页候选表块合并为逻辑表块，并汇总统计口径。"""
    log = _lazy_logger(logger, "pdf_parser")
    with log.enter("group_continued_tables",
                   {"raw_tables": len(raw_tables), "pages": len(page_texts)}) as span:
        blocks, stats = table_parser.build_table_blocks(raw_tables, logger=log)
        table_pages = sorted({b.page_start for b in raw_tables})
        stats["table_pages"] = len(table_pages)
        # max_run 以「含表页」为准，与本仓库 §8.8 全量扫描口径一致
        best = run = 0
        prev: int | None = None
        for page in table_pages:
            run = run + 1 if prev is not None and page == prev + 1 else 1
            best = max(best, run)
            prev = page
        stats["max_run"] = best
        span.set_output({"logical_tables": len(blocks), **stats})
        return blocks, stats


def parse_pdf(
    pdf: PdfSource | Path | str,
    *,
    cfg: AppConfig | None = None,
    scan_tables: bool = True,
    merge_continued: bool = True,
    pages: Iterable[int] | None = None,
    table_pages: set[int] | None = None,
    logger: Any = None,
) -> PdfParseResult:
    """解析单份 PDF：逐页文本 + 逐页表格 + 跨页合并 + 文本块。

    参数扩展说明（设计 §3 允许新增带默认值的参数）：
        ``pages``       只解析指定 **1-based 物理页**（调试/定点回归），启用时显式记入 warnings 与日志；
        ``table_pages`` 只在这些 **1-based 物理页**上调用 ``find_tables()``（供 ``--fast-table`` 跳过
                        「无绘制线且文本极短」的页）；``None`` = 逐页全扫（默认，避免漏表）。
    """
    config = cfg or get_config()
    log = _lazy_logger(logger, "pdf_parser")
    source = _as_source(pdf)
    with log.enter("parse_pdf", {"file_name": source.file_name, "scan_tables": scan_tables,
                                 "merge_continued": merge_continued,
                                 "pages": (list(pages) if pages is not None else None),
                                 "table_pages": (len(table_pages) if table_pages is not None else None)}) as span:
        t0 = time.perf_counter()
        doc = _open_doc(source, logger=log)
        pages_out: list[PageText] = []
        raw_tables: list[TableBlock] = []
        text_blocks: list[TextBlock] = []
        warnings: list[str] = []
        prev_tail = ""
        try:
            total = int(doc.page_count)
            if total <= 0:
                raise PdfParseError(f"PDF 页数异常：{total}", code="RAG-2001",
                                    detail={"file_name": source.file_name})
            if pages is None:
                wanted = list(range(1, total + 1))
            else:
                wanted = sorted({_assert_page(p, total) for p in pages})
                note = f"调试模式：只解析 {len(wanted)}/{total} 页（{wanted[:12]}{'...' if len(wanted) > 12 else ''}）"
                warnings.append(note)
                log.log_event("pdf.partial_parse", level="WARNING", file_name=source.file_name,
                              requested=len(wanted), total=total, pages=wanted[:50])
            for page_no in wanted:
                i = page_no - 1
                page_t0 = time.perf_counter()
                try:
                    page = doc.load_page(i)
                    text = page.get_text("text") or ""
                    # 标题检索文本 = 上一页末 3 行 + 本页全文（表标题常落在表格上方的上一页末尾）
                    title_text = f"{prev_tail}\n{text}" if prev_tail else text
                    table_count = 0
                    if scan_tables and (table_pages is None or page_no in table_pages):
                        finder = page.find_tables()
                        for idx, tab in enumerate(list(getattr(finder, "tables", []) or [])):
                            try:
                                raw_rows = tab.extract()
                                title = table_parser.table_title_for(title_text, tuple(tab.bbox), page_no)
                                block = table_parser.build_table_block(
                                    raw_rows, file_name=source.file_name, page=page_no, idx=idx,
                                    title=title, logger=log,
                                )
                                try:
                                    block.bbox = tuple(float(v) for v in tuple(tab.bbox))
                                except (TypeError, ValueError) as exc:  # bbox 拿不到不影响主流程，留痕即可
                                    warnings.append(f"p{page_no}#t{idx} bbox 解析降级：{exc}")
                                raw_tables.append(block)
                                table_count += 1
                            except Exception as exc:  # noqa: BLE001 —— 单表失败 → 显式降级为该页文本
                                log.log_event("table.error", level="ERROR", file_name=source.file_name,
                                              page=page_no, table_index=idx, error_type=type(exc).__name__,
                                              message=str(exc), degrade_to="text_only")
                                warnings.append(f"物理第 {page_no} 页第 {idx} 张表归一化失败，已降级为文本：{exc}")
                    for bi, item in enumerate(page.get_text("blocks") or []):
                        if len(item) < 7 or int(item[6]) != 0:
                            continue
                        block_text = str(item[4] or "").strip()
                        if not block_text:
                            continue
                        text_blocks.append(TextBlock(
                            block_id=f"{source.file_name.rsplit('.', 1)[0]}#p{page_no:04d}#b{bi:03d}",
                            file_name=source.file_name, page=page_no, block_index=bi, text=block_text,
                            bbox=(float(item[0]), float(item[1]), float(item[2]), float(item[3])),
                        ))
                    pages_out.append(PageText(
                        file_name=source.file_name, page=page_no, text=text, char_count=len(text),
                        table_count=table_count, has_table=table_count > 0, parsed_at=_now_iso(),
                    ))
                    log.log_event("pdf.page_done", file_name=source.file_name, page=page_no, chars=len(text),
                                  tables=table_count, elapsed_ms=round((time.perf_counter() - page_t0) * 1000, 2))
                    # 记录本页末 3 行，供下一页做表标题检索
                    tail_lines = [ln for ln in text.splitlines() if ln.strip()]
                    prev_tail = "\n".join(tail_lines[-3:]) if tail_lines else ""
                except Exception as exc:  # noqa: BLE001 —— 单页失败显式降级为空文本
                    log.log_event("pdf.page_error", level="ERROR", file_name=source.file_name, page=page_no,
                                  error_type=type(exc).__name__, message=str(exc))
                    warnings.append(f"物理第 {page_no} 页解析失败：{exc}")
                    pages_out.append(PageText(file_name=source.file_name, page=page_no, text="", char_count=0,
                                          table_count=0, has_table=False, parsed_at=_now_iso()))
                if page_no % 50 == 0:
                    log.log_event("pdf.progress", file_name=source.file_name, page=page_no, total=total,
                                  tables=len(raw_tables), text_blocks=len(text_blocks),
                                  elapsed_ms=round((time.perf_counter() - t0) * 1000, 2))
        finally:
            doc.close()

        page_map = {p.page: p.text for p in pages_out}
        if merge_continued:
            logical_tables, stats = group_continued_tables(raw_tables, page_map, logger=log)
        else:
            logical_tables = list(raw_tables)
            stats = {"raw_tables": len(raw_tables), "logical_tables": len(raw_tables),
                     "continued_tables": 0, "repeated_headers_removed": 0,
                     "degenerate_tables": sum(1 for b in logical_tables if b.degenerate),
                     "merge_rejected": 0, "max_run": 0,
                     "table_pages": len({b.page_start for b in raw_tables})}
            log.log_event("pdf.merge_skipped", level="WARNING", file_name=source.file_name,
                          reason="merge_continued=False（调试模式）", raw_tables=len(raw_tables))

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        result = PdfParseResult(
            file_name=source.file_name, page_count=len(pages_out), pages=pages_out, tables=logical_tables,
            text_blocks=text_blocks, elapsed_ms=elapsed_ms, warnings=warnings, stats=stats,
        )
        log.log_event("pdf.parse_done", file_name=source.file_name, pages=len(pages_out),
                      tables=len(logical_tables), raw_tables=stats.get("raw_tables"),
                      text_blocks=len(text_blocks), stats=stats, elapsed_ms=elapsed_ms,
                      warnings=len(warnings))
        span.set_output({"file_name": source.file_name, "pages": len(pages_out),
                         "tables": len(logical_tables), "text_blocks": len(text_blocks),
                         "stats": stats, "warnings": len(warnings)})
        return result


def parse_all(
    raw_dir: Path | str | None = None,
    *,
    cfg: AppConfig | None = None,
    out_dir: Path | str | None = None,
    overwrite: bool = False,
    logger: Any = None,
) -> list[PdfParseResult]:
    """发现并解析 ``raw_dir`` 下**全部** PDF，落盘 JSONL 产物并写入 SQLite。"""
    config = cfg or get_config()
    log = _lazy_logger(logger, "pdf_parser")
    directory = Path(raw_dir) if raw_dir is not None else config.paths.raw_dir
    target_dir = Path(out_dir) if out_dir is not None else config.paths.processed_dir
    with log.enter("parse_all", {"raw_dir": str(directory), "out_dir": str(target_dir),
                                 "overwrite": overwrite}) as span:
        sources = discover_pdfs(directory, logger=log)
        if not sources:
            log.log_event("pdf.no_source", level="ERROR", raw_dir=str(directory),
                          reason="目录下没有任何 PDF，不能对空语料宣称成功")
            raise PdfParseError(f"语料目录下没有 PDF：{directory}", code="RAG-2001",
                                detail={"raw_dir": str(directory)})
        results: list[PdfParseResult] = []
        for source in sources:
            if not source.readable:
                log.log_event("pdf.skip_unreadable", level="ERROR", file_name=source.file_name,
                              path=str(source.path), reason="discover 阶段试开失败")
                continue
            result = parse_pdf(source, cfg=config, logger=log)
            write_parse_artifacts(result, target_dir, overwrite=overwrite)
            persist_parse_result(result, db_path=config.paths.index_dir / "rag.sqlite3",
                                 source=source, logger=log)
            results.append(result)
        if not results:
            raise PdfParseError(f"全部 PDF 均不可解析：{directory}", code="RAG-2000",
                                detail={"raw_dir": str(directory), "found": len(sources)})
        log.log_event("pdf.parse_all_done", files=len(results),
                      pages=sum(r.page_count for r in results),
                      tables=sum(len(r.tables) for r in results),
                      text_blocks=sum(len(r.text_blocks) for r in results))
        span.set_output({"files": len(results), "pages": sum(r.page_count for r in results),
                         "tables": sum(len(r.tables) for r in results)})
        return results


def write_parse_artifacts(
    result: PdfParseResult,
    out_dir: Path | str,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """写 ``{stem}.pages.jsonl`` / ``{stem}.tables.jsonl`` / ``{stem}.text_blocks.jsonl``。"""
    target = Path(out_dir)
    stem = result.file_name.rsplit(".", 1)[0]
    paths = {
        "pages": target / f"{stem}.pages.jsonl",
        "tables": target / f"{stem}.tables.jsonl",
        "text_blocks": target / f"{stem}.text_blocks.jsonl",
    }
    write_jsonl(paths["pages"], (p.to_dict() for p in result.pages), overwrite=overwrite)
    write_jsonl(paths["tables"], (t.to_dict() for t in result.tables), overwrite=overwrite)
    write_jsonl(paths["text_blocks"], (b.to_dict() for b in result.text_blocks), overwrite=overwrite)
    return paths


# ---------------------------------------------------------------------------
# SQLite 最小落库（DDL 与设计 §5.3 一致；完整 sqlite_manager 由 T7 交付）
# ---------------------------------------------------------------------------
def connect_sqlite(db_path: Path | str) -> sqlite3.Connection:
    """打开（必要时创建）SQLite 库并确保 schema 存在（幂等）。"""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(str(path))
        conn.executescript(SQLITE_DDL)
        return conn
    except sqlite3.Error as exc:
        raise StorageError(f"SQLite 初始化失败：{path}（{exc}）", detail={"db_path": str(path)}) from exc


def persist_parse_result(
    result: PdfParseResult,
    *,
    db_path: Path | str | None = None,
    source: PdfSource | None = None,
    logger: Any = None,
) -> dict[str, int]:
    """把 ``documents`` / ``pages`` / ``tables`` 写入 SQLite（UPSERT，可重复跑）。

    ``source`` 由 ``parse_all`` 传入，避免重复发现/重复哈希 14 MB 的 PDF；
    缺省时把 ``size_bytes``/``sha256_16`` 记 0/空串并写 WARN（显式降级，不静默）。
    """
    log = _lazy_logger(logger, "pdf_parser")
    target = Path(db_path) if db_path is not None else get_config().paths.index_dir / "rag.sqlite3"
    with log.enter("persist_parse_result", {"file_name": result.file_name, "db": str(target),
                                            "pages": len(result.pages), "tables": len(result.tables)}) as span:
        t0 = time.perf_counter()
        conn = connect_sqlite(target)
        try:
            import json as _json

            parsed_at = _now_iso()
            size_bytes = source.size_bytes if source is not None else 0
            sha = source.sha256_16 if source is not None else ""
            if source is None:
                log.log_event("pdf.persist_degrade", level="WARNING", file_name=result.file_name,
                              reason="未收到 PdfSource，documents.size_bytes/sha256_16 记 0/空串")
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO documents(file_name,size_bytes,sha256_16,page_count,parsed_at)"
                    " VALUES(?,?,?,?,?)",
                    (result.file_name, size_bytes, sha, result.page_count, parsed_at),
                )
                conn.executemany(
                    "INSERT OR REPLACE INTO pages(file_name,page,text,char_count,table_count)"
                    " VALUES(?,?,?,?,?)",
                    [(p.file_name, p.page, p.text, p.char_count, p.table_count) for p in result.pages],
                )
                conn.executemany(
                    "INSERT OR REPLACE INTO tables(table_id,file_name,page,page_start,page_end,n_rows,n_cols,"
                    "logical_cols,markdown,title,key_numbers,degenerate,continued_from,header_inherited,row_pages)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [
                        (
                            t.table_id, t.file_name, t.page, t.page_start, t.page_end, t.n_rows, t.n_cols,
                            t.logical_cols, t.markdown, t.title, _json.dumps(t.key_numbers, ensure_ascii=False),
                            1 if t.degenerate else 0, t.continued_from, 1 if t.header_inherited else 0,
                            _json.dumps(t.row_pages, ensure_ascii=False),
                        )
                        for t in result.tables
                    ],
                )
        except sqlite3.Error as exc:
            log.log_event("sqlite.error", level="ERROR", db=str(target), table="documents/pages/tables",
                          error_type=type(exc).__name__, message=str(exc))
            raise wrap(exc, code="RAG-7000", stage="storage", db_path=str(target)) from exc
        finally:
            conn.close()
        counts = {"documents": 1, "pages": len(result.pages), "tables": len(result.tables)}
        log.log_event("sqlite.upsert", db=str(target), **counts,
                      elapsed_ms=round((time.perf_counter() - t0) * 1000, 2))
        span.set_output({"db": str(target), **counts})
        return counts
