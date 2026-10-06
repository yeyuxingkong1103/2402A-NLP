# -*- coding: utf-8 -*-
"""工单3 分块、关键词与元数据（设计/接口设计.md §2.2、§3.7；系统架构 §6.4/§7 冻结）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

策略：
    * **表格**：一个逻辑表块 = 1 个 chunk（含跨页合并块），``content`` 按架构 §6.4 组装
      （表标题 + 来源行 + Markdown + 关键数字）；超过 ``max_table_chars`` 才按行批次拆，
      每部分都带同一 ``table_id`` 与「第 k/n 部分」标注；
    * **文本**：逐页聚合文本块 → 清洗页眉页脚 → 按句切分（size=500 / overlap=80），
      尾巴 < ``min_size`` 并入前一块；
    * 页码一律 1-based 物理页，越界抛 ``ChunkError``（不静默钳制）。
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field, fields as dc_fields
from typing import Any, Iterable, Mapping, Sequence

from .config import AppConfig, get_config
from .errors import ChunkError, wrap
from .pdf_parser import PdfParseResult, TextBlock, TableBlock, connect_sqlite
from .text_utils import (
    collapse_whitespace,
    extract_keywords as _text_extract_keywords,
    split_sentences,
    strip_printed_page,
    write_jsonl,
)

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

# 关键字段加权词典（任务书 15 个词；命中即作为关键词候选并加权）
KEYWORD_WEIGHTS: dict[str, float] = {
    "收入": 1.5,
    "占比": 1.5,
    "注册资本": 1.6,
    "法定代表人": 1.6,
    "募集资金": 1.6,
    "补充流动资金": 1.8,
    "上游": 1.4,
    "下游": 1.4,
    "供应商": 1.4,
    "客户": 1.3,
    "技术标准": 1.4,
    "国家科技进步一等奖": 1.8,
    "发行股数": 1.6,
    "关联方": 1.5,
    "持股比例": 1.5,
    "本公司关系": 1.6,
}

_SECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^第[一二三四五六七八九十百零〇0-9]{1,4}[节章]\s*\S{1,30}$"),
    re.compile(r"^[一二三四五六七八九十]{1,3}、\S{1,30}$"),
    re.compile(r"^（[一二三四五六七八九十]{1,3}）\S{1,30}$"),
)

# 编号/小标题标记：用于「段落内再分小点」时的**优先切分点**（设计 §7「按段落/小标题聚合」）。
# 实测依据（T5）：PDF1 物理 129 的段落把「经营模式…」与「①行业集中度高的影响 发行人产品主要应用于军队的信息化建设…
# 82.10%…」粘在同一个 chunk 里，导致答案句被前文主题稀释（BM25 仅第 5、向量前 100 都不进）→ 题 33/260 丢命中。
# 把该句切出来后：**BM25 排名第 2/2392**（实测），命中恢复。
_ENUM_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]"),
    re.compile(r"（[一二三四五六七八九十]{1,3}）(?=[^\s])"),
    re.compile(r"(?<=[。；;！？!?])\s*[0-9]{1,2}、(?=[^\s])"),
)


@dataclass(slots=True)
class Chunk:
    """检索/生成共用的最小单元（字段为设计 §2.2 冻结项）。"""

    chunk_id: str
    file_name: str
    page: int                       # = page_start（对外引用页，1-based 物理页）
    page_start: int
    page_end: int
    type: str                       # "text" | "table"
    section: str
    content: str
    keywords: list[str]
    table_id: str | None
    char_count: int
    order: int
    source_block_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Chunk":
        names = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def _lazy_logger(logger: Any, module: str) -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


def chunk_id_for(file_name: str, page: int, kind: str, idx: int) -> str:
    """``f"{stem}_p{page:04d}_{kind}{idx:02d}"``（kind ∈ {"t","x"}）。"""
    stem = file_name.rsplit(".", 1)[0]
    if kind not in {"t", "x"}:
        raise ChunkError(f"非法 chunk 类型：{kind}", detail={"kind": kind})
    return f"{stem}_p{int(page):04d}_{kind}{int(idx):02d}"


def split_segments(text: str, *, min_segment: int = 60) -> list[str]:
    """把一段正文按「编号/小标题标记」切成若干语义小点（供分块时优先在这些位置切开）。

    返回的片段保持原文顺序；长度 < ``min_segment`` 的碎片会并回前一片段（避免切得过碎）。
    """
    body = str(text or "")
    if not body:
        return []
    cuts: set[int] = set()
    for pattern in _ENUM_MARKERS:
        for match in pattern.finditer(body):
            cuts.add(match.start())
    if not cuts:
        return [body]
    pieces: list[str] = []
    last = 0
    for position in sorted(cuts):
        if position - last < min_segment:
            continue                          # 与上一片段太近 → 不切（防碎片化）
        pieces.append(body[last:position])
        last = position
    pieces.append(body[last:])
    merged: list[str] = []
    for piece in pieces:
        if merged and len(piece.strip()) < min_segment:
            merged[-1] = merged[-1] + piece
        elif piece.strip():
            merged.append(piece)
    return [p.strip() for p in merged if p.strip()]


def detect_section(page_text: str, *, default: str = "") -> str:
    """从页文本中取页内最后一个「节/章/条」级标题；取不到回退 ``default``。"""
    found = ""
    for line in (str(page_text or "").splitlines()):
        text = line.strip()
        if not text or len(text) > 40:
            continue
        if any(pattern.match(text) for pattern in _SECTION_PATTERNS):
            found = text
    return found or default


def split_text(text: str, *, size: int = 500, overlap: int = 80, min_size: int = 120) -> list[str]:
    """按句切分文本：目标块长 ``size``、相邻块共享 ``overlap`` 字符、尾块 < ``min_size`` 并入前块。"""
    body = collapse_whitespace(text)
    if not body:
        return []
    size = max(int(size), 50)
    overlap = max(min(int(overlap), size - 1), 0)
    pieces: list[str] = []
    current = ""
    for sentence in split_sentences(body):
        if len(sentence) > size:
            # 单句超长：先冲掉当前块，再对超长句硬切
            if current:
                pieces.append(current)
                current = ""
            step = max(size - overlap, 1)
            for start in range(0, len(sentence), step):
                pieces.append(sentence[start:start + size])
            current = ""
            continue
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= size:
            current = f"{current} {sentence}"
        else:
            pieces.append(current)
            # overlap 前缀必须**按剩余空间裁剪**：否则 80 字重叠 + 接近 size 的句子会产出
            # 超过上限的块（实测出现 606 字文本块，正是此处无上限判断所致）。
            keep = max(size - len(sentence) - 1, 0)
            tail = current[-min(overlap, keep):] if keep > 0 else ""
            current = f"{tail} {sentence}".strip() if tail else sentence
    if current:
        pieces.append(current)
    if len(pieces) > 1 and len(pieces[-1]) < int(min_size):
        merged = f"{pieces[-2]} {pieces[-1]}".strip()
        pieces[-2] = merged
        pieces.pop()
    return [p for p in pieces if p.strip()]


def extract_keywords(text: str, *, topk: int = 8, weight_dict: Mapping[str, float] | None = None) -> list[str]:
    """关键词（委托 ``text_utils.extract_keywords``）：默认使用工单的关键字段加权词典。"""
    weights = KEYWORD_WEIGHTS if weight_dict is None else weight_dict
    return _text_extract_keywords(text, topk=topk, weight_dict=weights)


def _keyword_source(content: str) -> str:
    """关键词投影：去掉 content 模板标签（``[表标题]`` 等）与 Markdown 竖线/分隔行。

    实测教训：直接用 content 抽关键词会得到 ``---``、``1``、``2`` 这类噪声。
    """
    kept: list[str] = []
    for line in str(content or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and "]" in stripped:
            continue
        if stripped and set(stripped) <= set("|-: "):
            continue
        kept.append(line.replace("|", " "))
    return " ".join(kept)


# ---------------------------------------------------------------------------
# 表格 chunk
# ---------------------------------------------------------------------------
def _table_source_line(block: TableBlock) -> str:
    """来源行（架构 §6.4）：单页写「第 N 页」，跨页写「第 A–B 页（表头页 A）」。"""
    if block.page_start == block.page_end:
        return f"{block.file_name} 第 {block.page_start} 页"
    return f"{block.file_name} 第 {block.page_start}-{block.page_end} 页（表头页 {block.page_start}）"


def _table_content(block: TableBlock, *, part: int | None = None, parts: int | None = None,
                   markdown: str | None = None, numbers: Sequence[str] | None = None) -> str:
    """按 §6.4 组装表格块 content（表标题 + 来源 + Markdown + 关键数字）。"""
    source = _table_source_line(block)
    if part is not None and parts is not None and parts > 1:
        source = f"{source}（第 {part}/{parts} 部分）"
    body = markdown if markdown is not None else block.markdown
    nums = list(numbers if numbers is not None else block.key_numbers)
    return "\n".join([
        f"[表标题] {block.title}",
        f"[来源] {source}",
        "[Markdown]",
        body,
        f"[关键数字] {' '.join(nums)}",
    ])


def _split_table_parts(block: TableBlock, cfg: AppConfig) -> list[tuple[str, list[str]]]:
    """超长逻辑表 → 按行批次拆（每部分保留表标题与来源行，共享 table_id）。"""
    limit = int(cfg.chunk.max_table_chars)
    full = _table_content(block)
    if len(full) <= limit or not block.rows:
        return [(full, list(block.key_numbers))]
    batches: list[list[list[str]]] = []
    current: list[list[str]] = []
    current_len = len(_table_content(block, markdown="", numbers=[]))
    for row in block.rows:
        row_len = len(" | ".join(str(c) for c in row)) + 4
        if current and current_len + row_len > limit:
            batches.append(current)
            current = []
            current_len = len(_table_content(block, markdown="", numbers=[]))
        current.append(row)
        current_len += row_len
    if current:
        batches.append(current)
    total = len(batches)
    parts: list[tuple[str, list[str]]] = []
    for index, rows in enumerate(batches, 1):
        markdown = _markdown_for(block.header, rows)
        numbers = _numbers_of(markdown, block.header)
        parts.append((_table_content(block, part=index, parts=total, markdown=markdown, numbers=numbers), numbers))
    return parts


def _markdown_for(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    from .table_parser import table_to_markdown

    return table_to_markdown(header, rows)


def _numbers_of(markdown: str, header: Sequence[str]) -> list[str]:
    from .text_utils import extract_numbers

    return extract_numbers(markdown + " " + " ".join(header))[:24]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build_chunks(
    parse_result: PdfParseResult,
    *,
    cfg: AppConfig | None = None,
    logger: Any = None,
) -> list[Chunk]:
    """把解析结果转成 chunk 列表（表格整体成块，文本按 400~600 字切分）。"""
    config = cfg or get_config()
    log = _lazy_logger(logger, "chunker")
    file_name = parse_result.file_name
    stem = file_name.rsplit(".", 1)[0]
    with log.enter("build_chunks", {"file_name": file_name, "pages": parse_result.page_count,
                                    "tables": len(parse_result.tables),
                                    "text_blocks": len(parse_result.text_blocks)}) as span:
        t0 = time.perf_counter()
        chunks: list[Chunk] = []
        seen_ids: set[str] = set()
        table_idx = 0
        text_chunks = 0
        table_chunks = 0
        total_chars = 0

        # ---- 1) 表格块：逻辑表块整块成 1 chunk（超长才拆部分）----
        for block in sorted(parse_result.tables, key=lambda b: (b.page_start, b.table_id)):
            if block.degenerate:
                log.log_event("chunker.skip_degenerate", level="WARNING", table_id=block.table_id,
                              page=block.page_start, reason="退化表不进正文 chunk")
                continue
            parts = _split_table_parts(block, config)
            if len(parts) > 1:
                log.log_event("chunker.table_split", level="WARNING", table_id=block.table_id,
                              page_start=block.page_start, page_end=block.page_end, parts=len(parts),
                              chars=len(_table_content(block)))
            section = detect_section(parse_result.pages[block.page_start - 1].text if
                                     block.page_start - 1 < len(parse_result.pages) else "", default="")
            for content, _numbers in parts:
                cid = chunk_id_for(file_name, block.page_start, "t", table_idx)
                if cid in seen_ids:
                    raise ChunkError(f"chunk_id 重复：{cid}", code="RAG-2201", detail={"chunk_id": cid})
                seen_ids.add(cid)
                chunk = Chunk(
                    chunk_id=cid, file_name=file_name, page=block.page_start, page_start=block.page_start,
                    page_end=block.page_end, type="table", section=section, content=content,
                    keywords=extract_keywords(_keyword_source(content), topk=8, weight_dict=KEYWORD_WEIGHTS),
                    table_id=block.table_id, char_count=len(content), order=-1,
                    source_block_ids=[block.table_id],
                )
                if block.page_start != block.page_end:
                    log.log_event("chunker.continued_chunk", chunk_id=cid, table_id=block.table_id,
                                  page_start=block.page_start, page_end=block.page_end,
                                  absorbed_pages=block.absorbed_pages)
                chunks.append(chunk)
                table_idx += 1
                table_chunks += 1
                total_chars += len(content)

        # ---- 2) 文本块：**按段落（TextBlock）边界聚合** → 清洗页眉页脚 → 仅在段落超长时按句切分 ----
        # 设计 §7 要求「按段落/小标题聚合并按 chunk_size 切分」。
        # 实测教训（T5）：早期实现只按字数累积、忽略段落边界，会把「答案句」粘在
        # 另一个主题段落的长块尾部（如 PDF1 物理 129：477 字块前 380 字讲民品客户，
        # 真正的「军用收入占比 82.10%…」只在最后 95 字）→ 该块语义被前文主导，检索排到第 8~21 名，题 33/260 丢命中。
        by_page: dict[int, list[TextBlock]] = {}
        for block in parse_result.text_blocks:
            by_page.setdefault(block.page, []).append(block)
        last_section = ""
        for page in sorted(by_page):
            blocks = sorted(by_page[page], key=lambda b: b.block_index)
            raw_text = "\n".join(b.text for b in blocks)
            page_text = strip_printed_page(raw_text, page=page)
            if not page_text.strip():
                continue
            section = detect_section(page_text, default=last_section)
            last_section = section
            pieces: list[str] = []
            buffer = ""
            used_blocks: list[str] = []
            min_size = int(config.chunk.min_size)
            for block in blocks:
                # 逐块清洗（页眉页脚只在块首/块尾出现，按页清洗更稳 → 用整页清洗结果兜底）
                block_text = strip_printed_page(block.text, page=page)
                block_text = collapse_whitespace(block_text)
                if not block_text:
                    continue
                if len(block_text) > int(config.chunk.size):
                    # 超长单段：先把已有缓冲冲掉，再按句切分该段
                    if buffer:
                        pieces.append(buffer)
                        buffer, used_blocks = "", []
                    for piece in split_text(block_text, size=config.chunk.size,
                                            overlap=config.chunk.overlap,
                                            min_size=config.chunk.min_size):
                        pieces.append(piece)
                    continue
                # **小标题/编号点优先切分**：让每个语义小点尽量独立成块，避免答案句被前文主题稀释
                for segment in split_segments(block_text):
                    starts_with_marker = bool(
                        any(pattern.match(segment[:2]) for pattern in _SECTION_PATTERNS)
                        or any(marker.match(segment) for marker in _ENUM_MARKERS)
                    )
                    if buffer and starts_with_marker and len(buffer) >= min_size:
                        pieces.append(buffer)
                        buffer, used_blocks = "", []
                    if buffer and len(buffer) + 1 + len(segment) > int(config.chunk.size):
                        pieces.append(buffer)                    # **在段落边界切开**
                        buffer, used_blocks = "", []
                    buffer = f"{buffer} {segment}".strip() if buffer else segment
                    if block.block_id not in used_blocks:
                        used_blocks.append(block.block_id)
            if buffer:
                pieces.append(buffer)
            # 最终硬上限兜底：任何仍超过 size+100 的块再按句切一刀
            # （覆盖 overlap 前缀、页内再平衡、尾块并入等所有路径；实测修复后 >600 的块 14 → 0）
            max_chars = int(config.chunk.size) + 100
            if any(len(piece) > max_chars for piece in pieces):
                normalized: list[str] = []
                for piece in pieces:
                    if len(piece) > max_chars:
                        normalized.extend(split_text(piece, size=config.chunk.size,
                                                    overlap=config.chunk.overlap,
                                                    min_size=config.chunk.min_size))
                    else:
                        normalized.append(piece)
                pieces = normalized
            # 页内再平衡：末块 < min_size 且前块有余量时，从句边界把前块的尾句搬给末块，
            # 使**两侧**都落在 [min_size, size+100]。这样既避免「末块过小」，也避免「并入前块后越界」
            # （设计 §3.7 的 400~600 与「尾块并入前块」两条在边界上会冲突，本函数同时满足两者的可行区间）。
            if len(pieces) >= 2 and len(pieces[-1]) < min_size:
                max_chars = int(config.chunk.size) + 100
                prev_text, tail_text = pieces[-2], pieces[-1]
                guard = 0
                while len(tail_text) < min_size and len(prev_text) > min_size and guard < 8:
                    cut = max(prev_text.rfind("。"), prev_text.rfind("；"), prev_text.rfind("！"))
                    if cut <= 0 or (len(prev_text) - cut) > 150:
                        break                       # 找不到合适的句边界或移出量过大 → 停止
                    moved = prev_text[cut + 1:]
                    if len(tail_text) + len(moved) > max_chars:
                        break                       # 搬过来会越上限 → 停止
                    tail_text = moved + tail_text
                    prev_text = prev_text[:cut + 1]
                    guard += 1
                pieces[-2], pieces[-1] = prev_text.strip(), tail_text.strip()
                log.log_event("chunker.page_rebalance", file_name=file_name, page=page,
                              moved_sentences=guard, prev_chars=len(pieces[-2]),
                              tail_chars=len(pieces[-1]))
            for index, piece in enumerate(pieces):
                is_tail = index == len(pieces) - 1
                # 尾块（< min_size）并入**本文件最近的前一个文本块**，但必须守住设计 §3.7 的文本块上限
                # （size=500 → 上限 600）。实测教训（captain 复核发现）：早期实现无条件并入，
                # 导致 14 个文本块越界（最大 697，特征 = 块内含尾块并入留下的换行符）。
                # 越界时宁可为独立小尾块，也不破坏尺寸契约；并写 chunker.tail_standalone 留痕。
                if is_tail and len(piece) < min_size and chunks:
                    max_chars = int(config.chunk.size) + 100
                    target = None
                    for candidate in reversed(chunks):
                        if candidate.file_name != file_name:
                            break
                        if candidate.type == "text":
                            target = candidate
                            break
                    if target is not None and len(target.content) + 1 + len(piece) <= max_chars:
                        # 尾巴并入前块（设计 §3.7）：延长前块的 page_end，不新开块
                        target.content = f"{target.content}\n{piece}".strip()
                        target.char_count = len(target.content)
                        target.page_end = max(target.page_end, page)
                        target.keywords = extract_keywords(_keyword_source(target.content), topk=8,
                                                           weight_dict=KEYWORD_WEIGHTS)
                        target.source_block_ids.append(f"{stem}#p{page:04d}#tail")
                        total_chars = total_chars + len(piece)
                        continue
                    log.log_event("chunker.tail_standalone", level="WARNING",
                                  file_name=file_name, page=page, chars=len(piece),
                                  reason=("前块并入后超上限" if target is not None else "无同类前块可并入"),
                                  prev_chars=(len(target.content) if target is not None else 0),
                                  max_chars=max_chars)
                cid = chunk_id_for(file_name, page, "x", text_chunks)
                if cid in seen_ids:
                    raise ChunkError(f"chunk_id 重复：{cid}", code="RAG-2201", detail={"chunk_id": cid})
                seen_ids.add(cid)
                chunks.append(Chunk(
                    chunk_id=cid, file_name=file_name, page=page, page_start=page, page_end=page,
                    type="text", section=section, content=piece,
                    keywords=extract_keywords(_keyword_source(piece), topk=8, weight_dict=KEYWORD_WEIGHTS),
                    table_id=None, char_count=len(piece), order=-1,
                    source_block_ids=used_blocks or [b.block_id for b in blocks],
                ))
                text_chunks += 1
                total_chars += len(piece)

        # ---- 3) 全局 order：按 (物理页, 表格优先, chunk_id) 排序 ----
        chunks.sort(key=lambda c: (c.page_start, 0 if c.type == "table" else 1, c.chunk_id))
        for order, chunk in enumerate(chunks):
            chunk.order = order
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.log_event("chunker.done", file_name=file_name, text_chunks=text_chunks,
                      table_chunks=table_chunks, total_chunks=len(chunks), total_chars=total_chars,
                      elapsed_ms=elapsed_ms)
        span.set_output({"file_name": file_name, "text_chunks": text_chunks,
                         "table_chunks": table_chunks, "total_chunks": len(chunks),
                         "total_chars": total_chars})
        return chunks


def write_chunks(chunks: Sequence[Chunk], path: Path | str, *, overwrite: bool = False) -> int:
    """写 ``chunks.jsonl``（按 order 递增），返回行数。"""
    ordered = sorted(chunks, key=lambda c: c.order)
    return write_jsonl(path, (c.to_dict() for c in ordered), overwrite=overwrite)


def persist_chunks(
    chunks: Sequence[Chunk],
    *,
    db_path: Path | str | None = None,
    logger: Any = None,
) -> int:
    """把 chunk 写入 SQLite ``chunks`` 表（DDL 复用设计 §5.3；``order`` → ``ord``）。"""
    log = _lazy_logger(logger, "chunker")
    target = db_path if db_path is not None else get_config().paths.index_dir / "rag.sqlite3"
    with log.enter("persist_chunks", {"db": str(target), "chunks": len(chunks)}) as span:
        t0 = time.perf_counter()
        conn = connect_sqlite(target)
        try:
            # 先删除这批 chunk 所属文件的**旧行**：重建（重新分块）后 chunk_id 会变化，
            # 只做 INSERT OR REPLACE 会把上一版的分块永久留在表里（实测出现 4565 行 vs 实际 2391）。
            files = sorted({c.file_name for c in chunks})
            with conn:
                if files:
                    conn.executemany("DELETE FROM chunks WHERE file_name = ?", [(name,) for name in files])
                conn.executemany(
                    "INSERT OR REPLACE INTO chunks(chunk_id,file_name,page,page_start,page_end,type,section,"
                    "content,keywords,table_id,char_count,ord,source_block_ids) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [
                        (
                            c.chunk_id, c.file_name, c.page, c.page_start, c.page_end, c.type, c.section,
                            c.content, json.dumps(c.keywords, ensure_ascii=False), c.table_id, c.char_count,
                            c.order, json.dumps(c.source_block_ids, ensure_ascii=False),
                        )
                        for c in chunks
                    ],
                )
        except sqlite3.Error as exc:
            log.log_event("sqlite.error", level="ERROR", db=str(target), table="chunks",
                          error_type=type(exc).__name__, message=str(exc))
            raise wrap(exc, code="RAG-7000", stage="storage", db_path=str(target)) from exc
        finally:
            conn.close()
        log.log_event("sqlite.upsert", db=str(target), chunks=len(chunks),
                      elapsed_ms=round((time.perf_counter() - t0) * 1000, 2))
        span.set_output({"db": str(target), "chunks": len(chunks)})
        return len(chunks)


def iter_chunk_dicts(chunks: Iterable[Chunk]) -> Iterable[dict[str, Any]]:
    """便捷：chunk → dict 迭代。"""
    for chunk in chunks:
        yield chunk.to_dict()
