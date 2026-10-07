"""
分块模块
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

策略：标题感知（heading-aware）+ 滑动窗口兜底。

为什么要标题感知：招股说明书是强结构文档。一个块如果横跨
「第四节 风险因素」和「第五节 业务与技术」，向量会被两个主题拉扯，
检索时既不像上面也不像下面。按标题切分能让每个块的语义单一。

为什么必须有**长度上限兜底**（实测坑）：
中文年报/招股书里有大量「整段只有逗号、没有句号」的长枚举（财务表格残留、
并列指标清单）。句子切分器会把它们整体当成一个句子，于是原样穿过所有长度检查，
产生几千字的巨块。这类块危害不是「大」，而是入库时整批失败。

因此本模块的兜底分两层，缺一层都不行：
  1) 每段文本都要过 _split_long；
  2) _split_long 自身要能对「无标点长文本」硬切，且优先断在 ，、；： 等软边界。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from .config import (
    CHUNK_MAX_LENGTH,
    CHUNK_MIN_LENGTH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    WORK_ORDER_NO,  # noqa: F401
)
from .pdf_parser import ParsedDoc

# ------------------------------------------------------------------ 标题识别

# 招股说明书常见层级：第X节 / 第X章 / 一、 / （一） / 1、 / （1）
_HEADING_PATTERNS = [
    (1, re.compile(r"^第[一二三四五六七八九十百]+[节章]\s*\S")),
    (2, re.compile(r"^[一二三四五六七八九十]+、\s*\S")),
    (3, re.compile(r"^（[一二三四五六七八九十]+）\s*\S")),
    (4, re.compile(r"^\d+[、.]\s*\S{2,}")),
    (5, re.compile(r"^（\d+）\s*\S")),
]

_SENT_END = re.compile(r"(?<=[。！？；!?;])")
# 软边界：断在这里，引用原文仍然可读
_SOFT_BREAK = re.compile(r"[，,、；;：:\u3001\s]")


def _is_heading(line: str) -> tuple[int, str] | None:
    line = line.strip()
    if not line or len(line) > 60:
        return None
    for level, pat in _HEADING_PATTERNS:
        if pat.match(line):
            return level, line
    return None


# ------------------------------------------------------------------ 长度兜底


def _split_long(text: str, max_len: int) -> list[str]:
    """
    把超长文本切开。优先在软边界（，、；：/空白）下刀，位置太靠前才硬切。

    关键：本函数必须能处理「整段没有一个句号」的输入 —— 这正是原文里
    表格残留、并列清单的形态。若只在句子之间切，这类文本会原样穿透长度检查。
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    pieces: list[str] = []
    rest = text
    while len(rest) > max_len:
        window = rest[:max_len]
        cut = -1
        for m in _SOFT_BREAK.finditer(window):
            cut = m.end()
        # 软边界太靠前（不足一半）→ 说明这段没有可读的断点，退回硬切
        if cut < max_len * 0.5:
            cut = max_len
        pieces.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        pieces.append(rest)
    return pieces


def _split_sentences(text: str) -> list[str]:
    parts = [p for p in _SENT_END.split(text) if p and p.strip()]
    return parts or ([text] if text.strip() else [])


def _pack(
    sentences: list[str],
    size: int,
    overlap: int,
    max_len: int,
    min_len: int,
) -> list[str]:
    """把句子打包成块，带尾部重叠。所有产出都保证 len <= max_len。"""
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    overlap = min(overlap, max(0, size - 1))

    for sent in sentences:
        for piece in _split_long(sent, max_len):
            if buf and buf_len + len(piece) > size:
                chunks.append("".join(buf))
                tail: list[str] = []
                tail_len = 0
                for x in reversed(buf):
                    if tail_len + len(x) > overlap:
                        break
                    tail.insert(0, x)
                    tail_len += len(x)
                buf, buf_len = tail, tail_len
            buf.append(piece)
            buf_len += len(piece)

    if buf:
        tail_text = "".join(buf)
        if len(tail_text) >= min_len or not chunks:
            chunks.append(tail_text)
        else:
            # 太短且已有前块 → 并进前一块（前提是不超上限）
            if len(chunks[-1]) + len(tail_text) <= max_len:
                chunks[-1] += tail_text
            else:
                chunks.append(tail_text)
    return chunks


def _final_guard(chunks: list[str], max_len: int) -> list[str]:
    """最后一道闸：保证**每一个**块都不超上限。返回前会断言。"""
    out: list[str] = []
    for c in chunks:
        out.extend(_split_long(c, max_len))
    for c in out:
        assert len(c) <= max_len, f"分块超上限：{len(c)} > {max_len}"
    return out


# ------------------------------------------------------------------ 主流程


@dataclass
class Chunk:
    """一个检索单元。"""

    chunk_id: str
    doc: str            # 来源文档（人类可读全称，如「武汉力源信息技术股份有限公司招股说明书」）
    doc_key: str        # 来源文档短键（xingtu / liyuan），工单03 两文档语料下用于消歧
    page: int           # 起始页（1-based）
    page_end: int       # 结束页
    section: str        # 所属标题路径
    type: str           # text | table
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_document(
    parsed: ParsedDoc,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_len: int = CHUNK_MIN_LENGTH,
    max_len: int = CHUNK_MAX_LENGTH,
    doc_key: str = "",
    seq_start: int = 0,
) -> list[Chunk]:
    """
    把 ParsedDoc 切成 Chunk 列表。

    打包策略：**跨小节累积**，而不是「遇到标题就断」。
    招股说明书里大量小节只有一两句话（如「（3）技术风险」下面一段），
    逐标题断开会产生大量 100 字以下的碎块，向量语义不足、BM25 也难命中。
    这里的做法是：小节切换时先看当前缓冲是否已够一半容量，不够就继续往里装，
    并把新的章节标题作为「定位前缀」写进块首，保证块内仍然知道自己在讲什么章节。

    doc_key / seq_start：工单03 起语料是**两份招股说明书**，两份文档里都存在
    「第 22 页」这种页码，因此块 ID 必须**全局唯一**（`rag.retrieve_for` 用
    chunk_id 去重，两份文档的 c00001 会互相顶掉）。做法是由调用方
    （scripts/build_index.py）累加 seq_start，本文档内的块接在上一篇后面编号。
    """
    chunks: list[Chunk] = []
    seq = seq_start
    doc_label = parsed.doc_name or parsed.source

    # 章节路径：记录每一层的当前标题，形成 "第X节 > 一、xxx" 这样的定位串
    heading_stack: dict[int, str] = {}

    def current_section() -> str:
        if not heading_stack:
            return ""
        return " > ".join(heading_stack[k] for k in sorted(heading_stack))

    # ---- 1) 先线性化：产出 (页码, 文本, 章节, 是否标题) 的单元序列
    @dataclass
    class Unit:
        page: int
        text: str
        section: str
        heading: bool
        force_break: bool = False   # True 表示这里必须先落一块（大节边界）

    units: list[Unit] = []
    for page in parsed.pages:
        if not page.text:
            continue
        para_lines: list[str] = []

        def flush_para(pg=page.page) -> None:
            if para_lines:
                units.append(Unit(pg, "\n".join(para_lines), current_section(), False))
                para_lines.clear()

        for ln in page.text.splitlines():
            head = _is_heading(ln)
            if head:
                flush_para()
                level, title = head
                for k in [k for k in heading_stack if k >= level]:
                    heading_stack.pop(k, None)
                heading_stack[level] = title
                # 一级标题（第X节/第X章）是硬边界：跨节拼接会让块内出现两个主题，
                # 向量被拉扯，实测会导致关键事实被挤出召回。
                units.append(Unit(page.page, title, current_section(), True, force_break=(level == 1)))
            else:
                para_lines.append(ln)
        flush_para()

    # ---- 2) 打包：容量驱动的滑动窗口
    buf: list[str] = []
    buf_len = 0
    buf_pages: list[int] = []
    buf_section = ""

    def flush(carry_overlap: bool = True) -> None:
        """落一块，并把尾部 overlap 字作为下一块的开头（保持跨块语义连续）。"""
        nonlocal seq, buf, buf_len, buf_pages, buf_section
        if not buf:
            return
        body = "".join(buf).strip()
        # 块首带章节定位前缀（若正文里还没有出现该标题）
        if buf_section and buf_section not in body[: len(buf_section) + 8]:
            body = f"【{buf_section}】\n{body}"
        for piece in _final_guard([body], max_len):
            if len(piece.strip()) < min_len and chunks and len(chunks[-1].text) + len(piece) <= max_len:
                chunks[-1].text += piece
                chunks[-1].page_end = buf_pages[-1] if buf_pages else chunks[-1].page_end
                continue
            seq += 1
            chunks.append(
                Chunk(
                    chunk_id=f"c{seq:05d}",
                    doc=doc_label,
                    doc_key=doc_key,
                    page=buf_pages[0] if buf_pages else 0,
                    page_end=buf_pages[-1] if buf_pages else 0,
                    section=buf_section,
                    type="text",
                    text=piece.strip(),
                )
            )

        tail = ""
        if carry_overlap and overlap > 0:
            for s in reversed(_split_sentences(body)):
                if len(tail) + len(s) > overlap:
                    break
                tail = s + tail
        last_page = buf_pages[-1] if buf_pages else 0
        buf = [tail] if tail else []
        buf_len = len(tail)
        buf_pages = [last_page] if tail else []

    for u in units:
        # 一级标题无条件断块；其余章节变化时，缓冲过半才断，避免产生大量碎块。
        # 两种情况都不携带 overlap 尾巴：跨节了，尾部上下文对下一节没有价值。
        if u.force_break and buf:
            flush(carry_overlap=False)
        elif u.section != buf_section and buf_len >= size * 0.5:
            flush(carry_overlap=False)
        if not buf:
            buf_section = u.section
        buf.append(u.text + ("\n" if not u.heading else "\n"))
        buf_pages.append(u.page)
        buf_len += len(u.text) + 1
        if buf_len >= size:
            flush()

    flush()

    # ---- 2) 表格：结构化表格单独成块
    #
    # 工单03 起，表格不再是「一段 Markdown」，而是**行级语义文本**：
    #     关联方名称：赵马克；持股比例：42.35%；与本公司关系：公司控股股东
    # 这样列名与值是绑定的，BM25 能命中「持股比例」这类列名，
    # 生成模型也不必自己去数列（数错列是表格问答最常见的错法）。
    #
    # 超长表按行切，**每块重复表头行**：从表格中间断开会让后半段丢掉列名，
    # 单看那一块就不知道每个值属于哪一列了。
    for tb in getattr(parsed, "tables", []) or []:
        head = _table_header_line(tb)
        section = f"第{tb.page}页 表格{tb.index + 1}" + (f" {tb.caption}" if tb.caption else "")
        for piece in split_table_text(head, tb.semantic_rows(), max_len):
            seq += 1
            chunks.append(
                Chunk(
                    chunk_id=f"c{seq:05d}",
                    doc=doc_label,
                    doc_key=doc_key,
                    page=tb.page,
                    page_end=tb.page,
                    section=section,
                    type="table",
                    text=piece,
                )
            )

    return chunks


def _table_header_line(tb) -> str:
    """
    表格块的头部：来源 + 表名 + 表类型 + 列名清单。

    把**列名清单**单独写一行是有意的 —— 像「与武汉力源信息技术股份有限公司不存在
    控制关系的关联方企业有哪些？」这种问题，问法与列名（「企业名称」「与本公司关系」）
    并不逐字相同，把列名集中列出来能显著提高关键词路的命中率。
    """
    kind = f"[{tb.kind}]" if tb.kind else ""
    caption = tb.caption or "表格"
    cols = " | ".join(h for h in tb.headers if h and not h.startswith("列"))
    doc_name = tb.doc
    lines = [f"【表】{caption}{kind}（第{tb.page}页）"]
    if doc_name:
        lines.append(f"来源文档：{doc_name}")
    if cols:
        lines.append(f"列名：{cols}")
    return "\n".join(lines)


def split_table_text(head: str, rows: list[str], max_len: int) -> list[str]:
    """把表头 + 语义行打包成若干块，每块都带完整表头。"""
    if not rows:
        return [head] if head.strip() else []
    out: list[str] = []
    cur = [head]
    cur_len = len(head)
    for r in rows:
        if cur_len + len(r) + 1 > max_len and len(cur) > 1:
            out.append("\n".join(cur))
            cur = [head]
            cur_len = len(head)
        cur.append(r)
        cur_len += len(r) + 1
    if len(cur) > 1:
        out.append("\n".join(cur))
    # 极端情况：表头加一行仍超上限 → 再兜一次长度
    final: list[str] = []
    for piece in out:
        final.extend(_split_long(piece, max_len) if len(piece) > max_len else [piece])
    return final
