"""
分块模块
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

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
    doc: str            # 来源文件名
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
) -> list[Chunk]:
    """
    把 ParsedDoc 切成 Chunk 列表。

    打包策略：**跨小节累积**，而不是「遇到标题就断」。
    招股说明书里大量小节只有一两句话（如「（3）技术风险」下面一段），
    逐标题断开会产生大量 100 字以下的碎块，向量语义不足、BM25 也难命中。
    这里的做法是：小节切换时先看当前缓冲是否已够一半容量，不够就继续往里装，
    并把新的章节标题作为「定位前缀」写进块首，保证块内仍然知道自己在讲什么章节。
    """
    chunks: list[Chunk] = []
    seq = 0

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
                    doc=parsed.source,
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

    # ---- 2) 表格：单独成块，带独立序号，便于溯源到「这是表里的数」
    # 表格同样受长度上限约束，但**不能硬切** —— 从表格中间断开会让行列对不上。
    # 做法是按行分块，并在每一块头部重复表头（Markdown 表格的 header + 分隔行），
    # 这样每一块单独看都是自洽的表格。
    for page in parsed.pages:
        for ti, md in enumerate(page.tables, 1):
            section = f"第{page.page}页 表格{ti}"
            for piece in _split_markdown_table(md, max_len):
                seq += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"c{seq:05d}",
                        doc=parsed.source,
                        page=page.page,
                        page_end=page.page,
                        section=section,
                        type="table",
                        text=piece,
                    )
                )

    return chunks


def _split_markdown_table(md: str, max_len: int) -> list[str]:
    """把超长 Markdown 表格按行拆开，每块重复表头。"""
    md = md.strip()
    if len(md) <= max_len:
        return [md] if md else []

    lines = [ln for ln in md.splitlines() if ln.strip()]
    if len(lines) < 3:
        # 不是标准表格（无表头分隔行），退回普通切分
        return _split_long(md, max_len)

    header, sep, rows = lines[0], lines[1], lines[2:]
    prefix = f"{header}\n{sep}"
    out: list[str] = []
    cur = [prefix]
    cur_len = len(prefix)
    for r in rows:
        if cur_len + len(r) + 1 > max_len and len(cur) > 1:
            out.append("\n".join(cur))
            cur = [prefix]
            cur_len = len(prefix)
        cur.append(r)
        cur_len += len(r) + 1
    if len(cur) > 1:
        out.append("\n".join(cur))
    # 极端情况：表头加一行仍超上限 → 最后再兜一次
    final: list[str] = []
    for piece in out:
        final.extend(_split_long(piece, max_len) if len(piece) > max_len else [piece])
    return final
