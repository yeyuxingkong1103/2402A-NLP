# -*- coding: utf-8 -*-
"""文档分块：固定长度 / 句子 / 段落 / 法条 四种策略，支持父子块与 overlap。

策略（``strategy``）：
  * ``fixed``     —— 定长硬切（保底，最差）；
  * ``sentence``  —— 句子级窗口；
  * ``paragraph`` —— 段落优先、超长段落退回句子级（通用文本默认）；
  * ``article``   —— **法条感知**：以「第X条」为原子单元，**一个块 = 一整条**
                     （超长条按句子切开并标「（续N）」），每块自带
                     「编/章/节 + 条号」出处前缀；文本不含条号时自动退化为
                     ``paragraph``，不报错、不丢内容。

为什么要 ``article``：法律文本本身按「条」组织（《民法典》111,360 字 / 1322 条，
平均 81.8 字）。用固定窗口（如 300 字）切会**把条切碎**——实测
``paragraph/300`` 下 90.0% 的子块横跨 ≥2 条、39.8% 从半句起头、且存在 1693 字的
巨块，导致检索命中的片段缺条号、引用无法落到具体条文。``article`` 下这些指标≈0。

父子块策略：
  * 父块（is_parent=True）保留大段上下文，用于喂给大模型；
  * 子块用于检索，命中后可通过 parent_id 取回父块补齐上下文。
overlap 只作用于通用窗口策略；``article`` 每块是完整条文，不做重叠（否则会出现
半条拼接、条号与正文错位）。
"""
from __future__ import annotations

import re

from ..config import ChunkConfig
from ..schemas import Chunk
from ..utils import make_summary, now_ts, stable_id
from .loaders import Document

# 中英文句末标点
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])|\n{2,}")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")

# ---- 法条感知用到的三个正则 ----
# 条号本身（含中文数字与「〇/零/两」）
_ARTICLE_NO_RE = re.compile(r"第[一二三四五六七八九十百千零〇两]+条")
# 条的开头（用于把文本切成「条」）
_ARTICLE_SPLIT_RE = re.compile(r"(?=第[一二三四五六七八九十百千零〇两]+条)")
# 编 / 分编 / 章 / 节 的标题行（允许 markdown 前缀与全角空格）
_HEADING_RE = re.compile(
    r"^[ \t#>*　]*(第[一二三四五六七八九十百千零〇两]+(?:分?[编章节]))[ \t　]*([^\n]*)$",
    re.M,
)

STRATEGIES = ("fixed", "sentence", "paragraph", "article")

#: **Markdown 表格行**（解析器把 PDF 表格转成 Markdown 后就是这种形态）
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


def is_table_line(line: str) -> bool:
    """这一行是不是 Markdown 表格行（表头/数据/分隔行都算）。"""
    return bool(_TABLE_ROW_RE.match(line or ""))


def split_markdown_tables(text: str) -> list[tuple[bool, str]]:
    """把文本切成 ``[(是否表格, 片段)]``，**表格是连续的一段**。

    为什么要单独切出来：分块策略（尤其 ``article``）会按"第X条"切，
    而表格单元格里**正好会有"第X条"** ⇒ 表格被从中间切开、**表头与数据分家**
    （实测：表头落在一个块里、数据行落在另一个块里，检索时拿到"没有表头的行"）。
    """
    segments: list[tuple[bool, str]] = []
    buffer: list[str] = []
    current_is_table: bool | None = None

    def flush() -> None:
        if buffer:
            segments.append((bool(current_is_table), "\n".join(buffer)))

    for line in (text or "").splitlines():
        line_is_table = is_table_line(line)
        if current_is_table is None or line_is_table == current_is_table:
            current_is_table = line_is_table
            buffer.append(line)
            continue
        flush()
        buffer = [line]
        current_is_table = line_is_table
    flush()
    return segments


def table_blocks(segment: str, chunk_size: int) -> list[str]:
    """把一张表格切成若干块，**每块都重复表头**（"表头随块"的硬保证）。

    表格没超长就整块返回（最常见）；超长时按行切，但**每一块都带上表头与分隔行** ——
    否则检索命中的是"凭空的一行数据"，模型看不出这一列是什么。
    """
    lines = [line for line in (segment or "").splitlines() if line.strip()]
    if not lines:
        return []
    size = max(int(chunk_size), 1)
    whole = "\n".join(lines)
    if len(whole) <= size:
        return [whole]
    header = lines[:2] if len(lines) > 1 and is_table_line(lines[1]) else lines[:1]
    body = lines[len(header):]
    prefix = "\n".join(header)
    blocks: list[str] = []
    current: list[str] = []
    for row in body:
        candidate = "\n".join([prefix, *current, row])
        if current and len(candidate) > size:
            blocks.append("\n".join([prefix, *current]))
            current = [row]
        else:
            current.append(row)
    if current:
        blocks.append("\n".join([prefix, *current]))
    return blocks or [whole]


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text or "") if p.strip()]


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for block in _PARAGRAPH_SPLIT_RE.split(text or ""):
        for piece in _SENTENCE_SPLIT_RE.split(block):
            piece = (piece or "").strip()
            if piece:
                parts.append(piece)
    return parts


def _windows(units: list[str], size: int, overlap: int) -> list[str]:
    """把语义单元拼成带重叠的窗口。"""
    if not units:
        return []
    size = max(size, 1)
    overlap = max(min(overlap, size - 1), 0)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for unit in units:
        unit_len = len(unit)
        if current and current_len + unit_len > size:
            chunks.append("".join(current).strip())
            # 保留尾部若干字符作为重叠
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(current):
                if tail_len >= overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev)
            current = list(tail)
            current_len = sum(len(x) for x in current)
        current.append(unit)
        current_len += unit_len

    if current:
        chunks.append("".join(current).strip())
    return [c for c in chunks if c]


# --------------------------------------------------------------------------
# 法条感知（article）策略
# --------------------------------------------------------------------------
def split_articles(text: str) -> tuple[str, list[str], list[int]]:
    """按「第X条」切开文本。

    返回 ``(前言, [每条正文...], [每条在原文中的起始偏移...])``；
    文本里没有条号时返回 ``(text, [], [])``。
    """
    text = text or ""
    matches = list(_ARTICLE_NO_RE.finditer(text))
    if not matches:
        return text, [], []

    articles: list[str] = []
    offsets: list[int] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.start():end].strip()
        if body:
            articles.append(body)
            offsets.append(match.start())
    return text[: matches[0].start()], articles, offsets


# 层级：编(0) → 分编(1) → 章(2) → 节(3)；出现更粗的层级时要清掉更细的，否则会拼出
# 「物权编的分编 + 合同编的章」这种不存在的出处（探针第一次跑就撞到了）。
_HEADING_KIND = {"编": 0, "分编": 1, "章": 2, "节": 3}


def heading_context_before(text: str, offset: int) -> str:
    """取 ``offset`` 之前有效的「编 + 分编 + 章 + 节」标题，拼成出处前缀。

    规则：新出现的**编**清掉分编/章/节；新的**分编**清掉章/节；新的**章**清掉节。
    """
    if not text:
        return ""
    levels: dict[str, str] = {}
    for match in _HEADING_RE.finditer(text, 0, max(offset, 0)):
        token = match.group(1)
        title = (match.group(2) or "").strip()
        kind = "分编" if token.endswith("分编") else token[-1]
        order = _HEADING_KIND.get(kind, 2)
        for finer, finer_order in _HEADING_KIND.items():
            if finer_order > order:
                levels.pop(finer, None)
        levels[kind] = (f"{token} {title}").strip() if title else token
    return " ".join(levels[kind] for kind in ("编", "分编", "章", "节") if levels.get(kind))


#: 单块 **UTF-8 字节数** 硬上限：Milvus 的 varchar ``text`` 上限是 8192，而且**按字节算**
#: （2026-09-17 实测：一个 3,394 汉字的块被报 ``length: 10182`` = 3,394×3，整篇 upsert 失败）。
#: 取 6000 字节留足余量（≈2000 汉字）；**任何策略**产出的块都在 :func:`chunk_text`
#: 出口统一强制 ≤ 该值。为什么不直接调大 Milvus 上限：8k 级的块对检索本身就是灾难。
MAX_CHUNK_BYTES = 6000
#: 兼容旧名（历史上是"字符数"上限，现已改为字节）：仅供外部引用，不要再用于判断。
MAX_CHUNK_CHARS = MAX_CHUNK_BYTES


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _hard_split(text: str, limit: int) -> list[str]:
    """把过长文本切成 **UTF-8 字节数 ≤ limit** 的片段（先按句子，句子仍超长再逐字累加）。

    这是"语义切分"失败时的兜底 —— **宁可切碎，也不能让整篇文档被向量库拒收**。
    逐字累加而不是按字符切片，是为了不切出半个汉字（截断处必须落在字符边界）。
    """
    if _utf8_len(text) <= limit:
        return [text]
    pieces: list[str] = []
    for sentence in split_sentences(text) or [text]:
        if _utf8_len(sentence) <= limit:
            pieces.append(sentence)
            continue
        current, used = "", 0
        for char in sentence:
            width = len(char.encode("utf-8"))
            if used + width > limit:
                pieces.append(current)
                current, used = "", 0
            current += char
            used += width
        if current:
            pieces.append(current)
    merged: list[str] = []
    current = ""
    for piece in pieces:
        if current and _utf8_len(current) + _utf8_len(piece) > limit:
            merged.append(current)
            current = piece
        else:
            current += piece
    if current:
        merged.append(current)
    return merged


def _enforce_max(chunks: list[str], limit: int = MAX_CHUNK_BYTES) -> list[str]:
    """出口保险：任何策略都不许产出超过硬上限（**UTF-8 字节**）的块。"""
    out: list[str] = []
    for chunk in chunks:
        if _utf8_len(chunk) <= limit:
            out.append(chunk)
        else:
            out.extend(_hard_split(chunk, limit))
    return out


def _split_oversized_article(body: str, size: int, prefix: str, tag_base: str = "") -> list[str]:
    """单条超过 ``size`` 时切开；每片都带出处前缀，续片标「（续N）」。

    用 :func:`_hard_split`：句子级切分之后**仍超长**（例如整篇没有句号的名单/决定）
    会退到字符级 —— 否则会出现 4 万字单片、整篇被向量库拒收。
    """
    parts = _hard_split(body, size)
    if len(parts) <= 1:
        return [f"{prefix}{tag_base} {body}".strip()] if prefix else [body]
    out: list[str] = []
    for index, part in enumerate(parts):
        tag = "" if index == 0 else f"（续{index}）"
        out.append(f"{prefix} {tag} {part}".strip() if prefix else f"{tag} {part}".strip())
    return out


def chunk_articles(text: str, size: int, context: str = "") -> list[str]:
    """把含法条的文本切成「一条一块」的块；文本无法条结构时返回空列表。"""
    size = max(size, 1)
    preamble, articles, offsets = split_articles(text)
    if not articles:
        return []

    # 首条之前的内容**不得静默丢弃**：去掉纯标题行后若还剩**实质**文字，并入第一块。
    # 阈值取 30 字：文档标题（如「中华人民共和国民法典」10 字）与残留标记不并入——
    # 它们已由 source 与出处前缀承载，并进来只会让"每块以出处起头"失效、并可能
    # 造出"标题块"这种既无用又高分的检索噪声。
    leftover = _HEADING_RE.sub("", preamble or "")
    leftover = re.sub(r"^[ \t#>*　]+", "", leftover, flags=re.M).strip()
    lead = leftover if len(leftover) >= 30 else ""

    chunks: list[str] = []
    for index, (body, offset) in enumerate(zip(articles, offsets)):
        no_match = _ARTICLE_NO_RE.match(body)
        no = no_match.group(0) if no_match else ""
        heading = heading_context_before(text, offset) or context
        head = heading.strip()

        # 正文自带条号时，前缀只补「编/章/节」，避免条号重复
        if len(body) + len(head) + 1 <= size:
            chunk = f"{head} {body}".strip() if head else body
            if index == 0 and lead:
                chunk = f"{lead}\n{chunk}"
            chunks.append(chunk)
            continue

        # 超长条：续片必须自带条号，否则读者（与模型）看不出依据哪一条
        prefix = f"{head} {no}".strip() if no else head
        parts = _split_oversized_article(body, size, prefix)
        if index == 0 and lead and parts:
            parts[0] = f"{lead}\n{parts[0]}"
        chunks.extend(parts)
    return chunks


def chunk_text(text: str, strategy: str = "paragraph",
               chunk_size: int = 300, overlap: int = 50,
               context: str = "") -> list[str]:
    """按策略把文本切成子块；``context`` 是继承来的「编/章/节」出处前缀。

    **出口一律过 :func:`_enforce_max`**：无论用哪种策略、无论文本多畸形，都不会产出
    超过 :data:`MAX_CHUNK_CHARS` 的块（否则整篇会被 Milvus 的 varchar(8192) 拒收）。
    """
    strategy = (strategy or "paragraph").lower()
    if strategy not in STRATEGIES:
        raise ValueError(f"未知分块策略: {strategy!r}（可选 {' | '.join(STRATEGIES)}）")

    text = text or ""
    if not text.strip():
        return []

    # ⚠️ **表格必须整块保留**（2026-09-26 实测）：`article` 策略按"第X条"切，
    #    而表格单元格里就会有"第X条"（"| 第一条 | 如实申报 | 警告 |"）
    #    ⇒ 表格被从中间切开、**表头与数据分家**，检索命中的是"凭空的一行数据"。
    #    做法：先把表格段切出来单独成块（超长则按行切但**每块重复表头**），
    #    其余文本照原策略走。表格段不参与 `_windows` 的重叠逻辑（重叠会重复行）。
    segments = split_markdown_tables(text)
    if any(is_table for is_table, _segment in segments):
        result: list[str] = []
        for is_table, segment in segments:
            if is_table:
                result.extend(table_blocks(segment, chunk_size))
                continue
            if segment.strip():
                result.extend(chunk_text(segment, strategy=strategy,
                                         chunk_size=chunk_size, overlap=overlap,
                                         context=context))
        return _enforce_max(result)

    if strategy == "fixed":
        size = max(chunk_size, 1)
        step = max(size - max(overlap, 0), 1)
        result = [text[i:i + size].strip() for i in range(0, len(text), step) if text[i:i + size].strip()]
        return _enforce_max(result)

    if strategy == "article":
        chunks = chunk_articles(text, chunk_size, context=context)
        if chunks:
            return _enforce_max(chunks)
        # 非法律文本（没有条号）→ 退化为段落窗口，不报错、不丢内容
        strategy = "paragraph"

    if strategy == "sentence":
        return _enforce_max(_windows(split_sentences(text), chunk_size, overlap))

    # paragraph：段落优先，超长段落退回句子级；单元本身超长由 _enforce_max 兜底
    units: list[str] = []
    for para in split_paragraphs(text):
        if len(para) <= chunk_size:
            units.append(para + "\n")
        else:
            units.extend(s + "\n" for s in split_sentences(para))
    return _enforce_max(_windows(units, chunk_size, overlap))


def build_chunks(document: Document, config: ChunkConfig | None = None) -> list[Chunk]:
    """把 Document 切成「父块 + 子块」两级 chunk 列表。"""
    config = config or ChunkConfig()
    now = now_ts()
    chunks: list[Chunk] = []

    parent_blocks = chunk_text(
        document.text,
        strategy="paragraph",
        chunk_size=config.parent_size,
        overlap=0,
    ) or [document.text]

    # 预先定位每个父块在全文中的偏移：article 策略要据此继承「编/章/节」，
    # 否则跨父块的章（父块边界落在章中间）会丢掉章节出处。
    offsets: list[int] = []
    cursor = 0
    for block in parent_blocks:
        position = document.text.find(block, cursor)
        if position < 0:
            position = cursor
        offsets.append(position)
        cursor = position + len(block)

    for p_index, parent_text in enumerate(parent_blocks):
        if not parent_text.strip():
            continue
        parent_id = stable_id(document.doc_id, "parent", p_index)
        chunks.append(Chunk(
            id=parent_id,
            text=parent_text,
            source=document.source,
            doc_id=document.doc_id,
            chunk_index=p_index,
            parent_id=None,
            is_parent=True,
            summary=make_summary(parent_text),
            created_at=now,
            updated_at=now,
        ))

        children = chunk_text(
            parent_text,
            strategy=config.strategy,
            chunk_size=config.chunk_size,
            overlap=config.overlap,
            context=heading_context_before(document.text, offsets[p_index]),
        )
        for c_index, child_text in enumerate(children):
            chunks.append(Chunk(
                id=stable_id(document.doc_id, parent_id, c_index, child_text),
                text=child_text,
                source=document.source,
                doc_id=document.doc_id,
                chunk_index=c_index,
                parent_id=parent_id,
                is_parent=False,
                summary=make_summary(child_text),
                created_at=now,
                updated_at=now,
            ))

    return chunks
