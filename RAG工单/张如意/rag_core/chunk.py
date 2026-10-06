# -*- coding: utf-8 -*-
"""
文本分块模块
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

提供 4 种分块策略，工单02 通过对比实验选出最优策略：
  fixed      —— 固定窗口 + 重叠（Baseline，工单01 使用）
  recursive  —— 按 段落→句子 递归切分（通用改良）
  semantic   —— 基于相邻句向量相似度的语义断点切分
  structure  —— 招股书章节结构感知切分（工单02 最终采用）

关键设计：
  1. 表格块整体保留，绝不切碎（工单03 表格检索的前提）
  2. 每个 chunk 携带 (文档, 页码, 章节路径)，供答案溯源
  3. 结构分块会在 chunk 头部注入「章节路径」，提升关键词匹配率
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from . import config
from .pdf_parse import PageBlock


@dataclass
class Chunk:
    text: str
    doc: str
    page: int
    type: str = "text"
    chunk_id: str = ""
    section: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id, "text": self.text, "doc": self.doc,
            "page": self.page, "type": self.type, "section": self.section,
            **self.meta,
        }


# ---------------------------------------------------------------------------
# 招股说明书章节标题识别
# ---------------------------------------------------------------------------
# 形如：第五节 业务与技术 / 一、主营业务 / （一）主要产品 / 1、产品分类
# 第 4 条要求编号后**不能紧跟数字**，否则 "1.00 元" 这类小数会被误判成标题
_SECTION_PATS = [
    (re.compile(r"^第[一二三四五六七八九十百]+[节章]\s*[、.．]?\s*(.{0,30})"), 1),
    (re.compile(r"^([一二三四五六七八九十]{1,3})[、.．]\s*(.{0,30})"), 2),
    (re.compile(r"^（([一二三四五六七八九十]{1,3})）\s*(.{0,30})"), 3),
    (re.compile(r"^(\d{1,2})[、.．](?!\d)\s*(.{0,30})"), 4),
]

# 标题行的长度上限：超过此长度几乎不可能是标题，而是被误合的正文
_MAX_HEADING_LEN = 40


def _detect_heading(line: str) -> tuple[int, str] | None:
    """
    返回 (层级, 标题文本)，不是标题则返回 None。

    判据（三重过滤，宁可漏判也不误判）：
      1. 长度不超过 _MAX_HEADING_LEN
      2. 匹配章节/编号模式
      3. 标题部分不能以数字或标点开头（排除 "1.00 元"、"3.5 亿元" 之类）
    """
    s = line.strip()
    if not s or len(s) > _MAX_HEADING_LEN:
        return None
    for pat, lvl in _SECTION_PATS:
        m = pat.match(s)
        if m:
            tail = (m.group(m.lastindex) or "").strip()
            # 编号后应当跟真实标题词；以数字/标点开头说明是数值而非标题
            if tail and re.match(r"^[\d\s、，,。.．%％)]", tail):
                return None
            return lvl, s
    return None


# ---------------------------------------------------------------------------
# 策略 1：固定窗口
# ---------------------------------------------------------------------------
def chunk_fixed(blocks: list[PageBlock],
                size: int = config.CHUNK_SIZE,
                overlap: int = config.CHUNK_OVERLAP) -> list[Chunk]:
    """Baseline：纯字符窗口滑动，不考虑语义边界（工单01 使用）。"""
    chunks: list[Chunk] = []
    for b in blocks:
        if b.type != "text":
            chunks.append(_block_to_single_chunk(b))
            continue
        text, step = b.content, size - overlap
        for i in range(0, len(text), step):
            seg = text[i:i + size]
            if len(seg.strip()) < 20:
                continue
            chunks.append(Chunk(text=seg, doc=b.doc, page=b.page, type=b.type))
    return _assign_ids(chunks)


# ---------------------------------------------------------------------------
# 策略 2：递归切分（段落 → 句子）
# ---------------------------------------------------------------------------
_SENT_END = re.compile(r"(?<=[。！？；!?;])")


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_END.split(text) if s.strip()]


def chunk_recursive(blocks: list[PageBlock],
                    size: int = config.CHUNK_SIZE,
                    overlap: int = config.CHUNK_OVERLAP) -> list[Chunk]:
    """按段落聚合，超长段落再按句子切，尽量不切断句子。"""
    chunks: list[Chunk] = []
    for b in blocks:
        if b.type != "text":
            chunks.append(_block_to_single_chunk(b))
            continue
        buf = ""
        for para in b.content.split("\n"):
            if not para.strip():
                continue
            if len(buf) + len(para) <= size:
                buf += para
                continue
            if buf:
                chunks.append(Chunk(text=buf, doc=b.doc, page=b.page, type=b.type))
                buf = buf[-overlap:] if overlap else ""
            if len(para) <= size:
                buf += para
            else:                       # 超长段落 -> 句子级切分
                cur = buf
                for sent in _split_sentences(para):
                    if len(cur) + len(sent) > size and cur.strip():
                        chunks.append(Chunk(text=cur, doc=b.doc, page=b.page, type=b.type))
                        cur = cur[-overlap:] if overlap else ""
                    cur += sent
                buf = cur
        if buf.strip():
            chunks.append(Chunk(text=buf, doc=b.doc, page=b.page, type=b.type))
    return _assign_ids(chunks)


# ---------------------------------------------------------------------------
# 策略 3：语义分块
# ---------------------------------------------------------------------------
def chunk_semantic(blocks: list[PageBlock],
                   size: int = config.CHUNK_SIZE,
                   threshold: float = 0.72) -> list[Chunk]:
    """
    按相邻句向量的余弦相似度找语义断点：
    相似度骤降处即为话题切换点，在此断开，保证 chunk 内语义一致。
    """
    from . import embed

    chunks: list[Chunk] = []
    for b in blocks:
        if b.type != "text":
            chunks.append(_block_to_single_chunk(b))
            continue
        sents = _split_sentences(b.content)
        if len(sents) <= 2:
            chunks.append(Chunk(text=b.content, doc=b.doc, page=b.page, type=b.type))
            continue

        vecs = embed.encode(sents, show_progress=False)
        sims = np.sum(vecs[:-1] * vecs[1:], axis=1)

        cur, cur_len = sents[0], len(sents[0])
        for k, sim in enumerate(sims):
            nxt = sents[k + 1]
            # 语义跳变 或 长度超限 -> 断块
            if (sim < threshold and cur_len > size * 0.4) or cur_len + len(nxt) > size * 1.5:
                chunks.append(Chunk(text=cur, doc=b.doc, page=b.page, type=b.type))
                cur, cur_len = nxt, len(nxt)
            else:
                cur += nxt
                cur_len += len(nxt)
        if cur.strip():
            chunks.append(Chunk(text=cur, doc=b.doc, page=b.page, type=b.type))
    return _assign_ids(chunks)


# ---------------------------------------------------------------------------
# 策略 4：章节结构感知（工单02 最终方案）
# ---------------------------------------------------------------------------
def chunk_structure(blocks: list[PageBlock],
                    size: int = config.CHUNK_SIZE,
                    overlap: int = config.CHUNK_OVERLAP,
                    min_size: int = 120) -> list[Chunk]:
    """
    利用招股书天然章节层级切分，并在 chunk 文本头部注入章节路径。

    为什么有效：原问题「武汉兴图新科…来自军用领域的收入分别是多少？」
    中的实体词在正文里常被简称为「公司」「发行人」，单纯向量检索会漂移；
    注入「第五节 业务与技术 > 一、主营业务 > （三）军用领域收入」这样的
    路径后，关键词与语义信号同时增强，命中率显著提升。

    Args:
        min_size: 最小块长（不含章节路径前缀）。短于此长度的片段不单独成块，
            而是并入下一块——避免招股书里大量「一、二级标题紧挨着」的
            情形产生一堆只有标题没有正文的碎块。
    """
    chunks: list[Chunk] = []
    stack: dict[int, str] = {}
    buf = ""
    cur_section = ""
    pending = ""          # 过短片段，暂存后并入下一块

    def content_len(s: str) -> int:
        """去掉开头的【章节路径】前缀后的有效正文长度。"""
        return len(re.sub(r"^【[^】]*】", "", s).strip())

    def flush(page: int, doc: str, section: str, force: bool = False):
        nonlocal buf, pending
        text = buf.strip()
        if text and (force or content_len(text) >= min_size or pending):
            chunks.append(Chunk(text=(pending + text), doc=doc, page=page,
                                type="text", section=section))
            pending = ""
        elif text:
            pending += text
        buf = ""

    for b in blocks:
        if b.type != "text":
            flush(b.page, b.doc, cur_section)
            c = _block_to_single_chunk(b)
            c.section = cur_section
            chunks.append(c)
            continue

        for line in b.content.split("\n"):
            if not line.strip():
                continue
            hd = _detect_heading(line)
            if hd:
                lvl, title = hd
                stack[lvl] = title
                for k in list(stack):
                    if k > lvl:
                        del stack[k]
                flush(b.page, b.doc, cur_section)
                cur_section = " > ".join(stack[k] for k in sorted(stack))
                buf = f"【{cur_section}】"
                continue

            if len(buf) + len(line) > size and len(buf) > size * 0.5:
                flush(b.page, b.doc, cur_section)
                buf = f"【{cur_section}】" + (buf[-overlap:] if overlap else "")
            buf += line

    flush(blocks[-1].page if blocks else 1,
          blocks[-1].doc if blocks else "", cur_section, force=True)
    return _assign_ids(chunks)


# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------
def _block_to_single_chunk(b: PageBlock) -> Chunk:
    """表格 / 图像块整体成为一个 chunk，不切分。"""
    return Chunk(text=b.content or "", doc=b.doc, page=b.page, type=b.type,
                 meta={k: v for k, v in b.extra.items() if k != "image_path"})


def _assign_ids(chunks: list[Chunk]) -> list[Chunk]:
    from collections import defaultdict
    counters: dict[str, int] = defaultdict(int)
    for c in chunks:
        if not c.text.strip():
            continue
        key = f"{c.doc}|{c.type}"
        counters[key] += 1
        c.chunk_id = f"{c.doc}-{c.type[:3]}-{counters[key]:05d}"
    return [c for c in chunks if c.text.strip()]


STRATEGIES = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
    "semantic": chunk_semantic,
    "structure": chunk_structure,
}


def chunk_blocks(blocks: list[PageBlock], strategy: str = "structure", **kw) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"未知分块策略：{strategy}，可选 {list(STRATEGIES)}")
    return STRATEGIES[strategy](blocks, **kw)
