"""文本分块：固定长度 / 句子 / 段落 / 标题 / 语义 / 父子块。

分块大小以 token 估算（中文 1 字符≈1 token，英文 4 字符≈1 token），
与 .env 中的 CHUNK_SIZE / CHUNK_OVERLAP / PARENT_CHUNK_SIZE / CHILD_CHUNK_SIZE 对应。
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger("rag.chunker")

_SENTENCE_END = "。！？!?；;\n"
_HEADING_RE = re.compile(r"^\s*(第[一二三四五六七八九十百]+[章节讲部分]|[0-9]+(\.[0-9]+)*[、.．]?\s|\#{1,6}\s)")


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数：CJK 字符按 1，其他按 4 字符 1 token。

    为什么不精确分词？这里只需在"分块大小"层面做近似控制，
    中文 1 字≈1 token、英文 4 字符≈1 token 是业界常用粗略比例，
    足够保证块体量落在模型上下文可接受范围内，且零额外依赖、极快。
    """
    if not text:
        return 0
    cjk = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uff00-\uffef]", text))
    other = len(text) - cjk
    if not other:
        return cjk
    return cjk + max(1, other // 4)


@dataclass
class Chunk:
    """一个文本块：content 是正文，chunk_index 是块序号，parent_index 指向父块。

    父子分块时，子块(parent 的 child)用于检索，parent_index 指向父块序号，
    extra 里存父块完整内容，供回答时展开更大上下文。
    """
    content: str
    chunk_index: int
    parent_index: Optional[int] = None
    summary: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.content)


def split_sentences(text: str) -> List[str]:
    """按中英文标点切句。"""
    sentences, buf = [], ""
    for ch in text:
        buf += ch
        if ch in _SENTENCE_END:
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())
    return [s for s in sentences if s]


def split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in parts if p]


def _clean(text: str) -> str:
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_by_sentences(text: str, chunk_size: int, overlap: int) -> List[Chunk]:
    """句子级滑窗：按句子聚合到目标大小，句间重叠。

    先按标点切句，再逐句累加；一旦加下一句会超 chunk_size，就把当前缓冲
    落成一个块。然后"回填"：从缓冲末尾反向挑句子，凑够 overlap 大小的
    重叠，作为下一个块的起点——保证块与块之间有上下文衔接，避免语义被硬切。
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    chunks: List[Chunk] = []
    buf: List[str] = []
    buf_tokens = 0

    for sent in sentences:
        st = estimate_tokens(sent)
        if buf and buf_tokens + st > chunk_size:
            chunks.append(Chunk(content=_clean("".join(buf)), chunk_index=len(chunks)))
            # 回填重叠句子
            tail: List[str] = []
            tail_tokens = 0
            for s in reversed(buf):
                s_tokens = estimate_tokens(s)
                if tail_tokens + s_tokens > overlap:
                    break
                tail.insert(0, s)
                tail_tokens += s_tokens
            buf, buf_tokens = tail, tail_tokens
        buf.append(sent)
        buf_tokens += st

    if buf:
        chunks.append(Chunk(content=_clean("".join(buf)), chunk_index=len(chunks)))
    return [c for c in chunks if c.content]


def chunk_fixed(text: str, chunk_size: int, overlap: int) -> List[Chunk]:
    """固定长度滑窗（按字符近似 token）。

    最简单、最通用的分块法：以 step = chunk_size - overlap 为步长滑动，
    每个窗口取 chunk_size 个字符。代价是完全不顾语义，可能把一句话从中间切开，
    适合无结构的纯文本兜底。
    """
    text = _clean(text)
    if not text:
        return []
    step = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(text), step):
        piece = text[i:i + chunk_size]
        if piece.strip():
            chunks.append(Chunk(content=piece.strip(), chunk_index=len(chunks)))
        if i + chunk_size >= len(text):
            break
    return chunks


def chunk_by_paragraph(text: str, chunk_size: int, overlap: int) -> List[Chunk]:
    """段落聚合：优先保持段落完整，超长段落内部再按句子切。

    段落通常是语义的最小完整单元，优先按段落边界切分可保持语义完整；
    但个别段落可能极长（如整页无换行），此时单独一段就超过 chunk_size，
    需要把它交给 chunk_by_sentences 进一步切成句子级小块。
    """
    chunks: List[Chunk] = []
    buf, buf_tokens = [], 0
    for para in split_paragraphs(text):
        pt = estimate_tokens(para)
        if pt > chunk_size:
            if buf:
                chunks.append(Chunk(content=_clean("\n\n".join(buf)), chunk_index=len(chunks)))
                buf, buf_tokens = [], 0
            for sub in chunk_by_sentences(para, chunk_size, overlap):
                sub.chunk_index = len(chunks)
                chunks.append(sub)
            continue
        if buf and buf_tokens + pt > chunk_size:
            chunks.append(Chunk(content=_clean("\n\n".join(buf)), chunk_index=len(chunks)))
            buf, buf_tokens = [], 0
        buf.append(para)
        buf_tokens += pt
    if buf:
        chunks.append(Chunk(content=_clean("\n\n".join(buf)), chunk_index=len(chunks)))
    return [c for c in chunks if c.content]


def chunk_by_heading(text: str, chunk_size: int, overlap: int) -> List[Chunk]:
    """标题分块：把标题行作为分块边界，块内过大再按段落切。

    书籍/论文常以"第一章""1.1"或 # 号作标题，标题天然是章节边界；
    按标题切分能让检索出的块自带章节语境。若某章节仍然过大，再退回段落级切分。
    """
    lines = text.split("\n")
    sections: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        if _HEADING_RE.match(line) and current:
            sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(current)

    chunks: List[Chunk] = []
    for section in sections:
        section_text = "\n".join(section).strip()
        if not section_text:
            continue
        if estimate_tokens(section_text) <= chunk_size:
            chunks.append(Chunk(content=_clean(section_text), chunk_index=len(chunks)))
        else:
            for sub in chunk_by_paragraph(section_text, chunk_size, overlap):
                sub.chunk_index = len(chunks)
                chunks.append(sub)
    return chunks


def chunk_semantic(text: str, chunk_size: int, overlap: int) -> List[Chunk]:
    """语义分块（轻量近似）：段落内句子相似度突变处切分，此处用长度+段落边界近似。

    真正的语义分块需要向量模型计算相邻句相似度（昂贵）；本实现用
    "段落边界 + 长度上限"做轻量近似，兼顾效果与成本，避免每次建库都跑模型。
    """
    paragraphs = split_paragraphs(text) or [text]
    chunks: List[Chunk] = []
    buf, buf_tokens = [], 0
    for para in paragraphs:
        pt = estimate_tokens(para)
        if buf and buf_tokens + pt > chunk_size:
            chunks.append(Chunk(content=_clean("\n\n".join(buf)), chunk_index=len(chunks)))
            buf, buf_tokens = [], 0
        if pt > chunk_size:
            for sub in chunk_by_sentences(para, chunk_size, overlap):
                sub.chunk_index = len(chunks)
                chunks.append(sub)
            continue
        buf.append(para)
        buf_tokens += pt
    if buf:
        chunks.append(Chunk(content=_clean("\n\n".join(buf)), chunk_index=len(chunks)))
    return [c for c in chunks if c.content]


def chunk_parent_child(text: str, parent_size: Optional[int] = None,
                       child_size: Optional[int] = None) -> List[Chunk]:
    """父子分块：父块用于上下文，子块用于检索，子块 parent_index 指向父块序号。

    动机：检索精度与上下文完整度是一对矛盾——小块检索更精准但缺上下文，
    大块上下文更完整但检索易跑偏。父子分块用「小块检索、大块供给上下文」
    两全：先切大"父块"（如 800 token），再把每个父块切成若干"子块"（如 200 token），
    检索时命中子块，回答时通过 parent_index 取回父块全文作为背景。
    """
    parent_size = parent_size or settings.parent_chunk_size
    child_size = child_size or settings.child_chunk_size
    parents = chunk_by_paragraph(text, parent_size, int(parent_size * 0.15))
    children: List[Chunk] = []
    for p_idx, parent in enumerate(parents):
        sub_chunks = chunk_by_sentences(parent.content, child_size, int(child_size * 0.2))
        if not sub_chunks:
            sub_chunks = [Chunk(content=parent.content, chunk_index=0)]
        for sub in sub_chunks:
            children.append(Chunk(
                content=sub.content,
                chunk_index=len(children),
                parent_index=p_idx,
                extra={"parent_content": parent.content},
            ))
    return children


STRATEGIES = {
    # 策略名 → 分块函数 的注册表。chunk_text 按 strategy 字符串查表分发，
    # 新增策略只需在此登记即可，符合开闭原则（对扩展开放、对修改封闭）。
    "fixed": chunk_fixed,
    "sentence": chunk_by_sentences,
    "paragraph": chunk_by_paragraph,
    "heading": chunk_by_heading,
    "semantic": chunk_semantic,
}


def chunk_text(text: str, strategy: str = "paragraph",
               chunk_size: Optional[int] = None,
               overlap: Optional[int] = None) -> List[Chunk]:
    """按策略分块，默认段落+句子混合（对中文心理类文本效果稳定）。

    parent_child 是特殊策略（需要两个尺寸参数），单独分支处理；
    其余策略统一查 STRATEGIES 表。最后重排 chunk_index 保证序号连续。
    """
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    text = _clean(text or "")
    if not text:
        return []
    if strategy == "parent_child":
        chunks = chunk_parent_child(text)
    else:
        func = STRATEGIES.get(strategy, chunk_by_paragraph)
        chunks = func(text, chunk_size, overlap)
    for i, c in enumerate(chunks):
        c.chunk_index = i
    logger.debug("分块完成 strategy=%s 块数=%d", strategy, len(chunks))
    return chunks


def summarize_chunk(content: str, max_len: int = 120) -> str:
    """规则摘要：取首句 + 关键片段，避免额外调用大模型。

    首句通常是段落主题句，截断即可得到足够好的摘要；
    用规则而不是 LLM，是为了在建库批处理时保持零成本、零延迟。
    """
    sentences = split_sentences(content)
    summary = sentences[0] if sentences else content
    return summary[:max_len]