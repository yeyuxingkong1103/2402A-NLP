"""
chunking.py — 文本分块策略

提供六种切分方式，统一返回 Chunk 结构（dict）：text / chunk_type / index 是通用字段；
heading 记录所属标题；parent_index 与 parent_text 仅父子块策略使用 —— 检索命中子块时，
把 parent_text 一并交给大模型，兼顾检索精度与上下文完整度。
所有策略都不产生空块，长度小于 MIN_CHUNK_LENGTH 的碎片会被丢弃。
"""

from __future__ import annotations

import re
from typing import Sequence

import config
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 句子结束标点：中英文句号、问号、叹号、分号，以及换行
_SENTENCE_END = re.compile(r"(?<=[。！？；!?;])\s*")
# 段落分隔：一个及以上空行
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
# 标题行：第X章/节、一、二、、1.1、1)、# Markdown 标题等
# 编号与标题正文之间允许有空格（"第一章 总则"），故用 [ \t　]* 而非 \S
_HEADING = re.compile(
    r"^[ \t　]*(?:"
    r"第[一二三四五六七八九十百零\d]+[章节部分篇]"
    r"|[一二三四五六七八九十]+[、.．]"
    r"|\d+(?:\.\d+)*[、.． \t　]"
    r"|#{1,6}[ \t　]"
    r")[ \t　]*\S.*$",
    re.MULTILINE,
)

DEFAULT_STRATEGY = "parent_child"


def _clean(text: str) -> str:
    """合并多余空白，去掉零宽字符与首尾空格。"""
    text = text.replace("​", "").replace("﻿", "")
    text = re.sub(r"[ \t　]+", " ", text)
    return text.strip()


def _make_chunk(text: str, chunk_type: str, index: int, **extra) -> dict | None:
    """构造统一结构的块，过短的碎片返回 None。"""
    text = _clean(text)
    if len(text) < config.MIN_CHUNK_LENGTH:
        return None
    chunk = {"text": text, "chunk_type": chunk_type, "index": index}
    chunk.update(extra)
    return chunk


def _pack_sentences(sentences: Sequence[str], size: int, overlap: int) -> list[str]:
    """把句子按顺序打包成接近 size 的块，块间保留 overlap 个字符的上下文。"""
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if not sentence:
            continue
        # 单句超长时单独成块，避免块内混入无关内容
        if len(sentence) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence)
            continue

        if len(current) + len(sentence) <= size:
            current += sentence
        else:
            chunks.append(current)
            # 用块尾作为下一块的开头，保持上下文连续
            current = (current[-overlap:] if overlap > 0 else "") + sentence

    if current:
        chunks.append(current)
    return chunks


class Chunker:
    """文本分块器。所有方法返回 list[dict]，结构见模块文档。"""

    def __init__(self, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP):
        self.size = size
        self.overlap = overlap

    # ------------------------------------------------------------ 固定长度

    def chunk_fixed(self, text: str, size: int | None = None, overlap: int | None = None) -> list[dict]:
        """按固定长度切分，相邻块保留 overlap 个字符的重叠。

        改由 LangChain 的 RecursiveCharacterTextSplitter 承担：它在段落、句子这类自然
        边界断开，比等宽截断少切碎语义。中文没有空格，分隔符要显式补上中文句读。
        """
        size = size or self.size
        overlap = self.overlap if overlap is None else overlap
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            # splitter 要求 overlap < size，入参越界时压回去，别让它抛异常
            chunk_overlap=min(max(0, overlap), max(0, size - 1)),
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            keep_separator=True,
        )

        chunks: list[dict] = []
        for piece in splitter.split_text(_clean(text)):
            chunk = _make_chunk(piece, "fixed", len(chunks))
            if chunk:
                chunks.append(chunk)
        return chunks

    # ------------------------------------------------------------ 按句子

    def chunk_by_sentence(self, text: str) -> list[dict]:
        """按句子边界切分，再把句子聚合成接近 size 的块。"""
        sentences = [s for s in _SENTENCE_END.split(_clean(text)) if s.strip()]
        chunks: list[dict] = []
        for piece in _pack_sentences(sentences, self.size, self.overlap):
            chunk = _make_chunk(piece, "sentence", len(chunks))
            if chunk:
                chunks.append(chunk)
        return chunks

    # ------------------------------------------------------------ 按段落

    def chunk_by_paragraph(self, text: str) -> list[dict]:
        """按空行分段；超长段落再按句子二次切分。"""
        chunks: list[dict] = []
        for paragraph in _PARAGRAPH_SPLIT.split(text):
            paragraph = _clean(paragraph)
            if len(paragraph) < config.MIN_CHUNK_LENGTH:
                continue

            if len(paragraph) <= self.size * 1.5:
                chunk = _make_chunk(paragraph, "paragraph", len(chunks))
                if chunk:
                    chunks.append(chunk)
                continue

            for piece in _pack_sentences(
                [s for s in _SENTENCE_END.split(paragraph) if s.strip()], self.size, self.overlap
            ):
                chunk = _make_chunk(piece, "paragraph", len(chunks))
                if chunk:
                    chunks.append(chunk)
        return chunks

    # ------------------------------------------------------------ 按标题

    def chunk_by_heading(self, text: str) -> list[dict]:
        """按标题切分，每个块带上它所属的标题，便于回答时保留章节语境。"""
        text = _clean(text)
        matches = list(_HEADING.finditer(text))

        if not matches:
            # 没有识别到标题结构，退化为按段落分块
            return self.chunk_by_paragraph(text)

        chunks: list[dict] = []
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            section = text[start:end].strip()
            heading = match.group().strip()[:80]

            if len(section) <= self.size * 1.5:
                chunk = _make_chunk(section, "heading", len(chunks), heading=heading)
                if chunk:
                    chunks.append(chunk)
                continue

            # 章节过长时按句子切，每块都补上标题作为前缀
            body = section[match.end() - start :]
            for piece in _pack_sentences(
                [s for s in _SENTENCE_END.split(body) if s.strip()], self.size - len(heading), self.overlap
            ):
                chunk = _make_chunk(f"{heading}\n{piece}", "heading", len(chunks), heading=heading)
                if chunk:
                    chunks.append(chunk)
        return chunks

    # ------------------------------------------------------------ 语义分块

    def chunk_semantic(self, text: str, threshold: float = 0.55) -> list[dict]:
        """在语义低谷处切分：相邻句子相似度骤降的位置视为话题边界。

        需要向量化支持；若编码失败则自动退化为按段落分块。
        """
        sentences = [s.strip() for s in _SENTENCE_END.split(_clean(text)) if s.strip()]
        if len(sentences) <= 2:
            return self.chunk_by_paragraph(text)

        try:
            import embeddings as emb

            vectors = emb.encode_texts(sentences)
        except Exception:
            return self.chunk_by_paragraph(text)

        import numpy as np

        matrix = np.asarray(vectors, dtype=np.float32)
        boundaries = [0]
        for i in range(1, len(sentences)):
            similarity = float(np.dot(matrix[i - 1], matrix[i]))
            # 相似度明显低于阈值时认为话题切换
            if similarity < threshold:
                boundaries.append(i)
        boundaries.append(len(sentences))

        chunks: list[dict] = []
        for start, end in zip(boundaries, boundaries[1:]):
            merged = "".join(sentences[start:end])
            # 合并后仍然过短时，与后一块合并处理由 _pack 兜底
            chunk = _make_chunk(merged, "semantic", len(chunks))
            if chunk:
                chunks.append(chunk)

        if not chunks:
            return self.chunk_by_paragraph(text)

        # 相邻的短块再打包一次，避免产生大量碎片
        packed: list[dict] = []
        for piece in _pack_sentences([c["text"] for c in chunks], self.size, self.overlap):
            chunk = _make_chunk(piece, "semantic", len(packed))
            if chunk:
                packed.append(chunk)
        return packed

    # ------------------------------------------------------------ 父子块

    def chunk_parent_child(self, text: str, ratio: int | None = None) -> list[dict]:
        """父子块：大块保留完整语境，小块用于精确检索。

        返回的每个元素都是子块，附带 parent_index / parent_text；
        检索命中子块后，把 parent_text 一并交给大模型，兼顾精度与上下文。
        """
        ratio = ratio or config.PARENT_CHILD_RATIO
        parent_size = self.size * ratio

        # 父块按段落边界切，尽量保证语义完整
        parent_texts: list[str] = []
        buffer = ""
        for paragraph in _PARAGRAPH_SPLIT.split(_clean(text)):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(buffer) + len(paragraph) <= parent_size:
                buffer += ("\n" if buffer else "") + paragraph
            else:
                if buffer:
                    parent_texts.append(buffer)
                buffer = paragraph
        if buffer:
            parent_texts.append(buffer)

        chunks: list[dict] = []
        for parent_index, parent_text in enumerate(parent_texts):
            if len(parent_text) < config.MIN_CHUNK_LENGTH:
                continue
            for piece in _pack_sentences(
                [s for s in _SENTENCE_END.split(parent_text) if s.strip()], self.size, self.overlap
            ):
                chunk = _make_chunk(
                    piece,
                    "parent_child",
                    len(chunks),
                    parent_index=parent_index,
                    parent_text=parent_text,
                )
                if chunk:
                    chunks.append(chunk)

        return chunks if chunks else self.chunk_by_paragraph(text)

    # ------------------------------------------------------------ 统一入口

    def chunk(self, text: str, strategy: str = DEFAULT_STRATEGY) -> list[dict]:
        """按名称选择分块策略。"""
        strategies = {
            "fixed": self.chunk_fixed,
            "sentence": self.chunk_by_sentence,
            "paragraph": self.chunk_by_paragraph,
            "heading": self.chunk_by_heading,
            "semantic": self.chunk_semantic,
            "parent_child": self.chunk_parent_child,
        }
        if strategy not in strategies:
            raise ValueError(f"未知分块策略：{strategy}，可选：{', '.join(strategies)}")
        return strategies[strategy](text)


def chunk_text(text: str, strategy: str = DEFAULT_STRATEGY) -> list[dict]:
    """便捷函数：对单段文本分块。"""
    return Chunker().chunk(text, strategy)
