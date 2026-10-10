# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""三模态差异化分块。

策略：
  - 文本：full_04 按语义边界切分（500~800字，重叠15%）；baseline_03 固定 512 字
  - 表格：整表不切分（破坏表结构会丢失语义）
  - 图像：整图一个块，正文 = VLM描述 + 题注 + 章节路径
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable

from rag04.config import Settings
from rag04.schema import Chunk, FigureBlock, TableBlock, TextBlock

_CJK = re.compile(r"[一-鿿]")
_SENT_END = re.compile(r"(?<=[。！？；!?;])")

# RC5：句末标点与可后置的收尾符号。行尾若只差这些收尾符，仍视为句末。
_END_PUNCT = "。！？；!?;"
_CLOSERS = "”’」』）)】》〉\"'"


def _ends_sentence(text: str) -> bool:
    """行尾是否收句（允许句末标点后跟引号/括号等收尾符）。"""
    t = (text or "").rstrip()
    i = len(t) - 1
    while i >= 0 and t[i] in _CLOSERS:
        i -= 1
    return i >= 0 and t[i] in _END_PUNCT


def _join_lines(prev: str, cur: str) -> str:
    """拼行。ASCII 字母数字相接处补空格（英文行块切断时避免粘词）。"""
    if (prev and cur and prev[-1].isascii() and prev[-1].isalnum()
            and cur[0].isascii() and cur[0].isalnum()):
        return prev + " " + cur
    return prev + cur


def merge_line_blocks(blocks: Iterable[TextBlock]) -> list[TextBlock]:
    """RC5：把 PDF 行级块合并为段落级块（重建第二阶段）。

    根因（`_scratch/root-cause-retrieval.md` RC5）：PyMuPDF 在本语料上把每个
    排版行产出一个块，而答案句/清单常跨行——金额续句（id260/33）、要点清单
    （id2）、上下游清单（id34/793）被切断后单独成块，正确块召回后仍缺要点。

    判据：**上一行末尾没有句末标点（。！？；!?;，允许后置引号/括号）即未收句**，
    与下一行合并；已收句的行不吞并下一行。只在同一 (doc_id, page) 内合并，
    不跨页（页眉页脚会横插在页边界上）。合并块沿用首块 bbox/页码以保持溯源，
    旋转标志按行取或。
    """
    out: list[TextBlock] = []
    for b in blocks:
        text = (b.text or "").strip()
        if not text:
            continue
        if out and out[-1].doc_id == b.doc_id and out[-1].page == b.page \
                and not _ends_sentence(out[-1].text):
            prev = out[-1]
            out[-1] = TextBlock(
                doc_id=prev.doc_id, page=prev.page, bbox=prev.bbox,
                text=_join_lines(prev.text, text),
                has_rotated_text=prev.has_rotated_text or b.has_rotated_text,
                section_path=prev.section_path or b.section_path,
            )
        else:
            out.append(TextBlock(
                doc_id=b.doc_id, page=b.page, bbox=b.bbox, text=text,
                has_rotated_text=b.has_rotated_text,
                section_path=b.section_path,
            ))
    return out


def detect_lang(text: str) -> str:
    """含中日韩汉字即判为中文，否则英文。"""
    return "zh" if _CJK.search(text or "") else "en"


def _cid(doc_id: str, page: int, block_type: str, source_id: str, idx: int) -> str:
    raw = f"{doc_id}|{page}|{block_type}|{source_id}|{idx}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _split_sentences(text: str) -> list[str]:
    parts = [p for p in _SENT_END.split(text) if p and p.strip()]
    return parts or ([text] if text.strip() else [])


def _hard_split_sentence(text: str, hi: int, overlap: int) -> list[str]:
    """病态长句（无标点/OCR 乱码）硬切：每片 <= hi，相邻片重叠 overlap 字。

    尾部若已被上一片完全覆盖（剩余长度 <= overlap）则不产出，避免重复块。
    """
    step = max(hi - overlap, 1)
    pieces: list[str] = []
    for start in range(0, len(text), step):
        piece = text[start:start + hi]
        if not piece:
            break
        if pieces and len(piece) <= overlap:
            break
        pieces.append(piece)
        if start + hi >= len(text):
            break
    return pieces


def _text_chunk(b: TextBlock, bi: int, ci: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=_cid(b.doc_id, b.page, "text", f"p{b.page}b{bi}", ci),
        doc_id=b.doc_id, page=b.page, block_type="text",
        source_id=f"text#{b.page}#{bi}", text=text.strip(),
        section_path=b.section_path, lang=detect_lang(text),
    )


def chunk_fixed(blocks: Iterable[TextBlock], s: Settings) -> list[Chunk]:
    """baseline_03：定长切分，代表工单01/02/03的能力基线。"""
    out: list[Chunk] = []
    size = s.fixed_chunk_size
    overlap = int(size * s.chunk_overlap_ratio)
    for bi, b in enumerate(blocks):
        text = b.text.strip()
        if not text:
            continue
        step = max(size - overlap, 1)
        for ci, start in enumerate(range(0, len(text), step)):
            seg = text[start:start + size]
            if not seg.strip():
                continue
            out.append(Chunk(
                chunk_id=_cid(b.doc_id, b.page, "text", f"p{b.page}b{bi}", ci),
                doc_id=b.doc_id, page=b.page, block_type="text",
                source_id=f"text#{b.page}#{bi}", text=seg,
                section_path=b.section_path, lang=detect_lang(seg),
            ))
    return out


def chunk_text_blocks(blocks: Iterable[TextBlock], s: Settings) -> list[Chunk]:
    """full_04：按句子边界累积到目标长度，相邻块重叠 15%。"""
    out: list[Chunk] = []
    lo, hi = s.semantic_min_chars, s.semantic_max_chars
    overlap = int(hi * s.chunk_overlap_ratio)

    for bi, b in enumerate(blocks):
        text = b.text.strip()
        if not text:
            continue
        sents = _split_sentences(text)
        buf, ci = "", 0
        for sent in sents:
            if len(sent) > hi:
                # 病态长句：缓冲区先结清，再按上限硬切（片间保留重叠），保证不超 hi
                if buf.strip():
                    out.append(_text_chunk(b, bi, ci, buf))
                    ci += 1
                    buf = ""
                for piece in _hard_split_sentence(sent, hi, overlap):
                    if not piece.strip():
                        continue
                    out.append(_text_chunk(b, bi, ci, piece))
                    ci += 1
                continue
            if len(buf) + len(sent) > hi and len(buf) >= lo:
                out.append(_text_chunk(b, bi, ci, buf))
                ci += 1
                buf = buf[-overlap:] if overlap else ""
            buf += sent
        if buf.strip():
            out.append(_text_chunk(b, bi, ci, buf))
    return out


def chunk_tables(tables: Iterable[TableBlock], s: Settings) -> list[Chunk]:
    """整表为一个块——切分表格会破坏行列语义。"""
    out: list[Chunk] = []
    for ti, t in enumerate(tables):
        if not t.markdown.strip():
            continue
        head = t.markdown.splitlines()[0] if t.markdown else ""
        text = f"[表格] 第{t.page}页 共{t.n_rows}行×{t.n_cols}列\n表头：{head}\n{t.markdown}"
        out.append(Chunk(
            chunk_id=_cid(t.doc_id, t.page, "table", f"tbl{t.page}_{ti}", ti),
            doc_id=t.doc_id, page=t.page, block_type="table",
            source_id=f"table#{t.page}#{ti}", text=text,
            section_path=t.section_path, lang=detect_lang(text),
        ))
    return out


def _figure_body(f: FigureBlock) -> str:
    """图块正文：页码 + [图]题注 + [章节] + VLM 描述（图像块与文本块共用）。"""
    parts = []
    if f.caption:
        parts.append(f"[图] {f.caption}")
    if f.section_path:
        parts.append(f"[章节] {f.section_path}")
    if f.description:
        parts.append(f.description)
    if not parts:
        parts.append(f"[图] 第{f.page}页 图片（未解析出内容）")
    return f"第{f.page}页 " + "\n".join(parts)


def chunk_figures(figs: Iterable[FigureBlock], s: Settings) -> list[Chunk]:
    """整图为一个块：正文 = 题注 + VLM描述，CLIP向量与图路径存入 extra。"""
    out: list[Chunk] = []
    for i, f in enumerate(figs):
        text = _figure_body(f)
        out.append(Chunk(
            chunk_id=_cid(f.doc_id, f.page, "image", f"fig{f.page}_{i}", 0),
            doc_id=f.doc_id, page=f.page, block_type="image",
            source_id=f"image#{f.page}#fig{i}", text=text,
            section_path=f.section_path, lang=detect_lang(text),
            extra={
                "image_path": f.image_path,
                "bbox": list(f.bbox),
                "clip_vector": list(f.clip_vector or []),
                "figure_id": f.figure_id,
                "detect_method": f.detect_method,
                "confidence": f.confidence,
                "parse_warning": f.parse_warning,
            },
        ))
    return out


def chunk_figure_texts(figs: Iterable[FigureBlock], s: Settings) -> list[Chunk]:
    """RC2：把图块的 VLM 中文描述同时写入 text_chunks（bge-m3/BM25 可检索）。

    根因（RC2，实测）：中文问句下 CLIP 文图对齐失效（金标图 p39 全量排
    136/162、p72 第 100），而 VLM 描述此前**只**存在于 image_chunks（CLIP
    512 维单通道），中文提问时图上的答案文本根本进不了文本通道。故为每张图
    额外产出一个同正文的文本块（1024 维 bge-m3），使「图里写了什么」能被
    dense 与 BM25 直接召回到。

    ``extra.dedup_key`` = 对应 image 块的 chunk_id：融合阶段据此把同一张图的
    图像命中与文本命中合占**一个**名额（Task 8 遗留：figure_id 同页多图不
    唯一，chunk_id 才唯一，故用 chunk_id 作去重键）。
    """
    out: list[Chunk] = []
    for i, f in enumerate(figs):
        img_id = _cid(f.doc_id, f.page, "image", f"fig{f.page}_{i}", 0)
        text = _figure_body(f)
        out.append(Chunk(
            chunk_id=_cid(f.doc_id, f.page, "text", f"figtext{f.page}_{i}", 0),
            doc_id=f.doc_id, page=f.page, block_type="text",
            source_id=f"figuretext#{f.page}#fig{i}", text=text,
            section_path=f.section_path, lang=detect_lang(text),
            extra={
                "figure_chunk_id": img_id,
                "dedup_key": img_id,
                "image_path": f.image_path,
            },
        ))
    return out


def build_chunks(text_blocks, tables, figures, s: Settings) -> list[Chunk]:
    """按 pipeline_mode 组装三模态分块。

    ``figures`` 会被遍历两次（RC2 的图描述文本块 + 图像块），若调用方传生成器
    第二次遍历将静默为空——故先物化，保证图像块不会凭空消失。
    """
    figures = list(figures)
    if s.pipeline_mode == "full_04":
        # RC5：先做行块→段落合并，再按语义切分（baseline_03 是对照组，不合并）
        chunks = chunk_text_blocks(merge_line_blocks(text_blocks), s)
    else:
        chunks = chunk_fixed(text_blocks, s)
    if s.use_tables:
        chunks += chunk_tables(tables, s)
    if s.use_figures:
        chunks += chunk_figure_texts(figures, s)   # RC2：描述入文本通道
        chunks += chunk_figures(figures, s)
    return chunks
