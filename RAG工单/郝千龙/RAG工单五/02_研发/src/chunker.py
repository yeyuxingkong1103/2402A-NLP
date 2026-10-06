# -*- coding: utf-8 -*-
# 【分块模块 · chunker.py】标题感知切分 + 滑动重叠 + 公司实体上下文注入
# 工单编号：人工智能NLP-RAG-Query理解优化任务

"""分块层：把解析层 Block 切分为携带标题路径与公司实体的检索块。

- 正文：按句号/分号切句后贪心打包，块间保留 overlap 字重叠；
- 表格：整张 Markdown 表独立成块；
- 组织结构图：独立成块，便于“销售处归属”类问题精确召回；
- 每个块拼接“标题路径 + 主体公司名”前缀，解决多文档主体区分问题。
"""
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List

from config import CONFIG
from pdf_parser import Block

_SENT_SPLIT = re.compile(r"(?<=[。；;！？!?])|(?<=\.)\s+")
_COMPANY_RE = re.compile(r"[\u4e00-\u9fa5]{2,20}(?:股份有限公司|有限责任公司|有限公司)")


def _main_companies(blocks: List[Block], top_k: int = 4) -> List[str]:
    """基于词频选出文档主体公司（发行人），并剔除粘连实体。"""
    counter: Counter = Counter()
    for b in blocks:
        counter.update(_COMPANY_RE.findall(b.text))
    ranked = counter.most_common()
    chosen: List[str] = []
    for name, cnt in ranked:
        if cnt < 3:
            continue
        if any(name != other and name in other and counter[other] >= cnt
               for other, _ in ranked):
            continue
        chosen.append(name)
        if len(chosen) >= top_k:
            break
    return chosen


@dataclass
class Chunk:
    """检索块：子块文本用于向量化，父文本用于返回。"""

    chunk_id: int
    text: str
    parent_text: str
    page_no: int
    heading_path: str
    chunk_type: str = "text"
    companies: List[str] = field(default_factory=list)


def _split_sentences(text: str) -> List[str]:
    """按中/英文句读切句，过短句子并入相邻句。"""
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    merged: List[str] = []
    for part in parts:
        if merged and len(part) < 8:
            merged[-1] += part
        else:
            merged.append(part)
    return merged


def _pack(sentences: List[str], size: int, overlap: int) -> List[str]:
    """贪心打包句子为目标长度块，块间按字符滑窗重叠。"""
    packs: List[str] = []
    buf = ""
    for sent in sentences:
        if buf and len(buf) + len(sent) > size:
            packs.append(buf)
            tail = buf[-overlap:] if overlap > 0 else ""
            buf = tail + sent
        else:
            buf += sent
    if buf.strip():
        packs.append(buf.strip())
    return packs


def build_chunks(blocks: List[Block]) -> List[Chunk]:
    """把结构化 Block 列表构建为检索 Chunk 列表。"""
    company_set = _main_companies(blocks)
    chunks: List[Chunk] = []
    cid = 0

    # 计算该批 blocks 的主体公司（频次最高），用于缺省块的主体标注。
    # 多文档场景下应分批调用 build_chunks，确保每批主体公司正确。
    doc_main_company = company_set[0] if company_set else ""

    parents: dict = {}
    for b in blocks:
        if b.type == "text":
            parents.setdefault((b.heading_path, b.page_no), []).append(b.text)

    for b in blocks:
        # 优先用块文本中出现的公司；缺省时用该文档的主体公司
        present = [c for c in company_set if c in b.text]
        block_company = present[0] if present else doc_main_company
        present = present or [doc_main_company]
        ent = f"主体：{block_company}\n" if block_company else ""
        if b.type in ("table", "orgchart"):
            cid += 1
            prefix = f"（{b.heading_path or '组织结构图'}）" if b.heading_path else ""
            text = f"{prefix}\n{ent}{b.text}"
            chunks.append(Chunk(
                chunk_id=cid, text=text, parent_text=text, page_no=b.page_no,
                heading_path=b.heading_path, chunk_type=b.type,
                companies=present,
            ))
            continue

        sentences = _split_sentences(b.text)
        for pack in _pack(sentences, CONFIG.chunk_size, CONFIG.chunk_overlap):
            cid += 1
            parent = "".join(parents.get((b.heading_path, b.page_no), [b.text]))
            head = f"【{b.heading_path}】" if b.heading_path else ""
            chunks.append(Chunk(
                chunk_id=cid,
                text=f"{head}{ent}{pack}",
                parent_text=parent if len(parent) <= 1200 else parent[:1200],
                page_no=b.page_no, heading_path=b.heading_path,
                companies=present,
            ))
    return chunks
