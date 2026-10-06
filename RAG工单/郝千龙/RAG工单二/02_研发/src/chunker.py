# -*- coding: utf-8 -*-
# 【分块优化模块 · chunker.py】标题感知切分 + 滑动重叠 + 父块上下文增强 + 表格独立成块
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

"""分块层：把解析层 Block 切分为携带标题路径与公司实体的检索块。

对应设计文档 4.2：
- 正文：按句号/分号切句后贪心打包，块间保留 overlap 字重叠；
- 表格：整张 Markdown 表独立成块，不切断数字关系；
- Small-to-Big：检索用子块、返回父段落，并为每块拼接“标题路径 + 公司名”前缀。
"""
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List

from config import CONFIG
from pdf_parser import Block

# 句子切分：保留中文句读与英文句点
_SENT_SPLIT = re.compile(r"(?<=[。；;！？!?])|(?<=\.)\s+")
# 从文本中粗略抽取公司全称，用于给每个块补实体上下文
_COMPANY_RE = re.compile(r"[\u4e00-\u9fa5]{2,20}(?:股份有限公司|有限责任公司|有限公司)")


def _main_companies(blocks: List[Block], top_k: int = 3) -> List[str]:
    """基于词频选出文档主体公司（通常是发行人），并剔除粘连实体。

    粘连实体示例：“汉口银行科技金融服务中心武汉兴图新科电子股份有限公司”，
    其中包含更高频的短实体“武汉兴图新科电子股份有限公司”，应保留后者。

    :param blocks: 全部文档块
    :param top_k: 返回公司数量
    :return: 公司全称列表（按频次降序）
    """
    counter: Counter = Counter()
    for b in blocks:
        counter.update(_COMPANY_RE.findall(b.text))
    ranked = counter.most_common()
    chosen: List[str] = []
    for name, cnt in ranked:
        if cnt < 3:
            continue
        # 若该实体被另一个更高频实体包含，则视为粘连噪声
        if any(name != other and name in other and counter[other] >= cnt
               for other, _ in ranked):
            continue
        chosen.append(name)
        if len(chosen) >= top_k:
            break
    return chosen


@dataclass
class Chunk:
    """检索块：子块文本用于向量化，父文本用于返回给大模型。"""

    chunk_id: int
    text: str                 # 子块（含上下文前缀，直接用于检索）
    parent_text: str          # 父段落（完整归属段，Small-to-Big 返回）
    page_no: int
    heading_path: str
    chunk_type: str = "text"  # text / table
    companies: List[str] = field(default_factory=list)


def _split_sentences(text: str) -> List[str]:
    """按中文/英文句读切句，过短句子并入相邻句。

    :param text: 段落原文
    :return: 句子列表
    """
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    merged: List[str] = []
    for part in parts:
        if merged and len(part) < 8:
            merged[-1] += part
        else:
            merged.append(part)
    return merged


def _pack(sentences: List[str], size: int, overlap: int) -> List[str]:
    """贪心打包句子为目标长度块，块间按字符滑窗重叠。

    :param sentences: 句子列表
    :param size: 目标块字符数
    :param overlap: 重叠字符数
    :return: 块文本列表
    """
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
    """把结构化 Block 列表构建为检索 Chunk 列表。

    :param blocks: pdf_parser 输出的 Block 列表
    :return: 带编号、页码、标题路径与公司实体的 Chunk 列表
    """
    # 全局主体公司表（频次法选发行人）：用于给缺少主语的块补实体上下文
    company_set = _main_companies(blocks)
    main_company = company_set[0] if company_set else ""
    chunks: List[Chunk] = []
    cid = 0

    # 按标题路径聚合正文段，形成“父段落”
    parents: dict = {}
    for b in blocks:
        if b.type == "text":
            parents.setdefault((b.heading_path, b.page_no), []).append(b.text)

    for b in blocks:
        present = [c for c in company_set if c in b.text] or company_set[:1]
        if b.type == "table":
            cid += 1
            prefix = f"（{b.heading_path or '表格'}）" if b.heading_path else ""
            text = f"{prefix}\n{b.text}"
            chunks.append(Chunk(
                chunk_id=cid, text=text, parent_text=text, page_no=b.page_no,
                heading_path=b.heading_path, chunk_type="table",
                companies=present,
            ))
            continue

        sentences = _split_sentences(b.text)
        for pack in _pack(sentences, CONFIG.chunk_size, CONFIG.chunk_overlap):
            cid += 1
            parent = "".join(parents.get((b.heading_path, b.page_no), [b.text]))
            head = f"【{b.heading_path}】" if b.heading_path else ""
            ent = f"主体：{main_company}\n" if main_company else ""
            chunks.append(Chunk(
                chunk_id=cid,
                text=f"{head}{ent}{pack}",
                parent_text=parent if len(parent) <= 1200 else parent[:1200],
                page_no=b.page_no, heading_path=b.heading_path,
                companies=present,
            ))
    return chunks
