# -*- coding: utf-8 -*-
# 【多字段分块模块 · chunker.py】标题感知切分、滑动重叠、表格独立成块，并为每块标注标题/正文/表格/公司实体/摘要字段
# 工单编号：人工智能NLP-RAG-混合检索任务

"""分块层：把解析层 Block 切分为携带多字段标注的检索块。

工单六在工单二“标题感知分块”基础上新增多字段标注，每个 Chunk 持有：
- ``title``   标题字段：所属章节标题路径（如“第五节 发行人基本情况 > 一、…”）；
- ``body``    正文字段：贪心打包后的正文文本；
- ``table``   表格字段：表格块的 Markdown 文本（正文块为空）；
- ``company`` 公司实体字段：块内出现的公司全称/文档主体公司；
- ``summary`` 摘要字段：标题 + 首句（表格为标题），供摘要路加权检索。
分块策略沿用“正文按句切分贪心打包+块间重叠、整张表独立成块、Small-to-Big”。
"""
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

from config import CONFIG
from pdf_parser import Block

# 句子切分：保留中文句读与英文句点
_SENT_SPLIT = re.compile(r"(?<=[。；;！？!?])|(?<=\.)\s+")
# 从文本中粗略抽取公司全称
_COMPANY_RE = re.compile(r"[一-龥]{2,20}(?:股份有限公司|有限责任公司|有限公司)")


def _main_companies_per_doc(blocks: List[Block], top_k: int = 3) -> Dict[str, List[str]]:
    """按来源文档分别统计主体公司（发行人），避免两份招股书实体互相干扰。

    :param blocks: 全部文档块
    :param top_k: 每份文档保留的公司数量
    :return: {doc_name: [公司全称, ...]} 按频次降序
    """
    counters: Dict[str, Counter] = defaultdict(Counter)
    for b in blocks:
        counters[b.doc_name].update(_COMPANY_RE.findall(b.text))
    result: Dict[str, List[str]] = {}
    for doc_name, counter in counters.items():
        ranked = counter.most_common()
        chosen: List[str] = []
        for name, cnt in ranked:
            if cnt < 3:
                continue
            # 粘连实体（如“汉口银行…武汉兴图新科…”）被更高频实体包含时剔除
            if any(name != other and name in other and counter[other] >= cnt
                   for other, _ in ranked):
                continue
            chosen.append(name)
            if len(chosen) >= top_k:
                break
        result[doc_name] = chosen
    return result


@dataclass
class Chunk:
    """检索块：携带多字段文本，支持多字段倒排与稠密检索。"""

    chunk_id: int
    text: str                 # 稠密向量用的检索文本（标题+实体+正文/表格）
    parent_text: str          # 父段落（Small-to-Big 返回与答案抽取用）
    page_no: int
    heading_path: str
    doc_name: str = ""
    chunk_type: str = "text"  # text / table
    companies: List[str] = field(default_factory=list)
    # 多字段标注：title/body/table/company/summary
    fields: Dict[str, str] = field(default_factory=dict)


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


def _make_summary(heading: str, first_sentence: str) -> str:
    """生成摘要字段：标题路径 + 首句，截断到配置长度。

    :param heading: 标题路径
    :param first_sentence: 块内首句
    :return: 摘要文本
    """
    summary = f"{heading}。{first_sentence}" if heading else first_sentence
    return summary[: CONFIG.summary_max_chars]


def build_chunks(blocks: List[Block]) -> List[Chunk]:
    """把结构化 Block 列表构建为多字段标注的 Chunk 列表。

    :param blocks: pdf_parser 输出的 Block 列表（可来自多份 PDF）
    :return: 带编号、页码、标题路径、来源文档与五字段标注的 Chunk 列表
    """
    # 各文档主体公司表：用于给缺少主语的块补实体上下文
    company_map = _main_companies_per_doc(blocks)

    # 按（文档、标题路径、页码）聚合正文段形成“父段落”
    parents: Dict[tuple, List[str]] = defaultdict(list)
    for b in blocks:
        if b.type == "text":
            parents[(b.doc_name, b.heading_path, b.page_no)].append(b.text)

    chunks: List[Chunk] = []
    cid = 0
    for b in blocks:
        doc_companies = company_map.get(b.doc_name, [])
        present = [c for c in doc_companies if c in b.text] or doc_companies[:1]
        company_field = " ".join(present)

        if b.type == "table":
            cid += 1
            prefix = f"（{b.heading_path or '表格'}）" if b.heading_path else ""
            text = f"{prefix}\n{b.text}"
            title_field = b.heading_path or b.table_title or "表格"
            chunks.append(Chunk(
                chunk_id=cid, text=text, parent_text=text, page_no=b.page_no,
                heading_path=b.heading_path, doc_name=b.doc_name,
                chunk_type="table", companies=present,
                fields={"title": title_field, "body": "", "table": b.text,
                        "company": company_field,
                        "summary": (b.table_title or title_field)},
            ))
            continue

        sentences = _split_sentences(b.text)
        for order, pack in enumerate(_pack(sentences, CONFIG.chunk_size,
                                           CONFIG.chunk_overlap)):
            cid += 1
            parent = "".join(parents.get(
                (b.doc_name, b.heading_path, b.page_no), [b.text]))
            head = f"【{b.heading_path}】" if b.heading_path else ""
            ent = f"主体：{present[0]}\n" if present else ""
            chunks.append(Chunk(
                chunk_id=cid,
                text=f"{head}{ent}{pack}",
                parent_text=parent if len(parent) <= 1200 else parent[:1200],
                page_no=b.page_no, heading_path=b.heading_path,
                doc_name=b.doc_name, companies=present,
                fields={"title": b.heading_path,
                        "body": pack,
                        "table": "",
                        "company": company_field,
                        # 同一 Block 拆出的重叠块共享首句摘要
                        "summary": _make_summary(b.heading_path, sentences[0]
                                                 if sentences else pack[:40])},
            ))
    return chunks
