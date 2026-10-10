# -*- coding: utf-8 -*-
"""
工单 15 · 查询理解优化：视觉引用识别与图像增强查询
（人工智能NLP-RAG-优化技术图纸与文本的跨模态检索流程）

思路：
  当用户在问题中出现「图N / 第N页 / 图示 / 编号N / 部件N」等视觉引用时，
  正则抽取「图号 / 页码 / 部件号」，据此在解析结果里定位图表描述块，
  把图像描述文本作为「增强查询」与原问题拼成双查询，一起做向量召回。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

# —— 视觉引用正则（配置化：可外置到 cross_modal_config.yaml）——
RE_FIG = re.compile(r"图\s*(\d{1,3})")
RE_PAGE = re.compile(r"第\s*(\d{1,4})\s*页")
RE_ILLU = re.compile(r"(图示|附图|图纸|示意图|结构图)")
RE_PART = re.compile(r"(?:部件|零件|元件|编号)\s*(\d{1,3})")
RE_CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
             "七": 7, "八": 8, "九": 9, "十": 10}


@dataclass
class VisualRef:
    """一次查询中识别到的视觉引用。"""
    figure_no: List[int] = field(default_factory=list)
    page_no: List[int] = field(default_factory=list)
    part_no: List[int] = field(default_factory=list)
    has_illustration: bool = False

    @property
    def is_visual(self) -> bool:
        return bool(self.figure_no or self.page_no or self.part_no or self.has_illustration)

    def as_query_hint(self) -> str:
        toks = []
        if self.figure_no:
            toks += [f"图{n}" for n in self.figure_no]
        if self.page_no:
            toks += [f"第{n}页" for n in self.page_no]
        if self.has_illustration:
            toks.append("技术图纸 图示")
        if self.part_no:
            toks += [f"部件{n}" for n in self.part_no]
        return " ".join(toks)


def extract_visual_ref(question: str) -> VisualRef:
    ref = VisualRef()
    ref.figure_no = [int(x) for x in RE_FIG.findall(question)]
    ref.page_no = [int(x) for x in RE_PAGE.findall(question)]
    ref.part_no = [int(x) for x in RE_PART.findall(question)]
    ref.has_illustration = bool(RE_ILLU.search(question))
    return ref


def build_enhanced_queries(question: str,
                           figure_desc_lookup=None) -> List[str]:
    """
    返回检索用的查询列表： [原问题] 或 [原问题, 图像描述增强查询]。
    figure_desc_lookup: callable(ref) -> List[str]，返回命中的图像描述文本。
                        默认实现按 (页码, 图号) 在本地索引里查（此处留空，接入 RAGFlow 后替换）。
    """
    ref = extract_visual_ref(question)
    queries = [question]
    if not ref.is_visual:
        return queries

    descs: List[str] = []
    if callable(figure_desc_lookup):
        descs = list(figure_desc_lookup(ref) or [])
    if descs:
        # 图像描述作为强化检索条件，与原始问题一同检索（双路召回之一）
        queries.append(question + " " + " ".join(descs))
    else:
        # 无描述时退化为「视觉引用关键词」增强
        queries.append(question + " " + ref.as_query_hint())
    return queries


if __name__ == "__main__":
    for q in ["在文件中第11页图3中，编号13的部件相对于编号12的部件的位置关系是？",
              "根据专利文本，本发明主要涉及哪种物料的分配装置？"]:
        ref = extract_visual_ref(q)
        print(q)
        print("  ->", ref, "| queries:", build_enhanced_queries(q))
