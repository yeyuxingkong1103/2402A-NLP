# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-15-优化技术图纸与文本的跨模态检索流程
# 模块职责：查询理解——识别问题中的视觉引用（图3 / 第11页图3 / 图示），
#           定位对应图纸块，并用图纸的图像描述文本构造"图像增强查询"，
#           解决"仅凭文本相似度检索会漏掉图纸页"的故障 1。

import re                                                        # 引用抽取
import config                                                    # 正则与关键词
from logger import get_logger                                    # 日志

logger = get_logger("rag15.query")

_PAGE_RE = re.compile(config.PAGE_REF_RE)                        # "第N页"
_FIG_RE = re.compile(config.FIG_REF_RE)                          # "图N"


def parse_visual_refs(query):
    """解析问题中的视觉引用。

    返回 {has_visual, labels, pages, keywords}
        has_visual —— 是否出现视觉引用（决定要不要走图像增强检索）
        labels     —— 提到的图号，如 ["图3"]
        pages      —— 提到的页码，如 [11]
        keywords   —— 命中的视觉关键词
    """
    labels = [f"图{m}" for m in _FIG_RE.findall(query)]           # 图号
    pages = [int(p) for p in _PAGE_RE.findall(query)]             # 页码
    keywords = [k for k in config.VISUAL_REF_KEYWORDS if k in query]  # 关键词
    has_visual = bool(labels or pages or keywords)                # 任一命中即视为有视觉引用
    refs = {"has_visual": has_visual, "labels": labels,          # 汇总
            "pages": pages, "keywords": keywords}
    if has_visual:
        logger.info("检测到视觉引用：图号=%s 页码=%s 关键词=%s",
                    labels or "无", pages or "无", keywords or "无")
    return refs


def locate_figures(refs, chunks):
    """按图号/页码定位图纸块，返回匹配到的块下标列表（故障 1 的"定位该图表"）。"""
    hits = []                                                    # 命中的块下标
    for i, c in enumerate(chunks):                               # 遍历块
        if c["type"] != "figure":                                # 只看图纸块
            continue
        by_label = any(lb in (c.get("labels") or []) for lb in refs["labels"])  # 图号匹配
        by_page = c["page_no"] in refs["pages"]                  # 页码匹配
        if by_label or by_page:                                  # 任一命中
            hits.append(i)
    if hits:
        logger.info("按视觉引用定位到 %d 个图纸块：%s", len(hits),
                    [chunks[i]["chunk_id"] for i in hits])
    return hits


def build_figure_query(query, refs, chunks, figure_hits, max_chars=600):
    """构造"图像增强查询"：问题文本 + 对应图纸的图像描述摘要。

    这就是工单要求的"将该图表的图像描述文本作为强化的检索条件，与原始问题文本
    一同进行检索"——图纸描述由解析模块（VLM）生成，存放在图形块中。
    """
    if not refs["has_visual"]:                                   # 无视觉引用则不需要增强
        return query, ""
    parts = []                                                   # 图纸描述片段
    for i in figure_hits:                                        # 优先用精确命中的图纸块
        parts.append(chunks[i]["text"])                          # 图形块正文即图像描述
    if not parts:                                                # 未精确命中则退化为全部图纸块
        parts = [c["text"] for c in chunks if c["type"] == "figure"]
    desc = " ".join(parts)[:max_chars]                           # 截断，避免淹没原问题
    if not desc:                                                 # 没有任何图纸描述
        return query, ""
    enhanced = f"{query}\n【图纸内容参考】{desc}"                  # 拼接增强查询
    logger.info("已构造图像增强查询（追加图纸描述 %d 字）", len(desc))
    return enhanced, desc
