# -*- coding: utf-8 -*-
"""pipeline/chunking.py —— 分块。

在链路中的位置：
    backend/pipeline 构建管线的第三步：把按页文本切成带章节信息的 chunk。

切分策略（先按语义、再按长度）：
    1. 优先在章节标题处断开 —— 语义边界天然对齐，检索命中后能直接告诉用户
       "这条依据来自 3.2 设备要求"
    2. 同一章节内累积到 MAX_CHUNK 字符就切一刀，让 chunk 大小可控
    3. 仍有超长段就硬切，并保留 OVERLAP 字符重叠，避免一句话正好落在切点上被劈成两半

产出的每个 chunk 都带 page / section / chunk_id，这是答案可溯源的基础。
"""
from __future__ import annotations

from .config import HEADING_RE, MAX_CHUNK, OVERLAP

def chunk_pages(pages: list[dict]) -> list[dict]:
    """把按页文本切成带章节信息的 chunk。

    参数：
        pages: 清洗后的按页结构
    返回：
        [{"page": N, "section": "3.2 设备要求", "text": "...", "chunk_id": "chunk-007"}, ...]

    切分策略（先按语义、再按长度）：
        1. 优先在章节标题处断开 —— 语义边界天然对齐，检索命中后能直接告诉用户
           "这条依据来自 3.2 设备要求"，这是引用可读性的关键
        2. 同一章节内累积到 MAX_CHUNK(400) 字符就切一刀，让 chunk 大小可控
        3. 仍有超长段（整段没标题、又超过 400 字）就硬切，并保留 OVERLAP(40) 字符重叠，
           避免一句话正好落在切点上被劈成两半、两边都搜不到

    注意 chunk_id 是切分完成后统一编号的：
        硬切会产生新的子块，如果边切边编号，编号顺序会和最终顺序错位。
    """
    chunks = []
    for page in pages:
        buffer, section = [], "正文"  # 没遇到标题前，章节名兜底为"正文"
        for line in page["text"].splitlines():
            line = line.strip()
            if not line:
                continue
            heading = HEADING_RE.match(line)
            # len(line) < 60 是为了排除正文中恰好以数字开头的长句子，只认短标题
            if heading and len(line) < 60 and buffer:
                chunks.append({"page": page["page"], "section": section, "text": " ".join(buffer)})
                buffer, section = [], line
            buffer.append(line)
            if sum(map(len, buffer)) >= MAX_CHUNK:
                chunks.append({"page": page["page"], "section": section, "text": " ".join(buffer)})
                buffer = []
        if buffer:  # 收尾：本页最后未满 400 字的残留也要成块，否则尾部内容会丢
            chunks.append({"page": page["page"], "section": section, "text": " ".join(buffer)})

    final = []
    for chunk in chunks:
        text = chunk["text"]
        while len(text) > MAX_CHUNK:
            final.append(chunk | {"text": text[:MAX_CHUNK]})
            text = text[MAX_CHUNK - OVERLAP:]  # 回退 40 字符形成重叠，保住跨切点的句子
        if text.strip():
            final.append(chunk | {"text": text})
    for number, chunk in enumerate(final):
        chunk["chunk_id"] = f"chunk-{number:03d}"  # 编号补零到 3 位，保证字典序与数值序一致
    return final
