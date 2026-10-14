# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""
模块：LLM 抽取的候选块优先级排序（控制第2层成本）

全量 7757 块送 LLM 需 3~6 小时；本模块把候选收敛到**最可能含关系**的
2000~3000 块（约 1~2 小时），依据三条信号加权：

  1. 章节白名单命中（公司治理/股东/董监高/组织结构/子公司…）—— 权重最高；
  2. 图谱触发词（控股、持股、任职、下设、发行…）；
  3. 第1层实体密度（公司/指标命中数）。

目录、释义、纯页码等噪声块直接判负分剔除。
"""
from __future__ import annotations

from src.kg_dict import (GRAPH_TRIGGERS, SECTION_WHITELIST, find_companies,
                         find_metrics)
from src.kg_extract import extract_block

# 明确排除的章节/块类型（噪声，不含关系）
NOISE_HEADINGS = ("目录", "释义", "备查文件", "声明", "封面", "封底")
NOISE_BLOCK_TYPES = ("figure", "image")

W_SECTION = 10        # 章节白名单命中（每个）
W_TRIGGER = 3         # 触发词（每个，上限 5 个）
W_ENTITY = 2          # 第1层实体（每个，上限 6 个）


def score_chunk(chunk: dict, layer1: dict | None = None) -> tuple[int, list[str]]:
    """返回 (优先级分, 命中原因列表)。分数 < 0 表示应剔除。"""
    text = chunk.get("text") or chunk.get("parent_text") or ""
    if not text.strip():
        return -100, ["空块"]

    heading_path = chunk.get("heading_path") or []
    heading = " / ".join(str(h) for h in heading_path) or (chunk.get("heading") or "")

    for bad in NOISE_HEADINGS:
        if bad in heading:
            return -50, [f"噪声章节：{bad}"]
    if chunk.get("block_type") in NOISE_BLOCK_TYPES:
        return -50, [f"噪声块类型：{chunk.get('block_type')}"]

    score = 0
    reasons: list[str] = []

    hit_sections = [s for s in SECTION_WHITELIST if s in heading]
    if hit_sections:
        score += W_SECTION * len(hit_sections)
        reasons.append(f"章节命中：{'/'.join(hit_sections)}")

    hit_triggers = [t for t in GRAPH_TRIGGERS if t in text][:5]
    if hit_triggers:
        score += W_TRIGGER * len(hit_triggers)
        reasons.append(f"触发词：{'/'.join(hit_triggers)}")

    if layer1 is None:
        layer1 = extract_block(chunk)
    n_ent = len(layer1.get("entities", []))
    if n_ent:
        score += W_ENTITY * min(n_ent, 6)
        reasons.append(f"第1层实体 {n_ent} 个")

    if not reasons:
        reasons.append("无明显信号")

    return score, reasons


def select_chunks(chunks: list[dict], limit: int | None = None,
                  max_blocks: int | None = None) -> list[dict]:
    """按优先级降序返回候选块。

    limit：只考察前 N 块（调试用）；
    max_blocks：最多返回多少块（默认 config.KG_LLM_MAX_BLOCKS）。
    """
    from src import config

    max_blocks = max_blocks if max_blocks is not None else config.KG_LLM_MAX_BLOCKS
    subset = chunks[:limit] if limit is not None else chunks

    scored: list[dict] = []
    for chunk in subset:
        s, reasons = score_chunk(chunk)
        if s < 0:
            continue
        scored.append({"chunk": chunk, "score": s, "reasons": reasons})

    scored.sort(key=lambda x: (-x["score"], x["chunk"].get("chunk_id", "")))
    return scored[:max_blocks]
