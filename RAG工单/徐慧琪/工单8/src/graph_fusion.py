# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""
模块：三路融合 —— 图谱通道 + 向量通道 + 全文通道（硬性要求3「图+向量融合」）

为什么需要融合而不是替代：
  · 图谱通道给**结构化、可验证**的事实（指标数值、股权链、组织树），
    但对长尾叙述性内容覆盖不到；
  · 向量通道给**语义相近**的文本块，对数值精确性不敏感；
  · 全文通道给**字面命中**的文本块，对同义表述不敏感。
三者互补。融合时按 config.KG_FUSION_WEIGHTS 加权，并**按「文档+页」去重**，
避免同一页被两路命中而挤占上下文预算。

上下文分两区组装，让 LLM 清楚区分「确定性事实」与「待引用证据」。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src import config
from src.fusion import weighted_fuse


@dataclass
class FusedContext:
    graph_facts: list[str] = field(default_factory=list)
    text_hits: list[dict] = field(default_factory=list)
    weights: dict = field(default_factory=dict)


def _page_key(hit: dict) -> tuple:
    return (hit.get("source") or hit.get("source_doc") or "",
            int(hit.get("page_idx", hit.get("page", -1))))


def fuse(graph_facts: list[str], vector_hits: list[dict],
         fulltext_hits: list[dict], top_k: int = 8,
         weights: dict | None = None) -> FusedContext:
    """三路融合。

    vector_hits / fulltext_hits 为 strategies.retrieve 风格的命中（含 chunk_id /
    text / source / page_idx / score）。
    """
    weights = dict(weights or config.KG_FUSION_WEIGHTS)

    # 文本两路先融合（复用 07 的加权融合，保持与 07 可比）
    if vector_hits and fulltext_hits:
        fused = weighted_fuse([("vector", vector_hits),
                               ("fulltext", fulltext_hits)])
    elif vector_hits:
        fused = list(vector_hits)
    elif fulltext_hits:
        fused = list(fulltext_hits)
    else:
        fused = []

    # 按「文档+页」去重，保留得分最高者
    best: dict[tuple, dict] = {}
    for h in fused:
        k = _page_key(h)
        if k not in best or float(h.get("score", 0)) > float(best[k].get("score", 0)):
            best[k] = h
    deduped = sorted(best.values(), key=lambda x: -float(x.get("score", 0)))

    # 图谱通道给出的事实，其来源页若尚未在文本命中中出现，
    # 不应凭空造文本块；图谱事实单独成区（见 build_context_text）。
    return FusedContext(graph_facts=list(graph_facts),
                        text_hits=deduped[:top_k], weights=weights)


def build_context_text(fused: FusedContext, max_facts: int = 40,
                       max_chars_per_block: int = 700) -> str:
    """组装 LLM 上下文：图谱事实区 + 文本证据区。"""
    parts: list[str] = []

    if fused.graph_facts:
        parts.append("【图谱事实】（来自知识图谱的结构化事实，均为语料原文抽取）")
        for i, f in enumerate(fused.graph_facts[:max_facts], start=1):
            parts.append(f"{i}. {f}")
        parts.append("")

    if fused.text_hits:
        parts.append("【文本证据】（原文片段，按相关性排序）")
        for i, h in enumerate(fused.text_hits, start=1):
            doc = h.get("source") or h.get("source_doc") or "未知文档"
            page = h.get("page_idx", h.get("page", -1))
            text = (h.get("text") or "")[:max_chars_per_block]
            parts.append(f"[{i}] 来源：{doc} 第 {page} 页")
            parts.append(text)
            parts.append("")

    return "\n".join(parts).strip()
