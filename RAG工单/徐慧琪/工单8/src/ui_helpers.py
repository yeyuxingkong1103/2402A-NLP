# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""模块：界面渲染辅助（把 QaResult 字典渲染为 Markdown）。"""
from __future__ import annotations


def render_answer_md(result: dict, lang: str = "zh") -> str:
    """答案区（含降级提示与耗时）。"""
    parts: list[str] = []
    if result.get("degraded"):
        label = "降级" if lang == "zh" else "Degraded"
        reason = result.get("degraded_reason", "")
        parts.append(f"> ⚠️ **{label}**：{reason}\n")
    parts.append(result.get("answer") or "（无答案）")

    t = (result.get("timings") or {}).get("total")
    if t is not None:
        mark = "✅" if t <= 3.0 else "⚠️"
        unit = "秒" if lang == "zh" else "s"
        parts.append(f"\n\n---\n{mark} 响应时间：**{t:.2f} {unit}**"
                     f"（目标 ≤ 3 {unit}）")
    return "\n".join(parts)


def render_facts_md(facts: list[str]) -> str:
    if not facts:
        return "_未检索到图谱事实_"
    # 逐条列表渲染并保留行尾换行：单条事实时也是合法的 Markdown 列表。
    return "\n".join(f"{i}. {f}" for i, f in enumerate(facts, start=1)) + "\n"


def render_hits_md(hits: list[dict]) -> str:
    if not hits:
        return "_未检索到文本块_"
    out: list[str] = []
    for i, h in enumerate(hits, start=1):
        doc = h.get("source", "未知")
        page = h.get("page", h.get("page_idx", -1))
        score = h.get("score")
        score_txt = f"（相似度 {float(score):.3f}）" if score is not None else ""
        out.append(f"**[{i}] {doc} 第 {page} 页**{score_txt}\n\n"
                   f"> {(h.get('text') or '').strip()[:500]}\n")
    return "\n".join(out)


def answer_with_sources(result: dict, max_hits: int = 5) -> str:
    """答案 + 来源清单（供复制导出）。"""
    out = [result.get("answer", "")]
    hits = result.get("text_hits") or []
    if hits:
        out.append("\n\n## 来源\n")
        for i, h in enumerate(hits[:max_hits], start=1):
            out.append(f"[{i}] {h.get('source', '')} 第 "
                       f"{h.get('page', h.get('page_idx', -1))} 页")
    return "\n".join(out)


def lang_label(lang: str) -> str:
    return "中文" if lang == "zh" else "English"
