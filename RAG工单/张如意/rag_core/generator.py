# -*- coding: utf-8 -*-
"""
答案生成模块
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

职责：
  1. RAG 生成 —— 严格基于检索到的上下文作答，并给出引用来源
  2. 纯 LLM 生成 —— 工单01 要求「对比基于pdf 的返回结果和只使用 LLM 返回的答案」
  3. 拒答机制 —— 上下文不足以回答时明确说「文档中未找到」，抑制幻觉
  4. 多语言 —— 中文提问中文答，英文提问英文答（工单验收项）

Prompt 工程要点：
  · 强制「先定位再作答」：要求模型先指出信息在哪个片段，再给答案
  · 表格数据要求逐行核对，避免把「2019年」的数字安到「2018年」头上
  · 数值类问题要求保留原始单位与口径
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import llm

# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------
RAG_SYSTEM_ZH = """你是一个严谨的金融文档问答助手，服务于招股说明书、年度报告等专业文档的问答。

【作答规则】
1. 只依据【参考资料】作答，绝不使用参考资料之外的知识，也不要做常识性推测。
2. 先定位后作答：先在内部确认所需信息出现在哪个片段，再组织答案。
3. 数值必须精确：涉及金额、比例、年份、股数时，逐字核对参考资料中的数字、单位和口径，
   不要把不同年份/不同口径的数字混在一起。若资料给出多期数据，按期间分行列出。
4. 表格类资料要按行读取，注意表头与数据行的对应关系。
5. 若参考资料不足以回答问题，直接回复：「根据提供的文档内容，未能找到该问题的答案。」
   不要编造，也不要给出「可能」「大概」的猜测。
6. 回答要简洁，先给结论，再给支撑细节；不要复述问题，不要输出无关内容。
7. 回答末尾用【来源】标注引用的片段编号与页码，格式：来源：片段2（第58页）。

【输出格式】
<答案正文>
【来源】片段X（第Y页）"""

RAG_SYSTEM_EN = """You are a rigorous financial document QA assistant specialized in
prospectuses and annual reports.

Rules:
1. Answer ONLY from the provided reference material. Never use outside knowledge.
2. Locate first, then answer. Verify figures character by character.
3. For numeric questions, preserve exact values, units and periods. List multi-period
   data line by line rather than merging them.
4. If the references are insufficient, reply exactly:
   "Based on the provided documents, the answer to this question was not found."
5. Be concise. End with a 【Source】 line citing chunk numbers and page numbers."""

NO_CONTEXT_REPLY = "根据提供的文档内容，未能找到该问题的答案。"
NO_CONTEXT_REPLY_EN = "Based on the provided documents, the answer to this question was not found."


def _is_english(text: str) -> bool:
    """判断提问语言：中文字符占比低于 15% 视为英文提问。"""
    if not text:
        return False
    zh = sum(1 for c in text if "一" <= c <= "鿿")
    return zh / max(len(text), 1) < 0.15


# ---------------------------------------------------------------------------
# 生成结果
# ---------------------------------------------------------------------------
@dataclass
class Generation:
    question: str
    answer: str
    mode: str = "rag"                # rag | llm_only
    contexts: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    latency: float = 0.0
    refused: bool = False

    def to_dict(self) -> dict:
        return {
            "question": self.question, "mode": self.mode, "answer": self.answer,
            "refused": self.refused, "latency": round(self.latency, 3),
            "n_contexts": len(self.contexts),
            "citations": self.citations,
        }


# ---------------------------------------------------------------------------
# RAG 生成
# ---------------------------------------------------------------------------
def generate_rag(
    question: str,
    contexts: list[dict],
    context_text: str,
    history: list[tuple[str, str]] | None = None,
    model: str | None = None,
    temperature: float = 0.0,
) -> Generation:
    """
    基于检索上下文生成答案。

    Args:
        contexts: 检索到的片段列表（用于回填引用）
        context_text: 已拼装好的上下文文本
        history: [(问题, 答案)] 多轮历史，用于承接语气
    """
    import time

    en = _is_english(question)
    system = RAG_SYSTEM_EN if en else RAG_SYSTEM_ZH

    if not context_text.strip():
        return Generation(
            question=question,
            answer=NO_CONTEXT_REPLY_EN if en else NO_CONTEXT_REPLY,
            mode="rag", contexts=contexts, refused=True,
        )

    messages = [{"role": "system", "content": system}]
    for q, a in (history or [])[-3:]:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a[:600]})

    user = (
        f"【参考资料】\n{context_text}\n\n"
        f"【问题】\n{question}\n\n"
        f"请严格依据参考资料作答。"
    )
    messages.append({"role": "user", "content": user})

    t0 = time.perf_counter()
    answer = llm.chat(messages, model=model, temperature=temperature,
                      max_tokens=1500, tag="generate")
    latency = time.perf_counter() - t0

    refused = ("未能找到该问题" in answer) or ("was not found" in answer)
    return Generation(
        question=question, answer=answer.strip(), mode="rag",
        contexts=contexts, citations=_extract_citations(answer, contexts),
        latency=latency, refused=refused,
    )


def generate_llm_only(question: str, model: str | None = None) -> Generation:
    """
    纯 LLM 基线：不提供任何文档上下文，直接让模型回答。
    工单01 验收要求与 RAG 结果做对比，用于证明 RAG 的必要性。
    """
    import time

    en = _is_english(question)
    system = (
        "You are a helpful assistant. Answer the question concisely."
        if en else
        "你是一个乐于助人的助手，请简洁准确地回答用户的问题。"
    )
    t0 = time.perf_counter()
    answer = llm.chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": question}],
        model=model, temperature=0.0, max_tokens=800, tag="llm_only",
    )
    return Generation(question=question, answer=answer.strip(), mode="llm_only",
                      latency=time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# 引用解析
# ---------------------------------------------------------------------------
def _extract_citations(answer: str, contexts: list[dict]) -> list[dict]:
    """从答案的【来源】标注中解析出结构化引用信息。"""
    import re

    cits = []
    for m in re.finditer(r"片段\s*(\d+)", answer):
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(contexts):
            d = contexts[idx]
            cits.append({
                "chunk_id": d.get("chunk_id", ""), "doc": d.get("doc", ""),
                "page": d.get("page", ""), "type": d.get("type", "text"),
                "section": d.get("section", ""),
                "snippet": (d.get("text", "")[:100] + "…") if d.get("text") else "",
            })
    # 去重
    seen, out = set(), []
    for c in cits:
        k = (c["chunk_id"], c["page"])
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# 高并发/多子问题场景：批量生成
# ---------------------------------------------------------------------------
def generate_multi_hop(question: str, sub_questions: list[str],
                       retrieve_fn, reranker: str = "tfidf",
                       top_k: int = 5) -> Generation:
    """
    多跳生成（工单05 问题分解后用）：每个子问题独立检索，合并上下文后统一作答。
    相比单次检索，能回答「A 的 X 是多少，B 的 Y 又是多少」这类跨实体问题。
    """
    all_docs: list[dict] = []
    for sq in sub_questions:
        res = retrieve_fn(sq, strategy="hybrid", reranker=reranker, top_k=top_k)
        docs = res.docs if hasattr(res, "docs") else res
        all_docs.extend(docs)

    # 按 chunk_id 去重，保留首次出现顺序
    seen, merged = set(), []
    for d in all_docs:
        if d.get("chunk_id") not in seen:
            seen.add(d.get("chunk_id"))
            merged.append(d)

    ctx = "\n\n---\n\n".join(
        f"[片段{i}] 《{d.get('doc')}》第{d.get('page')}页\n{d.get('text', '')}"
        for i, d in enumerate(merged[: top_k * len(sub_questions)], 1)
    )
    return generate_rag(question, merged, ctx)
