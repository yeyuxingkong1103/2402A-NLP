# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务

模块：工单09 端到端问答链路（三层优化后的完整实现）

链路：
  实体识别去噪 → 问题类型判定 → 查询分解 → 子图检索(k-hop 自适应)
  → 多路召回(逐侧面配额) → 跨侧面 round-robin → 统一精排
  → 四分区上下文 → 生成 Prompt v2 → LLM → 答案 + 引用 + 检索证据

容错（沿用并扩展 08）：
  · 图谱不可用 → 退化为纯 RAG，如实标记 degraded；
  · 子图检索失败 → 跳过图谱通道；
  · 检索失败 → 空命中，仍尝试生成（并由 Prompt 触发兜底话术）；
  · LLM 失败/超时 → 抽取式降级答案，不抛异常；
  · 全链路**任何异常都不向调用方抛出**（工单「容错机制」要求）。

性能：检索 1.3s 量级（见 retrieve_v2），生成 ~1s，合计控制在 3s 预算内。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from w9 import config9
from w9.prompts_loader import load_prompt
from w9.retrieve_v2 import RetrievalResult, retrieve_v2

from src import config as c8
from src.kg_store import KnowledgeGraph

# 惰性单例：09 图谱
_kg9: KnowledgeGraph | None = None
_kg9_error: str | None = None


def get_kg9(load: bool = True) -> KnowledgeGraph | None:
    """加载 09 图谱（含第2层合并）；失败返回 None 并记录原因。"""
    global _kg9, _kg9_error
    if _kg9 is not None:
        return _kg9
    if not load:
        return None
    try:
        import os
        if not os.path.isfile(config9.KG9_PICKLE):
            _kg9_error = (f"09 图谱不存在：{config9.KG9_PICKLE}\n"
                          f"请先运行 python -m w9.kg_merge")
            return None
        _kg9 = KnowledgeGraph.load(config9.KG9_PICKLE)
        _kg9_error = None
    except Exception as exc:                           # noqa: BLE001
        _kg9_error = f"{type(exc).__name__}: {exc}"
        return None
    return _kg9


@dataclass
class QaResult9:
    question: str = ""
    answer: str = ""
    lang: str = "zh"
    qtype: str = config9.QTYPE_FACTUAL
    queries: list[str] = field(default_factory=list)
    graph_facts: list[str] = field(default_factory=list)
    text_hits: list[dict] = field(default_factory=list)
    subgraph: dict = field(default_factory=dict)
    linked: list[dict] = field(default_factory=list)
    context: str = ""
    timings: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    degraded: bool = False
    degraded_reason: str = ""

    def contexts(self) -> list[str]:
        """供 RAGAS 使用的上下文列表。

        图谱事实与文本块**分列**：图谱事实是结构化断言，文本块是原文片段。
        RAGAS 的 context precision/recall 面向「检索到的上下文片段」，
        两者都是被送入 LLM 的真实上下文，故都计入。
        """
        out: list[str] = []
        for f in self.graph_facts:
            out.append(f"【图谱事实】{f}")
        for h in self.text_hits:
            doc = h.get("source") or h.get("source_doc") or ""
            page = h.get("page_idx", h.get("page", -1))
            bt = h.get("block_type", "text")
            cid = h.get("chunk_id", "-")
            out.append(f"【文本证据】来源：{doc} 第 {page} 页"
                       f"（block_type={bt}, id={cid}）\n{h.get('text') or ''}")
        return out

    def to_dict(self) -> dict:
        return {
            "question": self.question, "answer": self.answer, "lang": self.lang,
            "qtype": self.qtype, "queries": list(self.queries),
            "n_graph_facts": len(self.graph_facts),
            "n_text_hits": len(self.text_hits),
            "contexts": self.contexts(),
            "text_hits": [{"chunk_id": h.get("chunk_id"),
                           "source": h.get("source") or h.get("source_doc"),
                           "page": h.get("page_idx", h.get("page")),
                           "block_type": h.get("block_type"),
                           "score": h.get("score"),
                           "text": (h.get("text") or "")[:400]}
                          for h in self.text_hits],
            "subgraph": self.subgraph,
            "timings": dict(self.timings), "info": dict(self.info),
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
        }


def _answer_lang(question: str) -> str:
    try:
        from src.rag import answer_lang
        return answer_lang(question)
    except Exception:                                  # noqa: BLE001
        s = (question or "").strip()
        return "en" if s and s.isascii() else "zh"


def _call_llm(messages: list[dict], **kw) -> str:
    from src import llm
    res = llm.chat(messages, **kw)
    return str(res.get("answer", "")) if isinstance(res, dict) else str(res)


def build_messages_v2(question: str, context: str, lang: str,
                      has_graph: bool, qtype: str,
                      prompt: dict | None = None) -> list[dict]:
    """构造生成 messages（Prompt v2）。"""
    prompt = prompt or load_prompt("graph_qa_v2")
    system = prompt["system_zh"] if lang == "zh" else prompt["system_en"]

    if not has_graph:
        system += "\n\n" + (prompt.get("no_graph_hint_zh", "")
                            if lang == "zh" else prompt.get("no_graph_hint_en", ""))
    # 全局型问题追加「覆盖每个方面」的强化指令（最怕只答一个侧面）
    if qtype == config9.QTYPE_GLOBAL:
        extra = prompt.get("global_hint_zh" if lang == "zh" else "global_hint_en", "")
        if extra:
            system += "\n\n" + extra

    tpl = prompt.get("user_template_zh" if lang == "zh" else "user_template_en")
    user = (tpl.format(context=context, question=question) if tpl
            else f"{context}\n\n问题：{question}")
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


# 证据编号 → 真实来源。编号顺序**必须与 retrieve_v2.build_context_v2 的编号一致**：
# 该函数按 text → table → image 的顺序全局连续编号，这里照同一顺序展开。
def citation_index(ret: RetrievalResult) -> list[dict]:
    """构造 [n] → 真实来源 的映射表（供答案末尾附「引用来源」）。"""
    groups: dict[str, list[dict]] = {"table": [], "image": [], "text": []}
    for h in ret.text_hits:
        bt = str(h.get("block_type") or "text").lower()
        groups.setdefault(bt if bt in groups else "text", []).append(h)

    out: list[dict] = []
    n = 0
    for key in ("text", "table", "image"):
        for h in groups[key]:
            n += 1
            out.append({
                "n": n,
                "source": h.get("source") or h.get("source_doc") or "",
                "page": h.get("page_idx", h.get("page", -1)),
                "block_type": h.get("block_type", "text"),
                "chunk_id": h.get("chunk_id", "-"),
            })
    return out


_CITE_RE = re.compile(r"\[(\d+)\]")


def _append_citations(answer: str, cites: list[dict], lang: str) -> str:
    """把答案里出现的 [n] 解析成真实来源清单，附在答案末尾。

    这是工单「引用机制」的落地方式：**页码 + block_type + ID 由代码生成，
    不由模型生成**。模型只负责指出「这句话依据哪条证据」，
    来源元数据从检索结果直接取 —— 因此引用**不可能被编造**。
    答案里没引用任何编号时不追加（避免给纯兜底回答挂一张空表）。
    """
    if not answer or not cites:
        return answer
    used = sorted({int(m) for m in _CITE_RE.findall(answer)})
    if not used:
        return answer
    by_n = {c["n"]: c for c in cites}
    rows: list[str] = []
    for n in used:
        c = by_n.get(n)
        if not c:
            continue
        rows.append(f"[{n}] {c['source']} 第 {c['page']} 页 "
                    f"（block_type={c['block_type']}, id={c['chunk_id']}）")
    if not rows:
        return answer
    head = "\n\n**引用来源**：" if lang == "zh" else "\n\n**Sources**:"
    return answer + head + "\n" + "\n".join(rows)


def _extractive_answer(ret: RetrievalResult, lang: str) -> str:
    """LLM 不可用时的抽取式降级答案（无 LLM 也能交付）。"""
    parts: list[str] = []
    if ret.graph_facts:
        parts.append("【图谱事实】" if lang == "zh" else "[GRAPH FACTS]")
        parts.extend(f"- {f}" for f in ret.graph_facts[:8])
    if ret.text_hits:
        parts.append("【原文片段】" if lang == "zh" else "[TEXT EVIDENCE]")
        for h in ret.text_hits[:4]:
            doc = h.get("source") or h.get("source_doc") or ""
            page = h.get("page_idx", h.get("page", -1))
            snip = (h.get("text") or "").strip().replace("\n", " ")[:220]
            parts.append(f"- （{doc} 第 {page} 页）{snip}")
    if not parts:
        return ("未能检索到相关内容。" if lang == "zh"
                else "No relevant content found.")
    return "\n".join(parts)


def _retrieve_with_mode(question: str, kg: KnowledgeGraph | None,
                        mode: str, use_cache: bool) -> RetrievalResult:
    """按档位临时收紧检索规模后调用 retrieve_v2。

    与 08 的 `KG_SUBGRAPH_HOPS` 同款做法：改的是 `config9` 上的运行期参数，
    调用结束**必定还原**（放在 finally 里），避免 fast 档的设置泄漏到后续 full 档请求。
    """
    spec = config9.GEN9_MODES.get(mode, {})
    pool = spec.get("rerank_pool")
    topk = spec.get("final_top_k")
    if pool is None and topk is None:
        return retrieve_v2(question, kg, use_cache=use_cache)

    old = (config9.RERANK9_POOL, config9.FINAL9_TOP_K)
    try:
        if pool:
            config9.RERANK9_POOL = pool
        if topk:
            config9.FINAL9_TOP_K = topk
        # retrieve_v2 在函数体内读取这两个常量，故改 config 即生效
        return retrieve_v2(question, kg, use_cache=use_cache)
    finally:
        config9.RERANK9_POOL, config9.FINAL9_TOP_K = old


def ask_v2(question: str, kg: KnowledgeGraph | None = None,
           call_llm=None, use_cache: bool = True,
           num_predict: int | None = None, mode: str | None = None) -> QaResult9:
    """09 优化后的 Graph RAG 问答主入口。全链路不抛异常。

    mode: "full"（默认，完整作答）| "fast"（压缩输出守住 ≤3 秒）
          两档的取舍依据见 config9.GEN9_MODES 的注释。
    """
    mode = mode or config9.GEN9_MODE_DEFAULT
    if num_predict is None:
        num_predict = config9.GEN9_MODES.get(
            mode, {}).get("num_predict", config9.GEN9_NUM_PREDICT)
    t_start = time.perf_counter()
    lang = _answer_lang(question)
    out = QaResult9(question=question, lang=lang)

    if not (question or "").strip():
        out.degraded = True
        out.degraded_reason = "空问题"
        out.answer = ("请输入问题。" if lang == "zh" else "Please enter a question.")
        out.timings = {"total": time.perf_counter() - t_start}
        return out

    kg = kg if kg is not None else get_kg9()
    if kg is None:
        out.degraded = True
        out.degraded_reason = f"09 图谱不可用：{_kg9_error}"

    # 1-5) 检索（实体 → 类型 → 分解 → 子图 → 多路召回 → 融合 → 上下文）
    # fast 档要一并收紧检索规模（精排候选数与上下文块数），否则光是精排
    # 40 条候选就要 0.5s，加上更长的上下文带来的 prompt eval，仍然超 3 秒。
    t0 = time.perf_counter()
    try:
        ret = _retrieve_with_mode(question, kg, mode, use_cache)
    except Exception as exc:                           # noqa: BLE001
        out.degraded = True
        out.degraded_reason = f"检索失败：{type(exc).__name__}: {exc}"
        ret = RetrievalResult()
    out.timings["retrieve_total"] = time.perf_counter() - t0
    out.timings.update({k: v for k, v in ret.timings.items()
                        if k not in out.timings})

    out.qtype = ret.qtype
    out.queries = ret.queries
    out.graph_facts = ret.graph_facts
    out.text_hits = ret.text_hits
    out.subgraph = ret.subgraph
    out.linked = ret.linked
    out.context = ret.context
    out.info = ret.info

    # 6) LLM 生成
    t0 = time.perf_counter()
    llm_fn = call_llm or _call_llm
    try:
        messages = build_messages_v2(question, ret.context, lang,
                                     bool(ret.graph_facts), ret.qtype)
        answer = llm_fn(messages, temperature=config9.GEN9_TEMPERATURE,
                        num_predict=num_predict)
        out.answer = (answer or "").strip()
        if not out.answer:
            raise RuntimeError("LLM 返回空答案")
        # 把模型写的 [n] 解析成真实来源（页码/block_type/ID 由代码生成 → 不可编造）
        out.answer = _append_citations(out.answer, citation_index(ret), lang)
    except Exception as exc:                           # noqa: BLE001
        out.degraded = True
        out.degraded_reason = (out.degraded_reason + "；" if
                               out.degraded_reason else "") + \
                              f"LLM 生成失败：{type(exc).__name__}"
        out.answer = _extractive_answer(ret, lang)
    out.timings["generate"] = time.perf_counter() - t0
    out.timings["total"] = time.perf_counter() - t_start
    return out


def warmup() -> dict:
    """预热 09 链路。"""
    out: dict = {}
    t0 = time.perf_counter()
    kg = get_kg9()
    out["graph"] = {"ok": kg is not None,
                    "nodes": kg.g.number_of_nodes() if kg else 0,
                    "edges": kg.g.number_of_edges() if kg else 0,
                    "error": _kg9_error,
                    "seconds": round(time.perf_counter() - t0, 2)}
    try:
        from src import rag
        out["rag"] = rag.warmup()
    except Exception as exc:                           # noqa: BLE001
        out["rag"] = {"ok": False, "error": str(exc)}
    return out


def status() -> dict:
    kg = get_kg9(load=False)
    import os
    return {
        "workorder": config9.WORKORDER_ID,
        "graph_loaded": kg is not None,
        "graph_nodes": kg.g.number_of_nodes() if kg else 0,
        "graph_edges": kg.g.number_of_edges() if kg else 0,
        "graph_error": _kg9_error,
        "kg9_pickle": config9.KG9_PICKLE,
        "kg9_exists": os.path.isfile(config9.KG9_PICKLE),
        "qdrant": config9.QDRANT9_LOCAL_PATH,
        "qdrant_is_copy": config9.QDRANT9_USING_COPY,
    }
