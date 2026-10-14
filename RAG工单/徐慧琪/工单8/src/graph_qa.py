# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""
模块：Graph RAG 问答主链路（硬性要求2、3）

链路（对齐工单要求的流程图）：
  Query 理解 → 实体识别+链接 → 子图检索 → 向量+全文检索
  → 三路融合 → Reranker 精排 → LLM 生成 → 答案+检索结果+图谱结构

容错（硬性要求7）：
  · 图谱加载失败 → 退化为纯 RAG（07 能力），如实标记 degraded；
  · 子图检索失败 → 跳过图谱通道，仅用文本通道；
  · LLM 失败   → 降级为抽取式答案（用图谱事实/文本首句拼装），不抛异常。

性能（硬性要求4：≤3 秒）：
  · 图谱查询为内存操作；实体链接与子图结果均带缓存；
  · 文本检索复用 07 的 warmup/显存闸门，Reranker 与 LLM 常驻。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from src import config
from src.graph_fusion import build_context_text, fuse
from src.graph_retriever import (SubgraphResult, link_entities,
                                 retrieve_subgraph)
from src.kg_store import KnowledgeGraph

# ---------------------------------------------------------------------------
# 惰性单例
# ---------------------------------------------------------------------------
_kg: KnowledgeGraph | None = None
_kg_error: str | None = None

# 子图缓存：同实体集合的问题复用同一子图（支撑 ≤3s 目标）
_subgraph_cache: dict[str, SubgraphResult] = {}
_SUBGRAPH_CACHE_MAX = 128


def get_kg(load: bool = True) -> KnowledgeGraph | None:
    """惰性加载图谱；失败返回 None 并记录错误（不抛异常）。"""
    global _kg, _kg_error
    if _kg is not None:
        return _kg
    if not load:
        return None
    try:
        _kg = KnowledgeGraph.load()
        _kg_error = None
    except Exception as exc:                       # noqa: BLE001
        _kg_error = f"{type(exc).__name__}: {exc}"
        return None
    return _kg


def clear_caches() -> None:
    _subgraph_cache.clear()


# ---------------------------------------------------------------------------
# 结果对象
# ---------------------------------------------------------------------------
@dataclass
class QaResult:
    question: str = ""
    answer: str = ""
    lang: str = "zh"
    mode: str = "none"
    strategy: str = "hybrid"
    graph_facts: list[str] = field(default_factory=list)
    subgraph: dict = field(default_factory=dict)
    text_hits: list[dict] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    degraded: bool = False
    degraded_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question, "answer": self.answer, "lang": self.lang,
            "mode": self.mode, "strategy": self.strategy,
            "graph_facts": list(self.graph_facts), "subgraph": self.subgraph,
            "text_hits": [{"chunk_id": h.get("chunk_id"),
                           "source": h.get("source"),
                           "page": h.get("page_idx"),
                           "score": h.get("score"),
                           "text": (h.get("text") or "")[:400]}
                          for h in self.text_hits],
            "timings": dict(self.timings),
            "degraded": self.degraded, "degraded_reason": self.degraded_reason,
        }


# ---------------------------------------------------------------------------
# 可注入的检索 / LLM（测试时打桩）
# ---------------------------------------------------------------------------
def _retrieve_text(question: str, strategy: str = "hybrid",
                   top_k: int | None = None) -> list[dict]:
    """文本检索（复用 07 的 strategies）。失败返回空列表（容错）。

    注意上游契约（已核对 工单7/src/strategies.py:188）：
      · RetrievalConfig 的字段是 `strategy` / `top_k_recall` / `top_k_rerank`
        —— **没有 `top_k` 字段**；
      · `strategies.retrieve()` 返回 **dict** `{"hits": [...], "info": {...}}`，
        不是 list —— 这里取 `["hits"]`。
    """
    top_k = top_k or config.RERANKER_TOP_K_OUT
    try:
        from src import strategies
        cfg = strategies.RetrievalConfig(strategy=strategy, top_k_recall=top_k)
        res = strategies.retrieve(question, cfg)
        return res.get("hits", []) if isinstance(res, dict) else list(res)
    except Exception as exc:                       # noqa: BLE001
        print(f"[graph_qa] 文本检索失败，降级为空结果：{exc}", flush=True)
        return []


def _call_llm(messages: list[dict], **kwargs) -> str:
    """LLM 调用契约：`messages -> str`。

    上游 `llm.chat()` 返回 dict（工单7/src/llm.py:137），这里取出 `answer`。
    """
    from src import llm
    res = llm.chat(messages, **kwargs)
    return str(res.get("answer", "")) if isinstance(res, dict) else str(res)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
def _load_qa_prompt() -> dict:
    from src.kg_prompt import load_prompt
    return load_prompt("graph_qa")


def _build_messages(question: str, context_text: str, lang: str,
                    has_graph: bool) -> list[dict]:
    prompt = _load_qa_prompt()
    system = prompt["system_zh"] if lang == "zh" else prompt["system_en"]
    if not has_graph:
        system += "\n\n" + (prompt["no_graph_hint_zh"] if lang == "zh"
                            else prompt["no_graph_hint_en"])
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"{context_text}\n\n问题：{question}"}]


# ---------------------------------------------------------------------------
# 主链路
# ---------------------------------------------------------------------------
def _cache_key(question: str, linked: list[dict]) -> str:
    return question + "|" + ",".join(sorted(x["entity_id"] for x in linked))


def ask(question: str, kg: KnowledgeGraph | None = None, state=None,
        use_rerank: bool = True, use_cache: bool = True,
        strategy: str = "hybrid") -> QaResult:
    """Graph RAG 问答。

    state：可选的 dialogue.DialogueState，用于多轮对话（指代消解）。
    """
    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    lang = "zh"
    try:
        from src.rag import answer_lang
        lang = answer_lang(question)
    except Exception:                              # noqa: BLE001
        lang = "en" if question.strip().isascii() and question.strip() else "zh"

    result = QaResult(question=question, lang=lang, strategy=strategy)

    # 图谱不可用 → 退化纯 RAG
    kg = kg if kg is not None else get_kg()
    if kg is None:
        result.degraded = True
        result.degraded_reason = f"图谱不可用：{_kg_error}"

    # 1) Query 理解（多轮：指代消解）
    if state is not None:
        try:
            from src import dialogue
            dialogue.update_focus(state, question)
        except Exception:                          # noqa: BLE001
            pass

    # 2) 实体识别 + 链接
    t0 = time.perf_counter()
    linked = link_entities(question, kg) if kg is not None else []
    timings["link"] = time.perf_counter() - t0

    # 3) 子图检索（带缓存）
    t0 = time.perf_counter()
    if kg is None:
        sub = SubgraphResult(mode="none")
    else:
        key = _cache_key(question, linked)
        sub = _subgraph_cache.get(key) if use_cache else None
        if sub is None:
            try:
                sub = retrieve_subgraph(question, linked, kg)
            except Exception as exc:               # noqa: BLE001
                print(f"[graph_qa] 子图检索失败，跳过图谱通道：{exc}", flush=True)
                sub = SubgraphResult(mode="none")
                result.degraded = True
                result.degraded_reason = f"子图检索失败：{exc}"
            if use_cache:
                if len(_subgraph_cache) >= _SUBGRAPH_CACHE_MAX:
                    _subgraph_cache.clear()
                _subgraph_cache[key] = sub
    timings["subgraph"] = time.perf_counter() - t0
    result.mode = sub.mode
    result.graph_facts = list(sub.facts)
    result.subgraph = sub.to_html_data()

    # 4) 文本检索（复用 07）
    t0 = time.perf_counter()
    hits = _retrieve_text(question, strategy, config.RERANKER_TOP_K_OUT)
    timings["retrieve"] = time.perf_counter() - t0

    # 5) 三路融合（向量/全文已在 strategies 内融合，此处叠加图谱通道）
    fused = fuse(sub.facts, hits, [], top_k=config.RERANKER_TOP_K_OUT)
    result.text_hits = fused.text_hits
    context_text = build_context_text(fused)

    # 6) LLM 生成
    t0 = time.perf_counter()
    try:
        messages = _build_messages(question, context_text, lang,
                                   bool(sub.facts))
        answer = _call_llm(messages, temperature=config.OLLAMA_TEMPERATURE,
                           num_predict=config.OLLAMA_NUM_PREDICT)
        result.answer = (answer or "").strip()
        if not result.answer:
            raise RuntimeError("LLM 返回空答案")
    except Exception as exc:                       # noqa: BLE001
        result.degraded = True
        result.degraded_reason = (result.degraded_reason + "；" if
                                  result.degraded_reason else "") + \
                                 f"LLM 生成失败：{type(exc).__name__}"
        result.answer = _extractive_answer(sub, fused.text_hits, lang)
    timings["generate"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - t_start
    result.timings = timings
    return result


def ask_uploaded(question: str, index, call_llm=None,
                 top_k: int | None = None) -> QaResult:
    """就**上传文档**提问（会话级，不触碰全局图谱与语料索引）。

    与 ask() 的唯一区别是检索来源：命中来自上传文档的内存索引。
    生成链路（prompt / 语言判定 / 降级）与 ask() 完全复用，
    保证上传文档的答案口径与主链路一致。
    """
    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    lang = "zh"
    try:
        from src.rag import answer_lang
        lang = answer_lang(question)
    except Exception:                              # noqa: BLE001
        lang = "en" if question.strip().isascii() and question.strip() else "zh"

    result = QaResult(question=question, lang=lang, strategy="upload")
    result.mode = "upload"

    if index is None or not getattr(index, "n_chunks", 0):
        result.degraded = True
        result.degraded_reason = "未上传文档（或文档无可用内容）"
        result.answer = ("尚未上传可用的文档：请先在「上传文档」区域选择一个 PDF，"
                         "再就它提问。" if lang == "zh" else
                         "No uploaded document is available yet — please upload "
                         "a PDF first.")
        timings["total"] = time.perf_counter() - t_start
        result.timings = timings
        return result

    t0 = time.perf_counter()
    hits = index.search(question, top_k=top_k or config.RERANKER_TOP_K_OUT)
    timings["retrieve"] = time.perf_counter() - t0

    fused = fuse([], hits, [], top_k=config.RERANKER_TOP_K_OUT)
    result.text_hits = fused.text_hits
    context_text = build_context_text(fused)

    t0 = time.perf_counter()
    llm_fn = call_llm or _call_llm
    try:
        messages = _build_messages(question, context_text, lang,
                                   False)          # 上传模式无图谱事实
        answer = llm_fn(messages, temperature=config.OLLAMA_TEMPERATURE,
                        num_predict=config.OLLAMA_NUM_PREDICT)
        result.answer = (answer or "").strip()
        if not result.answer:
            raise RuntimeError("LLM 返回空答案")
    except Exception as exc:                       # noqa: BLE001
        result.degraded = True
        result.degraded_reason = f"LLM 生成失败：{type(exc).__name__}"
        result.answer = _extractive_answer(SubgraphResult(mode="upload"),
                                           fused.text_hits, lang)
    timings["generate"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - t_start
    result.timings = timings
    return result


def _extractive_answer(sub: SubgraphResult, text_hits: list[dict],
                       lang: str = "zh") -> str:
    """抽取式降级答案：把图谱事实与文本首句拼成答案（无 LLM 时仍可交付）。"""
    parts: list[str] = []
    if sub.facts:
        parts.append("【图谱事实】" if lang == "zh" else "[GRAPH FACTS]")
        parts.extend(f"- {f}" for f in sub.facts[:10])
    if text_hits:
        parts.append("【原文片段】" if lang == "zh" else "[TEXT EVIDENCE]")
        for h in text_hits[:3]:
            doc = h.get("source", "")
            # 检索命中带的是 1 基 `page`（语料命中无 page_idx，上传命中两者都有），
            # 只读 page_idx 会让降级答案引用「第 -1 页」。
            page = h.get("page", h.get("page_idx", -1))
            snippet = (h.get("text") or "").strip().replace("\n", " ")[:200]
            parts.append(f"- （{doc} 第 {page} 页）{snippet}")
    if not parts:
        return ("未能检索到相关内容。" if lang == "zh"
                else "No relevant content found.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 预热与状态
# ---------------------------------------------------------------------------
def warmup() -> dict:
    """预热：加载图谱 + 触发 07 的模型预热，避免首问超时。"""
    out: dict = {}
    t0 = time.perf_counter()
    kg = get_kg()
    out["graph"] = {"ok": kg is not None,
                    "nodes": kg.g.number_of_nodes() if kg else 0,
                    "edges": kg.g.number_of_edges() if kg else 0,
                    "error": _kg_error,
                    "seconds": round(time.perf_counter() - t0, 2)}
    try:
        from src import rag
        out["rag"] = rag.warmup()
    except Exception as exc:                       # noqa: BLE001
        out["rag"] = {"ok": False, "error": str(exc)}
    return out


def status() -> dict:
    kg = get_kg(load=False)
    return {
        "graph_loaded": kg is not None,
        "graph_nodes": kg.g.number_of_nodes() if kg else 0,
        "graph_edges": kg.g.number_of_edges() if kg else 0,
        "graph_error": _kg_error,
        "subgraph_cache": len(_subgraph_cache),
        "kg_pickle": config.KG_PICKLE,
        "kg_pickle_exists": os.path.isfile(config.KG_PICKLE),
    }


# ---------------------------------------------------------------------------
# 金融研报生成（工单任务描述：「包括但不限于问题问答以及金融研报生成」）
# ---------------------------------------------------------------------------
@dataclass
class ReportResult:
    title: str = ""
    markdown: str = ""
    sections: list[dict] = field(default_factory=list)
    source_table: list[dict] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    degraded: bool = False
    degraded_reason: str = ""

    def to_dict(self) -> dict:
        return {"title": self.title, "markdown": self.markdown,
                "sections": list(self.sections),
                "source_table": list(self.source_table),
                "timings": dict(self.timings), "degraded": self.degraded,
                "degraded_reason": self.degraded_reason}


def _collect_source_table(kg: KnowledgeGraph, companies: list[str],
                          metrics: list[str]) -> list[dict]:
    """从图谱直接取数，生成「每个数字可回溯」的来源表。

    Graph RAG 相对普通 RAG 在研报生成上的优势：数据由图谱提供，
    结构（哪家公司/哪个指标/哪个期间）确定，不靠 LLM 自由发挥。
    """
    rows: list[dict] = []
    for metric in metrics:
        for r in kg.metric_compare(metric, companies=companies or None):
            rows.append({"company": r["company"], "metric": metric,
                         "value": r["value"], "unit": r["unit"],
                         "period": r["period"], "yoy": r["yoy"],
                         "source_doc": r["source_doc"], "page": r["page"],
                         "evidence": r.get("evidence", "")})
    return rows


def _graph_block(rows: list[dict]) -> str:
    if not rows:
        return "（图谱中未检索到相关指标数据）"
    lines = ["| 公司 | 指标 | 数值 | 单位 | 报告期 | 同比 | 来源 | 页码 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        yoy = f"{r['yoy']}%" if r.get("yoy") is not None else "—"
        lines.append(f"| {r['company']} | {r['metric']} | {r['value']} | "
                     f"{r['unit'] or '—'} | {r['period'] or '—'} | {yoy} | "
                     f"{r['source_doc']} | 第{r['page']}页 |")
    return "\n".join(lines)


def _parse_sections(markdown: str) -> list[dict]:
    """按 `## ` 标题切分 Markdown，返回 [{"name","content"}]。"""
    sections: list[dict] = []
    cur_name: str | None = None
    buf: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            if cur_name is not None:
                sections.append({"name": cur_name,
                                 "content": "\n".join(buf).strip()})
            cur_name = line[3:].strip()
            buf = []
        elif cur_name is not None:
            buf.append(line)
    if cur_name is not None:
        sections.append({"name": cur_name, "content": "\n".join(buf).strip()})
    return sections


def _inject_data_section(markdown: str, rows: list[dict]) -> str:
    """LLM 未按结构产出「数据概览」时，补入图谱取数的确定性数据表。

    保证研报里始终有可回溯的结构化数字 —— 这是 Graph RAG 相对
    纯生成的核心价值；小模型（qwen2.5:3b）实测会自由发挥章节结构。
    """
    if not rows or "数据概览" in markdown:
        return markdown
    block = ("## 一、数据概览\n\n" + _graph_block(rows) +
             "\n\n> 本节数据来自知识图谱，每个数字均可回溯到来源文档与页码。\n")
    if not markdown.strip():
        return block
    lines = markdown.split("\n", 1)
    if lines[0].startswith("#"):        # 报告标题行之后插入
        rest = lines[1] if len(lines) > 1 else ""
        return lines[0] + "\n\n" + block + "\n" + rest
    return block + "\n" + markdown


def generate_report(topic: str, kg: KnowledgeGraph | None = None,
                    companies: list[str] | None = None,
                    metrics: list[str] | None = None,
                    strategy: str = "hybrid") -> ReportResult:
    """生成金融研报：图谱做骨架、文本块做填充，每个数字可回溯。"""
    from src.kg_dict import METRIC_TABLE
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    out = ReportResult(title=topic)

    kg = kg if kg is not None else get_kg()
    if kg is None:
        out.degraded = True
        out.degraded_reason = f"图谱不可用：{_kg_error}"

    if not companies:
        companies = [kg.g.nodes[n].get("name", n)
                     for n in kg.nodes_of_type("Company")] if kg else []
    if not metrics:
        metrics = ["不良贷款率", "拨备覆盖率", "营业收入",
                   "归属于母公司股东的净利润", "资产总计"]

    # 1) 图谱取数（确定性）
    t = time.perf_counter()
    rows = _collect_source_table(kg, companies, metrics) if kg else []
    timings["graph"] = time.perf_counter() - t
    out.source_table = rows

    # 2) 文本检索补充叙述性证据
    t = time.perf_counter()
    query = topic + " " + " ".join(companies[:5])
    hits = _retrieve_text(query, strategy, config.RERANKER_TOP_K_OUT)
    timings["retrieve"] = time.perf_counter() - t

    evidence_block = "\n".join(
        f"[{i}] {h.get('source')} 第 {h.get('page_idx')} 页："
        f"{(h.get('text') or '')[:400]}"
        for i, h in enumerate(hits[:8], start=1)) or "（无）"

    # 3) LLM 撰写
    t = time.perf_counter()
    try:
        prompt = _load_qa_prompt()
        messages = [
            {"role": "system", "content": prompt["report_system_zh"]},
            {"role": "user", "content": prompt["report_user_template_zh"].format(
                topic=topic, companies="、".join(companies) or "（全部）",
                graph_block=_graph_block(rows), evidence_block=evidence_block)},
        ]
        out.markdown = (_call_llm(messages,
                                  temperature=config.OLLAMA_TEMPERATURE,
                                  num_predict=2048) or "").strip()
        if not out.markdown:
            raise RuntimeError("LLM 返回空研报")
        out.markdown = _inject_data_section(out.markdown, rows)
        out.sections = _parse_sections(out.markdown)
    except Exception as exc:                       # noqa: BLE001
        out.degraded = True
        out.degraded_reason = f"LLM 撰写失败：{type(exc).__name__}"
        out.markdown = _fallback_report(topic, rows, evidence_block)
        out.sections = _parse_sections(out.markdown)
    timings["generate"] = time.perf_counter() - t

    timings["total"] = time.perf_counter() - t0
    out.timings = timings
    return out


def _fallback_report(topic: str, rows: list[dict], evidence: str) -> str:
    """LLM 不可用时的降级研报：只输出确定性数据部分。"""
    return (f"# {topic}\n\n"
            f"> ⚠️ 生成模型不可用，以下为**图谱取数的确定性结果**"
            f"（未含分析性文字）。\n\n"
            f"## 一、数据概览\n\n{_graph_block(rows)}\n\n"
            f"## 五、数据来源\n\n{evidence}")
