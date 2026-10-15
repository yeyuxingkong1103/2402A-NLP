# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务

模块：工单09 交互界面（Streamlit，中英双语）

四个页签：
  1. **问答** —— 提问（中/英），查看答案、图谱事实、检索到的原文块（含页码与块 ID）
  2. **优化对比** —— 同一问题分别跑 08 基线与 09 优化链路，左右并列对比
  3. **指标看板** —— RAGAS 指标、辅助检索指标、响应时间（读取评估产物）
  4. **系统状态** —— 健康检查、图谱规模、容错清单

启动：
    streamlit run app.py
"""
from __future__ import annotations

import io
import json
import os
import time

import streamlit as st

from w9 import compat  # noqa: F401  必须最先：置离线 + 挂载 08 路径
from w9 import config9

st.set_page_config(page_title="Graph RAG 优化（工单09）", page_icon="📊",
                   layout="wide")


# ---------------------------------------------------------------------------
# 文案（中英双语）
# ---------------------------------------------------------------------------
T = {
    "zh": {
        "title": "Graph RAG 优化 · 金融年报问答",
        "caption": "工单编号：人工智能NLP-RAG-Graph RAG优化任务　|　"
                   "语料：9 份上市公司年报　|　图数据库：NetworkX　|　"
                   "向量库：Qdrant　|　LLM：qwen2.5:3b（本机）",
        "tab_qa": "问答", "tab_cmp": "优化对比", "tab_metric": "指标看板",
        "tab_status": "系统状态",
        "ask": "请输入问题", "placeholder": "例如：平安银行2019年年度报告的董事长致辞中，"
                                            "提到其盈利增长的关键因素有哪些？",
        "run": "提问", "running": "检索与生成中…",
        "answer": "答案", "graph_facts": "图谱事实", "evidence": "检索到的原文块",
        "qtype": "问题类型", "queries": "子查询（查询分解）",
        "timings": "耗时分解", "degraded": "降级原因",
        "cmp_run": "同时运行 08 基线与 09 优化", "baseline": "08 基线（优化前）",
        "optimized": "09 优化后", "metric_req": "工单要求",
        "no_eval": "尚无评估产物。请先运行：python -m w9.run_eval --set main "
                   "--systems wo08,wo09 --ragas core，再运行 python -m w9.compare",
        "health": "健康检查", "graph_scale": "图谱规模",
        "faults": "容错机制清单", "lang": "界面语言",
    },
    "en": {
        "title": "Graph RAG Optimization · Financial Annual-Report QA",
        "caption": "Work order: AI-NLP-RAG Graph RAG Optimization　|　"
                   "Corpus: 9 listed-company annual reports　|　Graph: NetworkX　|　"
                   "Vector store: Qdrant　|　LLM: qwen2.5:3b (local)",
        "tab_qa": "Ask", "tab_cmp": "A/B Compare", "tab_metric": "Metrics",
        "tab_status": "System",
        "ask": "Enter your question",
        "placeholder": "e.g. What were Postal Savings Bank's NPL ratio and "
                       "provision coverage ratio at end-2019?",
        "run": "Ask", "running": "Retrieving and generating…",
        "answer": "Answer", "graph_facts": "Graph facts",
        "evidence": "Retrieved passages",
        "qtype": "Question type", "queries": "Sub-queries (decomposition)",
        "timings": "Timings", "degraded": "Degraded",
        "cmp_run": "Run both 08 baseline and 09 optimized",
        "baseline": "08 baseline (before)", "optimized": "09 optimized (after)",
        "metric_req": "Requirement",
        "no_eval": "No evaluation artifacts yet. Run: python -m w9.run_eval "
                   "--set main --systems wo08,wo09 --ragas core, then "
                   "python -m w9.compare",
        "health": "Health check", "graph_scale": "Graph scale",
        "faults": "Fault-tolerance matrix", "lang": "UI language",
    },
}


def _lang() -> str:
    return st.session_state.get("lang", "zh")


def t(key: str) -> str:
    return T[_lang()].get(key, key)


# ---------------------------------------------------------------------------
# 惰性加载（避免页面一打开就吃满显存）
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="加载模型与索引…")
def _warmup():
    from src import rag
    t0 = time.perf_counter()
    info = {}
    try:
        info["rag"] = rag.warmup()
    except Exception as exc:                           # noqa: BLE001
        info["rag"] = {"ok": False, "error": str(exc)}
    try:
        from w9.qa_v2 import get_kg9
        kg = get_kg9()
        info["graph"] = {"ok": kg is not None,
                         "nodes": kg.g.number_of_nodes() if kg else 0,
                         "edges": kg.g.number_of_edges() if kg else 0}
    except Exception as exc:                           # noqa: BLE001
        info["graph"] = {"ok": False, "error": str(exc)}
    info["seconds"] = round(time.perf_counter() - t0, 1)
    return info


def _render_subgraph(sub: dict) -> None:
    """渲染图谱结构（工单用户手册要求「查看知识图谱」）。

    优先用 `st.graphviz_chart` 画 DOT（Streamlit 在**浏览器端**渲染，
    不需要装 Python graphviz 包）；不可用时退回「节点表 + 边表」，
    保证任何环境下都能看到图谱结构。
    """
    nodes = (sub or {}).get("nodes") or []
    edges = (sub or {}).get("edges") or []
    if not nodes and not edges:
        st.caption("本次未检索到图谱结构（该问题未命中图谱通道）。")
        return

    st.caption(f"子图：{len(nodes)} 节点 / {len(edges)} 边 · 模式 {sub.get('mode')}")

    def _esc(s: str) -> str:
        return str(s).replace('"', "'").replace("\n", " ")[:28]

    lines = ["digraph G {", '  rankdir=LR;',
             '  node [shape=box, style=rounded, fontsize=10];']
    for n in nodes[:60]:
        lines.append(f'  "{_esc(n.get("id"))}" [label="{_esc(n.get("label"))}",'
                     f' color="{n.get("color", "#9AA0A6")}"];')
    for e in edges[:120]:
        lines.append(f'  "{_esc(e.get("from"))}" -> "{_esc(e.get("to"))}" '
                     f'[label="{_esc(e.get("label"))}", fontsize=8];')
    lines.append("}")
    dot = "\n".join(lines)

    try:
        st.graphviz_chart(dot, use_container_width=True)
    except Exception:                                  # noqa: BLE001
        st.caption("（Graphviz 渲染不可用，改用表格展示）")
        st.dataframe([{"节点": n.get("label"), "类型": n.get("type")}
                      for n in nodes[:80]], use_container_width=True)
        st.dataframe([{"头": e.get("from"), "关系": e.get("label"),
                       "尾": e.get("to"), "页": e.get("page")}
                      for e in edges[:120]], use_container_width=True)

    with st.expander("子图数据（DOT / JSON）"):
        st.code(dot, language="dot")


def _render_result(r, lang: str) -> None:
    """渲染一次问答结果。"""
    if r.degraded:
        st.warning(f"{t('degraded')}: {r.degraded_reason}")
    st.markdown(f"### {t('answer')}")
    st.markdown(r.answer or "—")

    c1, c2, c3 = st.columns(3)
    c1.metric(t("qtype"), r.qtype)
    c2.metric("graph facts", len(r.graph_facts))
    c3.metric("passages", len(r.text_hits))

    if len(r.queries) > 1:
        with st.expander(f"{t('queries')}（{len(r.queries)}）"):
            for i, q in enumerate(r.queries, 1):
                st.write(f"{i}. {q}")

    if r.graph_facts:
        with st.expander(f"{t('graph_facts')}（{len(r.graph_facts)}）"):
            for f in r.graph_facts:
                st.write("• " + f)

    if r.text_hits:
        with st.expander(f"{t('evidence')}（{len(r.text_hits)}）"):
            for i, h in enumerate(r.text_hits, 1):
                src = h.get("source") or h.get("source_doc") or ""
                st.markdown(f"**[{i}] {src} 第 {h.get('page_idx')} 页** "
                            f"`block_type={h.get('block_type')}` "
                            f"`id={h.get('chunk_id')}` "
                            f"score={h.get('score')}")
                st.text((h.get("text") or "")[:800])

    if r.subgraph and (r.subgraph.get("nodes") or r.subgraph.get("edges")):
        with st.expander(f"知识图谱结构（{len(r.subgraph.get('nodes') or [])} 节点 / "
                         f"{len(r.subgraph.get('edges') or [])} 边）"):
            _render_subgraph(r.subgraph)

    if r.timings:
        with st.expander(t("timings")):
            st.json({k: round(v, 3) for k, v in r.timings.items()})


# ---------------------------------------------------------------------------
# 主界面
# ---------------------------------------------------------------------------
def main() -> None:
    with st.sidebar:
        st.session_state["lang"] = st.selectbox(
            T["zh"]["lang"] + " / UI language", ["zh", "en"], index=0)
        st.divider()
        try:
            st.json(config9.summary())
        except Exception as exc:                       # noqa: BLE001
            st.error(str(exc))

    lang = _lang()
    st.title(t("title"))
    st.caption(t("caption"))

    tab_qa, tab_cmp, tab_metric, tab_status = st.tabs(
        [t("tab_qa"), t("tab_cmp"), t("tab_metric"), t("tab_status")])

    # ---------------- 问答 ----------------
    with tab_qa:
        q = st.text_area(t("ask"), placeholder=t("placeholder"), height=90)
        if st.button(t("run"), type="primary"):
            from w9.runtime9 import InputError, validate_question
            try:
                qs = validate_question(q)
            except InputError as exc:
                st.error(str(exc))
            else:
                with st.spinner(t("running")):
                    _warmup()
                    from w9.qa_v2 import ask_v2
                    try:
                        r = ask_v2(qs)
                        _render_result(r, lang)
                    except Exception as exc:           # noqa: BLE001
                        st.error(f"处理失败：{type(exc).__name__}: {exc}")

        # ---- 上传 PDF 问答（工单用户手册要求覆盖的操作之一）----
        # 复用 08 的 `src/upload_index`（会话级索引，不写入知识图谱与全局语料库），
        # 生成链路沿用 09 的 qa_v2 口径，保证上传文档与主链路的答案风格一致。
        with st.expander("📄 " + ("上传 PDF 并就它提问" if lang == "zh"
                                  else "Upload a PDF and ask about it")):
            up = st.file_uploader("PDF", type=["pdf"], key="pdf_up")
            if up is not None:
                if st.button(t("run") + " (upload)", key="ask_up"):
                    with st.spinner(t("running")):
                        _warmup()
                        try:
                            from src.upload_index import build_index_from_pdf
                            idx = build_index_from_pdf(up.getvalue(), up.name)
                            st.session_state["up_index"] = idx
                            st.success(f"已索引 {idx.n_chunks} 个块：{up.name}")
                        except Exception as exc:     # noqa: BLE001
                            # 容错（工单硬性要求2）：解析失败给出可读提示，不中断会话
                            st.error(f"文档解析失败：{type(exc).__name__}: {exc}")
                            st.session_state.pop("up_index", None)

            idx = st.session_state.get("up_index")
            if idx is not None:
                qu = st.text_input(
                    "就上传文档提问" if lang == "zh" else "Ask about the document",
                    key="up_q")
                if st.button(t("run"), key="ask_up2"):
                    from w9.runtime9 import InputError, validate_question
                    try:
                        qu = validate_question(qu)
                    except InputError as exc:
                        st.error(str(exc))
                    else:
                        with st.spinner(t("running")):
                            try:
                                hits = idx.search(qu, top_k=config9.FINAL9_TOP_K)
                                ctx = "\n\n".join(
                                    f"[{i}] 来源：{h.get('source', '')} "
                                    f"第 {h.get('page_idx')} 页\n{h.get('text', '')}"
                                    for i, h in enumerate(hits, 1))
                                from w9.qa_v2 import _answer_lang, _call_llm, build_messages_v2
                                lg = _answer_lang(qu)
                                msgs = build_messages_v2(qu, ctx, lg, False, "factual")
                                st.markdown(_call_llm(
                                    msgs, temperature=config9.GEN9_TEMPERATURE,
                                    num_predict=config9.GEN9_NUM_PREDICT))
                                with st.expander(t("evidence")):
                                    for i, h in enumerate(hits, 1):
                                        st.markdown(
                                            f"**[{i}] 第 {h.get('page_idx')} 页** "
                                            f"`id={h.get('chunk_id')}`")
                                        st.text((h.get("text") or "")[:600])
                            except Exception as exc:     # noqa: BLE001
                                st.error(f"处理失败：{type(exc).__name__}: {exc}")

    # ---------------- 优化对比 ----------------
    with tab_cmp:
        q2 = st.text_area(t("ask"), key="cmp_q",
                          placeholder=T[lang]["placeholder"], height=90)
        if st.button(t("cmp_run")):
            from w9.runtime9 import InputError, validate_question
            try:
                qs = validate_question(q2)
            except InputError as exc:
                st.error(str(exc))
            else:
                with st.spinner(t("running")):
                    _warmup()
                    from w9 import baseline08
                    from w9.qa_v2 import ask_v2
                    left, right = st.columns(2)
                    with left:
                        st.subheader(t("baseline"))
                        try:
                            t0 = time.perf_counter()
                            r8 = baseline08.ask(qs)
                            st.caption(f"{time.perf_counter() - t0:.2f}s")
                            st.markdown(r8.answer or "—")
                            st.caption(f"graph facts {len(r8.graph_facts)} · "
                                       f"passages {len(r8.text_hits)} · "
                                       f"mode {r8.mode}")
                        except Exception as exc:       # noqa: BLE001
                            st.error(f"{type(exc).__name__}: {exc}")
                    with right:
                        st.subheader(t("optimized"))
                        try:
                            t0 = time.perf_counter()
                            r9 = ask_v2(qs)
                            st.caption(f"{time.perf_counter() - t0:.2f}s")
                            st.markdown(r9.answer or "—")
                            st.caption(f"graph facts {len(r9.graph_facts)} · "
                                       f"passages {len(r9.text_hits)} · "
                                       f"qtype {r9.qtype}")
                        except Exception as exc:       # noqa: BLE001
                            st.error(f"{type(exc).__name__}: {exc}")

    # ---------------- 指标看板 ----------------
    with tab_metric:
        p = os.path.join(config9.EVAL9_DIR, "compare_summary.json")
        if not os.path.isfile(p):
            st.info(t("no_eval"))
        else:
            with io.open(p, encoding="utf-8") as f:
                cmp = json.load(f)
            s08 = cmp["systems"].get("wo08", {})
            s09 = cmp["systems"].get("wo09", {})
            st.subheader("RAGAS（工单核心指标）")
            cols = st.columns(2)
            for col, (key, need) in zip(cols, (("context_precision", 0.8),
                                               ("context_recall", 0.9))):
                b = (s08.get("ragas") or {}).get(key)
                a = (s09.get("ragas") or {}).get(key)
                col.metric(f"{key} （{t('metric_req')} ≥ {need}）",
                           "—" if a is None else f"{a:.4f}",
                           delta=None if (a is None or b is None) else f"{a - b:+.4f}",
                           help=f"{t('baseline')}: {'—' if b is None else f'{b:.4f}'}")
            st.subheader("辅助检索指标")
            st.json({"08": s08.get("aux", {}), "09": s09.get("aux", {})})
            st.subheader("响应时间")
            st.json({"08": {"mean_s": s08.get("time_mean_s"),
                            "max_s": s08.get("time_max_s"),
                            "over_3s": s08.get("n_over_budget")},
                     "09": {"mean_s": s09.get("time_mean_s"),
                            "max_s": s09.get("time_max_s"),
                            "over_3s": s09.get("n_over_budget")}})

    # ---------------- 系统状态 ----------------
    with tab_status:
        if st.button(t("health")):
            from w9.runtime9 import health_check
            with st.spinner("…"):
                st.json(health_check())
        st.subheader(t("graph_scale"))
        try:
            from w9.qa_v2 import get_kg9
            kg = get_kg9()
            if kg is None:
                st.warning("09 图谱未构建。请运行：python -m w9.kg_merge")
            else:
                st.json(kg.stats())
        except Exception as exc:                       # noqa: BLE001
            st.error(f"{type(exc).__name__}: {exc}")

        st.subheader(t("faults"))
        from w9.runtime9 import FAULT_MATRIX
        st.dataframe(FAULT_MATRIX, use_container_width=True)


main()
