# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
模块：中英双语问答界面（Streamlit）

页面：
  1. 问答（Ask）    —— 流式回答 + 引用页码 + 全链路耗时
  2. 对比（Compare）—— 优化后(工单2) / 优化前(工单1) / 纯 LLM 三方对比
  3. 知识库（KB）   —— 库状态、重建、指标

运行：streamlit run app.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import bootstrap  # noqa: E402  —— 必须最先导入（离线 + 栈修复）

import streamlit as st

from src import config

st.set_page_config(page_title="招股说明书智能问答系统 · 工单02", page_icon="📈",
                   layout="wide")

# ---------------------------------------------------------------------------
# i18n
# ---------------------------------------------------------------------------
I18N = {
    "zh": {
        "title": "招股说明书智能问答系统",
        "subtitle": "工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化",
        "tab_ask": "💬 问答",
        "tab_compare": "⚖️ 优化前后对比",
        "tab_kb": "📚 知识库",
        "ask_placeholder": "请输入关于招股说明书的问题，例如：公司注册资本是多少？",
        "ask_btn": "提问",
        "thinking": "检索与生成中…",
        "answer": "回答",
        "citations": "引用来源",
        "timings": "耗时明细",
        "contexts": "检索到的片段",
        "page": "第{page}页",
        "no_answer": "根据已有资料无法确定。",
        "compare_q": "选择对比问题",
        "run_compare": "开始对比",
        "opt_system": "优化后（工单02）",
        "base_system": "优化前（工单01）",
        "llm_only": "纯 LLM（无检索）",
        "time_s": "总耗时",
        "kb_status": "知识库状态",
        "rebuild": "重建知识库",
        "building": "建库中（终端可见进度）…",
        "examples": "示例问题",
        "settings": "参数",
        "rerank": "启用重排",
        "topk": "返回片段数",
        "warmup": "预热模型",
        "warmed": "预热完成",
    },
    "en": {
        "title": "Prospectus QA System",
        "subtitle": "Work Order: NLP-RAG PDF QA Optimization",
        "tab_ask": "💬 Ask",
        "tab_compare": "⚖️ Before / After",
        "tab_kb": "📚 Knowledge Base",
        "ask_placeholder": "Ask a question about the prospectus, e.g. What is the registered capital?",
        "ask_btn": "Ask",
        "thinking": "Retrieving and generating…",
        "answer": "Answer",
        "citations": "Citations",
        "timings": "Timings",
        "contexts": "Retrieved fragments",
        "page": "Page {page}",
        "no_answer": "Cannot be determined from the available materials.",
        "compare_q": "Choose a question",
        "run_compare": "Run comparison",
        "opt_system": "Optimized (WO-02)",
        "base_system": "Baseline (WO-01)",
        "llm_only": "LLM only (no retrieval)",
        "time_s": "Total time",
        "kb_status": "Knowledge base",
        "rebuild": "Rebuild index",
        "building": "Building (see terminal)…",
        "examples": "Examples",
        "settings": "Settings",
        "rerank": "Enable rerank",
        "topk": "Top-K fragments",
        "warmup": "Warm up models",
        "warmed": "Warmup done",
    },
}


@st.cache_resource(show_spinner=False)
def _warmup_once():
    """首次进入即预热（embedding / reranker / BM25 / Ollama），降低首题延迟。"""
    from src import rag
    return rag.warmup()


@st.cache_data(show_spinner=False)
def _load_questions():
    try:
        with open(config.EVAL_QUESTIONS_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return []


def _llm_only_answer(question: str) -> dict:
    """纯 LLM 对照：不检索，直接问模型（工单要求的三方对比之一）。"""
    from src import llm
    t0 = time.time()
    try:
        resp = llm.chat([
            {"role": "system",
             "content": "你是招股说明书问答助手。若不知道确切答案，请回答“不确定”。不要编造。"},
            {"role": "user", "content": question},
        ], num_predict=256)
        return {"answer": resp.get("answer", ""), "elapsed": round(time.time() - t0, 2)}
    except Exception as exc:  # noqa: BLE001
        return {"answer": f"（LLM 不可用：{exc}）", "elapsed": round(time.time() - t0, 2)}


def _render_citations(citations: list[dict], T: dict) -> None:
    if not citations:
        return
    pages = []
    for c in citations:
        p = c.get("page_display") or (int(c.get("page_idx", 0)) + 1)
        pages.append(T["page"].format(page=p))
    st.caption(f"📄 {T['citations']}：{'、'.join(dict.fromkeys(pages))}")


def _render_timings(t: dict) -> None:
    if not t:
        return
    parts = []
    label = {"analyze_ms": "查询理解", "retrieve_ms": "混合检索", "rerank_ms": "重排",
             "context_ms": "上下文", "gen_ms": "生成", "ttft_ms": "首字",
             "total_ms": "总计"}
    for k, name in label.items():
        if k in t and t[k] is not None:
            parts.append(f"{name} {t[k] / 1000:.2f}s")
    if parts:
        st.caption("⏱ " + " | ".join(parts))


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------
def page_ask(T: dict, opts: dict) -> None:
    st.text_input(" ", placeholder=T["ask_placeholder"], key="q_main",
                  label_visibility="collapsed")
    examples = [q["question"] for q in _load_questions()][:5]
    if examples:
        st.caption(T["examples"] + "：")
        cols = st.columns(min(len(examples), 3))
        for i, ex in enumerate(examples[:3]):
            if cols[i].button(ex[:22] + ("…" if len(ex) > 22 else ""), key=f"ex{i}",
                              use_container_width=True):
                st.session_state["q_main"] = ex
                st.rerun()

    q = (st.session_state.get("q_main") or "").strip()
    submitted = st.button(T["ask_btn"], type="primary", disabled=not q)
    if not (submitted and q):
        return

    from src import rag
    with st.spinner(T["thinking"]):
        result = rag.ask(q, use_rerank=opts["rerank"], top_k=opts["top_k"])

    st.markdown(f"### {T['answer']}")
    st.markdown(result.answer or T["no_answer"])
    _render_citations(result.citations, T)
    _render_timings(result.timings)
    with st.expander(T["contexts"], expanded=False):
        for i, b in enumerate(result.contexts, start=1):
            head = " / ".join(b.get("heading_path") or []) or "（无标题）"
            st.markdown(f"**【片段{i}】{T['page'].format(page=int(b.get('page_idx', 0)) + 1)}**"
                        f"　{head}")
            st.text((b.get("parent_text") or "")[:900])


def page_compare(T: dict, opts: dict) -> None:
    questions = _load_questions()
    if not questions:
        st.warning("缺少 data/test_questions.json")
        return
    labels = {f"[{q['id']}] {q['question'][:40]}…": q["question"] for q in questions}
    pick = st.selectbox(T["compare_q"], list(labels.keys()), key="cmp_q")
    question = labels[pick]

    if st.button(T["run_compare"], type="primary"):
        from src import baseline, rag
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"**{T['opt_system']}**")
            with st.spinner(T["thinking"]):
                r = rag.ask(question, use_rerank=opts["rerank"], top_k=opts["top_k"])
            st.write(r.answer)
            _render_citations(r.citations, T)
            _render_timings(r.timings)

        with col2:
            st.markdown(f"**{T['base_system']}**")
            with st.spinner(T["thinking"]):
                b = baseline.ask_baseline(question)
            if b.get("error"):
                st.warning(b["error"])
            else:
                st.write(b.get("answer", ""))
                cites = b.get("citations") or []
                if cites:
                    pages = [T["page"].format(page=int(c.get("page_idx", 0)) + 1)
                             for c in cites]
                    st.caption(f"📄 {T['citations']}：{'、'.join(dict.fromkeys(pages))}")
                st.caption(f"⏱ {T['time_s']} {b.get('wall_s', b.get('timings', {}).get('total_s', '?'))}s")

        with col3:
            st.markdown(f"**{T['llm_only']}**")
            with st.spinner(T["thinking"]):
                lo = _llm_only_answer(question)
            st.write(lo["answer"])
            st.caption(f"⏱ {T['time_s']} {lo['elapsed']}s")


def page_kb(T: dict) -> None:
    from src import rag, vector_store
    st.markdown(f"### {T['kb_status']}")
    try:
        st.json(rag.status())
    except Exception as exc:  # noqa: BLE001
        st.warning(f"状态获取失败：{exc}")

    if st.button(T["rebuild"]):
        st.info(T["building"])
        import subprocess
        subprocess.Popen([sys.executable, os.path.join(config.ROOT, "build_index.py")],
                         cwd=config.ROOT)


def main() -> None:
    lang = st.sidebar.selectbox("Language / 语言", ["zh", "en"],
                                format_func=lambda x: "中文" if x == "zh" else "English")
    T = I18N[lang]
    st.title(T["title"])
    st.caption(T["subtitle"])

    with st.sidebar:
        st.markdown(f"### {T['settings']}")
        opts = {
            "rerank": st.checkbox(T["rerank"], value=True),
            "top_k": st.slider(T["topk"], 3, 10, config.FINAL_TOP_K),
        }
        if st.button(T["warmup"]):
            with st.spinner("…"):
                _warmup_once()
            st.success(T["warmed"])

    tab1, tab2, tab3 = st.tabs([T["tab_ask"], T["tab_compare"], T["tab_kb"]])
    with tab1:
        page_ask(T, opts)
    with tab2:
        page_compare(T, opts)
    with tab3:
        page_kb(T)


if __name__ == "__main__":
    # 说明：Streamlit 的脚本执行发生在工作线程中；bootstrap 已通过
    # threading.stack_size(16MB) 提高了此后所有线程的栈，故此处直接调用即可，
    # 不需要（也不能）用 run_with_large_stack 另起线程包裹。
    main()
