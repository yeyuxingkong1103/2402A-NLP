# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
模块：中英双语问答界面（Streamlit）

页面：
  1. 问答（Ask）    —— 流式回答 + 引用页码 + **表格定位**（哪个PDF/哪页/哪个表/哪行）+ 全链路耗时
  2. 对比（Compare）—— 03系统（表格优化）/ 01·02系统（无表格优化）/ 纯 LLM 三方对比
  3. 表格（Tables） —— 表格库状态、按文档/表名的表格浏览（工单3 专项）
  4. 知识库（KB）   —— 库状态、重建、指标

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

st.set_page_config(page_title="招股说明书智能问答系统 · 工单03", page_icon="📈",
                   layout="wide")

# ---------------------------------------------------------------------------
# i18n
# ---------------------------------------------------------------------------
I18N = {
    "zh": {
        "title": "招股说明书智能问答系统",
        "subtitle": "工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化",
        "tab_ask": "💬 问答",
        "tab_compare": "⚖️ 优化前后对比",
        "tab_tables": "📊 表格检索",
        "tab_kb": "📚 知识库",
        "ask_placeholder": "请输入关于招股说明书的问题，例如：本次发行股数是多少，占发行后总股本的比例是多少？",
        "ask_btn": "提问",
        "thinking": "检索与生成中…",
        "answer": "回答",
        "citations": "引用来源",
        "timings": "耗时明细",
        "contexts": "检索到的片段",
        "page": "第{page}页",
        "table_badge": "表格",
        "no_answer": "根据已有资料无法确定。",
        "route": "文档路由",
        "compare_q": "选择对比问题",
        "run_compare": "开始对比",
        "opt_system": "工单03（表格优化）",
        "base_system": "工单01/02（无表格优化）",
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
        "table_browse": "浏览已入库表格",
        "table_pick_doc": "选择文档",
        "table_pick": "选择表格",
        "table_source": "来源",
        "table_page": "页码",
        "table_mode": "表型",
        "table_quality": "质量分",
        "table_rows": "行数",
        "table_header": "表头/字段",
        "table_body": "表格内容",
        "table_none": "还没有表格结构化产物，请先运行 python build_index.py",
    },
    "en": {
        "title": "Prospectus QA System",
        "subtitle": "Work Order: NLP-RAG PDF Table Parsing & Retrieval Optimization",
        "tab_ask": "💬 Ask",
        "tab_compare": "⚖️ Before / After",
        "tab_tables": "📊 Tables",
        "tab_kb": "📚 Knowledge Base",
        "ask_placeholder": "Ask a question about the prospectuses, e.g. How many shares are issued and what percentage of the post-issuance total share capital is that?",
        "ask_btn": "Ask",
        "thinking": "Retrieving and generating…",
        "answer": "Answer",
        "citations": "Citations",
        "timings": "Timings",
        "contexts": "Retrieved fragments",
        "page": "Page {page}",
        "table_badge": "Table",
        "no_answer": "Cannot be determined from the available materials.",
        "route": "Document routing",
        "compare_q": "Choose a question",
        "run_compare": "Run comparison",
        "opt_system": "WO-03 (table-optimized)",
        "base_system": "WO-01/02 (no table optimization)",
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
        "table_browse": "Browse indexed tables",
        "table_pick_doc": "Document",
        "table_pick": "Table",
        "table_source": "Source",
        "table_page": "Page",
        "table_mode": "Mode",
        "table_quality": "Quality",
        "table_rows": "Rows",
        "table_header": "Header / fields",
        "table_body": "Table content",
        "table_none": "No table artifact yet — run `python build_index.py` first.",
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


@st.cache_data(show_spinner=False)
def _load_tables():
    """读取表格结构化产物 data/tables/*_tables.json（按文档分组）。"""
    out: dict[str, list[dict]] = {}
    if not os.path.isdir(config.TABLE_DIR):
        return out
    for name in sorted(os.listdir(config.TABLE_DIR)):
        if not name.endswith("_tables.json"):
            continue
        try:
            with open(os.path.join(config.TABLE_DIR, name), encoding="utf-8") as fh:
                out[name[:-len("_tables.json")]] = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
    return out


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
    """引用来源：多文档下必须显示**哪份 PDF**；表格片段额外给出表格定位。"""
    if not citations:
        return
    items = []
    for c in citations:
        p = c.get("page_display") or (int(c.get("page_idx", 0)) + 1)
        src = (c.get("source") or "").replace(".pdf", "")
        label = f"{src} {T['page'].format(page=p)}"
        if c.get("is_table"):
            cap = c.get("table_caption") or c.get("table_id") or ""
            rows = c.get("row_indices") or []
            extra = f"｜{T['table_badge']}：{cap}" if cap else f"｜{T['table_badge']}"
            if rows:
                extra += f"（第 {', '.join(str(r + 1) for r in rows)} 行）"
            label += extra
        items.append(label)
    st.caption(f"📄 {T['citations']}：" + "；".join(dict.fromkeys(items)))


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

    route = (result.analysis or {}).get("doc_route") or {}
    if route.get("reason"):
        st.caption(f"🧭 {T['route']}：{route['reason']}")
    _render_timings(result.timings)

    with st.expander(T["contexts"], expanded=False):
        for i, b in enumerate(result.contexts, start=1):
            head = " / ".join(b.get("heading_path") or []) or "（无标题）"
            src = (b.get("source") or "").replace(".pdf", "")
            badge = ""
            if b.get("is_table"):
                badge = f"　📊 {T['table_badge']}：{b.get('table_caption') or b.get('table_id') or ''}"
            st.markdown(f"**【片段{i}】{src} {T['page'].format(page=int(b.get('page_idx', 0)) + 1)}**"
                        f"　{head}{badge}")
            st.text((b.get("parent_text") or "")[:1200])


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


def page_tables(T: dict) -> None:
    """工单3 专项：浏览已入库的表格（验证表格解析与结构化效果）。"""
    st.markdown(f"### {T['table_browse']}")
    tables = _load_tables()
    if not tables:
        st.info(T["table_none"])
        return

    total = sum(len(v) for v in tables.values())
    low = sum(1 for v in tables.values() for r in v if r.get("low_quality"))
    c1, c2, c3 = st.columns(3)
    c1.metric("文档数", len(tables))
    c2.metric("表格总数", total)
    c3.metric("低质量表", low)

    doc = st.selectbox(T["table_pick_doc"], list(tables.keys()), key="tbl_doc")
    recs = tables[doc]
    if not recs:
        return
    options = {f"[p{r['page_display']}] {r.get('caption') or r['table_id']}": i
               for i, r in enumerate(recs)}
    pick = st.selectbox(T["table_pick"], list(options.keys()), key="tbl_pick")
    rec = recs[options[pick]]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(T["table_source"], rec.get("source", doc))
    c2.metric(T["table_page"], rec.get("page_display"))
    c3.metric(T["table_mode"], rec.get("mode", "-"))
    c4.metric(T["table_rows"], len(rec.get("rows") or []))

    st.caption(f"{T['table_quality']}：{rec.get('quality')}　|　table_id：{rec.get('table_id')}")
    head = rec.get("header") or rec.get("fields") or []
    if head:
        st.markdown(f"**{T['table_header']}**：{'｜'.join(head)}")
    rows = rec.get("rows") or []
    if rows:
        st.markdown(f"**{T['table_body']}**")
        st.dataframe(rows[:60], use_container_width=True, height=320)


def page_kb(T: dict) -> None:
    from src import rag
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

    tab1, tab2, tab3, tab4 = st.tabs([T["tab_ask"], T["tab_compare"],
                                      T["tab_tables"], T["tab_kb"]])
    with tab1:
        page_ask(T, opts)
    with tab2:
        page_compare(T, opts)
    with tab3:
        page_tables(T)
    with tab4:
        page_kb(T)


if __name__ == "__main__":
    # 说明：Streamlit 的脚本执行发生在工作线程中；bootstrap 已通过
    # threading.stack_size(16MB) 提高了此后所有线程的栈，故此处直接调用即可，
    # 不需要（也不能）用 run_with_large_stack 另起线程包裹。
    main()
