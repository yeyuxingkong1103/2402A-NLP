# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-Query理解优化
         （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
模块：中英双语问答界面（Streamlit）

页面：
  1. 问答（Ask）    —— 流式回答 + 引用页码 + **表格定位**（哪个PDF/哪页/哪个表/哪行）+ 全链路耗时
  2. 多轮对话（Multi-turn）—— 工单5 核心：连续追问，「他 / 这个公司 / 那 XX 呢」自动消解，
                              并把**改写过程**摊开展示（原问题 → 独立问题 + 改写方式）
  3. 对比（Compare）—— 03系统（表格优化）/ 01·02系统（无表格优化）/ 纯 LLM 三方对比
  4. 表格（Tables） —— 表格库状态、按文档/表名的表格浏览（工单3 专项）
  5. 知识库（KB）   —— 库状态、重建、指标

运行：streamlit run app.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import bootstrap  # noqa: E402  —— 必须最先导入（离线 + 栈修复）

import streamlit as st

from src import config, dialogue

st.set_page_config(page_title="招股说明书智能问答系统 · 工单05", page_icon="📈",
                   layout="wide")

# ---------------------------------------------------------------------------
# i18n
# ---------------------------------------------------------------------------
I18N = {
    "zh": {
        "title": "招股说明书智能问答系统",
        "subtitle": "工单编号：人工智能NLP-RAG-Query理解优化",
        "tab_ask": "💬 问答",
        "tab_mt": "🧠 多轮对话",
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
        "nearby_table": "同页表格",
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
        # --- 多轮对话（工单5 核心页）---
        "mt_intro": "连续追问即可：系统会自动理解「他」「这个公司」「那 XX 呢」指的是哪家公司，"
                    "并把改写过程摊开显示。",
        "mt_input": "请输入追问，例如：这个公司的法定代表人是谁？",
        "mt_rewrite": "Query 理解（改写过程）",
        "mt_raw": "原问题",
        "mt_resolved": "独立问题",
        "mt_mode": "改写方式",
        "mt_entity": "指代实体",
        "mt_clear": "清空会话",
        "mt_regen": "重新生成",
        "mt_fb_up": "👍 有帮助",
        "mt_fb_down": "👎 没帮助",
        "mt_fb_thanks": "已记录反馈",
        "mt_unresolved": "指代无法解析：请补充公司名称后再问。",
        "mt_empty": "还没有对话。输入一个公司名开始，然后用「他」「这个公司」追问。",
        "mt_examples": "示例问法（点一下发起会话）",
        "mt_run_acceptance": "▶ 一键跑完工单验收的 5 轮序列",
        "mt_acceptance_hint": "工单要求的 5 轮连续对话（Q1→Q5），会清空当前会话后按序提问，"
                              "耗时约 7 秒。",
        "mt_turns": "本会话轮次",
    },
    "en": {
        "title": "Prospectus QA System",
        "subtitle": "Work Order: NLP-RAG Query Understanding Optimization",
        "tab_ask": "💬 Ask",
        "tab_mt": "🧠 Multi-turn",
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
        "nearby_table": "table on the same page",
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
        # --- Multi-turn chat (work-order 5 core page) ---
        "mt_intro": "Just keep asking follow-ups — pronouns like “he”, “this company” or "
                    "“what about X” are resolved automatically, and the rewrite is shown inline.",
        "mt_input": "Ask a follow-up, e.g. Who is the legal representative of this company?",
        "mt_rewrite": "Query understanding (rewrite)",
        "mt_raw": "Original",
        "mt_resolved": "Standalone",
        "mt_mode": "Rewrite mode",
        "mt_entity": "Resolved entity",
        "mt_clear": "Clear chat",
        "mt_regen": "Regenerate",
        "mt_fb_up": "👍 Helpful",
        "mt_fb_down": "👎 Not helpful",
        "mt_fb_thanks": "Feedback recorded",
        "mt_unresolved": "Could not resolve the reference — please name the company.",
        "mt_empty": "No conversation yet. Start with a company name, then follow up with "
                    "“he” / “this company”.",
        "mt_examples": "Example openers (click to start a session)",
        "mt_run_acceptance": "▶ Run the work order's 5-turn acceptance sequence",
        "mt_acceptance_hint": "The five consecutive questions required by the work order "
                              "(Q1→Q5). Clears the current chat, then asks them in order; "
                              "takes about 7 seconds.",
        "mt_turns": "Turns in this session",
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
        # 同页表格定位（工单5）：表格与正文数字冲突时表不入上下文（见
        # rag._nearby_table_notes），但这里告诉读者「表格在哪、哪几行命中」——
        # 需求文档要求 Q1 必须能溯源到收入表格。
        nt = c.get("nearby_table")
        if nt:
            cap = nt.get("table_caption") or nt.get("table_id") or ""
            rows = nt.get("row_indices") or []
            extra = f"｜{T['nearby_table']}：{cap}" if cap else f"｜{T['nearby_table']}"
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


def page_multiturn(T: dict, opts: dict) -> None:
    """多轮对话页（工单5 核心）。

    展示面 = `page_ask` 的答案/引用/路由/耗时四件，**外加改写过程**：
    多轮的价值就体现在「他」被换成了哪家公司，不摊开就没有说服力（演示视频的关键镜头）。

    会话隔离：`DialogueState` 存在 `src/dialogue` 的**进程级**表里，
    若所有浏览器会话共用一个固定 id，两个标签页会串味，故每个 Streamlit 会话
    生成独立 id（见 `_mt_state`）。
    """
    from src import rag

    st.header(T["tab_mt"])
    st.caption(T["mt_intro"])

    state = _mt_state()

    col_a, col_b, col_c = st.columns([1, 1, 4])
    if col_a.button(T["mt_clear"], key="mt_clear", disabled=not state.turns):
        dialogue.reset_state(state.session_id)
        st.session_state.pop("mt_feedback", None)
        st.rerun()
    if col_b.button(T["mt_regen"], key="mt_regen", disabled=not state.turns):
        # 回退一轮后重问：焦点与 last_aspect 必须从**剩余轮次**重建，
        # 否则被丢弃那一轮的焦点会残留，Q4 式的省略补全会用错模板。
        dropped = state.turns.pop()
        last = state.turns[-1] if state.turns else None
        state.focus_company = (last.focus_company or None) if last else None
        state.last_aspect = (last.aspect or None) if last else None
        st.session_state["mt_pending"] = dropped.raw_question
        st.rerun()
    col_c.caption(f"{T['mt_turns']}：{state.turn_count}")

    if not state.turns:
        st.info(T["mt_empty"])
        _mt_examples(T)

    feedback = st.session_state.setdefault("mt_feedback", {})
    for i, turn in enumerate(state.turns):
        with st.chat_message("user"):
            st.markdown(turn.raw_question)
        with st.chat_message("assistant"):
            if turn.rewrite_mode == "unresolved":
                st.warning(T["mt_unresolved"])
            st.markdown(turn.answer or T["no_answer"])
            _mt_show_rewrite(T, turn)
            _render_citations(turn.citations, T)
            _render_timings(turn.timings)
            c1, c2, c3 = st.columns([1, 1, 6])
            if c1.button(T["mt_fb_up"], key=f"mt_up_{i}"):
                feedback[i] = "up"
                st.toast(T["mt_fb_thanks"])
            if c2.button(T["mt_fb_down"], key=f"mt_down_{i}"):
                feedback[i] = "down"
                st.toast(T["mt_fb_thanks"])
            if feedback.get(i) == "up":
                c3.caption("👍")
            elif feedback.get(i) == "down":
                c3.caption("👎")

    # 「重新生成」/「一键跑验收序列」把待问问题放进 session_state，本轮直接复用
    # （chat_input 已被清空）。batch 非空时按序一次问完（验收序列 Q1→Q5 必须带上下文，
    # 不能拆成五次点击，否则 Q2 的「他」没有前文可指）。
    pending = st.session_state.pop("mt_pending", None)
    batch = st.session_state.pop("mt_batch", None)
    typed = st.chat_input(T["mt_input"], key="mt_chat_input")
    questions = batch or ([pending or typed] if (pending or typed) else [])
    if not questions:
        return
    questions = [q.strip() for q in questions if (q or "").strip()]
    if not questions:
        return

    for question in questions:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner(T["thinking"]):
                result, turn = rag.ask_turn(question, state, use_rerank=opts["rerank"])
            if turn.rewrite_mode == "unresolved":
                st.warning(T["mt_unresolved"])
            st.markdown(result.answer or T["no_answer"])
            _mt_show_rewrite(T, turn)
            _render_citations(result.citations, T)
            route = (result.analysis or {}).get("doc_route") or {}
            if route.get("reason"):
                st.caption(f"🧭 {T['route']}：{route['reason']}")
            _render_timings(result.timings)
    # 回答已写进 DialogueState，重跑后用上面的历史渲染分支重新画出（避免双份气泡）
    st.rerun()


def _acceptance_questions() -> list[str]:
    """工单验收序列的 5 个问题（单一来源：data/acceptance_dialogue.json）。

    从数据文件读而不是在 UI 里再抄一份——验收题改了不存在"界面还是老的"这种可能。
    """
    path = os.path.join(config.ROOT, "data", "acceptance_dialogue.json")
    try:
        with open(path, encoding="utf-8") as fh:
            spec = json.load(fh)
        return [t["question"] for t in spec.get("turns", []) if t.get("question")]
    except Exception:                                       # noqa: BLE001
        return []


def _mt_examples(T: dict) -> None:
    """空会话时的入口：单题起手，或一键跑完工单要求的 5 轮序列。"""
    starters = ["武汉兴图新科电子股份有限公司的法定代表人是谁？",
                "武汉力源信息技术股份有限公司的注册资本是多少？",
                "武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"]
    st.caption(T["mt_examples"] + "：")
    cols = st.columns(len(starters))
    for i, (col, q) in enumerate(zip(cols, starters)):
        if col.button(q[:18] + "…", key=f"mt_ex{i}", use_container_width=True):
            st.session_state["mt_pending"] = q
            st.rerun()

    seq = _acceptance_questions()
    if seq and st.button(T["mt_run_acceptance"], key="mt_run_acceptance", type="primary"):
        # 先清空：Q2 的「他」要靠 Q1 建立的焦点，混在旧会话里指代就可能指错
        dialogue.reset_state(_mt_state().session_id)
        st.session_state.pop("mt_feedback", None)
        st.session_state["mt_batch"] = seq
        st.rerun()
    if seq:
        st.caption(T["mt_acceptance_hint"])


def _mt_state() -> dialogue.DialogueState:
    """取（或建）**本浏览器会话**的 DialogueState。

    注意不能给所有会话用同一个 id：`dialogue` 的会话表是进程级的，
    固定 id 会让两个标签页共用一段对话历史。
    """
    sid = st.session_state.get("mt_session_id")
    if not sid:
        sid = f"ui-mt-{uuid.uuid4().hex[:8]}"
        st.session_state["mt_session_id"] = sid
    return dialogue.get_state(sid)


def _mt_show_rewrite(T: dict, turn: dialogue.Turn) -> None:
    """改写过程可见——演示视频里体现 Query 理解的关键。直通轮不占版面。"""
    if turn.rewrite_mode == "passthrough":
        return
    with st.expander(T["mt_rewrite"], expanded=turn.rewrite_mode == "unresolved"):
        st.markdown(f"**{T['mt_raw']}**：{turn.raw_question}")
        if turn.resolved_question:
            st.markdown(f"**{T['mt_resolved']}**：{turn.resolved_question}")
        span = turn.coref_span or {}
        st.caption(f"{T['mt_mode']}：`{turn.rewrite_mode}`"
                   + (f"　|　{T['mt_entity']}：{span.get('entity') or '—'}"
                      if span else ""))


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

    tab1, tab2, tab3, tab4, tab5 = st.tabs([T["tab_ask"], T["tab_mt"], T["tab_compare"],
                                            T["tab_tables"], T["tab_kb"]])
    with tab1:
        page_ask(T, opts)
    with tab2:
        page_multiturn(T, opts)
    with tab3:
        page_compare(T, opts)
    with tab4:
        page_tables(T)
    with tab5:
        page_kb(T)


if __name__ == "__main__":
    # 说明：Streamlit 的脚本执行发生在工作线程中；bootstrap 已通过
    # threading.stack_size(16MB) 提高了此后所有线程的栈，故此处直接调用即可，
    # 不需要（也不能）用 run_with_large_stack 另起线程包裹。
    main()
