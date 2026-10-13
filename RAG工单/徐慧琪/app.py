# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估
# （上游来源：工单06 混合检索；本工单复用于 ccf_competition 语料的功能测试与评估）
# 工单编号：人工智能NLP-RAG-混合检索任务
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

st.set_page_config(page_title="招股说明书智能问答系统 · 工单06 混合检索",
                   page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# i18n
# ---------------------------------------------------------------------------
I18N = {
    "zh": {
        "title": "招股说明书智能问答系统",
        "subtitle": "工单编号：人工智能NLP-RAG-混合检索任务",
        "tab_ask": "💬 问答",
        "tab_mt": "🧠 多轮对话",
        "tab_strategy": "🔀 检索策略",
        "tab_compare": "⚖️ 优化前后对比",
        "tab_tables": "📊 表格检索",
        "tab_kb": "📚 知识库",
        # --- 工单6：检索策略配置与应用 ---
        "st_title": "检索策略配置与应用（三种策略可切换、可调参）",
        "st_intro": "本页对应工单06 核心：向量检索（召回+重排）/ 全文检索（自研倒排索引）/ "
                    "混合检索（双路融合+重排）。改动立即生效，同一问题可三策略并排对比。",
        "st_strategy": "检索策略",
        "st_topk_recall": "召回候选数 top_k_recall",
        "st_topk_rerank": "精排后返回数 top_k_rerank",
        "st_threshold": "精排分数阈值 Score 阈值",
        "st_weights": "两路权重（仅混合检索生效）",
        "st_vector_w": "向量权重",
        "st_fulltext_w": "全文权重",
        "st_fusion": "融合算法",
        "st_rerank_algo": "精排算法",
        "st_rerank_model": "Reranker 模型",
        "st_embed_model": "嵌入模型",
        "st_apply": "应用配置",
        "st_applied": "配置已应用（下一次提问生效）",
        "st_query": "用于对比/试检索的问题",
        "st_run": "执行检索",
        "st_run3": "三种策略并排对比",
        "st_arms": "召回通路",
        "st_counts": "召回候选数",
        "st_recall_ms": "召回耗时",
        "st_fusion_ms": "融合耗时",
        "st_rerank_ms": "精排耗时",
        "st_nchanged": "精排前后名次变化",
        "st_degraded": "降级",
        "st_gap": "召回阶段候选数",
        "st_beforeafter": "精排前后 Top-K 变化（块 → 前名次 → 后名次）",
        "st_syntax": "全文检索语法：`A AND B`、`A OR B`、`NOT C`、`\"短语匹配\"`、"
                     "`模糊词~`（编辑距离1）、`title:标题字段`、`body:正文字段`、`summary:摘要字段`",
        "st_hint_hybrid": "混合检索：向量 + 全文双路召回 → 融合 → Reranker 精排（推荐）",
        "st_fusion_labels": {"rrf": "RRF 倒数排名融合", "weighted": "加权平均", "vote": "投票(Borda)"},
        "st_algo_labels": {"model": "Reranker 模型（生产默认）", "llm": "LLM 重排器（对比基线）",
                           "tfidf": "TF-IDF 重排器（对比基线）", "adaptive": "用户反馈自适应重排器"},
        "st_rerank_report": "精排报告（工单硬性要求2）",
        "st_no_hits": "无命中",
        "st_current": "当前生效配置",
        "fb_helpful": "👍 有帮助",
        "fb_unhelpful": "👎 没帮助",
        "fb_recorded": "反馈已记录（自适应重排器会用它调整排序）",
        "fb_stats": "用户反馈库",
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
        "subtitle": "Work Order: NLP-RAG Hybrid Retrieval",
        "tab_ask": "💬 Ask",
        "tab_mt": "🧠 Multi-turn",
        "tab_strategy": "🔀 Retrieval Strategy",
        "tab_compare": "⚖️ Before / After",
        "tab_tables": "📊 Tables",
        "tab_kb": "📚 Knowledge Base",
        # --- 工单6：检索策略配置与应用 ---
        "st_title": "Retrieval Strategy Configuration (3 switchable strategies)",
        "st_intro": "Work Order 06 core: Vector retrieval (recall + rerank) / Full-text retrieval "
                    "(self-built inverted index) / Hybrid retrieval (dual-path fusion + rerank). "
                    "Changes take effect immediately; compare all three on the same question.",
        "st_strategy": "Strategy",
        "st_topk_recall": "Recall candidates (top_k_recall)",
        "st_topk_rerank": "Returned after rerank (top_k_rerank)",
        "st_threshold": "Rerank score threshold",
        "st_weights": "Path weights (hybrid only)",
        "st_vector_w": "Vector weight",
        "st_fulltext_w": "Full-text weight",
        "st_fusion": "Fusion method",
        "st_rerank_algo": "Rerank algorithm",
        "st_rerank_model": "Reranker model",
        "st_embed_model": "Embedding model",
        "st_apply": "Apply configuration",
        "st_applied": "Configuration applied (effective from the next query)",
        "st_query": "Question for comparison / trial search",
        "st_run": "Run retrieval",
        "st_run3": "Compare three strategies",
        "st_arms": "Recall paths",
        "st_counts": "Recall counts",
        "st_recall_ms": "Recall",
        "st_fusion_ms": "Fusion",
        "st_rerank_ms": "Rerank",
        "st_nchanged": "Rank changes after rerank",
        "st_degraded": "Degraded",
        "st_gap": "Candidates before rerank",
        "st_beforeafter": "Top-K changes (chunk → rank before → rank after)",
        "st_syntax": "Full-text syntax: `A AND B`, `A OR B`, `NOT C`, `\"exact phrase\"`, "
                     "`typo~` (edit distance 1), `title:...`, `body:...`, `summary:...`",
        "st_hint_hybrid": "Hybrid: vector + full-text recall → fusion → Reranker (recommended)",
        "st_fusion_labels": {"rrf": "RRF (reciprocal rank fusion)", "weighted": "Weighted average",
                             "vote": "Voting (Borda count)"},
        "st_algo_labels": {"model": "Reranker model (production default)",
                           "llm": "LLM reranker (baseline)", "tfidf": "TF-IDF reranker (baseline)",
                           "adaptive": "User-feedback adaptive reranker"},
        "st_rerank_report": "Rerank report (hard requirement #2)",
        "st_no_hits": "No hits",
        "st_current": "Current configuration",
        "fb_helpful": "👍 Helpful",
        "fb_unhelpful": "👎 Not helpful",
        "fb_recorded": "Feedback recorded (the adaptive reranker will use it)",
        "fb_stats": "Feedback store",
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


def _render_rerank_info(analysis: dict, T: dict) -> None:
    """工单硬性要求2：展示「召回候选数 / 精排耗时 / 精排前后 Top-K 命中变化」。"""
    info = (analysis or {}).get("rerank_info") or {}
    if not info:
        return
    algo = info.get("algorithm")
    model = info.get("model") or ""
    label = {"model": "Reranker 模型", "llm": "LLM 重排器",
             "tfidf": "TF-IDF 重排器", "adaptive": "自适应重排器",
             "off": "未精排", "deferred": "分阶段"}.get(algo, algo)
    parts = [f"精排算法 {label}"]
    if algo == "model" and model:
        parts.append(f"模型 {model}")
    parts += [f"召回候选 {info.get('n_in', '-')}",
              f"精排耗时 {info.get('elapsed_ms', '-')}ms",
              f"名次变化 {info.get('n_changed', 0)} 条"]
    if info.get("degraded"):
        parts.append(f"⚠️ 降级：{info.get('degrade_reason') or '是'}")
    st.caption("🎯 " + " ｜ ".join(parts))

    changes = info.get("rank_changes") or []
    if changes:
        with st.expander(T["st_beforeafter"], expanded=False):
            rows = [{"chunk": c["chunk"], "before": c["before"], "after": c["after"],
                     "delta": c["delta"]} for c in changes[:30]]
            st.dataframe(rows, use_container_width=True)


def _render_feedback(question: str, result, T: dict) -> None:
    """用户反馈（👍/👎）：写入反馈库，供自适应重排器学习（工单 PDF 要求的三算法之一）。"""
    key = f"fb_{abs(hash(question)) % 10**8}"
    c1, c2, c3 = st.columns([1, 1, 4])
    if c1.button(T["fb_helpful"], key=key + "_up"):
        _record_feedback(question, result, True)
        st.success(T["fb_recorded"])
    if c2.button(T["fb_unhelpful"], key=key + "_down"):
        _record_feedback(question, result, False)
        st.success(T["fb_recorded"])
    try:
        from src import rerankers
        s = rerankers.feedback_stats()
        if s["n"]:
            c3.caption(f"{T['fb_stats']}：{s['n']} 条（有用 {s['helpful']} / "
                       f"无用 {s['unhelpful']}）")
    except Exception:  # noqa: BLE001 —— 反馈统计失败不影响主流程
        pass


def _record_feedback(question: str, result, helpful: bool) -> None:
    """把反馈绑定到「答案实际引用的块」——这些块才是排序该被奖惩的对象。"""
    from src import rerankers, runtime

    cfg = runtime.get()
    cit = (result.citations or [None])[0] if result.citations else None
    hit = {
        "source": (cit or {}).get("source", ""),
        "chunk_id": (result.analysis or {}).get("top_chunk_id", ""),
        "page_idx": (cit or {}).get("page_idx", 0),
        "block_type": (cit or {}).get("block_type", ""),
    }
    if not hit["chunk_id"]:
        top = (result.contexts or [{}])[0]
        hit["chunk_id"] = top.get("chunk_id", "")
        hit["source"] = hit["source"] or top.get("source", "")
        hit["page_idx"] = hit["page_idx"] or top.get("page_idx", 0)
    rerankers.record_feedback(question, hit, helpful,
                              strategy=cfg.strategy,
                              reranker=cfg.reranker_model)


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------
def page_strategy(T: dict, opts: dict) -> None:
    """工单06 核心页：三种检索策略的配置与应用（含三策略并排对比）。"""
    from src import config as cfg_mod
    from src import rerankers, runtime, strategies

    st.markdown(f"### {T['st_title']}")
    st.caption(T["st_intro"])

    desc = strategies.describe()
    cur = runtime.get()

    # ---------------- 配置区 ----------------
    with st.expander(T["st_current"], expanded=True):
        c1, c2, c3 = st.columns(3)
        strat = c1.selectbox(
            T["st_strategy"], list(desc["strategies"]),
            index=list(desc["strategies"]).index(cur.strategy),
            format_func=lambda k: desc["strategies"][k])
        topk_recall = c2.slider(T["st_topk_recall"], 5, 60, cur.top_k_recall, step=5)
        topk_rerank = c3.slider(T["st_topk_rerank"], 1, 20, cur.top_k_rerank)

        c4, c5, c6 = st.columns(3)
        threshold = c4.slider(T["st_threshold"], 0.0, 0.9, float(cur.score_threshold), 0.05)
        fusion = c5.selectbox(
            T["st_fusion"], list(desc["fusion_methods"]),
            index=list(desc["fusion_methods"]).index(cur.fusion),
            format_func=lambda k: T["st_fusion_labels"].get(k, k),
            disabled=(strat != "hybrid"))
        algo = c6.selectbox(
            T["st_rerank_algo"], list(desc["rerank_algorithms"]),
            index=list(desc["rerank_algorithms"]).index(cur.rerank_algorithm),
            format_func=lambda k: T["st_algo_labels"].get(k, k))

        w1, w2, w3 = st.columns(3)
        vw = w1.slider(T["st_vector_w"], 0.0, 1.0, float(cur.vector_weight), 0.05,
                       disabled=(strat != "hybrid"))
        fw = w2.slider(T["st_fulltext_w"], 0.0, 1.0, float(cur.fulltext_weight), 0.05,
                       disabled=(strat != "hybrid"))
        rmodel = w3.selectbox(
            T["st_rerank_model"], list(desc["reranker_models"]),
            index=list(desc["reranker_models"]).index(cur.reranker_model),
            format_func=lambda k: f"{k} — {desc['reranker_models'][k]}")

        emodel = st.selectbox(
            T["st_embed_model"], list(desc["embedding_models"]),
            index=list(desc["embedding_models"]).index(cur.embedding_model),
            format_func=lambda k: f"{k} — {desc['embedding_models'][k]}")

        if st.button(T["st_apply"], type="primary"):
            try:
                runtime.update(strategy=strat, top_k_recall=topk_recall,
                               top_k_rerank=topk_rerank, score_threshold=threshold,
                               fusion=fusion, rerank_algorithm=algo,
                               reranker_model=rmodel, embedding_model=emodel,
                               vector_weight=vw, fulltext_weight=fw)
                st.success(T["st_applied"])
            except Exception as exc:  # noqa: BLE001 —— 非法取值给出明确提示
                st.error(f"{exc}")

    st.caption(T["st_syntax"])
    if strat == "hybrid":
        st.caption(T["st_hint_hybrid"])

    # ---------------- 试检索 / 三策略对比 ----------------
    st.divider()
    examples = [q["question"] for q in _load_questions()]
    default_q = (examples[0] if examples else
                 "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？")
    # 输入框与示例按钮共用同一个 session key：按钮写入后 st.rerun()，
    # 输入框随即可见新值（不要用两个 key，否则 value= 会被旧控件状态覆盖）。
    if "q_strategy" not in st.session_state:
        st.session_state["q_strategy"] = default_q
    q = st.text_input(T["st_query"], key="q_strategy")
    if q and examples[:3]:
        cols = st.columns(3)
        for i, ex in enumerate(examples[:3]):
            if cols[i].button(ex[:24] + "…", key=f"stex{i}", use_container_width=True):
                st.session_state["q_strategy"] = ex
                st.rerun()

    b1, b2 = st.columns([1, 1])
    run_one = b1.button(T["st_run"], disabled=not q)
    run_all = b2.button(T["st_run3"], type="primary", disabled=not q)

    if run_one and q:
        with st.spinner(T["thinking"]):
            res = strategies.retrieve(q, runtime.get())
        _render_strategy_result(res, T, expanded=True)

    if run_all and q:
        cols = st.columns(3)
        for i, s in enumerate(("vector", "fulltext", "hybrid")):
            with cols[i]:
                st.markdown(f"#### {desc['strategies'][s]}")
                cfg_s = strategies.with_overrides(runtime.get(), strategy=s)
                with st.spinner("…"):
                    res = strategies.retrieve(q, cfg_s)
                _render_strategy_result(res, T, expanded=False)

    with st.expander("🧪 全文检索语法演示（布尔 / 短语 / 模糊 / 多字段）", expanded=False):
        for demo in ['注册资本 AND 法定代表人',
                     '"军用领域" AND 收入',
                     '(武汉博润 OR 上海博润) AND 关联方',
                     '法定代理人~',
                     'title:组织结构图']:
            st.code(demo, language="text")
        st.caption("把上面任一表达式粘贴到上方输入框并选择「全文检索」策略即可验证。")


def _render_strategy_result(res: dict, T: dict, expanded: bool = True) -> None:
    """渲染一次检索的策略观测信息 + 命中片段（召回/融合/精排三段耗时）。"""
    info = res.get("info") or {}
    hits = res.get("hits") or []
    st.caption(
        f"{T['st_arms']}：{'、'.join(info.get('arms') or ['-'])} ｜ "
        f"{T['st_gap']}：{info.get('n_candidates', '-')} ｜ "
        f"{T['st_recall_ms']} {info.get('recall_ms', '-')}ms"
        + (f" ｜ {T['st_fusion_ms']} {info.get('fusion_ms', '-')}ms"
           if info.get("fusion_ms") else "")
        + f" ｜ {T['st_rerank_ms']} {info.get('rerank_ms', '-')}ms ｜ "
        f"{T['st_nchanged']} {info.get('rerank', {}).get('n_changed', 0)}"
    )
    if info.get("rerank", {}).get("degraded"):
        st.warning(f"{T['st_degraded']}：{info['rerank'].get('degrade_reason') or '是'}")
    if not hits:
        st.info(T["st_no_hits"])
        return
    for i, h in enumerate(hits, start=1):
        src = (h.get("source") or "").replace(".pdf", "")
        page = int(h.get("page_idx", 0)) + 1
        bt = h.get("block_type") or ""
        badges = [f"{src} {T['page'].format(page=page)}", bt]
        if h.get("table_id"):
            badges.append(f"{T['table_badge']}:{h['table_id']}")
        if h.get("image_id"):
            badges.append(f"图像:{h['image_id']}")
        score_bits = []
        for k, name in (("rerank_score", "rerank"), ("fused_score", "fused"),
                        ("vector_score", "vec"), ("fulltext_score", "fulltext")):
            if h.get(k) is not None:
                score_bits.append(f"{name}={float(h[k]):.4f}")
        st.markdown(f"**[{i}] {' ｜ '.join(badges)}**  " + " ".join(score_bits))
        st.text((h.get("text") or "")[:300].replace("\n", " "))
        if h.get("image_path") and os.path.isfile(os.path.join(config.ROOT, h["image_path"])):
            if expanded:
                st.image(os.path.join(config.ROOT, h["image_path"]), width=520,
                         caption=h.get("image_id", ""))


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
    _render_rerank_info(result.analysis, T)
    _render_timings(result.timings)
    _render_feedback(q, result, T)

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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [T["tab_ask"], T["tab_mt"], T["tab_strategy"], T["tab_compare"],
         T["tab_tables"], T["tab_kb"]])
    with tab1:
        page_ask(T, opts)
    with tab2:
        page_multiturn(T, opts)
    with tab3:
        page_strategy(T, opts)
    with tab4:
        page_compare(T, opts)
    with tab5:
        page_tables(T)
    with tab6:
        page_kb(T)


if __name__ == "__main__":
    # 说明：Streamlit 的脚本执行发生在工作线程中；bootstrap 已通过
    # threading.stack_size(16MB) 提高了此后所有线程的栈，故此处直接调用即可，
    # 不需要（也不能）用 run_with_large_stack 另起线程包裹。
    main()
