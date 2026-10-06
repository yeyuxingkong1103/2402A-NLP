# -*- coding: utf-8 -*-
# 【Streamlit交互界面 · app.py】检索模式/融合方式/双路权重/重排器热切换，证据可点采纳/不采纳，中英双语
# 工单编号：人工智能NLP-RAG-混合检索任务

"""启动方式：

    streamlit run app.py

功能（对应验收“交互友好性：清晰简洁界面、支持多轮对话和用户反馈、中英文问答”）：
- 侧边栏热配：检索模式（向量/全文/混合）、融合方式（加权平均/投票/RRF）、
  双路权重滑块、重排器（LLM/TF-IDF/用户反馈自适应）、嵌入后端展示、中英文切换；
- 主区展示答案、Top-5 证据（来源招股书、页码、章节、双路名次、重排分）、
  阶段耗时与 3 秒验收线状态；
- 每条证据提供“采纳 / 不采纳”按钮，反馈实时持久化到 feedback.jsonl，
  切换“自适应重排”后立即参与排序，形成反馈闭环；
- 会话内保留多轮问答历史。
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG, RT_CONFIG
from embeddings import create_embedder
from feedback_store import FeedbackStore
from qa_engine import QAEngine
from retriever import HybridRetriever
from vector_store import IndexStore

# 双语文案
I18N = {
    "zh": {
        "title": "招股说明书混合检索问答系统（工单六）",
        "lang": "界面语言 Language",
        "mode": "检索模式", "fusion": "融合方式（混合检索生效）",
        "reranker": "重排策略",
        "wvec": "向量路权重", "wfull": "全文路权重",
        "input": "请输入问题，例如：武汉力源信息技术股份有限公司主要从事什么业务？",
        "button": "提交提问",
        "answer": "答案", "evidence": "检索证据 Top-3",
        "latency": "端到端耗时", "sla_ok": "满足 ≤3 秒验收线",
        "sla_bad": "超出 3 秒验收线",
        "dense_rank": "向量名次", "sparse_rank": "全文名次",
        "adopt": "采纳", "reject": "不采纳",
        "feedback_ok": "反馈已记录，切换“用户反馈自适应重排”后即刻生效",
        "feedback_count": "历史反馈条数（含模拟）",
        "empty_warn": "请输入问题后再提交。",
        "page": "第", "page_unit": "页", "from": "来源",
        "history": "本轮对话历史",
        "embed": "嵌入后端（需重新建库生效）",
        "strategy": "本次策略",
        "timeret": "检索耗时", "timeans": "答案组织耗时",
        "tip": "说明：权重调整即时生效；RRF/投票仅依赖名次，加权平均依赖归一分数。",
    },
    "en": {
        "title": "Prospectus Hybrid Retrieval Q&A (Work Order 6)",
        "lang": "Language 界面语言",
        "mode": "Retrieval mode", "fusion": "Fusion (for hybrid mode)",
        "reranker": "Reranker",
        "wvec": "Dense weight", "wfull": "Full-text weight",
        "input": "Ask a question, e.g. What business does Wuhan P&S engage in?",
        "button": "Ask",
        "answer": "Answer", "evidence": "Evidence Top-3",
        "latency": "End-to-end latency", "sla_ok": "Within 3s SLA",
        "sla_bad": "Exceeds 3s SLA",
        "dense_rank": "Dense rank", "sparse_rank": "BM25 rank",
        "adopt": "Adopt", "reject": "Reject",
        "feedback_ok": "Feedback saved; effective under adaptive reranker",
        "feedback_count": "Historical feedback (incl. simulated)",
        "empty_warn": "Please enter a question.",
        "page": "Page", "page_unit": "", "from": "Source",
        "history": "Session history",
        "embed": "Embedding backend (rebuild index to apply)",
        "strategy": "Strategy",
        "timeret": "Retrieval", "timeans": "Answer extraction",
        "tip": "Weights apply instantly; RRF/vote are rank-only, weighted-average uses normalized scores.",
    },
}

MODE_LABEL = {"vector": "向量检索 vector", "fulltext": "全文检索 fulltext",
              "hybrid": "混合检索 hybrid"}
FUSION_LABEL = {"weighted_avg": "加权平均 weighted_avg",
                "vote": "投票融合 vote(Borda)", "rrf": "RRF 倒数排名融合"}
RERANK_LABEL = {"llm": "LLM重排（Cross-Encoder）",
                "tfidf": "TF-IDF重排",
                "adaptive": "用户反馈自适应重排"}


@st.cache_resource
def load_components():
    """单例加载索引、反馈存储与问答引擎（常驻内存，不计入单次响应）。"""
    embedder = create_embedder()
    store = IndexStore.load(CONFIG.index_dir, embedder)
    feedback = FeedbackStore(CONFIG.feedback_file)
    retriever = HybridRetriever(store, feedback)
    return QAEngine(retriever), retriever, feedback


def main() -> None:
    """渲染 Streamlit 页面：热配侧栏 + 问答主区 + 反馈按钮 + 多轮历史。"""
    lang = st.sidebar.radio("中文 / English", ["zh", "en"],
                            format_func=lambda x: "中文" if x == "zh" else "English")
    t = I18N[lang]
    st.title(t["title"])

    engine, retriever, feedback = load_components()

    # ---------- 侧边栏：检索策略热配 ----------
    st.sidebar.markdown(f"**{t['mode']}**")
    mode_label = st.sidebar.radio("", list(MODE_LABEL.values()), index=2)
    RT_CONFIG.search_mode = {v: k for k, v in MODE_LABEL.items()}[mode_label]

    st.sidebar.markdown(f"**{t['fusion']}**")
    fusion_label = st.sidebar.selectbox("", list(FUSION_LABEL.values()), index=2)
    RT_CONFIG.fusion_method = {v: k for k, v in FUSION_LABEL.items()}[fusion_label]

    # 双路权重滑块（热配，提交时归一化）
    wv = st.sidebar.slider(t["wvec"], 0.0, 1.0, RT_CONFIG.vector_weight, 0.05)
    wf = st.sidebar.slider(t["wfull"], 0.0, 1.0, RT_CONFIG.fulltext_weight, 0.05)
    RT_CONFIG.set_weights(wv, wf)

    st.sidebar.markdown(f"**{t['reranker']}**")
    rr_label = st.sidebar.selectbox("", list(RERANK_LABEL.values()), index=0)
    RT_CONFIG.reranker_name = {v: k for k, v in RERANK_LABEL.items()}[rr_label]

    try:
        backend = retriever.store.embedder.backend
    except Exception:
        backend = "?"
    st.sidebar.caption(f"{t['embed']}: **{backend}**")
    st.sidebar.caption(t["tip"])
    st.sidebar.caption(f"{t['feedback_count']}: {feedback.count()} "
                       f"(simulated={feedback.simulated_count()})")

    # ---------- 多轮对话历史 ----------
    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.text_area(t["input"], height=80)
    if st.button(t["button"], type="primary"):
        if not question.strip():
            st.warning(t["empty_warn"])
            return
        result = engine.answer(question)
        st.session_state.history.append((question, result))

    # 展示最近一轮问答与反馈按钮
    if st.session_state.history:
        question, result = st.session_state.history[-1]
        st.markdown(f"### {t['answer']}")
        st.write(result.answer)
        sla_ok = result.latency_s <= CONFIG.response_timeout_s
        st.caption(
            f"{t['strategy']}: `{result.strategy}` ｜ "
            f"{t['latency']}: **{result.latency_s:.2f}s** "
            f"({t['timeret']} {result.timings['retrieval_s']}s / "
            f"{t['timeans']} {result.timings['answer_s']}s) ｜ "
            f"{'✅ ' + t['sla_ok'] if sla_ok else '⚠️ ' + t['sla_bad']}")

        st.markdown(f"### {t['evidence']}")
        for i, ev in enumerate(result.evidences, 1):
            with st.container(border=True):
                st.markdown(
                    f"**#{i}｜{t['from']} {ev.doc_name}｜{t['page']}"
                    f"{ev.page_no}{t['page_unit']}** ｜ {ev.heading_path[:40]}")
                st.write(ev.text[:280])
                st.caption(
                    f"{t['dense_rank']}: {ev.dense_rank if ev.dense_rank > 0 else '-'} ｜ "
                    f"{t['sparse_rank']}: {ev.sparse_rank if ev.sparse_rank > 0 else '-'} ｜ "
                    f"score: {ev.score:.4f} ｜ chunk_id: {ev.chunk_id}")
                c1, c2 = st.columns(2)
                if c1.button(f"👍 {t['adopt']} #{i}", key=f"adopt_{i}"):
                    feedback.record_adopt(question, ev.chunk_id)
                    st.success(t["feedback_ok"])
                if c2.button(f"👎 {t['reject']} #{i}", key=f"reject_{i}"):
                    feedback.record_reject(question, ev.chunk_id)
                    st.success(t["feedback_ok"])

        # 多轮历史折叠展示
        with st.expander(f"📜 {t['history']}（{len(st.session_state.history)}）"):
            for q, r in reversed(st.session_state.history):
                st.markdown(f"- **Q：{q}**")
                st.markdown(f"  - A：{r.answer[:120]}（{r.latency_s:.2f}s）")


if __name__ == "__main__":
    main()
