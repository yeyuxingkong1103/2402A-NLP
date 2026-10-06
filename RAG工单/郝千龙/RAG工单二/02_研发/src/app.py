# -*- coding: utf-8 -*-
# 【Streamlit交互界面 · app.py】中英文双语问答界面，展示答案/证据页码/阶段耗时
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

"""启动方式：

    streamlit run app.py

功能：加载索引 → 单例常驻 → 提问后展示抽取式/LLM 答案、
Top-3 证据块（页码、标题、双路排名、分数）与各阶段耗时，
满足“交互友好性、多语言、响应≤3秒”验收项。
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from embeddings import create_embedder
from qa_engine import QAEngine
from retriever import Retriever
from vector_store import IndexStore

# 双语文案
I18N = {
    "zh": {
        "title": "招股说明书智能问答系统（工单二优化版）",
        "lang": "界面语言",
        "input": "请输入问题，例如：武汉兴图新科电子股份有限公司注册资本是多少？",
        "button": "提交提问",
        "answer": "答案",
        "evidence": "检索证据（Top-3）",
        "timing": "耗时分解",
        "page": "第", "page_unit": "页",
        "empty_warn": "请输入问题后再提交。",
        "latency": "端到端耗时",
        "dense_rank": "向量排名", "sparse_rank": "关键词排名",
        "sla_ok": "满足 ≤3 秒验收线", "sla_bad": "超出 3 秒验收线",
    },
    "en": {
        "title": "Prospectus Q&A System (Work Order 2 Optimized)",
        "lang": "Language",
        "input": "Ask a question, e.g. What is the registered capital of Wuhan Xingtu Xinke?",
        "button": "Ask",
        "answer": "Answer",
        "evidence": "Retrieved Evidence (Top-3)",
        "timing": "Latency Breakdown",
        "page": "Page", "page_unit": "",
        "empty_warn": "Please enter a question.",
        "latency": "End-to-end latency",
        "dense_rank": "Dense rank", "sparse_rank": "BM25 rank",
        "sla_ok": "Within 3s SLA", "sla_bad": "Exceeds 3s SLA",
    },
}


@st.cache_resource
def load_engine() -> QAEngine:
    """加载并缓存问答引擎（模型/索引单例常驻，不计入单次响应）。

    :return: 已就绪的 QAEngine
    """
    embedder = create_embedder()
    store = IndexStore.load(CONFIG.index_dir, embedder)
    return QAEngine(Retriever(store))


def main() -> None:
    """渲染 Streamlit 页面并处理一次问答。"""
    lang = st.sidebar.radio("Language / 语言", ["zh", "en"], format_func=lambda x: "中文" if x == "zh" else "English")
    t = I18N[lang]
    st.title(t["title"])

    question = st.text_area(t["input"], height=80)
    if st.button(t["button"]):
        if not question.strip():
            st.warning(t["empty_warn"])
            return
        try:
            engine = load_engine()
        except FileNotFoundError:
            st.error("未找到索引，请先运行：python build_index.py --pdf <招股书路径>")
            return
        result = engine.answer(question.strip())

        st.subheader(t["answer"])
        st.success(result.answer or "—")
        sla = result.latency_s <= CONFIG.response_timeout_s
        st.caption(f"{t['latency']}: {result.latency_s:.3f}s　"
                   f"✅ {t['sla_ok']}" if sla else f"⛔ {t['sla_bad']}")

        st.subheader(t["evidence"])
        for i, ev in enumerate(result.evidences, start=1):
            with st.container(border=True):
                st.markdown(f"**#{i}** · {t['page']} **{ev.page_no}**{t['page_unit']} · "
                            f"{ev.heading_path or '-'}  ")
                st.caption(f"score={ev.score:.4f} ｜ {t['dense_rank']}: {ev.dense_rank} ｜ "
                           f"{t['sparse_rank']}: {ev.sparse_rank}")
                st.text(ev.parent_text[:500])

        st.subheader(t["timing"])
        st.json(result.timings)


if __name__ == "__main__":
    main()
