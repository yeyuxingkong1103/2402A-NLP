# -*- coding: utf-8 -*-
# 【Streamlit图文问答界面 · app.py】中英双语，展示文本/表格/图像证据与图片原图
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化

"""启动方式（在02_研发/src目录下执行）：

    streamlit run app.py

功能：加载图文双索引（单例常驻）→ 提问后展示答案、Top-3证据（含★图片证据
直接渲染原图）、所属文档/页码、双路排名、各阶段耗时与降级标记，
满足“交互友好性、中英文支持、响应≤3秒、图像证据展示”验收项。
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from multimodal_index import IndexStore, create_embedder
from qa_engine import QAEngine
from retriever import Retriever

I18N = {
    "zh": {
        "title": "招股说明书图文智能问答系统（工单四·图像内容解析版）",
        "lang": "界面语言",
        "input": "请输入问题，例如：组织结构图中销售部有几个部门构成？",
        "button": "提交提问",
        "answer": "答案",
        "evidence": "检索证据（Top-3，含图像证据）",
        "timing": "耗时分解",
        "page": "第", "page_unit": "页", "doc": "文档",
        "empty_warn": "请输入问题后再提交。",
        "latency": "端到端耗时",
        "dense_rank": "向量排名", "sparse_rank": "关键词排名",
        "sla_ok": "满足 ≤3 秒验收线", "sla_bad": "超出 3 秒验收线",
        "image_evidence": "图像证据（点击可放大）",
        "degraded": "⚠ 本题图像多模态/OCR不可用，答案基于图注+邻近正文降级生成",
        "type_map": {"text": "文本", "table": "表格", "image": "图像"},
        "examples": ["组织结构图中，销售部有几个部门构成，大客户销售部有几个销售处？",
                     "2008年中国IC市场应用结构与增长图中，增长率最快和负增长的分别是哪个行业？",
                     "本次发行股数是多少，占发行后总股本的比例是多少？",
                     "不存在控制关系的关联方企业有哪些？"],
    },
    "en": {
        "title": "Prospectus Multimodal Q&A (Work Order 4 · Image Parsing)",
        "lang": "Language",
        "input": "Ask a question, e.g. How many departments form the Sales Dept?",
        "button": "Ask",
        "answer": "Answer",
        "evidence": "Retrieved Evidence (Top-3, incl. images)",
        "timing": "Latency Breakdown",
        "page": "Page", "page_unit": "", "doc": "Doc",
        "empty_warn": "Please enter a question.",
        "latency": "End-to-end latency",
        "dense_rank": "Dense rank", "sparse_rank": "BM25 rank",
        "sla_ok": "Within 3s SLA", "sla_bad": "Exceeds 3s SLA",
        "image_evidence": "Image evidence (click to enlarge)",
        "degraded": "Degraded: OCR/multimodal unavailable; answer from caption and nearby text",
        "type_map": {"text": "Text", "table": "Table", "image": "Image"},
        "examples": [],
    },
}


@st.cache_resource
def load_engine() -> QAEngine:
    """加载并缓存问答引擎（模型/索引单例常驻，加载耗时不计入单次响应）。

    :return: 已就绪QAEngine
    """
    embedder = create_embedder()
    store = IndexStore.load(CONFIG.index_dir, embedder)
    return QAEngine(Retriever(store))


def main() -> None:
    """渲染Streamlit页面并处理一次图文问答。"""
    lang = st.sidebar.radio("Language / 语言", ["zh", "en"],
                            format_func=lambda x: "中文" if x == "zh" else "English")
    t = I18N[lang]
    st.title(t["title"])

    if lang == "zh" and t["examples"]:
        with st.expander("示例问题（点击展开）"):
            for ex in t["examples"]:
                st.markdown(f"- {ex}")

    question = st.text_area(t["input"], height=80)
    if st.button(t["button"]):
        if not question.strip():
            st.warning(t["empty_warn"])
            return
        try:
            engine = load_engine()
        except FileNotFoundError:
            st.error("未找到索引，请先运行：python build_index.py")
            return
        result = engine.answer(question.strip())

        st.subheader(t["answer"])
        st.success(result.answer or "—")
        sla = result.latency_s <= CONFIG.response_timeout_s
        st.caption(f"{t['latency']}: {result.latency_s:.3f}s　"
                   f"{'✅ ' + t['sla_ok'] if sla else '⛔ ' + t['sla_bad']}")
        if result.degraded:
            st.warning(t["degraded"])

        st.subheader(t["evidence"])
        for i, ev in enumerate(result.evidences, start=1):
            with st.container(border=True):
                type_cn = t["type_map"].get(ev.chunk_type, ev.chunk_type)
                st.markdown(
                    f"**#{i}** · {t['doc']}: `{ev.doc}` · "
                    f"{t['page']} **{ev.page_no}**{t['page_unit']} · "
                    f"[{type_cn}] · {ev.heading_path or '-'}")
                st.caption(f"score={ev.score:.4f} ｜ {t['dense_rank']}: "
                           f"{ev.dense_rank} ｜ {t['sparse_rank']}: {ev.sparse_rank}")
                if ev.chunk_type == "image" and ev.image_path and \
                        os.path.exists(ev.image_path):
                    st.markdown(f"**{t['image_evidence']}**")
                    if ev.image_caption:
                        st.caption(ev.image_caption)
                    st.image(ev.image_path, use_column_width=True)
                st.text(ev.parent_text[:500])

        st.subheader(t["timing"])
        st.json(result.timings)


if __name__ == "__main__":
    main()
