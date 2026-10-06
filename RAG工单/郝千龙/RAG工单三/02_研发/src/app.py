# -*- coding: utf-8 -*-
# 【问答演示应用 · app.py】Streamlit中英双语问答界面：答案、表格Markdown证据、页码、标题路径与耗时
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

"""招股书表格问答演示（工单三）。

启动（PowerShell）：
    cd 02_研发\\src
    streamlit run app.py
索引为离线构建产物，在线问答只做检索与抽取，不再解析 PDF。
"""
import os

import pandas as pd
import streamlit as st

from config import CONFIG
from vector_store import IndexStore
from retriever import Retriever
from qa_engine import QAEngine

# ---------------------------------------------------------------------------
# 中英双语文案
# ---------------------------------------------------------------------------
I18N = {
    "zh": {
        "title": "招股意向书 PDF 表格解析与检索问答系统（工单三）",
        "subtitle": "表格感知解析 · 逐行语义块 · 双路召回RRF · 表格感知精排",
        "lang_label": "界面语言 / Language",
        "input_label": "请输入问题",
        "button": "提问",
        "examples": "示例问题（点击填入）",
        "answer": "生成答案",
        "evidence": "证据片段（Small-to-Big：行块召回，整表溯源）",
        "page": "页码",
        "heading": "标题路径",
        "company": "发行人",
        "source_table": "来源整表（Markdown）",
        "row_claim": "命中表格行",
        "latency": "端到端耗时",
        "timeout_warn": "超过 3 秒验收预算，请检查索引规模或缩小候选数。",
        "type_map": {"table_row": "表格行", "table": "整表", "text": "正文"},
        "no_index": "未找到索引，请先运行：python build_index.py",
        "loading": "正在加载离线索引…",
    },
    "en": {
        "title": "Prospectus PDF Table Parsing & QA (Order 3)",
        "subtitle": "Table-aware parsing · row claims · hybrid retrieval "
                    "(TF-IDF+BM25/RRF) · table-aware reranking",
        "lang_label": "Language / 界面语言",
        "input_label": "Enter your question",
        "button": "Ask",
        "examples": "Example questions (click to fill)",
        "answer": "Answer",
        "evidence": "Evidence (small-to-big: row recall, whole-table source)",
        "page": "Page",
        "heading": "Section",
        "company": "Issuer",
        "source_table": "Source table (Markdown)",
        "row_claim": "Matched table row",
        "latency": "End-to-end latency",
        "timeout_warn": "Exceeds the 3s budget; check index size / candidates.",
        "type_map": {"table_row": "Table row", "table": "Table",
                     "text": "Text"},
        "no_index": "Index not found. Run first: python build_index.py",
        "loading": "Loading offline index…",
    },
}

EXAMPLES = [
    "力源信息本次发行股数是多少？占发行后总股本的比例是多少？",
    "力源信息本次募集资金拟投资哪些项目？",
    "力源信息与公司不存在控制关系的关联方企业有哪些？",
    "兴图新科来自军用领域的收入分别是多少？",
    "兴图新科军用产品收入占主营业务收入的比重分别是多少？",
    "兴图新科所处电子信息行业的上游涉及哪些企业？",
    "兴图新科的注册资本是多少？",
    "兴图新科募集资金中用于补充流动资金的金额是多少？",
]


@st.cache_resource(show_spinner=False)
def load_engine():
    """加载一次索引并构建问答引擎（缓存，避免每次提问重复加载）。

    :return: QAEngine 实例
    """
    store = IndexStore.load(CONFIG.index_dir)
    return QAEngine(Retriever(store))


def md_table_to_frame(md: str) -> pd.DataFrame:
    """把证据中的 Markdown 整表转为 DataFrame 供 st.dataframe 展示。

    :param md: Markdown 表格文本
    :return: DataFrame；无法解析时返回空表
    """
    rows = [ln.strip() for ln in md.splitlines()
            if ln.strip().startswith("|") and "---" not in ln]
    if len(rows) < 2:
        return pd.DataFrame()
    grid = [[c.strip() for c in ln.strip("|").split("|")] for ln in rows]
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]
    return pd.DataFrame(grid[1:], columns=grid[0])


def main() -> None:
    """Streamlit 应用主函数。"""
    st.set_page_config(page_title="工单三-表格问答", layout="wide")
    lang = st.sidebar.radio("Language / 界面语言", ["zh", "en"],
                            format_func=lambda x: "中文" if x == "zh"
                            else "English")
    t = I18N[lang]

    st.title(t["title"])
    st.caption(t["subtitle"])

    if not os.path.isdir(CONFIG.index_dir):
        st.error(t["no_index"])
        return
    with st.spinner(t["loading"]):
        engine = load_engine()

    st.markdown(f"**{t['examples']}**")
    chosen = st.session_state.get("q", "")
    cols = st.columns(4)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 4].button(ex, key=f"ex{i}"):
            chosen = ex
            st.session_state["q"] = ex

    with st.form("qa_form"):
        question = st.text_area(t["input_label"], value=chosen, height=80)
        submitted = st.form_submit_button(t["button"])

    if submitted and question.strip():
        result = engine.answer(question.strip())
        st.subheader(t["answer"])
        st.write(result.answer)
        c1, c2 = st.columns(2)
        c1.metric(t["latency"], f"{result.latency_s:.3f} s")
        c2.metric("Top-K", len(result.evidences))
        if result.latency_s > CONFIG.response_timeout_s:
            st.warning(t["timeout_warn"])

        st.subheader(t["evidence"])
        for i, ev in enumerate(result.evidences, 1):
            tag = t["type_map"].get(ev.chunk_type, ev.chunk_type)
            with st.expander(
                    f"#{i} [{tag}] P{ev.page_no} · {ev.company} · "
                    f"{ev.heading_path[:42]}", expanded=(i == 1)):
                st.write(
                    f"**{t['page']}**: {ev.page_no}　|　"
                    f"**{t['company']}**: {ev.company}　|　"
                    f"**{t['heading']}**: {ev.heading_path}")
                if ev.table_title:
                    st.caption(ev.table_title)
                if ev.row_claim:
                    st.markdown(f"**{t['row_claim']}**")
                    st.success(ev.row_claim)
                if ev.chunk_type in {"table", "table_row"} and \
                        ev.parent_text.count("|") >= 2:
                    st.markdown(f"**{t['source_table']}**")
                    frame = md_table_to_frame(ev.parent_text)
                    if not frame.empty:
                        st.dataframe(frame, use_container_width=True,
                                     hide_index=True)
                    else:
                        st.code(ev.parent_text, language="markdown")
                else:
                    st.write(ev.parent_text[:900])


if __name__ == "__main__":
    main()
