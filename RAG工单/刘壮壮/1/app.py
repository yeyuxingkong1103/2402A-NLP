# -*- coding: utf-8 -*-
"""
app.py — 问答系统交互界面（Streamlit）
工单编号：人工智能NLP-RAG-基于PDF 文档的问答系统

启动：streamlit run app.py
提供问题输入、答案展示、检索片段展示与反馈机制。
"""
from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import RAGPipeline, PDF_PATH, ATTACH_DIR  # noqa: E402
from rag_core import Config  # noqa: E402

st.set_page_config(page_title="PDF 问答系统", page_icon="📄", layout="wide")


@st.cache_resource
def get_pipeline():
    p = RAGPipeline(Config())
    p.build(PDF_PATH)
    return p


st.title("📄 基于 PDF 文档的问答系统")
st.caption("工单编号：人工智能NLP-RAG-基于PDF 文档的问答系统")

with st.spinner("正在加载文档并构建索引……"):
    pipeline = get_pipeline()

st.sidebar.header("说明")
st.sidebar.markdown(
    f"- 文档：`{os.path.basename(PDF_PATH)}`\n"
    f"- 分块数：{len(pipeline.chunks)}\n"
    f"- 嵌入器：{pipeline.embedder_name}\n"
    f"- LLM：{pipeline.config.llm_model or '离线抽取式'}"
)
st.sidebar.markdown("> 语音输入可安装 `streamlit-audiorecorder` 扩展；本示例提供文字输入。")

question = st.text_input("请输入你的问题：", placeholder="例如：武汉兴图新科电子股份有限公司法定代表人是谁？")

if question:
    res = pipeline.ask(question)
    st.markdown("### 🤖 回答")
    st.write(res["answer"])
    st.caption(f"意图：{res['intent']} · 耗时：{res['elapsed']:.2f}s")

    with st.expander("查看检索到的相关片段"):
        for i, h in enumerate(res["hits"][:4], 1):
            st.markdown(f"**[{i}]** 相似度 `{h['score']:.3f}`")
            st.write(h["text"])

    # 反馈机制
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 答案有帮助"):
            st.success("感谢反馈！")
    with col2:
        if st.button("👎 答案不准确"):
            st.warning("感谢反馈，我们将持续优化检索质量。")
