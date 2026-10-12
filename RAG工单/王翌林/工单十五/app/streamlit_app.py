# -*- coding: utf-8 -*-
"""
工单十五：Streamlit 界面（跨模态检索 RAG 演示）
"""
import os, json
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="工单十五·跨模态检索RAG", layout="wide")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag_engine import RAGEngine


@st.cache_resource
def get_engine():
    eng = RAGEngine(top_k=5)
    mapping_json = os.path.join(os.path.dirname(__file__), "..", "data", "kb_mapping.json")
    mapping_json = os.path.abspath(mapping_json)
    if os.path.isfile(mapping_json):
        with open(mapping_json, encoding="utf-8") as f:
            m = json.load(f)
        eng.set_page_images({int(k): v for k, v in m.get("page_images", {}).items()})
        eng.set_page_texts({int(k): v for k, v in m.get("page_texts", {}).items()})
    return eng


engine = get_engine()

st.title("🔧 工单十五：技术图纸与文本的跨模态检索 RAG")
st.caption("专利 CN100347506C（块状散料分配装置）· 图3 跨模态问答")

with st.sidebar:
    st.header("测试问题")
    questions = [
        "根据专利文本，本发明主要涉及哪种物料的分配装置？",
        "根据专利文本，本发明的分散装置包含以下哪个组件？",
        "在文件中第11页图3中，编号13的部件相对于编号12的部件的位置关系是？",
        "在文件中第11页图3中，编号14的部件位于整个装置的哪个位置？",
        "根据文件中第11页图3，散料从部件14进入后，下一步会经过哪个部件？",
        "在文件中第11页图3的装置中，如果需要调整链条的位置，需要操作哪个部件?",
    ]
    for i, q in enumerate(questions, 1):
        if st.button(f"Q{i}. {q[:20]}…", key=f"q{i}"):
            st.session_state["q"] = q

q = st.text_input("请输入问题", value=st.session_state.get("q", questions[2]))
if st.button("提问", type="primary"):
    with st.spinner("检索与推理中..."):
        r = engine.ask(q)
    st.success(f"**答案**：{r['answer']}")
    route = r.get("route", {})
    st.info(f"路由: {route.get('query_type')} | 模式: {r.get('mode')} | "
            f"图号: {route.get('fig_no')} | 页码: {route.get('page')} | "
            f"耗时: {r.get('latency_ms')}ms")
    fig = r.get("figure") or {}
    if fig.get("image_path") and os.path.isfile(fig["image_path"]):
        st.image(fig["image_path"], caption=f"图{fig.get('fig_no')}（第{fig.get('page')}页）", width=500)
    if fig.get("description"):
        with st.expander("图纸结构化描述"):
            st.write(fig["description"])
    if r.get("references"):
        with st.expander("检索参考片段"):
            for ref in r["references"]:
                st.write(f"- 第{ref.get('page')}页 (score={ref.get('score')})")
