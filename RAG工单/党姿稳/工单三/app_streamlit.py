# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
交互界面（Streamlit）：支持双文档知识库问答、表格块高亮、按文档过滤、检索片段溯源。
"""
import os
import socket
import streamlit as st
from pdf_parser import build_chunks, load_chunks
from retriever import Retriever
from rag_chain import rag_answer
from config import CHUNKS_FILE, EVAL_QUESTIONS

st.set_page_config(page_title="RAG问答系统（表格解析版）", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="正在加载知识库（含表格）与嵌入模型...")
def get_retriever():
    chunks = load_chunks() if os.path.exists(CHUNKS_FILE) else build_chunks()
    return Retriever(chunks)


def find_free_port(start=8501, end=8600):
    for p in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


st.title("📊 基于PDF文档的问答系统（工单三·表格解析版）")
st.caption("工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化")

with st.sidebar:
    st.header("⚙️ 检索设置")
    top_k = st.slider("返回片段数 Top-K", 1, 10, 5)
    doc_filter = st.selectbox("限定文档", ["全部", "兴图新科", "武汉力源"])
    st.markdown("---")
    st.subheader("⚡ 快捷问题")
    quick = st.selectbox("验收问题", [""] + [q["question"] for q in EVAL_QUESTIONS])
    st.markdown("---")
    st.caption("表格解析：PyMuPDF find_tables → Markdown，表标题拼入块中，表格不再被切碎")

retriever = get_retriever()
n_tab = sum(1 for c in retriever.chunks if c["type"] == "table")
st.success(f"知识库就绪：共 {len(retriever.chunks)} 块（其中表格 {n_tab} 块）")

question = st.text_input("请输入问题", value=quick,
                         placeholder="例如：武汉力源信息技术股份有限公司本次发行股数是多少？")

if st.button("🚀 提问", type="primary") and question:
    ans, ctxs, t = rag_answer(question, retriever, top_k=top_k, optimize=True)
    if doc_filter != "全部":
        ctxs = [c for c in ctxs if c[0]["doc"] == doc_filter] or ctxs
    st.info(ans)
    st.caption(f"检索+生成耗时：{t:.2f}s")
    st.subheader("📚 检索片段（含页码溯源）")
    for i, (c, s) in enumerate(ctxs, 1):
        tag = "📊 表格块" if c["type"] == "table" else "📄 文本块"
        with st.expander(f"片段{i}｜{tag}｜{c['doc']}·第{c['page']}页｜分数{s:.4f}"):
            st.markdown(c["text"][:800])

if __name__ == "__main__" and not st.runtime.exists():
    # `streamlit run` 与 `python app_streamlit.py` 都会把脚本以 __name__="__main__" 执行；
    # 用"Runtime 是否已存在"判断，可避免启动器递归调用导致的 Runtime instance already exists
    import streamlit.runtime
    from streamlit.web import cli as stcli
    import sys
    port = find_free_port()
    print(f"启动地址: http://localhost:{port}")
    sys.argv = ["streamlit", "run", __file__, "--server.port", str(port)]
    sys.exit(stcli.main())
