# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
交互界面（Streamlit）：支持 PDF 上传、问答、检索策略切换、优化前后对比、片段溯源。
"""
import os
import socket
import streamlit as st
from pdf_parser import build_chunks, load_chunks
from retriever import Retriever
from rag_chain import rag_answer
from config import CHUNKS_FILE, EVAL_QUESTIONS

st.set_page_config(page_title="RAG问答系统（优化版）", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="正在加载知识库与嵌入模型...")
def get_retriever():
    chunks = load_chunks() if os.path.exists(CHUNKS_FILE) else build_chunks()
    return Retriever(chunks)


def find_free_port(start=8501, end=8600):
    """端口冲突时自动寻找可用端口"""
    for p in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


st.title("📄 基于PDF文档的问答系统（工单二·检索优化版）")
st.caption("工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化")

with st.sidebar:
    st.header("⚙️ 检索策略")
    strategy = st.radio(
        "选择检索方式",
        ["优化后（混合检索+重排）", "优化前（仅向量检索）", "两者对比"],
        index=2,
    )
    top_k = st.slider("返回片段数 Top-K", 1, 10, 5)
    st.markdown("---")
    st.subheader("⚡ 快捷问题")
    quick = st.selectbox("验收问题", [""] + [q["question"] for q in EVAL_QUESTIONS])
    st.markdown("---")
    st.caption("优化点：段落级分块 · 向量+BM25混合召回 · 归一化融合 · MMR去冗余")

retriever = get_retriever()
st.success(f"知识库就绪：{len(retriever.chunks)} 个文本块")

question = st.text_input("请输入问题", value=quick,
                         placeholder="例如：武汉兴图新科电子股份有限公司注册资本是多少？")

def show_answer(question, optimize, top_k):
    """展示一次问答结果（含检索片段溯源）"""
    ans, ctxs, t = rag_answer(question, retriever, top_k=top_k, optimize=optimize)
    (st.info if optimize else st.warning)(ans)
    st.caption(f"检索+生成耗时：{t:.2f}s")
    with st.expander("检索片段（含页码溯源）"):
        for i, (c, s) in enumerate(ctxs, 1):
            st.markdown(f"**片段{i}｜第{c['page']}页｜分数{s:.4f}**\n\n{c['text'][:400]}")


if st.button("🚀 提问", type="primary") and question:
    if strategy == "两者对比":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ✅ 优化后（混合检索+重排）")
            show_answer(question, True, top_k)
        with col2:
            st.markdown("#### ⚠️ 优化前（仅向量检索）")
            show_answer(question, False, top_k)
    elif strategy == "优化后（混合检索+重排）":
        st.markdown("#### ✅ 优化后（混合检索+重排）")
        show_answer(question, True, top_k)
    else:
        st.markdown("#### ⚠️ 优化前（仅向量检索）")
        show_answer(question, False, top_k)


if __name__ == "__main__" and not st.runtime.exists():
    # `streamlit run` 与 `python app_streamlit.py` 都会把脚本以 __name__="__main__" 执行；
    # 用"Runtime 是否已存在"判断，可避免启动器递归调用导致的 Runtime instance already exists
    import streamlit.runtime
    # 供 `python app_streamlit.py` 直接启动时使用
    from streamlit.web import cli as stcli
    import sys
    port = find_free_port()
    print(f"启动地址: http://localhost:{port}")
    sys.argv = ["streamlit", "run", __file__, "--server.port", str(port)]
    sys.exit(stcli.main())
