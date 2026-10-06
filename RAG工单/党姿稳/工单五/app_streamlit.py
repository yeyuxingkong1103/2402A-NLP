# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Query理解优化任务
交互界面（Streamlit）：多轮对话问答。
每轮先做 Query 改写（指代消解/省略补全），再检索生成；可开关以对比效果。
"""
import os
import socket
import streamlit as st
from kb import load_all_chunks
from retriever import Retriever
from dialog import DialogSession
from config import MULTI_TURN_DEMO

st.set_page_config(page_title="RAG多轮对话问答", page_icon="💬", layout="wide")


@st.cache_resource(show_spinner="正在加载知识库与模型...")
def get_retriever():
    allc, _, _ = load_all_chunks()
    return Retriever(allc)


def find_free_port(start=8501, end=8600):
    for p in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


st.title("💬 基于PDF文档的多轮对话问答系统（工单五·Query理解优化）")
st.caption("工单编号：人工智能NLP-RAG-Query理解优化任务")

with st.sidebar:
    st.header("⚙️ 设置")
    use_rewrite = st.toggle("启用 Query 改写（指代消解/省略补全）", value=True)
    top_k = st.slider("返回片段数 Top-K", 1, 10, 5)
    if st.button("🗑️ 清空对话"):
        st.session_state.pop("history", None)
        st.session_state.pop("sess", None)
        st.rerun()
    st.markdown("---")
    st.subheader("⚡ 验收多轮问题")
    for i, q in enumerate(MULTI_TURN_DEMO, 1):
        st.caption(f"{i}. {q}")

retriever = get_retriever()
if "sess" not in st.session_state:
    st.session_state.sess = DialogSession(retriever, use_rewrite=use_rewrite, top_k=top_k)
    st.session_state.history = []

sess = st.session_state.sess
sess.use_rewrite = use_rewrite
sess.rewriter = sess.rewriter if use_rewrite else None
if use_rewrite and sess.rewriter is None:
    from query_rewriter import QueryRewriter
    sess.rewriter = QueryRewriter()
sess.top_k = top_k

for h in st.session_state.history:
    with st.chat_message(h["role"]):
        st.markdown(h["content"])
        if h.get("rewritten"):
            st.caption(f"🔍 Query改写：{h['rewritten']}")

question = st.chat_input("请输入问题（支持多轮追问，如“那XX公司呢？”）")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("检索与生成中..."):
            rec = sess.ask(question)
        st.markdown(rec["answer"])
        st.caption(f"🔍 Query改写：{rec['rewritten']}")
        with st.expander(f"📚 检索片段（页码：{rec['pages']}）"):
            st.caption(f"命中文档：{'、'.join(rec['docs'])}")
    st.session_state.history.append(
        {"role": "assistant", "content": rec["answer"], "rewritten": rec["rewritten"]})

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
