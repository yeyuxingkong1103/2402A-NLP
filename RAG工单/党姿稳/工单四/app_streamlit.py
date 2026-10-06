# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
交互界面（Streamlit）：知识库问答（文本+表格+图像块），并以文搜图展示最相关图表。
"""
import os
import socket
import streamlit as st
from kb import load_all_chunks
from retriever import Retriever
from rag_chain import rag_answer
from config import EVAL_QUESTIONS

st.set_page_config(page_title="RAG问答系统（图像解析版）", page_icon="🖼️", layout="wide")


@st.cache_resource(show_spinner="正在加载知识库与模型...")
def get_resources():
    allc, _, imgc = load_all_chunks()
    return Retriever(allc), imgc


def find_free_port(start=8501, end=8600):
    for p in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


st.title("🖼️ 基于PDF文档的问答系统（工单四·图像解析版）")
st.caption("工单编号：人工智能NLP-RAG-图像内容解析及检索优化")

retriever, img_chunks = get_resources()
n_img = sum(1 for c in retriever.chunks if c["type"] == "image")
st.success(f"知识库就绪：共 {len(retriever.chunks)} 块（其中图像块 {n_img} 块）")

with st.sidebar:
    st.header("⚙️ 设置")
    top_k = st.slider("返回片段数 Top-K", 1, 10, 5)
    st.markdown("---")
    st.subheader("⚡ 快捷问题")
    quick = st.selectbox("验收问题", [""] + [q["question"] for q in EVAL_QUESTIONS])
    st.caption("图像解析：定位含图页 → RapidOCR 提取图内文字 → 生成图像语义块；CLIP 支持以文搜图")

question = st.text_input("请输入问题", value=quick,
                         placeholder="例如：组织结构图中销售部有几个部门构成？")

if st.button("🚀 提问", type="primary") and question:
    ans, ctxs, t = rag_answer(question, retriever, top_k=top_k, optimize=True)
    st.info(ans)
    st.caption(f"检索+生成耗时：{t:.2f}s")
    st.subheader("📚 检索片段")
    for i, (c, s) in enumerate(ctxs, 1):
        tag = "🖼️ 图像块" if c["type"] == "image" else ("📊 表格块" if c["type"] == "table" else "📄 文本块")
        with st.expander(f"片段{i}｜{tag}｜{c['doc']}·第{c['page']}页｜分数{s:.4f}"):
            st.markdown(c["text"][:800])
            if c["type"] == "image" and c.get("image") and os.path.exists(c["image"]):
                st.image(c["image"], caption=f"第{c['page']}页原图")

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
