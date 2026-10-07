# -*- coding: utf-8 -*-
"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统
功能: 《招股说明书1.pdf》问答系统的 Streamlit 交互界面。

    - 侧边栏: 检索方式(向量/全文/混合) / top_k / 是否重排
    - 主区  : 对话框提问 -> R.rag_answer(...) -> 展示答案
    - 展开区: 展示命中的原文片段(页码 + 分数 + 文本)

运行: streamlit run app.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import rag_common as R
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(HERE, "index", "doc1_text")

st.set_page_config(page_title="招股说明书问答 (工单01)", page_icon="📄",
                   layout="wide")


@st.cache_resource(show_spinner="正在加载索引与嵌入模型 (仅首次)...")
def get_retriever(index_dir):
    """进程内只加载一次 Retriever (索引 + 嵌入模型 + BM25)"""
    return R.Retriever(index_dir)


# --------------------------------------------------------------------------
# 侧边栏: 检索参数
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ 检索设置")
MODE_MAP = {"向量检索": "vector", "全文检索": "fulltext", "混合检索": "hybrid"}
mode_label = st.sidebar.selectbox("检索方式", list(MODE_MAP), index=0)
mode = MODE_MAP[mode_label]
top_k = st.sidebar.slider("top_k (返回片段数)", 1, 10, 5)
RERANK_MAP = {"不重排": None, "TF-IDF 重排": "tfidf",
              "Cross-Encoder 重排": "cross-encoder"}
rerank_label = st.sidebar.selectbox("重排方式", list(RERANK_MAP), index=0)
rerank = RERANK_MAP[rerank_label]
st.sidebar.divider()
st.sidebar.caption(f"索引目录: index/doc1_text")
st.sidebar.caption("文档: 《招股说明书1.pdf》")

# --------------------------------------------------------------------------
# 主区: 问答
# --------------------------------------------------------------------------
st.title("📄 招股说明书智能问答")
st.caption("工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统 —— "
           "武汉兴图新科电子股份有限公司《招股说明书1.pdf》")

if not R.VectorStore.exists(INDEX_DIR):
    st.error(f"索引不存在: {INDEX_DIR}\n请先在命令行运行 `python build_index.py` 构建索引。")
    st.stop()

retriever = get_retriever(INDEX_DIR)
st.sidebar.caption(f"索引块数: {len(retriever.vs.chunks)}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 回放历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("retrieved"):
            with st.expander(f"📑 命中片段 (top {len(msg['retrieved'])})"):
                for i, d in enumerate(msg["retrieved"], 1):
                    st.markdown(f"**[{i}] 第 {d.get('page')} 页** · "
                                f"分数 {d.get('score', 0):.4f} · {d.get('source')}")
                    st.text(d.get("text", ""))
        if msg.get("meta"):
            st.caption(msg["meta"])

prompt = st.chat_input("请输入关于招股说明书的问题, 例如: 公司注册资本是多少?")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"{mode_label} 检索 + DeepSeek 生成中 ..."):
            res = R.rag_answer(prompt, retriever, top_k=top_k, mode=mode,
                               rerank=rerank)
        st.markdown(res["answer"])
        meta = (f"检索 {res['time_retrieval']}s · 生成 {res['time_generation']}s · "
                f"{mode_label}" + (f" + {rerank_label}" if rerank else ""))
        st.caption(meta)
        with st.expander(f"📑 命中片段 (top {len(res['retrieved'])})", expanded=True):
            for i, d in enumerate(res["retrieved"], 1):
                st.markdown(f"**[{i}] 第 {d.get('page')} 页** · "
                            f"分数 {d.get('score', 0):.4f} · {d.get('source')}")
                st.text(d.get("text", ""))

    st.session_state.messages.append({
        "role": "assistant", "content": res["answer"],
        "retrieved": res["retrieved"], "meta": meta,
    })
