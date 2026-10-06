# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-功能测试及评估
交互界面（Streamlit）：基于 ccf_competition 金融年报语料的 RAG 问答与检索测试，
可选择验收问题、切换检索策略，并展示检索片段与评估结果。
"""
import os
import sys
import socket
import streamlit as st

from kb import load_chunks
from hybrid_retriever import HybridRetriever
from rag_chain import call_llm, build_rag_prompt, RAG_SYSTEM_PROMPT
from config import EVAL_QUESTIONS

st.set_page_config(page_title="金融年报RAG测试评估", page_icon="📈", layout="wide")


@st.cache_resource(show_spinner="正在加载金融年报语料...")
def get_retriever():
    return HybridRetriever(load_chunks(), reranker="llm")


def find_free_port(start=8501, end=8600):
    for p in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


st.title("📈 金融年报 RAG 问答与测试（工单七·功能测试及评估）")
st.caption("工单编号：人工智能NLP-RAG-功能测试及评估　|　语料：ccf_competition（9 份金融年报）")

with st.sidebar:
    st.header("⚙️ 检索配置")
    strategy = st.radio("检索策略", ["hybrid", "vector", "fulltext"],
                        format_func=lambda x: {"hybrid": "混合检索", "vector": "向量检索（召回+重排）",
                                               "fulltext": "全文检索"}[x])
    reranker = st.selectbox("重排算法", ["llm", "tfidf", "none"])
    top_k = st.slider("Top-K", 1, 10, 5)
    st.markdown("---")
    st.subheader("验收问题（10 个）")
    for q in EVAL_QUESTIONS:
        st.caption(f"{q['id']}. {q['question'][:42]}…")

r = get_retriever()
r.reranker_name = reranker
from rerankers import build_reranker
r.reranker = build_reranker(reranker)
st.success(f"语料就绪：{len(r.chunks)} 个块 / {len({c['doc'] for c in r.chunks})} 份年报")

quick = st.selectbox("⚡ 选择验收问题", [""] + [q["question"] for q in EVAL_QUESTIONS])
question = st.text_input("请输入问题", value=quick)

if st.button("🚀 检索并回答", type="primary") and question:
    import time
    t0 = time.time()
    res = r.search(question, strategy=strategy, top_k=top_k)
    st.info(call_llm(RAG_SYSTEM_PROMPT, build_rag_prompt(question, res)))
    st.caption(f"检索 + 生成耗时：{time.time()-t0:.2f}s")
    st.subheader("📚 检索片段")
    for i, (c, s) in enumerate(res, 1):
        with st.expander(f"片段{i}｜{c['doc']}·第{c['page']}页｜分数{s:.4f}"):
            st.markdown(c["text"][:800])

if __name__ == "__main__" and not st.runtime.exists():
    # `streamlit run` 与 `python app_streamlit.py` 都会把脚本以 __name__="__main__" 执行；
    # 用"Runtime 是否已存在"判断，可避免启动器递归调用导致的 Runtime instance already exists
    import streamlit.runtime
    from streamlit.web import cli as stcli
    port = find_free_port()
    print(f"启动地址: http://localhost:{port}")
    sys.argv = ["streamlit", "run", __file__, "--server.port", str(port)]
    sys.exit(stcli.main())
