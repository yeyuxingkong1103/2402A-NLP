# 工单编号：人工智能 NLP-RAG-混合检索任务
import streamlit as st
from step3_rag_v6 import ask_rag, DialogState

st.set_page_config(page_title="招股说明书 RAG v6 混合检索", layout="wide")
st.title("招股说明书 RAG v6（混合检索 + 多轮对话）")

with st.sidebar:
    st.header("检索配置")
    mode = st.radio("检索模式", ["hybrid", "vector", "fulltext"], index=0)
    reranker = st.radio("重排算法", ["bge", "tfidf", "llm", "adaptive"], index=0)
    if mode == "hybrid":
        vw = st.slider("向量权重", 0.0, 1.0, 0.6, 0.1)
        fw = 1.0 - vw
        st.write(f"全文权重：{fw:.1f}")
    else:
        vw, fw = 0.6, 0.4

if "state" not in st.session_state:
    st.session_state.state = DialogState()
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("rewritten"):
            st.caption(f"→ 重写后：{m['rewritten']}")
        if m.get("contexts"):
            with st.expander("引用片段"):
                for meta, text in m["contexts"]:
                    st.markdown(f"**{meta['source']} 第 {meta['page']} 页 [{meta['type']}]**")
                    st.write(text[:400])
                    st.markdown("---")

if prompt := st.chat_input("请输入你的问题（支持多轮对话）"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"检索：{mode} / 重排：{reranker}"):
            ans, ctx, rewritten = ask_rag(
                prompt, st.session_state.state,
                mode=mode, reranker=reranker,
                vector_weight=vw, fulltext_weight=fw
            )
        st.write(ans)
        st.caption(f"→ 重写后：{rewritten}")
        with st.expander("引用片段"):
            for meta, text in ctx:
                st.markdown(f"**{meta['source']} 第 {meta['page']} 页 [{meta['type']}]**")
                st.write(text[:400])
                st.markdown("---")

    st.session_state.messages.append({
        "role": "assistant",
        "content": ans,
        "rewritten": rewritten,
        "contexts": ctx
    })