# 工单编号：人工智能 NLP-RAG-Query 理解优化任务
import streamlit as st
from step3_rag_v5 import ask_rag, DialogState

st.set_page_config(page_title="招股说明书 RAG v5 多轮对话", layout="wide")
st.title("招股说明书 RAG v5（多轮对话 + Query 理解）")

if "state" not in st.session_state:
    st.session_state.state = DialogState()
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("rewritten"):
            st.caption(f"→ 重写后：{m['rewritten']}")

if prompt := st.chat_input("请输入你的问题（支持多轮对话）"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("检索生成中..."):
            ans, ctx, rewritten = ask_rag(prompt, st.session_state.state)
        st.write(ans)
        st.caption(f"→ 重写后：{rewritten}")
        with st.expander("引用片段"):
            for m, c in ctx:
                st.markdown(f"**{m['source']} 第 {m['page']} 页 [{m['type']}]**")
                st.write(c[:400])
                st.markdown("---")

    st.session_state.messages.append({
        "role": "assistant",
        "content": ans,
        "rewritten": rewritten
    })