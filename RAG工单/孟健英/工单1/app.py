# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统
import streamlit as st
from step3_rag import ask_rag, ask_llm

st.set_page_config(page_title="招股说明书 RAG 问答系统", layout="wide")
st.title("招股说明书 RAG 问答系统")

question = st.text_input("请输入你的问题：")

col1, col2 = st.columns(2)
if st.button("提问") and question:
    with st.spinner("RAG 检索生成中..."):
        rag_ans, ctx = ask_rag(question)
    with st.spinner("纯 LLM 生成中..."):
        llm_ans = ask_llm(question)

    with col1:
        st.subheader("RAG 回答")
        st.write(rag_ans)
    with col2:
        st.subheader("纯 LLM 回答")
        st.write(llm_ans)

    with st.expander("引用片段"):
        for p, c in ctx:
            st.markdown(f"**第 {p} 页**")
            st.write(c[:500])
            st.markdown("---")

    st.caption("本回答仅供参考，不能替代专业意见。")