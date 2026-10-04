# 工单编号：人工智能 NLP-RAG-功能测试及评估
import streamlit as st
from step3_rag_v7 import ask_rag

st.set_page_config(page_title="RAG 功能测试", layout="wide")
st.title("RAG 功能测试及评估")

question = st.text_input("请输入问题：")
if st.button("提问") and question.strip():
    with st.spinner("检索生成中..."):
        ans, ctx = ask_rag(question)
    st.subheader("回答")
    st.write(ans)
    with st.expander("引用片段"):
        for m, c in ctx:
            st.markdown(f"**{m['source']} 第 {m['page']} 页 [{m['type']}]**")
            st.write(c[:400])
            st.markdown("---")