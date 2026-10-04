# 工单编号：人工智能 NLP-RAG-Graph RAG 优化任务
import streamlit as st
from step4_rag_v9 import ask_graph_rag

st.set_page_config(page_title="Graph RAG 问答系统", layout="wide")
st.title("Graph RAG 优化版（金融年报问答）")

question = st.text_input("请输入问题：")

if st.button("提问") and question.strip():
    with st.spinner("Graph RAG 检索生成中..."):
        ans, ctx = ask_graph_rag(question)
    st.subheader("回答")
    st.write(ans)
    with st.expander("引用片段"):
        for m, c in ctx:
            st.markdown(f"**{m['source']} 第 {m['page']} 页 [{m['type']}]**")
            st.write(c[:400])
            st.markdown("---")
    st.caption("本回答仅供参考，不能替代专业意见。")