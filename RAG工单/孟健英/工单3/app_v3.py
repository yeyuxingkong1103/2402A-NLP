# 工单编号：人工智能 NLP-RAG-PDF 文档的表格解析及检索优化
import streamlit as st
from step3_rag_v3 import ask_rag, ask_llm

st.set_page_config(page_title="招股说明书 RAG v3", layout="wide")
st.title("招股说明书 RAG 问答系统 v3（表格解析优化）")

lang = st.radio("语言 / Language", ["中文", "English"])
if lang == "English":
    question = st.text_input("Please enter your question:")
else:
    question = st.text_input("请输入你的问题：")

if st.button("提问 / Ask"):
    if not question.strip():
        st.warning("请输入问题 / Please enter a question")
        st.stop()
    try:
        with st.spinner("RAG 检索生成中..."):
            rag_ans, ctx = ask_rag(question)
        with st.spinner("纯 LLM 生成中..."):
            llm_ans = ask_llm(question)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("RAG 回答")
            st.write(rag_ans)
        with col2:
            st.subheader("纯 LLM 回答")
            st.write(llm_ans)
        with st.expander("引用片段"):
            for m, c in ctx:
                st.markdown(f"**{m['source']} 第 {m['page']} 页 [{m['type']}]**")
                st.write(c[:500])
                st.markdown("---")
        st.caption("本回答仅供参考，不能替代专业意见。")
    except Exception as e:
        st.error(f"检索失败：{e}")