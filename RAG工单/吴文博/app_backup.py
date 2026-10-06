import streamlit as st
from utils import init_rag

st.set_page_config(page_title="PDF 智能问答系统", layout="wide")
st.title("📄 基于 PDF 的 RAG 问答系统")
st.markdown("---")

# ---------- 侧边栏：文档上传与索引构建 ----------
with st.sidebar:
    st.header("📁 文档管理")
    uploaded_file = st.file_uploader("上传 PDF 文件", type=["pdf"])

    if uploaded_file is not None:
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("🚀 开始解析并建立索引"):
            with st.spinner("正在解析 PDF 并构建向量库，请稍候..."):
                try:
                    chain, retriever = init_rag("temp_uploaded.pdf")
                    st.session_state.chain = chain
                    st.session_state.retriever = retriever
                    st.success("✅ 文档解析完成，可以开始提问！")
                except Exception as e:
                    st.error(f"解析失败：{e}")

# ---------- 主界面：问答区 ----------
if "chain" not in st.session_state:
    st.info("👈 请先在左侧上传 PDF 并点击『开始解析并建立索引』")
else:
    st.header("💬 问答区")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("请输入你的问题，例如：公司的法定代表人是谁？"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    answer = st.session_state.chain.invoke(prompt)
                    st.markdown(answer)

                    docs = st.session_state.retriever.invoke(prompt)
                    with st.expander("📚 查看引用来源"):
                        for i, d in enumerate(docs):
                            page = d.metadata.get("page", "未知")
                            st.markdown(f"**来源 {i+1}（页码：{page}）**")
                            st.caption(d.page_content[:300] + "...")
                except Exception as e:
                    answer = f"❌ 出错了：{e}"
                    st.error(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})