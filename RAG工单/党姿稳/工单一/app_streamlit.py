# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
交互界面（Streamlit版）：
- 支持上传PDF文档并构建知识库
- 基于PDF内容的RAG问答
- RAG回答 与 纯LLM回答 对比展示
- 展示检索到的相关文档片段
- 支持中英文问答
"""
import os
import time
import streamlit as st
from pdf_parser import build_chunks, build_chunks_from_bytes, load_chunks
from vector_store import VectorStore
from rag_chain import rag_answer, llm_only_answer
from config import CHUNKS_FILE, EVAL_QUESTIONS

# ============ 页面配置 ============
st.set_page_config(
    page_title="基于PDF文档的问答系统 - RAG",
    page_icon="📄",
    layout="wide",
)

# ============ 初始化 session_state ============
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = "招股说明书1.pdf"
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


def load_default_knowledge_base():
    """加载默认的招股说明书知识库"""
    with st.spinner("正在加载默认知识库（招股说明书1.pdf）..."):
        if os.path.exists(CHUNKS_FILE):
            chunks = load_chunks()
        else:
            chunks = build_chunks()
        store = VectorStore(chunks, use_cache=True)
        st.session_state.vector_store = store
        st.session_state.chunk_count = len(chunks)
        st.session_state.doc_name = "招股说明书1.pdf"


def build_knowledge_base_from_upload(pdf_bytes, filename):
    """从上传的PDF构建知识库"""
    with st.spinner(f"正在解析PDF：{filename} ..."):
        chunks, raw_len, clean_len = build_chunks_from_bytes(pdf_bytes)
    with st.spinner(f"正在生成向量索引（{len(chunks)}个文本块）..."):
        store = VectorStore(chunks, use_cache=False)
    st.session_state.vector_store = store
    st.session_state.chunk_count = len(chunks)
    st.session_state.doc_name = filename
    return raw_len, clean_len


# ============ 侧边栏：PDF上传 + 知识库管理 ============
with st.sidebar:
    st.title("📚 知识库管理")
    st.markdown("---")

    # PDF 上传
    st.subheader("📤 上传PDF文档")
    uploaded_file = st.file_uploader(
        "选择PDF文件上传",
        type=["pdf"],
        help="上传后系统将自动解析并构建知识库"
    )

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.read()
        file_size_mb = len(pdf_bytes) / (1024 * 1024)
        st.success(f"✅ 上传成功：{uploaded_file.name}（{file_size_mb:.2f} MB）")

        if st.button("🔨 构建知识库", type="primary", use_container_width=True):
            raw_len, clean_len = build_knowledge_base_from_upload(pdf_bytes, uploaded_file.name)
            st.success(f"知识库构建完成！")
            st.info(f"原始文本：{raw_len:,} 字符\n清洗后：{clean_len:,} 字符\n文本块：{st.session_state.chunk_count} 个")

    st.markdown("---")

    # 加载默认知识库按钮
    st.subheader("📋 默认文档")
    if st.button("加载招股说明书1.pdf", use_container_width=True):
        load_default_knowledge_base()
        st.success("默认知识库加载完成！")

    st.markdown("---")

    # 知识库状态
    st.subheader("📊 当前知识库状态")
    if st.session_state.vector_store is not None:
        st.write(f"**文档：** {st.session_state.doc_name}")
        st.write(f"**文本块数：** {st.session_state.chunk_count}")
        st.write(f"**向量维度：** {st.session_state.vector_store.vectors.shape[1]}")
    else:
        st.warning("⚠️ 知识库未加载，请上传PDF或加载默认文档")

    st.markdown("---")

    # 快捷问题
    st.subheader("⚡ 快捷问题")
    quick_q_options = [q["question"] for q in EVAL_QUESTIONS]
    selected_q = st.selectbox("选择验收问题", [""] + quick_q_options)

# ============ 主区域 ============
st.title("📄 基于PDF文档的问答系统（RAG）")
st.caption("工单编号：人工智能NLP-RAG-基于PDF文档的问答系统")

st.markdown("""
本系统基于检索增强生成（RAG）技术，能够针对上传的PDF文档内容进行精准问答。
左侧上传PDF或加载默认文档后，即可在下方提问。
""")

st.markdown("---")

# 问题输入区
st.subheader("💬 提问")
col_input, col_btn = st.columns([6, 1])
with col_input:
    # 如果侧边栏选了快捷问题，填入输入框
    default_q = selected_q if selected_q else ""
    question = st.text_input(
        "请输入您的问题（支持中英文）",
        value=default_q,
        placeholder="例如：武汉兴图新科电子股份有限公司注册资本是多少？",
        label_visibility="collapsed"
    )
with col_btn:
    submit = st.button("🚀 提问", type="primary", use_container_width=True)

st.markdown("---")

# 回答展示区
if submit and question:
    if st.session_state.vector_store is None:
        st.error("❌ 请先在左侧上传PDF或加载默认知识库！")
    else:
        store = st.session_state.vector_store

        # 进度提示
        with st.spinner("🔍 正在检索文档并生成回答..."):
            # RAG 回答
            rag_ans, contexts, rag_time = rag_answer(question, store)
            # 纯 LLM 回答（对比）
            llm_ans, llm_time = llm_only_answer(question)

        # 展示回答对比
        st.subheader("🤖 回答对比")

        col_rag, col_llm = st.columns(2)
        with col_rag:
            st.markdown("##### ✅ RAG回答（基于PDF文档）")
            st.info(rag_ans)
            st.caption(f"⏱️ 响应时间：{rag_time:.2f} 秒")

        with col_llm:
            st.markdown("##### 💬 纯LLM回答（无文档参考）")
            st.warning(llm_ans)
            st.caption(f"⏱️ 响应时间：{llm_time:.2f} 秒")

        st.markdown("---")

        # 展示检索片段
        st.subheader("📚 检索到的相关文档片段")
        for i, (chunk, score) in enumerate(contexts, 1):
            with st.expander(f"片段 {i} — 相似度：{score:.4f}"):
                st.write(chunk)

        # 响应时间统计
        st.markdown("---")
        st.subheader("📈 响应时间统计")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RAG耗时", f"{rag_time:.2f}s")
        with col2:
            st.metric("纯LLM耗时", f"{llm_time:.2f}s")
        with col3:
            st.metric("检索片段数", f"{len(contexts)}")

# 首次加载时自动加载默认知识库
if st.session_state.vector_store is None:
    load_default_knowledge_base()
    st.rerun()
