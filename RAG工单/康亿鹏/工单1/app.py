# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块说明：Streamlit 交互界面。提供 PDF 管理、知识库管理、文字/语音提问、
          答案展示（含引用来源与耗时）以及用户反馈功能。

启动方式：
    streamlit run app.py
"""
import os
import tempfile
import time

import streamlit as st

import config
from src.knowledge_base import clear_knowledge_base, ingest_pdf, knowledge_base_stats
from src.rag_chain import RAGPipeline

st.set_page_config(page_title="PDF 文档问答系统", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# 资源加载（缓存，避免重复加载模型）
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline() -> RAGPipeline:
    return RAGPipeline()


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""


init_state()
pipeline = load_pipeline()


# ---------------------------------------------------------------------------
# 侧边栏：知识库管理 + 参数设置
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 PDF 文档问答系统")
    st.caption(f"工单编号：{config.WORK_ORDER_NO}")

    st.subheader("① 知识库管理")
    uploaded = st.file_uploader("上传 PDF 文档", type=["pdf"])
    if uploaded is not None:
        if st.button("解析并入库", type="primary", use_container_width=True):
            with st.spinner("正在解析并向量化，请稍候……"):
                tmp_path = os.path.join(tempfile.gettempdir(), uploaded.name)
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                try:
                    stats = ingest_pdf(tmp_path, drop_old=True)
                    st.success(f"入库完成：{stats['inserted']} 条片段，耗时 {stats['elapsed']}s")
                except Exception as exc:
                    st.error(f"入库失败：{exc}")

    col_a, col_b = st.columns(2)
    if col_a.button("刷新统计", use_container_width=True):
        st.rerun()
    if col_b.button("清空知识库", use_container_width=True):
        try:
            clear_knowledge_base()
            st.warning("知识库已清空。")
        except Exception as exc:
            st.error(f"清空失败：{exc}")

    try:
        stats = knowledge_base_stats()
        st.info(
            f"集合：{stats['collection']}\n\n"
            f"地址：{stats['uri']}\n\n"
            f"片段数：{stats['count']}"
        )
    except Exception as exc:
        st.error(f"Milvus 未就绪：{exc}")

    st.divider()
    st.subheader("② 检索参数")
    top_k = st.slider("向量召回数量 k", 3, 30, config.RETRIEVE_TOP_K)
    top_n = st.slider("重排后保留数量 top_n", 1, 10, config.RERANK_TOP_N)
    enable_understanding = st.checkbox("启用 Query 理解（意图识别/消歧/分解）", value=True)

    st.divider()
    st.subheader("③ 语音输入")
    audio_value = st.audio_input("录制问题（可选）")
    if audio_value is not None:
        if st.button("识别语音", use_container_width=True):
            try:
                import speech_recognition as sr

                recognizer = sr.Recognizer()
                with sr.AudioFile(audio_value) as source:
                    audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language="zh-CN")
                st.session_state.pending_question = text
                st.success(f"识别结果：{text}")
            except ImportError:
                st.warning("未安装 SpeechRecognition，语音输入不可用。")
            except Exception as exc:
                st.error(f"语音识别失败：{exc}")

    st.divider()
    st.caption("响应时间目标：≤ 3 秒（工单验收指标）")


# ---------------------------------------------------------------------------
# 主区域：问答
# ---------------------------------------------------------------------------
st.markdown("### 💬 基于《招股说明书》的智能问答")
st.caption("支持中文与英文提问。系统将基于 PDF 内容检索并生成回答，同时展示引用来源。")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("meta"):
            with st.expander("查看检索来源与耗时"):
                st.markdown(message["meta"]["sources"], unsafe_allow_html=False)
                st.caption(message["meta"]["timing"])


def render_sources(result) -> str:
    """将检索到的片段渲染为 Markdown 来源列表。"""
    lines = []
    for i, ctx in enumerate(result.contexts, 1):
        meta = ctx.get("metadata", {})
        lines.append(
            f"**[片段{i}]** 第 {meta.get('page', '?')} 页 · 类型 {meta.get('type', 'text')} · "
            f"重排分 {meta.get('rerank_score', 0):.4f}\n\n"
            f"> {ctx['content'][:300].replace(chr(10), ' ')}"
        )
    return "\n\n".join(lines) if lines else "未检索到相关片段。"


question = st.chat_input("请输入你的问题，例如：武汉兴图新科电子股份有限公司法定代表人是谁？")
if st.session_state.pending_question and not question:
    question = st.session_state.pending_question
    st.session_state.pending_question = ""

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer_text = ""
        result = None
        start = time.perf_counter()
        try:
            for res, piece in pipeline.stream_ask(
                question, top_n=top_n, enable_understanding=enable_understanding
            ):
                result = res
                answer_text += piece
                placeholder.markdown(answer_text + "▌")
            placeholder.markdown(answer_text)
        except Exception as exc:
            answer_text = f"回答生成失败：{exc}"
            placeholder.error(answer_text)
        elapsed = round(time.perf_counter() - start, 3)

        meta = None
        if result is not None:
            timing = (
                f"总耗时 {elapsed}s（召回 {result.retrieve_time}s / 重排 {result.rerank_time}s / "
                f"生成 {result.generate_time}s）｜引用页码：{result.pages or '无'}"
            )
            sources = render_sources(result)
            with st.expander("查看检索来源与耗时"):
                st.markdown(sources)
                st.caption(timing)
            if result.analysis is not None:
                st.caption(
                    f"Query 理解 → 意图：{result.analysis.intent}｜规范问题："
                    f"{result.analysis.rewritten_query}｜子问题：{result.analysis.sub_questions or '无'}"
                )
            meta = {"sources": sources, "timing": timing}

        # 用户反馈机制
        col1, col2, col3 = st.columns([1, 1, 6])
        if col1.button("👍 有帮助", key=f"up_{len(st.session_state.messages)}"):
            st.success("感谢你的反馈！")
        if col2.button("👎 需改进", key=f"down_{len(st.session_state.messages)}"):
            st.info("已记录，我们会持续优化检索与生成效果。")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer_text, "meta": meta}
        )
