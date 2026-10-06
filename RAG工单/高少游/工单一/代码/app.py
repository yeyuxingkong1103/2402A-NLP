# -*- coding: utf-8 -*-
"""RAG 问答系统 · Streamlit 交互界面
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统

功能：
- 文字提问（支持 RAG 检索增强 与 纯 LLM 对比两种模式）
- 语音提问（上传音频 → 本地 Whisper 转文字，可选；无依赖时自动隐藏）
- 回答依据展示（来源页码高亮 + 相似来源片段可折叠展开）
- 用户反馈（👍/👎 + 可选留言，写入 output/feedback.csv）

启动（项目根目录，激活 langchain2 环境，需已启动 Ollama）：
    streamlit run app.py
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import streamlit as st

from src import config
from src.knowledge_base import _kv_file, build_kb
from src.qa_engine import QAEngine


# ---------------- 页面初始化 ------------------------------------------------
st.set_page_config(
    page_title="RAG 招股说明书问答系统",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="正在加载问答引擎（首次较慢）...")
def get_engine() -> QAEngine:
    if not _kv_file(config.DB_DIR).exists():
        build_kb()  # 首次运行自动构建向量库（可执行 scripts/build_kb.py 重建）
    return QAEngine()


def save_feedback(question: str, answer: str, rating: str, note: str) -> None:
    """把用户反馈追加保存到 output/feedback.csv。"""
    config.ensure_dirs()
    path = Path(config.OUTPUT_DIR) / "feedback.csv"
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ts", "mode", "question", "rating", "note", "answer"],
        )
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": st.session_state.get("mode", "rag"),
                "question": question,
                "rating": rating,
                "note": note or "",
                "answer": answer[:500],
            }
        )


# ---------------- 语音转文字（可选依赖 whisper） ------------------------------
def transcribe_audio(file) -> str:
    """使用本地 whisper 对上传音频转文字；whisper 不可用时给出提示。"""
    try:
        import whisper  # type: ignore
    except Exception:
        st.warning("语音转文字需要安装 openai-whisper，当前仅支持文字输入。")
        return ""
    try:
        model = whisper.load_model(os.getenv("WHISPER_MODEL", "base"))
        result = model.transcribe(file, language="zh")
        return (result.get("text") or "").strip()
    except Exception as exc:  # 本地模型加载/音频解码失败
        st.warning(f"语音转文字失败：{exc}")
        return ""


# ---------------- 侧边栏 ------------------------------------------------------
with st.sidebar:
    st.title("📄 RAG 问答系统")
    st.caption("基于招股说明书 PDF 的检索增强问答")
    st.divider()
    mode = st.radio(
        "回答模式",
        ["rag", "llm_only"],
        format_func=lambda m: "RAG 检索增强" if m == "rag" else "纯 LLM 对比",
        help="RAG 会先检索 PDF 依据再回答；纯 LLM 仅靠模型知识（可作对照）。",
    )
    top_k = st.slider("检索片段数 Top-K", 1, 20, config.TOP_K, 1)
    st.divider()
    if st.button("🔄 重新加载知识库"):
        st.cache_resource.clear()
        st.toast("知识库缓存已清空，下一次提问将重新加载")
    if st.button("下载反馈记录"):
        fb = Path(config.OUTPUT_DIR) / "feedback.csv"
        if fb.exists():
            st.download_button("下载 feedback.csv", fb.read_bytes(encoding="utf-8-sig"),
                               file_name="feedback.csv", mime="text/csv")
        else:
            st.info("暂无反馈记录")

st.session_state["mode"] = mode


# ---------------- 主区域 ------------------------------------------------------
st.title("基于 PDF 文档的问答系统")
st.caption("本地 langchain2 + Ollama（bge-m3 · deepseek-r1） ｜ 单文档：招股说明书1.pdf")

# 顶部：模式说明
if mode == "rag":
    st.success("**RAG 模式**：系统先检索招股说明书相关片段，再严格依据文档生成带页码的回答。")
else:
    st.info("**纯 LLM 模式**：不注入文档上下文，仅靠模型自身知识回答，用于与 RAG 结果对比。")

# 提问输入
left, right = st.columns([4, 1])
with left:
    question = st.text_input(
        "请输入您的问题",
        placeholder="例如：本次发行募集资金计划如何使用？",
        key="question_input",
    )
with right:
    q_btn = st.button("🔍 提问", type="primary", use_container_width=True)

# 语音提问
with st.expander("🎤 语音提问（可选）"):
    voice_file = st.file_uploader(
        "上传音频（wav/mp3/ogg），转文字后自动填入问题框",
        type=["wav", "mp3", "ogg", "m4a"],
        key="voice_file",
    )
    if voice_file is not None and st.button("转文字"):
        with st.spinner("正在识别语音..."):
            text = transcribe_audio(voice_file)
        if text:
            st.session_state["question_input"] = text
            st.success(f"识别结果：{text}")

# ---------------- 执行问答 ----------------------------------------------------
if q_btn or (question and st.session_state.get("auto_run")):
    if not question:
        st.warning("请先输入问题。")
        st.stop()
    with st.spinner("正在分析并检索生成，请稍候..."):
        engine = get_engine()
        start = time.time()
        if mode == "rag":
            result = engine.answer_rag(question)
        else:
            result = engine.answer_llm_only(question)
        elapsed = result.elapsed

    # 参考耗时指标
    met = "✅ 达标" if elapsed <= config.TARGET_RESPONSE_SECONDS else "⚠️ 超时"
    st.markdown(
        f"<div style='font-size:.9rem;color:#888'>耗时 {elapsed:.2f}s ｜ "
        f"目标 ≤ {config.TARGET_RESPONSE_SECONDS}s ｜ {met}</div>",
        unsafe_allow_html=True,
    )

    st.subheader("💬 回答")
    st.write(result.answer)

    if mode == "rag" and result.sources:
        with st.expander(f"📚 检索依据（{len(result.sources)} 条，来源页码见标注）", expanded=False):
            for i, src in enumerate(result.sources, 1):
                st.markdown(
                    f"**依据 {i}** ｜ 第 {src['page']} 页 ｜ 相似度 {src['score']}"
                )
                st.caption(src["text"])
                st.divider()
        page_refs = "、".join(sorted({str(s["page"]) for s in result.sources if s.get("page")}))
        st.caption(f"关键数据来源页码：第 {page_refs} 页")

    # 反馈
    st.divider()
    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        ok = st.button("👍 有帮助")
    with c2:
        bad = st.button("👎 没帮助")
    with c3:
        note = st.text_input("补充反馈（可选）", key="fb_note")
    if ok or bad:
        rating = "up" if ok else "down"
        save_feedback(question, result.answer, rating, note or "")
        st.toast("感谢您的反馈！")

st.divider()
st.caption(
    "技术栈：LangChain 1.3 + FAISS + BM25 混合检索 + RRF 融合 ｜ "
    "向量模型 bge-m3 ｜ 生成模型 deepseek-r1 ｜ 评估框架 RAGAS"
)