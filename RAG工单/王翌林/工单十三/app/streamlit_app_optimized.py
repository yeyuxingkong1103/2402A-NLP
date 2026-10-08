# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
app/streamlit_app_optimized.py —— 工单二优化版前端（端口 8502）

启动：streamlit run app/streamlit_app_optimized.py --server.port 8502
功能：
  1. 界面中英文切换（侧栏 i18n）
  2. 优化前后答案对比（可折叠 expander，展开才调基线链路）
  3. 引用来源：页码 + 相关度 + 原文片段
  4. 响应时间指标（含检索/生成分解）
  5. 反馈按钮：点赞/点踩/评论（POST /api/feedback）
  6. 语音输入：st.audio_input 录音 + faster-whisper 转写（未安装时给出实现方式说明）
  7. 文档解析入口：上传 PDF → 调用工单二优化解析器展示统计
"""
import json
import os
import sys
import tempfile
import time

import requests
import streamlit as st

st.set_page_config(page_title="PDF 智能问答（工单二优化版）", page_icon="🚀", layout="wide")

# ---------------- i18n（工单二：中英文界面切换，人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------------
I18N = {
    "zh": {"title": "📄 基于 PDF 文档的智能问答", "sub": "工单二优化版：混合检索 · 重排序 · 父子块 · 多语言 · 语义缓存",
           "lang_label": "界面语言 / Language", "answer_lang": "回答语言",
           "qa": "💬 智能问答", "question": "请输入您的问题（支持中英文）：",
           "ask": "🚀 提问", "voice": "🎤 语音输入（可选）", "compare": "⚖️ 优化前后答案对比",
           "refs": "📚 引用来源（页码 / 相关度 / 原文片段）", "latency": "总耗时", "retrieve": "检索",
           "generate": "生成", "cache": "缓存", "feedback": "本条回答是否有帮助？",
           "good": "👍 点赞", "bad": "👎 点踩", "comment": "✍️ 补充评论", "submit": "提交反馈",
           "fb_ok": "感谢您的反馈！", "upload": "📎 文档解析（上传 PDF）",
           "upload_help": "上传 PDF 使用工单二优化解析器（版面分析+表格结构化+OCR兜底）",
           "voice_help": "实现方式：st.audio_input 录音 → faster-whisper 本地转写 → 填入问题框。"
                         "安装：pip install faster-whisper（首次运行自动下载 base 模型约 140MB）",
           "no_api": "API 服务未启动，请先运行：uvicorn src.api:app --port 8000"},
    "en": {"title": "📄 PDF Document QA", "sub": "Optimized: Hybrid Retrieval · Rerank · Parent-Child · Multilingual · Cache",
           "lang_label": "界面语言 / Language", "answer_lang": "Answer Language",
           "qa": "💬 Ask", "question": "Enter your question (Chinese or English):",
           "ask": "🚀 Ask", "voice": "🎤 Voice Input (optional)", "compare": "⚖️ Before / After Comparison",
           "refs": "📚 References (page / score / snippet)", "latency": "Total", "retrieve": "Retrieval",
           "generate": "Generation", "cache": "Cache", "feedback": "Was this answer helpful?",
           "good": "👍 Upvote", "bad": "👎 Downvote", "comment": "✍️ Comment", "submit": "Submit",
           "fb_ok": "Thanks for your feedback!", "upload": "📎 Document Parsing (upload PDF)",
           "upload_help": "Upload a PDF parsed by the optimized parser (layout + tables + OCR fallback)",
           "voice_help": "How it works: st.audio_input records audio → faster-whisper transcribes locally "
                         "→ fills the question box. Install: pip install faster-whisper (~140MB base model).",
           "no_api": "API not running. Start it: uvicorn src.api:app --port 8000"},
}
st.session_state.setdefault("ui_lang", "zh")
lang_ui = st.session_state["ui_lang"]
T = I18N[lang_ui]

# ---------------- 侧栏（工单二优化版，人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------------
with st.sidebar:
    st.header("⚙️ 设置 / Settings")
    lang_choice = st.radio(T["lang_label"], ["中文", "English"], horizontal=True)
    st.session_state["ui_lang"] = {"中文": "zh", "English": "en"}[lang_choice]
    if st.session_state["ui_lang"] != lang_ui:  # 切换语言后立即生效
        st.rerun()
    ans_lang = st.radio(T["answer_lang"], ["自动检测 / Auto", "中文", "English"], horizontal=True)
    st.session_state["answer_lang"] = {"自动检测 / Auto": None, "中文": "zh", "English": "en"}[ans_lang]
    # 工单二：API Base 支持环境变量注入（部署脚本 start_optimized.sh 固定 FastAPI=8000）
    _default_api_base = os.getenv("API_BASE", f"http://127.0.0.1:{os.getenv('APP_PORT', '8000')}")
    api_base = st.text_input("API Base", _default_api_base)
    top_k = st.slider("Top-K", 1, 20, 8)
    st.divider()
    # 工单二：文档上传解析入口（人工智能NLP-RAG-基于PDF文档的问答系统优化）
    with st.expander(T["upload"]):
        st.caption(T["upload_help"])
        up = st.file_uploader("PDF", type=["pdf"], key="pdf_upload")
        if up is not None and st.button("解析 / Parse"):
            import fitz  # PyMuPDF：工单二优化解析统计（人工智能NLP-RAG-基于PDF文档的问答系统优化）
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                tf.write(up.getvalue())
                tmp_path = tf.name
            try:
                doc = fitz.open(tmp_path)
                n_tables = sum(len(page.find_tables().tables) for page in doc)
                st.success(f"✅ {up.name}: {doc.page_count} pages, {n_tables} tables")
                doc.close()
            finally:
                os.unlink(tmp_path)

# ---------------- 主区：提问（工单二，人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------------
st.title(T["title"])
st.caption(T["sub"])
st.subheader(T["qa"])
question = st.text_area(T["question"], height=90, key="q_box")

# 语音输入：st.audio_input 录音 + faster-whisper 本地转写；
# 未安装转写模型时在面板内常驻展示「实现方式」说明（工单二，人工智能NLP-RAG-基于PDF文档的问答系统优化）
with st.expander(T["voice"]):
    try:
        audio = st.audio_input("🎤 Record / 录音")  # Streamlit >= 1.41
        try:
            from faster_whisper import WhisperModel  # noqa: F401
            _whisper_ok = True
        except ImportError:
            _whisper_ok = False
            st.info(T["voice_help"])  # 实现方式：录音→faster-whisper 转写→填入问题框
        if audio is not None:
            if not _whisper_ok:
                st.info(T["voice_help"])
            else:
                from faster_whisper import WhisperModel
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    tf.write(audio.getvalue())
                    wav = tf.name
                try:
                    segs, _ = WhisperModel("base", device="cpu", compute_type="int8").transcribe(wav)
                    text = "".join(s.text for s in segs)
                finally:
                    os.unlink(wav)
                if text.strip():
                    st.session_state["q_box"] = text
                    st.rerun()
                st.info(f"🗣️ {text}")
    except AttributeError:
        st.info(T["voice_help"])  # 旧版 Streamlit 无 st.audio_input 时说明实现方式

col1, col2 = st.columns([1, 4])
with col1:
    ask_clicked = st.button(T["ask"], type="primary", use_container_width=True)


def _post(path: str, payload: dict, timeout: int = 120):
    """统一 POST（人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    return requests.post(f"{api_base}{path}", json=payload, timeout=timeout)


if ask_clicked and question.strip():
    st.session_state["fb_rating"] = 0  # 新问题重置上一条反馈（工单二，人工智能NLP-RAG-基于PDF文档的问答系统优化）
    payload = {"question": question, "top_k": top_k, "use_rag": True,
               "lang": st.session_state.get("answer_lang")}
    try:
        t0 = time.time()
        r = _post("/api/ask", payload)  # 工单二优化链路（人工智能NLP-RAG-基于PDF文档的问答系统优化）
        r.raise_for_status()
        # 结果持久化到 session_state：点赞/点踩/提交反馈等交互触发 rerun 后，
        # 答案、引用、对比面板仍保留（此前渲染放在 if ask_clicked 内，rerun 即消失）
        # （工单二，人工智能NLP-RAG-基于PDF文档的问答系统优化）
        st.session_state["last_resp"] = r.json()
        st.session_state["last_payload"] = payload
        st.session_state["baseline_resp"] = None  # 新问题重置基线对比缓存
        st.session_state["last_latency"] = f"{(time.time() - t0):.1f}s"
    except requests.exceptions.ConnectionError:
        st.error(T["no_api"])
    except Exception as e:
        st.error(f"请求失败 / Request failed: {e}")

# 结果展示区独立于 ask_clicked：只要本轮会话已有回答就持续渲染
# （工单二，人工智能NLP-RAG-基于PDF文档的问答系统优化）
data = st.session_state.get("last_resp")
if data:
    st.subheader("📝 " + ("RAG 优化版回答" if lang_ui == "zh" else "Optimized RAG Answer"))
    st.markdown(data.get("answer", ""))
    # 4. 响应时间指标（总耗时 + 检索/生成分解；语义缓存命中时无分解阶段，显示 —）
    #    （工单二，人工智能NLP-RAG-基于PDF文档的问答系统优化）
    bd = data.get("breakdown") or {}
    m = st.columns(4)
    m[0].metric(T["latency"], f"{data.get('latency_ms', 0):.0f} ms")
    m[1].metric(T["retrieve"],
                f"{bd['retrieve_ms']:.0f} ms" if bd.get("retrieve_ms") is not None else "—")
    m[2].metric(T["generate"],
                f"{bd['llm_ms']:.0f} ms" if bd.get("llm_ms") is not None else "—")
    m[3].metric(T["cache"], "✓" if data.get("cache_hit") else "—")
    # 3. 引用来源：页码 + 相关度 + 原文片段（人工智能NLP-RAG-基于PDF文档的问答系统优化）
    with st.expander(f"{T['refs']}（{len(data.get('references', []))}）"):
        for ref in data.get("references", []):
            st.markdown(f"**第 {ref.get('page')} 页** · score={ref.get('score')}")
            st.text(ref.get("preview", ""))
            st.divider()
    # 2. 优化前后对比（可折叠，展开才调基线链路；结果按问题缓存，避免反复请求）
    #    （工单二，人工智能NLP-RAG-基于PDF文档的问答系统优化）
    with st.expander(T["compare"]):
        base_r = st.session_state.get("baseline_resp")
        if base_r is None:
            try:
                with st.spinner("基线对比生成中 / Generating baseline…"):
                    base_r = _post("/api/ask",
                                   dict(st.session_state.get("last_payload", {}),
                                        chain="baseline")).json()
                st.session_state["baseline_resp"] = base_r
            except Exception as e:
                st.warning(f"基线链路不可用: {e}")
        if base_r is not None:
            c1, c2 = st.columns(2)
            c1.markdown(f"**优化前（基线）** · {base_r.get('latency_ms', 0):.0f} ms")
            c1.info(base_r.get("answer", "")[:600] or "（无）")
            c2.markdown(f"**优化后（工单二）** · {data.get('latency_ms', 0):.0f} ms")
            c2.success(data.get("answer", "")[:600])
    # 5. 反馈按钮：点赞/点踩/评论（rating 持久化到 session_state；
    #    +1=赞 / -1=踩 / 0=仅评论）（工单二，人工智能NLP-RAG-基于PDF文档的问答系统优化）
    st.session_state.setdefault("fb_rating", 0)
    st.caption(T["feedback"])
    f1, f2, f3 = st.columns([1, 1, 5])
    if f1.button(T["good"] + ("　✅" if st.session_state["fb_rating"] == 1 else ""), key="fb_up"):
        st.session_state["fb_rating"] = 1
        st.rerun()
    if f2.button(T["bad"] + ("　✅" if st.session_state["fb_rating"] == -1 else ""), key="fb_down"):
        st.session_state["fb_rating"] = -1
        st.rerun()
    comment = f3.text_input(T["comment"], key="fb_text")
    if st.button(T["submit"], key="fb_submit"):
        try:
            _post("/api/feedback", {"qa_log_id": data.get("qa_log_id"),
                                    "rating": st.session_state["fb_rating"],
                                    "comment": comment}, timeout=15)
            st.success(T["fb_ok"])
            st.session_state["fb_rating"] = 0
        except Exception as e:
            st.warning(f"feedback: {e}")
