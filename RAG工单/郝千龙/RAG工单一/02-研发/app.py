# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 【Streamlit前端 · app.py】中英双语问答界面：语音输入、RAG vs 基线对比、反馈、知识库管理、RAGAS评估
# 编写日期：2026-09-28   修订日期：2026-10-04
from pathlib import Path

import httpx
import streamlit as st
import streamlit.components.v1 as components

import config

API_BASE = f"http://127.0.0.1:{config.API_PORT}"

# 注册本地静态语音组件（Web Speech API）
_voice_comp = components.declare_component(
    "voice_input", path=str(Path(__file__).parent / "voice_component")
)

# ---------- 双语文案 ----------
I18N = {
    "zh": {
        "title": "招股说明书 RAG 问答系统",
        "caption": "工单编号：人工智能NLP-RAG-基于PDF文档的问答系统",
        "tab_qa": "💬 智能问答",
        "tab_kb": "📚 知识库管理",
        "tab_eval": "📊 RAGAS 评估",
        "question": "❓ 你的问题",
        "placeholder": "例如：公司注册资本是多少？",
        "ask": "🚀 提问",
        "asking": "RAG 检索 + 生成中...",
        "rag_ans": "✅ RAG 答案",
        "base_ans": "⚠️ 仅 LLM 答案（无检索）",
        "refs": "📎 引用来源",
        "quick": "工单 10 题快速验证",
        "latency": "延迟",
        "cache_hit": "缓存命中",
        "config": "⚙️ 配置",
        "topk": "检索 Top-K",
        "use_cache": "启用缓存",
        "show_base": "显示基线对比（仅 LLM）",
        "feedback_q": "这个回答对你有帮助吗？",
        "comment": "补充意见（可选）",
        "submit_fb": "提交反馈",
        "fb_done": "✅ 反馈已记录，感谢！",
        "kb_docs": "已入库文档",
        "kb_upload": "上传新 PDF 入库",
        "kb_delete": "删除",
        "kb_feedback": "用户反馈记录",
        "eval_btn": "一键跑工单 10 题真实 RAGAS 评估（约需数分钟）",
        "eval_running": "正在评估，请耐心等待...",
        "health": "健康检查",
        "lang_name": "语言",
    },
    "en": {
        "title": "Prospectus RAG Q&A System",
        "caption": "Work Order: NLP-RAG-PDF-Document-QA-System",
        "tab_qa": "💬 Q&A",
        "tab_kb": "📚 Knowledge Base",
        "tab_eval": "📊 RAGAS Evaluation",
        "question": "❓ Your question",
        "placeholder": "e.g. What is the registered capital?",
        "ask": "🚀 Ask",
        "asking": "RAG retrieval + generation...",
        "rag_ans": "✅ RAG Answer",
        "base_ans": "⚠️ LLM-only Answer (no retrieval)",
        "refs": "📎 References",
        "quick": "Work Order: 10 Questions",
        "latency": "Latency",
        "cache_hit": "Cache hit",
        "config": "⚙️ Settings",
        "topk": "Top-K",
        "use_cache": "Enable cache",
        "show_base": "Show LLM baseline",
        "feedback_q": "Was this answer helpful?",
        "comment": "Additional comments (optional)",
        "submit_fb": "Submit feedback",
        "fb_done": "✅ Feedback recorded. Thanks!",
        "kb_docs": "Documents in KB",
        "kb_upload": "Upload a new PDF",
        "kb_delete": "Delete",
        "kb_feedback": "User Feedback",
        "eval_btn": "Run real RAGAS on 10 questions (several minutes)",
        "eval_running": "Evaluating, please wait...",
        "health": "Health check",
        "lang_name": "Language",
    },
}

# ---------- 页面配置 ----------
st.set_page_config(page_title="RAG QA", page_icon=":books:", layout="wide")

# 侧边栏：语言切换（置顶）
with st.sidebar:
    lang_choice = st.radio(
        "🌐 Language / 语言", ["中文", "English"], horizontal=True, label_visibility="collapsed"
    )
lang = "en" if lang_choice == "English" else "zh"
T = I18N[lang]

st.title(T["title"])
st.caption(T["caption"])

# 侧边栏：其余配置
with st.sidebar:
    st.divider()
    st.header(T["config"])
    top_k = st.slider(T["topk"], 1, 10, 5)
    use_cache = st.checkbox(T["use_cache"], value=True)
    show_baseline = st.checkbox(T["show_base"], value=True)

    st.divider()
    st.subheader(T["quick"])
    for q in config.WORKORDER_QUESTIONS:
        label = q["question"] if lang == "zh" else f"Q{q['id']}"
        if st.button(label[:32], key=f"q{q['id']}"):
            st.session_state["question"] = q["question"]

    st.divider()
    if st.button(T["health"]):
        try:
            st.json(httpx.get(f"{API_BASE}/api/health", timeout=3).json())
        except Exception as e:
            st.error(str(e))


def post_ask(question: str, baseline: bool = False):
    """调用后端问答接口"""
    url = f"{API_BASE}/api/ask" + ("/baseline" if baseline else "")
    return httpx.post(
        url,
        json={"question": question, "top_k": top_k, "use_cache": use_cache,
              "lang": lang},
        timeout=20,
    ).json()


# ---------- 三个主标签页 ----------
tab_qa, tab_kb, tab_eval = st.tabs([T["tab_qa"], T["tab_kb"], T["tab_eval"]])

# ============ 页签一：智能问答 ============
with tab_qa:
    col_in, col_out = st.columns([1, 2])

    with col_in:
        default_q = st.session_state.get("question", "")
        question = st.text_area(
            T["question"], value=default_q, height=100,
            placeholder=T["placeholder"], key="q_input",
        )

        # 语音输入组件
        voice_val = _voice_comp(lang=lang, default=None, key="voice")
        if voice_val and voice_val.get("status") == "done" and voice_val.get("text"):
            spoken = voice_val["text"].strip()
            # 与已有文本拼接，随后 rerun 让输入框刷新
            cur = st.session_state.get("question", "")
            st.session_state["question"] = (cur + " " + spoken).strip() if cur else spoken
            st.rerun()

        ask_btn = st.button(T["ask"], type="primary")

    with col_out:
        if ask_btn and question.strip():
            st.session_state["current_q"] = question
            with st.spinner(T["asking"]):
                try:
                    st.session_state["rag_resp"] = post_ask(question)
                except Exception as e:
                    st.error(f"{e}")
                if show_baseline:
                    try:
                        st.session_state["base_resp"] = post_ask(question, baseline=True)
                    except Exception as e:
                        st.warning(f"baseline: {e}")

        # 展示 RAG 答案
        rag = st.session_state.get("rag_resp")
        if rag and rag.get("code") == 0:
            d = rag["data"]
            st.success(
                f"{T['rag_ans']}  ·  {T['latency']} {d['latency_ms']}ms  ·  "
                f"{T['cache_hit']}: {d['cache_hit']}"
            )
            st.markdown(d["answer"])
            if d.get("refs"):
                with st.expander(f"{T['refs']}（{len(d['refs'])}）"):
                    for i, r in enumerate(d["refs"], 1):
                        st.markdown(f"**[{i}] p.{r['page']} · score={r['score']:.3f}**")
                        st.caption(r["text"][:240])

            # 反馈区
            st.divider()
            st.markdown(f"**{T['feedback_q']}**")
            fb_col1, fb_col2 = st.columns([1, 3])
            with fb_col1:
                rating = st.radio(
                    "rating", ["👍", "👎"], horizontal=True,
                    label_visibility="collapsed", key="fb_rating",
                )
            with fb_col2:
                comment_text = st.text_input(T["comment"], key="fb_comment")
            if st.button(T["submit_fb"], key="fb_submit"):
                try:
                    r = httpx.post(
                        f"{API_BASE}/api/feedback",
                        json={
                            "question": st.session_state.get("current_q", ""),
                            "answer": d["answer"],
                            "rating": "up" if rating == "👍" else "down",
                            "comment": comment_text,
                        },
                        timeout=10,
                    ).json()
                    if r["code"] == 0:
                        st.success(T["fb_done"])
                except Exception as e:
                    st.error(str(e))

        # 展示基线答案
        base = st.session_state.get("base_resp")
        if base and base.get("code") == 0:
            st.divider()
            b = base["data"]
            st.warning(f"{T['base_ans']}  ·  {T['latency']} {b['latency_ms']}ms")
            st.markdown(b["answer"])

# ============ 页签二：知识库管理 ============
with tab_kb:
    col_docs, col_up = st.columns([3, 2])

    with col_docs:
        st.subheader(T["kb_docs"])
        try:
            info = httpx.get(f"{API_BASE}/api/documents", timeout=5).json()
            if info["code"] == 0 and info["data"]["documents"]:
                st.dataframe(
                    [
                        {"doc_id": d["doc_id"], "name": d["name"],
                         "pages": d["pages"], "chunks": d["chunks"],
                         "ingest_at": d["ingest_at"]}
                        for d in info["data"]["documents"]
                    ],
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    f"total chunks: {info['data']['total_chunks']} · "
                    f"vector: {info['data']['vector_backend']} · "
                    f"cache: {info['data']['cache_backend']}"
                )
                # 删除文档
                del_id = st.selectbox(
                    "doc_id",
                    [d["doc_id"] for d in info["data"]["documents"]],
                    label_visibility="collapsed",
                )
                if st.button(f"🗑 {T['kb_delete']} {del_id}"):
                    httpx.delete(f"{API_BASE}/api/documents/{del_id}", timeout=30)
                    st.rerun()
            else:
                st.info("Empty")
        except Exception as e:
            st.error(f"backend offline: {e}")

    with col_up:
        st.subheader(T["kb_upload"])
        up_file = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")
        if up_file and st.button("📤 Upload & Ingest"):
            with st.spinner("ingesting..."):
                try:
                    r = httpx.post(
                        f"{API_BASE}/api/upload",
                        files={"file": (up_file.name, up_file.getvalue(), "application/pdf")},
                        timeout=600,
                    ).json()
                    if r["code"] == 0:
                        st.success(f"✅ {r['data']['name']} → {r['data']['chunks']} chunks")
                        st.rerun()
                    else:
                        st.error(r["message"])
                except Exception as e:
                    st.error(str(e))

    st.divider()
    st.subheader(T["kb_feedback"])
    try:
        fb_data = httpx.get(f"{API_BASE}/api/feedback", timeout=5).json()
        items = fb_data["data"]["items"]
        if items:
            st.dataframe(
                [{"rating": x["rating"], "question": x["question"],
                  "comment": x["comment"], "ts": x["ts"]} for x in reversed(items)],
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("No feedback yet")
    except Exception as e:
        st.error(str(e))

# ============ 页签三：RAGAS 评估 ============
with tab_eval:
    st.markdown("**RAGAS**: faithfulness · answer_relevancy · context_precision · context_recall")
    if st.button(T["eval_btn"]):
        with st.spinner(T["eval_running"]):
            try:
                r = httpx.post(
                    f"{API_BASE}/api/evaluate/workorder", timeout=600
                ).json()
                if r["code"] == 0:
                    data = r["data"]
                    cols = st.columns(4)
                    for i, (k_name, v) in enumerate(data["summary"].items()):
                        if k_name != "n":
                            cols[i % 4].metric(k_name, f"{v:.3f}")
                    st.dataframe(
                        [
                            {"id": x.get("id"), "question": x["question"],
                             "faithfulness": x["faithfulness"],
                             "answer_relevancy": x["answer_relevancy"],
                             "context_precision": x["context_precision"],
                             "context_recall": x["context_recall"],
                             "latency_ms": x.get("latency_ms")}
                            for x in data["per_question"]
                        ],
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.error(r["message"])
            except Exception as e:
                st.error(str(e))

# ====================================================================
# 技术备注：
# 1. RAG：同屏展示 RAG 与纯 LLM 答案（工单对比要求），引用页码可溯源；
#    反馈数据回流，支撑知识库迭代。
# 2. 语音输入使用浏览器 Web Speech API，经 Streamlit 静态组件桥接，零额外依赖。
# 3. Transformer：问答、语音识别与 RAGAS 评判背后均为 Transformer 架构。
# 4. Fine-tuning：评估页可量化对比微调前后指标变化。
# ====================================================================
