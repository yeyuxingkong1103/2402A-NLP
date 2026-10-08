# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Query 理解优化任务
app/streamlit_app_v5.py —— 工单五 多轮对话 Streamlit 界面（新增文件）

启动：streamlit run app/streamlit_app_v5.py --server.port 8505
功能：多轮对话（会话保持）/ 指代消解可视化 / 中英文 / 引用展示 / 用户反馈
"""
import json
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from dotenv import load_dotenv
load_dotenv()                                   # 工单五：加载 .env

st.set_page_config(page_title="多轮对话 v5", page_icon="💬", layout="wide")

WORK_ORDER = "人工智能NLP-RAG-Query 理解优化任务"

# ---------------- 工单五：中英文 i18n ----------------
I18N = {
    "中文": {"title": "💬 多轮对话问答", "session": "会话",
             "new_session": "新会话", "input_placeholder": "输入问题，支持多轮追问（如：他/这个公司/那XX呢）",
             "send": "发送", "answer": "答案", "refs": "引用来源",
             "ref_text": "文本", "ref_table": "表格", "ref_image": "图像",
             "latency": "响应(ms)", "resolved": "消解后问句", "entity": "当前实体",
             "strategy": "消解策略", "fb_up": "👍 点赞", "fb_down": "👎 点踩",
             "fb_comment": "评论", "fb_saved": "反馈已保存，感谢！",
             "caption": "图像描述", "ocr": "图内文字(OCR)", "vqa": "图表问答(VQA)"},
    "English": {"title": "💬 Multi-turn Q&A", "session": "Session",
                "new_session": "New Session",
                "input_placeholder": "Ask a question (supports follow-ups)",
                "send": "Send", "answer": "Answer", "refs": "References",
                "ref_text": "Text", "ref_table": "Table", "ref_image": "Image",
                "latency": "Latency(ms)", "resolved": "Resolved Query",
                "entity": "Entity", "strategy": "Strategy",
                "fb_up": "👍 Like", "fb_down": "👎 Dislike",
                "fb_comment": "Comment", "fb_saved": "Feedback saved. Thanks!",
                "caption": "Caption", "ocr": "OCR", "vqa": "VQA"},
}

# 工单五：演示用快捷多轮对话（对应工单五验收的 5 轮问答）
DEMO_DIALOG = [
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    "他参与的哪个工程荣获了国家科技进步一等奖？",
    "这个公司的法定代表人是谁？",
    "那武汉力源信息技术股份有限公司呢？",
    "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",
]


@st.cache_resource(show_spinner="加载多轮对话引擎（首次约1-2分钟）...")
def load_engine():
    """工单五：缓存 ConversationEngine（含 RAGEngineV4 全链路）"""
    from src.conversation_engine import ConversationEngine
    engine = ConversationEngine()
    engine._get_rag().warmup()                   # 工单五：预热模型
    return engine


def save_feedback(payload: dict) -> str:
    """工单五：点赞/点踩/评论 → data/feedback_v5/"""
    fb_dir = Path("data/feedback_v5")
    fb_dir.mkdir(parents=True, exist_ok=True)
    path = fb_dir / f"feedback_{datetime.now():%Y%m%d}.json"
    records = []
    if path.exists():
        records = json.loads(path.read_text(encoding="utf-8"))
    records.append(payload)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return str(path)


def render_refs(result: dict, t: dict):
    """工单五：引用展示（图像/表格/文本）"""
    images = result.get("retrieved_images", [])
    tables = result.get("retrieved_tables", [])
    texts = result.get("retrieved_text_chunks", [])
    if images:
        st.markdown(f"**{t['refs']} · {t['ref_image']}**")
        for i, im in enumerate(images, 1):
            with st.expander(f"图{i} | {im.get('doc_id','')} p{im.get('page','?')}"):
                p = Path(im.get("path", ""))
                if p.exists():
                    st.image(str(p), use_container_width=True)
                if im.get("caption"):
                    st.markdown(f"**{t['caption']}**：{im['caption'][:200]}")
                if im.get("ocr_text"):
                    st.markdown(f"**{t['ocr']}**：\n\n{im['ocr_text'][:400]}")
                if im.get("vqa_text"):
                    st.markdown(f"**{t['vqa']}**：{im['vqa_text'][:300]}")
    if tables:
        st.markdown(f"**{t['refs']} · {t['ref_table']}**")
        for i, tb in enumerate(tables, 1):
            with st.expander(f"表{i} | {tb.get('doc_id','')} p{tb.get('page','?')}"):
                st.code((tb.get("content") or tb.get("table_text") or "")[:500])
    if texts:
        st.markdown(f"**{t['refs']} · {t['ref_text']}**")
        for i, c in enumerate(texts, 1):
            with st.expander(f"资料{i} | {c.get('doc_id','')} p{c.get('page','?')}"):
                st.markdown((c.get("content") or "")[:400])


# ================= 工单五：侧栏 =================
st.sidebar.title("多轮对话 v5")
st.sidebar.caption(f"工单编号：{WORK_ORDER}")
lang = st.sidebar.radio("🌐 Language", list(I18N.keys()), horizontal=True)
t = I18N[lang]

# 工单五：会话管理
if "v5_session_id" not in st.session_state:
    st.session_state["v5_session_id"] = None
if st.sidebar.button(t["new_session"], type="primary"):
    st.session_state["v5_session_id"] = None
    st.session_state["v5_messages"] = []
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("🎬 演示对话（5轮）")
if st.sidebar.button("加载演示对话"):
    st.session_state["v5_demo_idx"] = 0
    st.session_state["v5_session_id"] = None
    st.session_state["v5_messages"] = []

# ================= 工单五：主页面 =================
st.header(t["title"])

if "v5_messages" not in st.session_state:
    st.session_state["v5_messages"] = []

# 工单五：渲染历史消息
for msg in st.session_state["v5_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 工单五：输入框
question = st.chat_input(t["input_placeholder"])
if question:
    engine = load_engine()
    st.session_state["v5_messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("对话理解 + 检索生成中..."):
            t0 = time.time()
            result = engine.chat(
                question,
                session_id=st.session_state["v5_session_id"],
                use_image=True)
            total_ms = (time.time() - t0) * 1000
            st.session_state["v5_session_id"] = result["session_id"]

        st.markdown(result["answer"])

        # 工单五：指代消解可视化
        if result.get("is_followup"):
            st.caption(
                f"🔄 {t['resolved']}: `{result.get('resolved_query','')}` ｜ "
                f"{t['entity']}: {result.get('entity','')} ｜ "
                f"{t['strategy']}: {result.get('coref_strategy','')}")

        render_refs(result, t)

        # 工单五：响应时间
        c1, c2 = st.columns(2)
        c1.metric(t["latency"], f"{total_ms:.0f}")
        c2.metric("会话", result["session_id"])

        # 工单五：用户反馈
        col1, col2, col3 = st.columns([1, 1, 4])
        comment = col3.text_input(t["fb_comment"], key=f"fb_{len(st.session_state['v5_messages'])}")
        if col1.button(t["fb_up"], key=f"up_{len(st.session_state['v5_messages'])}"):
            save_feedback({"ts": datetime.now().isoformat(), "session_id": result["session_id"],
                           "question": question, "answer": result["answer"],
                           "rating": "up", "comment": comment,
                           "latency_ms": round(total_ms, 1), "work_order": WORK_ORDER})
            st.success(t["fb_saved"])
        if col2.button(t["fb_down"], key=f"down_{len(st.session_state['v5_messages'])}"):
            save_feedback({"ts": datetime.now().isoformat(), "session_id": result["session_id"],
                           "question": question, "answer": result["answer"],
                           "rating": "down", "comment": comment,
                           "latency_ms": round(total_ms, 1), "work_order": WORK_ORDER})
            st.success(t["fb_saved"])

    st.session_state["v5_messages"].append(
        {"role": "assistant", "content": result["answer"]})
