"""
streamlit_app.py — 前端问答页面

直接调用 chat.chat_stream()，不经过 FastAPI，本地单进程即可运行：

    streamlit run streamlit_app.py

页面能力：
    登录页  只填姓名，老用户自动认出，开新对话无需重复注册
    左侧栏  上传 PDF 建库、开启新对话、退出登录
    主区域  对话输入、流式回答、标注领域与引用的资料来源
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

import chat
import config
import session_memory
import users
from knowledge_base import KnowledgeBase

st.set_page_config(page_title="多角色 RAG 智能问答系统", page_icon="🤖", layout="wide")

DOMAIN_LABELS = {
    "legal": "法律",
    "medical": "医疗",
    "english": "英语",
    "chat": "闲聊",
}


@st.cache_resource
def get_kb() -> KnowledgeBase:
    """知识库单例，避免每次交互重建分块器与解析器。"""
    return KnowledgeBase()


def init_state() -> None:
    st.session_state.setdefault("user_id", None)
    st.session_state.setdefault("user_name", "")
    st.session_state.setdefault("messages", [])


def load_history(user_id: str) -> list[dict]:
    """从短期记忆恢复该用户的对话，刷新页面不丢上下文。"""
    try:
        turns = session_memory.get_short_term(user_id)
    except Exception:
        return []
    return [
        {"role": t["role"], "content": t["content"]}
        for t in turns
        if t.get("role") in {"user", "assistant"} and t.get("content")
    ]


# ---------------------------------------------------------------- 登录页


def render_login() -> None:
    """只填姓名的登录 / 注册页：新用户自动注册，老用户直接进。"""
    st.title("多角色 RAG 智能问答系统")
    st.caption("请填写姓名后开始使用。首次填写会自动注册，之后填同一个姓名即可继续。")

    with st.form("login"):
        name = st.text_input("姓名", placeholder="例如：张三")
        submitted = st.form_submit_button("登录 / 注册", use_container_width=True)

    if not submitted:
        return

    try:
        user_id, created = users.login_or_register(name)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.session_state.user_id = user_id
    st.session_state.user_name = name.strip()
    st.session_state.messages = load_history(user_id)
    st.success(f"{'注册成功' if created else '欢迎回来'}，{name.strip()}（{user_id}）")
    st.rerun()


# ---------------------------------------------------------------- 侧边栏


def render_sidebar() -> None:
    with st.sidebar:
        st.title("🤖 RAG 问答系统")
        st.caption(f"当前用户：**{st.session_state.user_name}**")

        st.divider()
        st.subheader("知识库")
        domain = st.selectbox(
            "导入领域",
            options=list(config.KB_COLLECTIONS.keys()),
            format_func=lambda d: DOMAIN_LABELS.get(d, d),
        )
        uploaded = st.file_uploader("上传 PDF 建库", type=["pdf"])
        if uploaded is not None and st.button("导入知识库", use_container_width=True):
            import_pdf(uploaded, domain)

        st.divider()
        st.subheader("对话")
        st.caption(f"短期保留最近 {config.SHORT_TERM_ROUNDS} 轮，长期记忆不受影响")
        if st.button("开启新对话", use_container_width=True):
            # 只清短期记忆：换话题，但之前聊过的偏好和事实还留着
            session_memory.clear_short_term(st.session_state.user_id)
            st.session_state.messages = []
            st.success("已开启新对话")
            st.rerun()

        st.divider()
        if st.button("退出登录", use_container_width=True):
            for key in ("user_id", "user_name", "messages"):
                st.session_state.pop(key, None)
            st.rerun()


def import_pdf(uploaded, domain: str) -> None:
    """把上传的 PDF 落到 data/{domain}/ 再入库。"""
    target_dir = config.DATA_DIR / domain
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / Path(uploaded.name).name

    with st.spinner(f"正在解析并导入 {uploaded.name} ..."):
        try:
            target.write_bytes(uploaded.getbuffer())
            report = get_kb().import_pdf(target, domain)
        except Exception as exc:
            st.error(f"导入失败：{exc}")
            return

    st.success(
        f"导入完成：{report.get('pages', 0)} 页 → "
        f"原始 {report.get('raw_chunks', 0)} 块 → "
        f"过滤后 {report.get('filtered_chunks', 0)} 块 → "
        f"入库 {report.get('stored', 0)} 条"
    )


# ---------------------------------------------------------------- 对话渲染


def render_sources(sources: list[dict]) -> None:
    """渲染引用来源。一条资料都没检索到时提示"通用知识"。"""
    if not sources:
        return

    if sources[0].get("type") == "general":
        st.caption("📚 本轮未检索到知识库资料，回答基于通用知识")
        return

    with st.expander(f"📚 引用了 {len(sources)} 条资料"):
        for item in sources:
            where = item["file"]
            if item.get("page"):
                where += f" 第{item['page']}页"
            st.markdown(
                f"**[资料{item['index']}]** `{where}` · 相关度 {item['score']:.3f}"
            )
            st.caption(item["preview"])


def stream_reply(user_id: str, prompt: str) -> tuple[str, dict | None]:
    """驱动一次流式对话，实时渲染增量内容，返回完整回复与元信息。"""
    status = st.status("正在检索知识库...", expanded=False)
    meta: dict | None = None
    text = ""
    placeholder = st.empty()

    for event in chat.chat_stream(user_id, prompt):
        kind = event.get("type")

        if kind == "meta":
            meta = event
            label = DOMAIN_LABELS.get(event["domain"], event["domain"])
            status.update(
                label=f"识别领域：{label} · 角色：{event['role']} · "
                f"资料 {len(event['sources'])} 条"
            )

        elif kind == "delta":
            text += event["content"]
            placeholder.markdown(text + "▌")

        elif kind == "done":
            placeholder.markdown(text)
            status.update(label=f"完成 · 用时 {event['elapsed']}s", state="complete")

        elif kind == "error":
            placeholder.empty()
            status.update(label="生成失败", state="error")
            st.error(event.get("message", "未知错误"))
            return text, meta

    return text, meta


# ---------------------------------------------------------------- 主流程


def main() -> None:
    init_state()

    # 没有登录就先给登录页，后面的逻辑一律要求已有 user_id
    if not st.session_state.user_id:
        render_login()
        return

    render_sidebar()
    user_id = st.session_state.user_id

    st.title("多角色 RAG 智能问答")
    st.caption("输入问题后系统会自动识别领域，路由到法律顾问 / 医疗咨询 / 英语学习助手。")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])
            if message.get("domain"):
                label = DOMAIN_LABELS.get(message["domain"], message["domain"])
                st.caption(f"领域：{label} · 角色：{message.get('persona', '')}")

    prompt = st.chat_input("请输入你的问题，例如：劳动合同到期不续签需要赔偿吗？")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        text, meta = stream_reply(user_id, prompt)
        if meta:
            render_sources(meta.get("sources", []))
            if meta.get("rewritten_query") and meta["rewritten_query"] != prompt:
                st.caption(f"检索用的改写查询：{meta['rewritten_query']}")
            if meta.get("domain"):
                label = DOMAIN_LABELS.get(meta["domain"], meta["domain"])
                st.caption(f"领域：{label} · 角色：{meta.get('role', '')}")

    if text:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": text,
                "sources": meta.get("sources", []) if meta else [],
                "domain": meta.get("domain") if meta else None,
                "persona": meta.get("role") if meta else "",
            }
        )


if __name__ == "__main__":
    main()
