# -*- coding: utf-8 -*-
"""【网页聊天界面 · app.py】Streamlit 前端：登录注册、角色切换、多轮对话与检索资料展示。"""
from __future__ import annotations

import streamlit as st

from auth import authenticate, create_user, make_token
from config import TSV_PATH
from database import SessionLocal, User, init_db
from services import add_knowledge, answer, get_retriever, list_roles_with_data
from pathlib import Path


# 页面基础配置：标题、图标、居中布局
st.set_page_config(page_title="RAG 角色扮演", page_icon="📚", layout="centered")
init_db()  # 启动即建表 + 写种子数据（幂等）


def login_view() -> None:
    """登录 / 注册视图：两个标签页，成功后写入 session_state 并刷新。"""
    st.title("RAG 角色扮演系统")
    st.caption("基于 RAG 的多角色扮演系统 · 支持 9 种角色")
    tab_login, tab_reg = st.tabs(["登录", "注册"])
    with tab_login:
        username = st.text_input("用户名", key="login_user")
        password = st.text_input("密码", type="password", key="login_pwd")
        if st.button("登录"):
            with SessionLocal() as db:
                user = authenticate(db, username, password)  # 校验密码哈希
            if not user:
                st.error("用户名或密码错误")
            else:
                # 登录成功：缓存用户信息与 token
                st.session_state.user_id = user.id
                st.session_state.username = user.username
                st.session_state.token = make_token(user.id)
                st.rerun()
    with tab_reg:
        username = st.text_input("新用户名", key="reg_user")
        password = st.text_input("新密码", type="password", key="reg_pwd")
        if st.button("注册并登录"):
            try:
                with SessionLocal() as db:
                    if db.query(User).filter_by(username=username).first():
                        st.error("用户名已存在")
                        return
                    user = create_user(db, username, password)  # 注册即登录
                st.session_state.user_id = user.id
                st.session_state.username = user.username
                st.session_state.token = make_token(user.id)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))


def chat_view() -> None:
    """主聊天视图：侧边栏（角色/参数/知识库上传）+ 消息流 + 输入框。"""
    roles = list_roles_with_data()  # 只显示有知识数据的角色
    with st.sidebar:
        st.write(f"当前用户：**{st.session_state.username}**")
        if st.button("退出"):
            # 退出登录：清空所有会话状态
            for k in ("user_id", "username", "token", "messages", "session_id"):
                st.session_state.pop(k, None)
            st.rerun()
        # 角色选择（多角色扮演核心交互）
        role = st.selectbox("角色", roles, format_func=lambda r: f"{r['name']}（{r['domain']}）")
        st.session_state.role_code = role["code"]
        top_k = st.slider("检索条数", 3, 12, 6)
        show_hits = st.checkbox("显示检索资料", True)
        # 知识库动态更新：上传 txt/pdf/tsv 直接入库
        uploaded = st.file_uploader("知识库动态更新（txt/pdf/tsv）", type=["txt", "pdf", "tsv"])
        if uploaded is not None and st.button("写入知识库"):
            dest = Path("data/uploads") / uploaded.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(uploaded.getvalue())
            n = add_knowledge(dest)
            get_retriever.cache_clear()  # 知识库变了，重建 BM25 检索器单例
            st.success(f"已写入 {n} 条")
        st.caption(f"主数据：{TSV_PATH}")

    retriever = get_retriever()
    st.title(role["name"])
    st.caption(role["description"] + f" ｜ 知识库 {len(retriever.pairs)} 条")

    # 渲染历史消息（session_state 持久化在浏览器会话中）
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("输入问题，中英文都可以…")
    if not query:
        return
    # 先渲染用户消息
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    # 再调用后端 RAG 链路生成回答
    with st.chat_message("assistant"):
        with st.spinner("检索 + 生成中…"):
            result = answer(
                st.session_state.user_id,
                st.session_state.role_code,
                query,
                session_id=st.session_state.get("session_id"),
                top_k=top_k,
            )
        st.session_state.session_id = result["session_id"]  # 记住会话 id，实现多轮
        st.markdown(result["answer"])
        if show_hits:
            with st.expander("检索资料"):
                st.code(result["retrieved"], language=None)
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})


def main() -> None:
    """入口：未登录显示登录页，已登录进入聊天页。"""
    if "user_id" not in st.session_state:
        login_view()
    else:
        chat_view()


if __name__ == "__main__":
    main()
