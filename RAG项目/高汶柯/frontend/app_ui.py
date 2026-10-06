"""Streamlit 前端：聊天 / 知识库管理 / 运维监控（Redis + Milvus 可视化）。

启动：``streamlit run frontend/app_ui.py``
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG 多角色扮演系统", page_icon="🎭", layout="wide")


# ===== 通用请求 =====
def api_get(path: str, **params):
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001
        st.error(f"请求失败 {path}: {exc}")
        return None


def api_post(path: str, json=None, **kwargs):
    try:
        r = requests.post(f"{API}{path}", json=json, timeout=300, **kwargs)
        r.raise_for_status()
        return r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
    except Exception as exc:  # noqa: BLE001
        st.error(f"请求失败 {path}: {exc}")
        return None


# ===== 侧边栏 =====
st.sidebar.title("🎭 RAG 多角色扮演系统")
page = st.sidebar.radio("功能", ["💬 对话", "📚 知识库管理", "📊 运维监控"])
user_id = st.sidebar.text_input("用户 ID", value="u1")

roles = api_get("/roles") or []
role_map = {f"{r['id']} · {r['role_name']}": r for r in roles} if roles else {}


# ===== 对话页 =====
def page_chat() -> None:
    st.header("💬 角色对话")
    if not role_map:
        st.warning("未获取到角色，请确认后端已启动且 MySQL 可用。")
        return

    label = st.selectbox("选择角色", list(role_map.keys()))
    role = role_map[label]
    st.caption(role.get("description", ""))

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if st.session_state.get("current_role") != role["id"]:
        st.session_state.current_role = role["id"]
        st.session_state.messages = []

    use_rag = st.toggle("启用知识库检索", value=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("请输入你的问题……")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        text = ""
        try:
            with requests.post(
                f"{API}/chat_stream",
                json={"user_id": user_id, "role_id": role["id"], "message": prompt, "use_rag": use_rag},
                stream=True, timeout=300,
            ) as resp:
                for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        text += chunk
                        placeholder.markdown(text)
        except Exception as exc:  # noqa: BLE001
            text = f"请求失败：{exc}"
            placeholder.markdown(text)
    st.session_state.messages.append({"role": "assistant", "content": text})

    if st.button("清空当前会话"):
        api_post(f"/clear?user_id={user_id}&role_id={role['id']}")
        st.session_state.messages = []
        st.rerun()


# ===== 知识库管理页 =====
def page_kb() -> None:
    st.header("📚 知识库管理")
    col1, col2 = st.columns(2)
    domain = col1.text_input("知识域 domain", value="law")
    parser = col2.selectbox("解析引擎", ["auto", "pymupdf", "pdfplumber", "paddleocr", "mineru", "multimodal"])

    up = st.file_uploader("上传 PDF", type=["pdf"])
    if up is not None and st.button("上传并入库"):
        files = {"file": (up.name, up.getvalue(), "application/pdf")}
        result = api_post("/upload_pdf", data={"domain": domain, "parser": parser}, files=files)
        if result:
            st.success(result.get("msg", "完成"))

    if st.button("批量入库 laws 目录"):
        result = api_post("/kb/ingest_dir", data={"domain": domain, "parser": parser})
        if result:
            st.success(f"处理 {result.get('count')} 个文件")

    st.subheader("已有文档")
    docs = api_get("/kb/docs") or []
    if docs:
        st.dataframe(pd.DataFrame(docs), use_container_width=True)
        doc_id = st.selectbox("选择要删除的 doc_id", [d["doc_id"] for d in docs])
        if st.button("删除该文档"):
            requests.delete(f"{API}/kb/docs/{doc_id}", timeout=30)
            st.success("已删除")
            st.rerun()


# ===== 运维监控页 =====
def page_ops() -> None:
    st.header("📊 运维监控")

    st.subheader("Redis 可视化")
    stats = api_get("/ops/redis/stats") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("键总数", stats.get("dbsize", 0))
    c2.metric("内存", stats.get("used_memory_human", "-"))
    c3.metric("命中率", f"{stats.get('hit_rate', 0) * 100:.1f}%")
    c4.metric("命中/未命中", f"{stats.get('keyspace_hits', 0)}/{stats.get('keyspace_misses', 0)}")

    dist = stats.get("type_distribution") or {}
    left, right = st.columns(2)
    if dist:
        fig = px.pie(values=list(dist.values()), names=list(dist.keys()), title="键类型分布")
        left.plotly_chart(fig, use_container_width=True)
    top = stats.get("top_queries") or []
    if top:
        fig2 = px.bar(x=[t["count"] for t in top], y=[t["query"] for t in top],
                      orientation="h", title="热问题 TopN")
        right.plotly_chart(fig2, use_container_width=True)

    pattern = st.text_input("键模式", value="*")
    keys = api_get("/ops/redis/keys", pattern=pattern, limit=200) or []
    if keys:
        st.dataframe(pd.DataFrame(keys), use_container_width=True)
        key = st.selectbox("查看键详情", [k["key"] for k in keys])
        if key:
            st.json(api_get(f"/ops/redis/key/{key}"))

    st.divider()
    st.subheader("Milvus 可视化")
    mstats = api_get("/ops/milvus/stats") or {}
    m1, m2 = st.columns(2)
    m1.metric("实体总数", mstats.get("num_entities", 0))
    m2.metric("集合", mstats.get("collection", "-"))
    domains = mstats.get("domains") or {}
    if domains:
        st.plotly_chart(
            px.bar(x=list(domains.keys()), y=list(domains.values()), title="各知识域实体数"),
            use_container_width=True,
        )

    mdomain = st.text_input("按 domain 过滤实体（留空为全部）", value="")
    entities = api_get("/ops/milvus/entities", domain=mdomain or None, limit=50, offset=0) or []
    if entities:
        st.dataframe(pd.DataFrame(entities), use_container_width=True)

    st.markdown("**检索预览**")
    q = st.text_input("预览查询", value="猎捕野生动物怎么处罚？")
    if st.button("执行检索预览"):
        result = api_post("/ops/milvus/search", json={"query": q, "domain": mdomain or None, "top_k": 5})
        if result:
            st.dataframe(pd.DataFrame(result), use_container_width=True)


if page == "💬 对话":
    page_chat()
elif page == "📚 知识库管理":
    page_kb()
else:
    page_ops()
