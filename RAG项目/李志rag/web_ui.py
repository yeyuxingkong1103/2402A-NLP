import uuid

import requests
import streamlit as st

st.set_page_config(page_title="医疗知识 RAG", page_icon="🩺", layout="wide")
st.markdown(
    """
<style>
:root { --ink:#17233c; --brand:#3967d6; --mint:#e9f7f4; }
.stApp { background:linear-gradient(135deg,#f8fbff 0%,#fff 48%,#f3fbf9 100%); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#edf3ff,#f8fbff 55%,#eefaf7); }
.hero { padding:1.4rem 1.7rem; border-radius:22px; margin:.4rem 0 1.1rem;
  background:linear-gradient(115deg,#315cca,#668bef 58%,#37a993); color:white;
  box-shadow:0 12px 28px rgba(48,91,190,.18); }
.hero-kicker { font-size:.76rem; letter-spacing:.16em; opacity:.84; }
.hero h1 { margin:.35rem 0; font-size:2rem; color:white; }
.hero p { margin:0; opacity:.92; }
.notice { padding:.8rem 1rem; border-left:4px solid #39a995; border-radius:10px;
  background:var(--mint); color:#245c58; margin-bottom:1rem; }
.catalog-card { padding:.8rem 1rem; margin:.5rem 0; border:1px solid #e1e9f3;
  border-radius:13px; background:rgba(255,255,255,.88); box-shadow:0 4px 12px rgba(31,53,91,.04); }
.catalog-card strong { color:var(--ink); font-size:1.02rem; }
.stChatMessage { border-radius:16px; border:1px solid #e7edf6; }
div[data-testid="stExpander"] { border-radius:14px; border-color:#dfe8f4; }
[data-testid="stChatInput"] { background:white; border:1px solid #cfdcf0;
  border-radius:18px; box-shadow:0 8px 24px rgba(49,92,202,.10); }
.stButton button { border-radius:11px; font-weight:600; }
.stTabs [data-baseweb="tab-list"] { gap:.5rem; background:#edf3fb; padding:.35rem; border-radius:13px; }
.stTabs [data-baseweb="tab"] { border-radius:9px; padding:.35rem 1rem; }
.welcome { padding:1.1rem 1.25rem; border:1px solid #dfe8f4; border-radius:16px;
  background:linear-gradient(120deg,#fff,#f2f7ff); color:#43516a; margin:.5rem 0 1rem; }
</style>
<div class="hero">
  <div class="hero-kicker">MEDICAL KNOWLEDGE ASSISTANT · RAG</div>
  <h1>🩺 医疗健康教育助手</h1>
  <p>连接你的专属知识库，快速理解指南、表格与图片中的健康信息。</p>
</div>
<div class="notice">仅用于健康知识教育，不能替代医生诊断和治疗。出现急症信号请立即拨打 120。</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🩺 医疗知识 RAG")
    st.caption("本地运行 · Milvus 检索 · DeepSeek 生成")
    st.success("● 服务已连接")
    with st.expander("高级设置"):
        API = st.text_input("API 地址", "http://127.0.0.1:8000/api/v1").rstrip("/")

if "token" not in st.session_state:
    st.session_state.token = ""
    st.session_state.is_admin = False
    st.session_state.messages = []
    st.session_state.session_id = uuid.uuid4().hex


def headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def api_request(method: str, path: str, timeout: int = 180, **kwargs):
    try:
        response = requests.request(method, f"{API}{path}", timeout=timeout, **kwargs)
        if response.status_code >= 400:
            detail = response.json().get("detail", response.text)
            raise RuntimeError(detail)
        return response
    except (requests.RequestException, ValueError, RuntimeError) as error:
        st.error(str(error))
        return None


def show_knowledge_catalog() -> list[dict]:
    response = api_request("GET", "/knowledge/catalog", timeout=30)
    items = response.json() if response else []
    with st.expander("📚 当前知识库可解答的问题（点击展开/收起）", expanded=False):
        if not items:
            st.info("知识库目录为空。管理员上传并成功索引 PDF 后会自动显示。")
        categories: dict[str, list[dict]] = {}
        for item in items:
            categories.setdefault(item["category"], []).append(item)
        for category, documents in categories.items():
            st.markdown(f"#### 🗂️ {category}")
            for document in documents:
                with st.container(border=True):
                    st.markdown(f"**📄 {document['title']}**")
                    if document["summary"]:
                        st.caption(document["summary"])
                    for index, question in enumerate(document["questions"]):
                        if st.button(
                            f"💬 {question}",
                            key=f"quick_{document['document_id']}_{index}",
                            disabled=not st.session_state.token,
                            use_container_width=True,
                        ):
                            st.session_state.quick_question = question
                    if not st.session_state.token:
                        st.caption("登录后可点击问题直接提问。")
        st.caption("其他问题也可以提问；没有直接依据时将标注并使用 DeepSeek 通用知识回答。")
    return items


catalog_items = show_knowledge_catalog()
metric_docs, metric_categories, metric_questions = st.columns(3)
metric_docs.metric("📄 知识文档", len(catalog_items))
metric_categories.metric("🗂️ 内容分类", len({item["category"] for item in catalog_items}))
metric_questions.metric("💬 推荐问题", sum(len(item["questions"]) for item in catalog_items))


if not st.session_state.token:
    login_tab, register_tab = st.tabs(["登录", "注册"])
    with login_tab:
        username = st.text_input("用户名", key="login_user")
        password = st.text_input("密码", type="password", key="login_password")
        if st.button("登录", type="primary"):
            response = api_request(
                "POST", "/auth/login", json={"username": username, "password": password}
            )
            if response:
                data = response.json()
                st.session_state.token = data["access_token"]
                st.session_state.is_admin = data["is_admin"]
                st.rerun()
    with register_tab:
        new_user = st.text_input("新用户名")
        new_password = st.text_input("新密码（至少 8 位）", type="password")
        if st.button("注册"):
            response = api_request(
                "POST",
                "/auth/register",
                json={"username": new_user, "password": new_password},
            )
            if response:
                data = response.json()
                st.session_state.token = data["access_token"]
                st.rerun()
    st.stop()

if st.sidebar.button("退出登录"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

roles_response = api_request("GET", "/roles", headers=headers())
roles = roles_response.json() if roles_response else []
if not roles:
    st.warning("没有可用角色")
    st.stop()
role_map = {role["name"]: role["id"] for role in roles}
role_name = st.sidebar.selectbox("角色", list(role_map))
role_id = role_map[role_name]
if st.sidebar.button("＋ 新建对话", use_container_width=True):
    st.session_state.messages = []
    st.session_state.session_id = uuid.uuid4().hex
    st.rerun()

chat_tab, admin_tab = st.tabs(["对话", "知识库管理"])
with chat_tab:
    if not st.session_state.messages:
        st.markdown(
            "<div class='welcome'><b>👋 你好，我是你的知识库助手</b><br>"
            "可以手动输入问题，也可以展开上方知识目录，点击推荐问题快速开始。</div>",
            unsafe_allow_html=True,
        )
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    manual_question = st.chat_input("请输入健康知识问题")
    quick_question = st.session_state.pop("quick_question", None)
    question = quick_question or manual_question
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.spinner("正在检索知识并生成回答，首次使用需加载模型……"):
            response = api_request(
                "POST",
                "/chat",
                timeout=300,
                headers=headers(),
                json={
                    "message": question,
                    "role_id": role_id,
                    "session_id": st.session_state.session_id,
                },
            )
        if response:
            data = response.json()
            answer = data["answer"]
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
    if st.button("清空最近 10 轮记忆"):
        api_request(
            "DELETE",
            f"/chat/{role_id}/{st.session_state.session_id}/memory",
            headers=headers(),
        )
        st.session_state.messages = []
        st.rerun()

with admin_tab:
    if not st.session_state.is_admin:
        st.info("只有管理员可以管理知识库。")
    else:
        st.subheader("上传新资料")
        upload = st.file_uploader("上传 PDF（支持文字、表格、扫描件和图表理解）", type=["pdf"])
        source_url = st.text_input("来源 URL（建议填写）")
        st.caption("系统会自动解析并建立索引；复杂视觉页在纯 CPU 上可能需要数分钟。")
        if st.button("上传并建立索引", type="primary", disabled=upload is None):
            files = {"file": (upload.name, upload.getvalue(), "application/pdf")}
            data = {"role_id": role_id, "source_url": source_url}
            with st.spinner("正在解析 PDF 并建立索引，请勿重复点击……"):
                response = api_request(
                    "POST",
                    "/knowledge/documents",
                    timeout=900,
                    headers=headers(),
                    files=files,
                    data=data,
                )
            if response:
                st.success("PDF 已上传，正在后台解析和建立索引。可稍后刷新查看状态。")
                st.rerun()
        st.divider()
        st.subheader("已上传的知识库")
        st.caption("删除后会同时清理 PDF、MySQL 记录、Milvus 向量和首页分类，无法撤销。")
        if st.button("刷新处理状态"):
            st.rerun()
        docs_response = api_request("GET", "/knowledge/documents", headers=headers())
        if docs_response:
            documents = docs_response.json()
            if not documents:
                st.info("当前没有已上传的知识库资料。")
            for document in documents:
                with st.container(border=True):
                    info, action = st.columns([8, 1])
                    with info:
                        status = {"ready": "可检索", "processing": "处理中", "failed": "失败"}.get(
                            document["status"], document["status"]
                        )
                        st.markdown(f"**📄 {document['filename']}**")
                        st.caption(
                            f"ID：{document['id']}　状态：{status}　"
                            f"分块：{document['chunk_count']}"
                        )
                        detail = document.get("error_message", "")
                        if document["status"] == "processing" and detail:
                            st.info(f"当前进度：{detail}")
                        elif document["status"] == "failed" and detail:
                            st.error(f"失败原因：{detail}")
                        if document["source_url"]:
                            st.caption(f"来源：{document['source_url']}")
                    with action:
                        if st.button("删除", key=f"delete_{document['id']}"):
                            st.session_state.delete_document_id = document["id"]
                    if st.session_state.get("delete_document_id") == document["id"]:
                        st.warning(f"确定永久删除“{document['filename']}”吗？")
                        confirm, cancel, _ = st.columns([1, 1, 6])
                        if confirm.button("确认删除", type="primary", key=f"confirm_{document['id']}"):
                            response = api_request(
                                "DELETE",
                                f"/knowledge/documents/{document['id']}",
                                headers=headers(),
                            )
                            if response is not None:
                                st.session_state.delete_document_id = None
                                st.toast("知识库资料已删除")
                                st.rerun()
                        if cancel.button("取消", key=f"cancel_{document['id']}"):
                            st.session_state.delete_document_id = None
                            st.rerun()
