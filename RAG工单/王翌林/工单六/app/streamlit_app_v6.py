# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
app/streamlit_app_v6.py —— 工单六 混合检索策略配置 + 多轮对话界面（新增文件）

启动：streamlit run app/streamlit_app_v6.py --server.port 8506
功能：检索模式（向量/全文/混合）/ 融合算法（RRF/加权平均+权重滑块）/
     重排器（LLM/TF-IDF/自适应）/ 全文匹配（AND/OR/短语/模糊）可视化配置；
     多轮对话（沿用工单五）；检索通道命中数与策略回显；用户反馈。
"""
import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from dotenv import load_dotenv
load_dotenv()                                   # 工单六：加载 .env

st.set_page_config(page_title="混合检索 v6", page_icon="🔀", layout="wide")

WORK_ORDER = "人工智能NLP-RAG-混合检索任务"

st.sidebar.title("混合检索 v6")
st.sidebar.caption(f"工单编号：{WORK_ORDER}")

# ================= 工单六：检索策略配置 =================
st.sidebar.subheader("⚙️ 检索策略配置")
mode_label = st.sidebar.radio(
    "检索模式", ["混合检索 hybrid", "向量检索 vector", "全文检索 fulltext"],
    index=0)
MODE_MAP = {"混合检索 hybrid": "hybrid", "向量检索 vector": "vector",
            "全文检索 fulltext": "fulltext"}
mode = MODE_MAP[mode_label]

fusion = st.sidebar.selectbox("混合融合算法", ["rrf（投票机制）",
                                               "weighted（加权平均）"],
                              index=0, disabled=(mode != "hybrid"))
fusion_code = "weighted" if fusion.startswith("weighted") else "rrf"

if fusion_code == "weighted" and mode == "hybrid":
    vw = st.sidebar.slider("向量通道权重", 0.0, 1.0, 0.6, 0.05)
    fw = round(1.0 - vw, 2)
    st.sidebar.caption(f"全文通道权重：{fw}（自动归一化）")
else:
    vw, fw = 0.6, 0.4

reranker_label = st.sidebar.radio(
    "重排算法",
    ["LLM 交叉编码（bge-reranker）", "TF-IDF 重排器", "用户反馈自适应重排器"],
    index=0)
RERANK_MAP = {"LLM 交叉编码（bge-reranker）": "llm",
              "TF-IDF 重排器": "tfidf",
              "用户反馈自适应重排器": "adaptive"}
reranker = RERANK_MAP[reranker_label]

match_label = st.sidebar.selectbox(
    "全文匹配方式", ["AND（与）", "OR（或）", "短语匹配 phrase", "模糊匹配 fuzzy"],
    index=0, disabled=(mode == "vector"))
MATCH_MAP = {"AND（与）": "and", "OR（或）": "or",
             "短语匹配 phrase": "phrase", "模糊匹配 fuzzy": "fuzzy"}
match = MATCH_MAP[match_label]

top_k = st.sidebar.slider("返回条数 top_k", 3, 12, 8)

if st.sidebar.button("🔄 重置为默认策略"):
    st.rerun()

st.sidebar.divider()
if st.sidebar.button("🆕 新会话", type="primary"):
    st.session_state["v6_session_id"] = None
    st.session_state["v6_messages"] = []
    st.rerun()


@st.cache_resource(show_spinner="加载混合检索引擎 v6（首次约1-2分钟）...")
def load_engine():
    """工单六：缓存 RAGEngineV6（含 v4 全链路 + v6 可配置文本检索）"""
    from src.rag_engine_v6 import RAGEngineV6
    engine = RAGEngineV6(top_k=5)
    engine.warmup()
    return engine


def build_retrieval_cfg():
    return {
        "mode": mode, "fusion": fusion_code, "reranker": reranker,
        "match": match, "vector_weight": vw, "fulltext_weight": fw,
        "top_k": top_k,
    }


# ================= 工单六：主页面 =================
st.header("🔀 混合检索策略问答 v6")
st.caption(f"当前策略：**{mode}** ｜ 融合：{fusion_code} ｜ "
           f"重排器：{reranker} ｜ 全文匹配：{match} ｜ "
           f"权重 向量{vw}/全文{fw} ｜ top_k={top_k}")

if "v6_messages" not in st.session_state:
    st.session_state["v6_messages"] = []
if "v6_session_id" not in st.session_state:
    st.session_state["v6_session_id"] = None

for msg in st.session_state["v6_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("输入问题，支持多轮追问与检索策略切换...")
if question:
    engine = load_engine()
    cfg = build_retrieval_cfg()
    st.session_state["v6_messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(f"{mode} 检索 + {reranker} 重排 + 生成中..."):
            conv = None
            from src.conversation_engine import ConversationEngine
            # 工单六：复用引擎单例构造会话引擎（避免重复加载模型）
            if "v6_conv" not in st.session_state:
                st.session_state["v6_conv"] = ConversationEngine(rag_engine=engine)
            conv = st.session_state["v6_conv"]
            result = conv.chat(
                question,
                session_id=st.session_state["v6_session_id"],
                use_image=True,
                retrieval_config=cfg)
            st.session_state["v6_session_id"] = result["session_id"]

        st.markdown(result["answer"])

        # 工单六：检索策略与通道命中回显
        rt = result.get("retrieval", {})
        if rt:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("检索模式", rt.get("mode", "-"))
            c2.metric("重排器", rt.get("reranker", "-"))
            c3.metric("向量命中", rt.get("vector_hits", 0))
            c4.metric("全文命中", rt.get("fulltext_hits", 0))
            st.caption(f"融合：{rt.get('fusion','-')} ｜ "
                       f"文本检索耗时 {rt.get('elapsed_ms', 0):.0f}ms ｜ "
                       f"总耗时 {result.get('latency_ms', 0):.0f}ms")

        # 工单六：引用折叠展示
        for i, c in enumerate(result.get("retrieved_text_chunks", [])[:5], 1):
            with st.expander(
                    f"资料{i} | {c.get('doc_id','')} p{c.get('page','?')} "
                    f"| {c.get('search_path','')} "
                    f"| score={c.get('rerank_score', c.get('score',0)):.3f}"):
                st.markdown((c.get("content") or "")[:400])
        for i, im in enumerate(result.get("retrieved_images", [])[:3], 1):
            with st.expander(f"图{i} | {im.get('doc_id','')} p{im.get('page','?')}"):
                p = Path(im.get("path", ""))
                if p.exists():
                    st.image(str(p), use_container_width=True)
                if im.get("caption"):
                    st.caption(im["caption"][:200])

        # 工单六：用户反馈（写入 data/feedback_v6 供自适应重排器学习）
        col1, col2, col3 = st.columns([1, 1, 4])
        comment = col3.text_input("评论", key=f"fb_{len(st.session_state['v6_messages'])}")
        if col1.button("👍 点赞", key=f"up_{len(st.session_state['v6_messages'])}"):
            fb_dir = Path("data/feedback_v6")
            fb_dir.mkdir(parents=True, exist_ok=True)
            fp = fb_dir / f"feedback_{datetime.now():%Y%m%d}.json"
            records = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else []
            records.append({"ts": datetime.now().isoformat(),
                            "session_id": result["session_id"], "query": question,
                            "answer": result["answer"], "rating": "up",
                            "comment": comment, "retrieval": cfg,
                            "work_order": WORK_ORDER})
            fp.write_text(json.dumps(records, ensure_ascii=False, indent=1),
                          encoding="utf-8")
            st.success("反馈已保存，自适应重排器将学习该偏好！")
        if col2.button("👎 点踩", key=f"dn_{len(st.session_state['v6_messages'])}"):
            fb_dir = Path("data/feedback_v6")
            fb_dir.mkdir(parents=True, exist_ok=True)
            fp = fb_dir / f"feedback_{datetime.now():%Y%m%d}.json"
            records = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else []
            records.append({"ts": datetime.now().isoformat(),
                            "session_id": result["session_id"], "query": question,
                            "answer": result["answer"], "rating": "down",
                            "comment": comment, "retrieval": cfg,
                            "work_order": WORK_ORDER})
            fp.write_text(json.dumps(records, ensure_ascii=False, indent=1),
                          encoding="utf-8")
            st.success("反馈已保存，自适应重排器将学习该偏好！")

    st.session_state["v6_messages"].append(
        {"role": "assistant", "content": result["answer"]})
