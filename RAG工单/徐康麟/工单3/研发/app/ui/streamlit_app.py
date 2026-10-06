# -*- coding: utf-8 -*-
"""工单3 Streamlit 界面（设计/接口设计.md §3.25 冻结；**真正的 Streamlit 应用**）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

* 模块顶层 ``import streamlit as st``（算力云 ``streamlit run`` 直接可跑）；
* 业务逻辑**只**来自 ``app.core.qa_engine``（与标准库备用界面同一套），本文件不实现检索/生成/引用；
* 功能：PDF 多选（来自 ``engine.files()``，自动发现而非硬编码）、问题输入、答案展示、
  可展开的引用原文、多轮历史、点赞/点踩、清空对话、首字响应时间、检索片段与页码展示；
* 预热：首次渲染前调用 ``engine.warmup()``（jieba + 嵌入 + 后端探测），满足验收项 4。

运行（算力云/本机有 streamlit 时）：
    streamlit run 研发/app/ui/streamlit_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st  # 模块顶层导入：算力云可跑；本机缺 streamlit 时该文件不会被备用界面导入

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core.config import get_config  # noqa: E402
from app.core.logging_conf import get_logger, setup_logging  # noqa: E402
from app.core.qa_engine import QAEngine  # noqa: E402

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"


@st.cache_resource(show_spinner="正在加载索引与模型（首次约需数秒）…")
def _engine() -> QAEngine:
    """进程级缓存引擎（预热一次，后续请求复用）。"""
    cfg = get_config()
    setup_logging(cfg, force=True)
    log = get_logger("streamlit_app")
    engine = QAEngine(cfg=cfg, logger=log)
    info = engine.warmup()
    log.log_event("ui.streamlit_warmup_done", **{k: v for k, v in info.items() if k != "backends"})
    return engine


def _init_state() -> None:
    """初始化会话状态（会话 id / 历史 / 上次检索结果）。"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "history" not in st.session_state:
        st.session_state.history = []          # [{"role","content","citations","first_token_ms","total_ms"}]
    if "last_retrieval" not in st.session_state:
        st.session_state.last_retrieval = None
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}


def _render_sidebar(engine: QAEngine) -> tuple[list[str], int]:
    """侧栏：PDF 多选（自动发现）、top-k、健康信息、清空对话。"""
    with st.sidebar:
        st.header("检索设置")
        files = engine.files()
        labels = [f"{f['file_name']}（{f['page_count'] or '?'} 页 / {f['chunk_count']} 块）" for f in files]
        picked = st.multiselect("PDF 范围（留空 = 全库）", options=labels, default=[])
        file_names = [files[labels.index(p)]["file_name"] for p in picked]
        top_k = st.slider("top-k 检索片段数", min_value=1, max_value=20, value=int(get_config().retrieval.top_k))
        st.divider()
        if st.button("清空对话", use_container_width=True):
            if st.session_state.session_id:
                engine.clear_conversation(st.session_state.session_id)
            st.session_state.session_id = None
            st.session_state.history = []
            st.session_state.last_retrieval = None
            st.session_state.feedback = {}
            st.success("已清空本次对话")
        st.divider()
        st.caption("引擎健康")
        health = engine.health()
        st.write({"检索块": health["retriever"].get("chunks"),
                  "向量维度": health["retriever"].get("dim"),
                  "BM25 词表": health["retriever"].get("bm25_vocab"),
                  "生成后端": health["llm"].get("backend"),
                  "模型": health["llm"].get("model")})
        if engine.warmup_info:
            st.caption("预热")
            st.write({k: engine.warmup_info.get(k) for k in ("tokenizer_ms", "embed_ms", "llm_probe_ms")})
    return file_names, top_k


def _render_citations(engine: QAEngine, citations: list[dict]) -> None:
    """引用展示：每个引用一个可展开块，展开即显示该页真实原文。"""
    if not citations:
        return
    st.markdown("**引用**：" + " ".join(f"`[{c['file_name']}: {c['page']}]`" for c in citations))
    for index, cite in enumerate(citations):
        with st.expander(f"引用原文 · {cite['file_name']} 第 {cite['page']} 页", expanded=(index == 0)):
            st.text(engine.citation_source(cite["file_name"], cite["page"], limit=1200) or "（该页无文本层内容）")


def _render_retrieval(retrieval: dict | None) -> None:
    """检索片段与页码展示（含数值锚定支持块）。"""
    if not retrieval:
        return
    with st.expander("检索片段与页码", expanded=False):
        for chunk in retrieval.get("support_chunks") or []:
            st.markdown(f"**[支持块] {chunk['file_name']}: {chunk['page']}**（数值锚定，不参与 top-k 排名）")
            st.code(str(chunk.get("content", ""))[:800], language="markdown")
        for chunk in retrieval.get("chunks") or []:
            st.markdown(f"**#{chunk.get('rank')} {chunk['file_name']}: {chunk['page']}** · "
                        f"{chunk.get('type')} · score {chunk.get('score')}")
            st.code(str(chunk.get("content", ""))[:800], language="markdown")
        stages = retrieval.get("stages") or {}
        if stages:
            st.caption("阶段耗时(ms)：" + ", ".join(f"{k}={v}" for k, v in stages.items()))


def _render_feedback(engine: QAEngine, index: int, entry: dict) -> None:
    """点赞/点踩（写 feedback 表并回显）。"""
    cols = st.columns([1, 1, 8])
    with cols[0]:
        if st.button("👍", key=f"up_{index}"):
            engine.feedback(rating="up", session_id=st.session_state.session_id,
                            answer_id=entry.get("answer_id"), trace_id=entry.get("trace_id"))
            st.session_state.feedback[index] = "up"
            st.toast("已记录点赞")
    with cols[1]:
        if st.button("👎", key=f"down_{index}"):
            engine.feedback(rating="down", session_id=st.session_state.session_id,
                            answer_id=entry.get("answer_id"), trace_id=entry.get("trace_id"))
            st.session_state.feedback[index] = "down"
            st.toast("已记录点踩")
    with cols[2]:
        if st.session_state.feedback.get(index):
            st.caption(f"已反馈：{'点赞' if st.session_state.feedback[index] == 'up' else '点踩'}")


def main() -> None:
    """页面主流程（Streamlit 每次交互重跑此函数）。"""
    st.set_page_config(page_title="工单3 · 招股说明书问答", page_icon="📄", layout="wide")
    st.title("工单3 · 招股说明书问答")
    st.caption("人工智能NLP-RAG-PDF文档的表格解析及检索优化 —— 表格解析 + 混合检索 + 带引用答案")

    engine = _engine()
    _init_state()
    file_names, top_k = _render_sidebar(engine)

    question = st.chat_input("请输入问题（支持中英文、多轮追问）")
    for index, entry in enumerate(st.session_state.history):
        with st.chat_message("user" if entry["role"] == "user" else "assistant"):
            st.markdown(entry["content"])
            if entry["role"] == "assistant":
                _render_citations(engine, entry.get("citations") or [])
                st.caption(f"首字 {entry.get('first_token_ms')} ms · 总 {entry.get('total_ms')} ms · "
                           f"trace {entry.get('trace_id')}")
                _render_feedback(engine, index, entry)
                _render_retrieval(entry.get("retrieval"))

    if question:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            started = time.perf_counter()
            with st.spinner("检索 + 生成中…"):
                answer = engine.ask(question, session_id=st.session_state.session_id,
                                    file_names=file_names or None, top_k=top_k)
            wall_ms = round((time.perf_counter() - started) * 1000, 2)
            st.session_state.session_id = getattr(answer, "conversation_id", None) or st.session_state.session_id
            st.markdown(answer.text)
            citations = [c.to_dict() for c in answer.citations]
            _render_citations(engine, citations)
            st.caption(f"首字 {answer.first_token_ms} ms · 总 {answer.total_ms} ms · 端到端 {wall_ms} ms · "
                       f"trace {answer.trace_id}"
                       + (f" · 不清楚原因：{answer.unknown_reason}" if answer.is_unknown else ""))
            retrieval = answer.retrieval.to_dict() if answer.retrieval is not None else None
            _render_retrieval(retrieval)
        entry = {"role": "assistant", "content": answer.text, "citations": citations,
                 "first_token_ms": answer.first_token_ms, "total_ms": answer.total_ms,
                 "trace_id": answer.trace_id, "answer_id": answer.answer_id, "retrieval": retrieval}
        st.session_state.history.append(entry)
        st.session_state.last_retrieval = retrieval
        st.rerun()


if __name__ == "__main__":
    main()
