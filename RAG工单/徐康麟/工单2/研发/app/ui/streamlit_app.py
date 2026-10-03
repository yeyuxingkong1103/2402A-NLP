"""Streamlit 前端（**真正的 Streamlit 应用**，算力云部署用）。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 接口层（对应 设计/接口设计.md §4）

运行（云端已安装 streamlit 的环境）::

    cd E:\\gao6gongdan\\工单2\\研发
    streamlit run app/ui/streamlit_app.py

本机说明（环境事实 2.2）：本机 **streamlit 不可用且断网无法安装**，因此：
- 本文件**不降级、不改写**为其它框架，保持标准 Streamlit 写法（评审可直接在云端运行）；
- 本机演示与在线测试请走同构的备用界面 ``app/ui/serve_fallback.py``（纯标准库 http.server）；
- 两者共用同一套 ``app/core`` 业务逻辑与同一份事件契约（``QAEngine.stream()``）。

交互契约（§4.2/§4.3）：只展示最终答案与引用；中间步骤默认折叠；兜底文案三种；
英文回答引用为 ``[Page: N]``；首字耗时来自 ``first_token`` 事件。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

try:  # 本机未安装 streamlit：保留模块可导入性，便于静态检查与 pytest 收集
    import streamlit as st

    HAS_STREAMLIT = True
except Exception:  # pragma: no cover - 本机路径
    st = None  # type: ignore[assignment]
    HAS_STREAMLIT = False

from app.core.config import get_settings  # noqa: E402
from app.core.errors import RAGError  # noqa: E402
from app.core.logging_conf import logger, setup_logging  # noqa: E402

UI_MODULE = "app.ui.streamlit_app"


# ==========================================================================
# 引擎句柄：把 QAEngine 的构造/失败状态收进一个可在 session_state 里存的对象
# ==========================================================================
class EngineHandle:
    """引擎句柄（``ready`` 表示可用；``error`` 记录最近一次失败文案）。"""

    def __init__(self) -> None:
        self.engine: Any = None
        self.ready = False
        self.error = ""
        self.warmed = False

    def ensure(self) -> Any:
        """惰性构造 QAEngine（首次调用会初始化 SQLite 与索引句柄）。"""
        if self.engine is not None:
            return self.engine
        try:
            from app.core.qa_engine import get_qa_engine

            self.engine = get_qa_engine()
            self.ready = True
        except Exception as exc:
            self.error = f"引擎初始化失败：{exc}"
            logger.exception(UI_MODULE, "引擎初始化失败")
        return self.engine

    def warmup(self) -> dict[str, Any]:
        """预热嵌入与 LLM（把冷启动移出首问）。"""
        engine = self.ensure()
        if engine is None:
            return {}
        try:
            info = engine.warmup()
            self.warmed = True
            return info
        except Exception:
            logger.exception(UI_MODULE, "预热失败")
            return {}


def _find_pdfs() -> list[Path]:
    """列出 研发/data/raw 下的 PDF（侧边栏文档选择用）。"""
    settings = get_settings()
    try:
        return sorted(settings.paths.data_raw.glob("*.pdf"))
    except Exception:
        logger.exception(UI_MODULE, "扫描 PDF 失败")
        return []


def _init_state() -> None:
    """初始化 st.session_state。"""
    defaults: dict[str, Any] = {
        "conversation_id": None,
        "messages": [],
        "engine_handle": None,
        "doc_id": "",
        "pdf_path": "",
        "pending_question": "",
        "last_error": "",
        "show_trace": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _sidebar(handle: EngineHandle) -> None:
    """侧边栏：文档、会话、健康、统计、调试开关。"""
    with st.sidebar:
        st.title("RAG 文档问答")
        st.caption("工单2：人工智能NLP-RAG-基于PDF文档的问答系统优化")
        settings = get_settings()

        pdfs = _find_pdfs()
        if pdfs:
            options = [str(path) for path in pdfs]
            current = st.session_state.get("pdf_path") or options[0]
            index = options.index(current) if current in options else 0
            selected = st.selectbox("文档", options, index=index)
            if selected != st.session_state.get("pdf_path"):
                st.session_state["pdf_path"] = selected
                st.session_state["doc_id"] = Path(selected).stem
                st.session_state["messages"] = []
                engine = handle.ensure()
                if engine is not None:
                    handle.ready = engine.load_index(Path(selected).stem)
                    if not handle.ready:
                        st.warning("该文档索引未就绪，请先执行：pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py")
        else:
            st.warning(f"未找到 PDF：{settings.paths.data_raw}")

        st.divider()
        st.subheader("会话")
        engine = handle.ensure()
        if engine is not None:
            if st.button("新建会话", use_container_width=True):
                try:
                    st.session_state["conversation_id"] = engine.new_conversation()
                    st.session_state["messages"] = []
                except RAGError as exc:
                    st.session_state["last_error"] = exc.user_message
            conversations = engine.list_conversations()
            if conversations:
                labels = {item.conversation_id: f"{item.title}（{item.message_count}）" for item in conversations}
                ids = list(labels)
                current_id = st.session_state.get("conversation_id")
                picked = st.selectbox(
                    "选择会话",
                    ids,
                    index=ids.index(current_id) if current_id in ids else 0,
                    format_func=lambda cid: labels[cid],
                )
                if picked != current_id:
                    st.session_state["conversation_id"] = picked
                    st.session_state["messages"] = []
            if st.session_state.get("conversation_id") and st.button("清空当前会话", use_container_width=True):
                try:
                    engine.clear_conversation(st.session_state["conversation_id"])
                    st.session_state["messages"] = []
                except RAGError as exc:
                    st.session_state["last_error"] = exc.user_message

        st.divider()
        st.subheader("健康状态")
        if engine is not None:
            health = engine.health()
            embedder = health.get("embedder", {})
            index = health.get("index", {})
            reranker = health.get("reranker", {})
            llm = health.get("llm", {}).get("backend", {})
            st.markdown(
                "\n".join(
                    [
                        f"- 嵌入：`{embedder.get('model')}` / {embedder.get('dimension')} 维"
                        + ("（**已降级**）" if embedder.get("degraded") else ""),
                        f"- 向量库：{index.get('backend')} / {index.get('count')} 条",
                        f"- BM25：{health.get('bm25_docs')} 条",
                        f"- LLM：`{llm.get('name')}` / `{llm.get('model')}`",
                        f"- 重排：`{reranker.get('mode')}`"
                        + ("（本机无重排模型权重，规则重排）" if reranker.get("mode") == "rule" else ""),
                        f"- 索引：{'就绪' if health.get('ready') else '未就绪'}",
                    ]
                )
            )
            st.caption("库表统计")
            st.json(engine.stats())

        st.divider()
        st.session_state["show_trace"] = st.checkbox(
            "显示检索调试信息（默认关闭）", value=bool(st.session_state.get("show_trace"))
        )


def _render_history() -> None:
    """渲染历史消息（含引用与耗时）。"""
    for message in st.session_state.get("messages", []):
        with st.chat_message(message.get("role", "assistant")):
            st.markdown(message.get("content", ""))
            citations = message.get("citations") or []
            if citations:
                _render_citations(citations)
            meta = message.get("meta")
            if meta:
                st.caption(meta)


def _render_citations(citations: list[dict[str, Any]]) -> None:
    """渲染引用：标签 + chunk_id + 原文摘要 + 章节，可展开查看完整块。"""
    with st.expander(f"引用来源（{len(citations)} 条）", expanded=False):
        for item in citations:
            label = f"[Page: {item.get('page')}]" if False else f"[页码: {item.get('page')}]"
            st.markdown(f"**{label}** `{item.get('chunk_id')}` — {item.get('section') or '-'}")
            st.caption((item.get("snippet") or "")[:240])


def _handle_question(handle: EngineHandle, question: str) -> None:
    """消费 ``engine.stream()`` 事件流并把答案渲染到界面。"""
    engine = handle.ensure()
    if engine is None:
        st.error(handle.error or "引擎不可用")
        return
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        meta_slot = st.empty()
        answer_text = ""
        first_token_ms = 0.0
        citations: list[dict[str, Any]] = []
        mode = ""
        total_ms = 0.0
        error_message = ""
        try:
            for event, payload in engine.stream(question, conversation_id=st.session_state.get("conversation_id")):
                if event == "status":
                    meta_slot.caption(payload.get("msg", ""))
                elif event == "first_token":
                    first_token_ms = float(payload.get("first_token_ms", 0.0))
                    meta_slot.caption(f"首字响应 {first_token_ms:.0f} ms")
                elif event == "delta":
                    answer_text += payload.get("text", "")
                    placeholder.markdown(answer_text + "▌")
                elif event == "citations":
                    citations = payload.get("citations", [])
                elif event == "error":
                    error_message = f"{payload.get('code')}: {payload.get('message')}"
                    st.session_state["last_error"] = payload.get("message", "")
                elif event == "done":
                    answer = payload.get("answer")
                    if answer is not None:
                        answer_text = answer.answer
                        citations = [item.model_dump(mode="json") for item in answer.citations]
                        mode = answer.mode
                        total_ms = answer.total_ms
                        first_token_ms = answer.first_token_ms or first_token_ms
                        if not st.session_state.get("conversation_id"):
                            st.session_state["conversation_id"] = None
        except Exception:
            logger.exception(UI_MODULE, "流式问答异常")
            st.session_state["last_error"] = "问答过程异常，请稍后再试"
        placeholder.markdown(answer_text or "（无内容）")
        _render_citations(citations)
        if mode:
            meta_slot.caption(
                f"模式 {mode} · 首字 {first_token_ms:.0f} ms · 端到端 {total_ms:.0f} ms"
                + (f" · {error_message}" if error_message else "")
            )
        if st.session_state.get("show_trace"):
            with st.expander("检索调试信息（评审取证用）", expanded=False):
                debug = engine._retriever.debug_snapshot()  # noqa: SLF001（仅调试展示）
                st.json(debug.as_dict())
                st.dataframe(
                    [
                        {
                            "chunk_id": item.chunk.chunk_id,
                            "page": item.chunk.page,
                            "type": item.chunk.type,
                            "score": round(item.score, 4),
                            "vector": round(item.vector_score, 4),
                            "bm25": round(item.bm25_score, 4),
                            "rerank": item.rerank_score,
                        }
                        for item in engine.last_retrieved()
                    ]
                )
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer_text,
            "citations": citations,
            "meta": f"模式 {mode} · 首字 {first_token_ms:.0f} ms · 端到端 {total_ms:.0f} ms",
        }
    )


def main() -> None:
    """Streamlit 应用入口。"""
    if not HAS_STREAMLIT:
        raise RuntimeError(
            "streamlit 未安装：本机演示请使用 `pwsh -NoProfile -File run_py.ps1 "
            "研发/app/ui/serve_fallback.py`（纯标准库）；云端安装 streamlit 后可 `streamlit run` 本文件。"
        )
    st.set_page_config(page_title="RAG 文档问答（工单2）", page_icon="📄", layout="wide")
    setup_logging()
    _init_state()
    if st.session_state.get("engine_handle") is None:
        st.session_state["engine_handle"] = EngineHandle()
    handle: EngineHandle = st.session_state["engine_handle"]

    _sidebar(handle)

    st.header("PDF 文档问答（RAG）")
    st.caption("支持中英文提问；无答案时回「不清楚」；答案带真实页码引用。")

    if not handle.warmed:
        with st.spinner("正在预热模型…"):
            info = handle.warmup()
        if info:
            logger.info(UI_MODULE, "预热完成", elapsed_ms=info.get("elapsed_ms"))

    if st.session_state.get("last_error"):
        st.error(f"⚠ {st.session_state['last_error']}")
        st.session_state["last_error"] = ""

    _render_history()

    question = st.chat_input("请输入问题（中文或英文）")
    if question:
        _handle_question(handle, question)
        st.rerun()


if __name__ == "__main__":
    main()
