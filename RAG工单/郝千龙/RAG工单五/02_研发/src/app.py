# -*- coding: utf-8 -*-
# 【多轮对话Streamlit界面 · app.py】多轮对话界面，侧边栏展示改写后查询与对话状态
# 工单编号：人工智能NLP-RAG-Query理解优化任务

"""启动方式：

    streamlit run app.py

功能：加载索引 → 单例常驻 → 多轮对话（历史滑窗）→ 展示答案、
Top-3 证据块（页码、标题、双路排名、分数）、各阶段耗时，
侧边栏实时展示“改写后的查询”“当前实体”“是否指代/省略”。
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from dialogue_manager import DialogueManager, DialogueState
from qa_engine import QAEngine
from retriever import Retriever
from vector_store import IndexStore


@st.cache_resource
def load_engine() -> QAEngine:
    """加载并缓存问答引擎（索引单例常驻，不计入单次响应）。"""
    store = IndexStore.load(CONFIG.index_dir)
    return QAEngine(Retriever(store))


def init_state() -> None:
    """初始化 session_state 中的对话状态与历史。"""
    if "dm" not in st.session_state:
        st.session_state.dm = DialogueManager()
    if "state" not in st.session_state:
        st.session_state.state = DialogueState()
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main() -> None:
    st.set_page_config(page_title="招股说明书多轮对话问答系统", layout="wide")
    init_state()
    st.title("招股说明书多轮对话问答系统（工单五 Query 理解优化版）")

    # 侧边栏：展示改写后的查询与对话状态
    with st.sidebar:
        st.header("Query 理解状态")
        if st.session_state.state.turns:
            last = st.session_state.state.turns[-1]
            st.subheader("本轮改写后查询")
            st.info(last.rewritten_query)
            st.caption(f"当前实体：{last.entity or '—'}")
            st.caption(f"是否指代消解：{'是' if '他' in last.user_query or '这个公司' in last.user_query or '该公司' in last.user_query else '否'}")
            st.caption(f"是否省略补全：{'是' if ('呢' in last.user_query or '怎么样' in last.user_query) else '否'}")
        else:
            st.caption("尚未开始对话")

        if st.button("清空对话历史"):
            st.session_state.state = DialogueState()
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption(f"索引目录：{CONFIG.index_dir}")
        st.caption(f"响应预算：{CONFIG.response_timeout_s}s")

    # 历史消息展示
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 输入框
    question = st.chat_input("请输入问题，支持多轮对话（例如：那武汉力源信息技术股份有限公司呢？）")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        try:
            engine = load_engine()
        except FileNotFoundError:
            st.error("未找到索引，请先运行：python build_index.py")
            return

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                parsed = st.session_state.dm.process_turn(
                    st.session_state.state, question)
                result = engine.answer(question, parsed.rewritten)
                st.session_state.dm.record_answer(
                    st.session_state.state, parsed, result.answer, result.latency_s)

            st.markdown(result.answer or "—")
            sla = result.latency_s <= CONFIG.response_timeout_s
            st.caption(f"耗时：{result.latency_s:.3f}s ｜ "
                       f"{'✅ 满足 ≤3s' if sla else '⛔ 超出 3s'}")

            with st.expander("查看改写后查询与证据"):
                st.markdown(f"**改写后查询**：{parsed.rewritten}")
                st.markdown(f"**当前实体**：{parsed.entity or '—'}")
                if parsed.has_coreference:
                    st.markdown("✅ 已进行指代消解")
                if parsed.has_ellipsis:
                    st.markdown("✅ 已进行省略补全")
                st.markdown("**检索证据（Top-3）**：")
                for i, ev in enumerate(result.evidences, start=1):
                    st.markdown(f"**#{i}** · 第 {ev.page_no} 页 · {ev.heading_path or '-'}")
                    st.caption(f"score={ev.score:.4f} ｜ 向量排名: {ev.dense_rank} ｜ "
                               f"关键词排名: {ev.sparse_rank}")
                    st.text(ev.parent_text[:500])

        st.session_state.messages.append(
            {"role": "assistant", "content": result.answer})


if __name__ == "__main__":
    main()
