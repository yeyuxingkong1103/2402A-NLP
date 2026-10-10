# -*- coding: utf-8 -*-
"""Streamlit 多轮对话演示界面。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务

启动：
    streamlit run app.py            # 交互式
    streamlit run app.py            # 浏览器访问 http://localhost:8501/?demo=1 自动回放
"""
from __future__ import annotations

import json

import streamlit as st

from src import config
from src.conversation import Conversation
from src.i18n import to_retrieval_query, translate_answer
from src.qa_engine import MultiTurnQA

st.set_page_config(page_title="招股说明书多轮问答系统", page_icon="📑", layout="wide")

TXT = {
    "zh": {
        "title": "招股说明书多轮检索问答系统",
        "sub": "工单编号：人工智能 NLP-RAG-Query 理解优化任务",
        "lang": "界面语言 / Language",
        "kb": "知识库",
        "reset": "清空对话",
        "placeholder": "请输入问题，例如：报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
        "rewrite": "多轮改写",
        "evidence": "检索证据",
        "latency": "响应耗时",
        "doc": "命中文档",
        "type": "答案类型",
        "examples": "示例问题",
        "translate": "英文模式自动翻译答案",
    },
    "en": {
        "title": "Prospectus Multi-turn Retrieval QA",
        "sub": "Work Order: NLP-RAG Query Understanding Optimization",
        "lang": "界面语言 / Language",
        "kb": "Knowledge Base",
        "reset": "Reset Conversation",
        "placeholder": "Ask a question, e.g. What was the military revenue of Wuhan Xingtu Xinke during the reporting period?",
        "rewrite": "Rewritten Query",
        "evidence": "Evidence",
        "latency": "Latency",
        "doc": "Source Doc",
        "type": "Answer Type",
        "examples": "Examples",
        "translate": "Auto-translate answer in English mode",
    },
}

EXAMPLES = [
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    "他参与的哪个工程荣获了国家科技进步一等奖？",
    "这个公司的法定代表人是谁？",
    "那武汉力源信息技术股份有限公司呢？",
    "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",
]


@st.cache_resource(show_spinner="正在加载知识库与模型…")
def get_qa():
    return MultiTurnQA()


def _render_history(t):
    for role, text, meta in st.session_state.display:
        with st.chat_message(role):
            st.markdown(text)
            if meta:
                with st.expander(f"🔎 {t['rewrite']} / {t['evidence']}"):
                    if meta.get("rewritten") and meta["rewritten"] != meta["question"]:
                        st.markdown(f"**{t['rewrite']}：** {meta['rewritten']}")
                    st.markdown(f"**{t['doc']}：** {meta.get('doc') or '-'} 　"
                                f"**{t['type']}：** {meta.get('answer_type')} 　"
                                f"**{t['latency']}：** {meta.get('latency')}s")
                    for e in meta.get("evidence", []):
                        st.markdown(f"- `{e['source']}` p.{e['page']}（{e['kind']}，"
                                    f"score={e['score']}）：{e['snippet']}")


def main() -> None:
    if "conv" not in st.session_state:
        st.session_state.conv = Conversation()
    if "display" not in st.session_state:
        st.session_state.display = []      # [(role, text, meta)]

    with st.sidebar:
        lang = st.radio(TXT["zh"]["lang"], ["中文", "English"], index=0)
        L = "zh" if lang == "中文" else "en"
        t = TXT[L]
        st.divider()
        st.markdown(f"**{t['kb']}**")
        try:
            meta = json.loads(config.META_PATH.read_text(encoding="utf-8"))
            st.write(f"- 知识块：{meta['chunks']}")
            st.write(f"- 向量维度：{meta['dim']}")
            st.write(f"- 嵌入模型：{meta['model']}")
            st.write(f"- 文档：{'、'.join(meta['sources'])}")
        except Exception:
            st.warning("知识库未构建，请先运行 build_kb.py")
        st.divider()
        auto_translate = st.checkbox(t["translate"], value=False) if L == "en" else False
        if st.button(t["reset"]):
            st.session_state.conv = Conversation()
            st.session_state.display = []
            st.rerun()
        st.divider()
        st.markdown(f"**{t['examples']}**")
        for i, ex in enumerate(EXAMPLES):
            if st.button(f"{i + 1}. {ex[:22]}…", key=f"ex{i}", use_container_width=True):
                st.session_state.pending = ex
                st.rerun()

    st.title(t["title"])
    st.caption(t["sub"])

    qa = get_qa()

    # 演示模式：?demo=1 时自动回放 5 轮验收对话（用于截图 / 录屏）
    if st.query_params.get("demo") and not st.session_state.display:
        for ex in EXAMPLES:
            st.session_state.display.append(("user", ex, None))
            r = qa.ask(ex, st.session_state.conv)
            st.session_state.display.append(("assistant", r.answer, {
                "question": ex, "rewritten": r.rewritten, "doc": r.doc,
                "answer_type": r.answer_type, "latency": r.latency,
                "evidence": r.evidence,
            }))

    _render_history(t)

    prompt = st.chat_input(t["placeholder"])
    if st.session_state.get("pending"):
        prompt = st.session_state.pending
        st.session_state.pending = None

    if prompt:
        st.session_state.display.append(("user", prompt, None))
        with st.chat_message("user"):
            st.markdown(prompt)

        lang_in, retrieval_q = to_retrieval_query(prompt)
        with st.chat_message("assistant"):
            with st.spinner("检索中…" if L == "zh" else "Retrieving…"):
                res = qa.ask(retrieval_q, st.session_state.conv)
            answer = res.answer
            if L == "en" and auto_translate:
                answer = translate_answer(answer, "en")
            st.markdown(answer)
            meta = {
                "question": prompt, "rewritten": res.rewritten, "doc": res.doc,
                "answer_type": res.answer_type, "latency": res.latency,
                "evidence": res.evidence,
            }
            with st.expander(f"🔎 {t['rewrite']} / {t['evidence']}"):
                if res.rewritten and res.rewritten != retrieval_q:
                    st.markdown(f"**{t['rewrite']}：** {res.rewritten}")
                    for r in res.rewrite_reasons:
                        st.caption(f"· {r}")
                st.markdown(f"**{t['doc']}：** {res.doc or '-'} 　"
                            f"**{t['type']}：** {res.answer_type} 　"
                            f"**{t['latency']}：** {res.latency}s")
                for e in res.evidence:
                    st.markdown(f"- `{e['source']}` p.{e['page']}（{e['kind']}，"
                                f"score={e['score']}）：{e['snippet']}")
        st.session_state.display.append(("assistant", answer, meta))


main()