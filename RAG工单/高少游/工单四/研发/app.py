# -*- coding: utf-8 -*-
"""Streamlit 演示界面（图像内容解析及检索优化版）
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化

功能：
- 中英文双语界面（验收标准：多语言支持）；
- 输入问题 → 返回答案 / 来源页码 / 端到端耗时 / 命中线索（检索精确度）；
- 对图形类问题（如 id 5 组织结构图、id 6 IC 市场增长图）直接展示命中的图形区域；
- 展示知识库状态（文本块 / 表格块 / 图像语义块 / CLIP 图像向量）。

运行：
    streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src import config
from src.evaluate_keys import KEY_TOKENS, hit_detail
from src.qa_engine import QAEngine
from src.knowledge_base import kb_stats
from src.image_index import get_image_index

st.set_page_config(page_title="招股说明书图像内容解析问答系统", page_icon="📊", layout="wide")

# ---------------- 文案（中/英） ----------------
TEXT = {
    "zh": {
        "title": "招股说明书图像内容解析问答系统",
        "subtitle": "工单编号：人工智能 NLP-RAG-图像内容解析及检索优化",
        "q_label": "请输入问题",
        "placeholder": "例如：武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成？",
        "ask": "提问",
        "answer": "答案",
        "sources": "检索来源",
        "elapsed": "响应时间",
        "hit": "命中线索（检索精确度）",
        "passed": "✅ 命中参考答案",
        "failed": "❌ 未命中参考答案",
        "figure": "命中图形",
        "kb": "知识库状态",
        "samples": "验收问题（点击直接提问）",
        "mode": "问答链路",
        "page": "页码",
        "score": "得分",
        "ctype": "类型",
        "preview": "内容预览",
        "no_kb": "知识库尚未构建，请先运行 python build_kb.py",
    },
    "en": {
        "title": "Prospectus Figure-Aware RAG QA System",
        "subtitle": "Work order: NLP-RAG figure content parsing & retrieval optimization",
        "q_label": "Enter your question",
        "placeholder": "e.g. In the organization chart, how many departments form the Sales Dept.?",
        "ask": "Ask",
        "answer": "Answer",
        "sources": "Retrieved sources",
        "elapsed": "Response time",
        "hit": "Matched key tokens (retrieval precision)",
        "passed": "✅ Matches reference answer",
        "failed": "❌ Does not match reference answer",
        "figure": "Matched figure",
        "kb": "Knowledge base status",
        "samples": "Acceptance questions (click to ask)",
        "mode": "Pipeline",
        "page": "Page",
        "score": "Score",
        "ctype": "Type",
        "preview": "Preview",
        "no_kb": "Knowledge base not built yet. Run python build_kb.py first.",
    },
}

MODE_LABELS = {
    "optimized": "优化链路（图像+表格感知检索 · 抽取式）",
    "optimized_llm": "优化检索 + 本地LLM生成",
    "baseline": "优化前基线（LLM）",
}


@st.cache_resource(show_spinner="正在加载问答引擎与向量库…")
def get_engine() -> QAEngine:
    return QAEngine()


@st.cache_data(show_spinner=False)
def load_questions() -> list[dict]:
    try:
        return json.loads(Path(config.QUESTIONS_PATH).read_text(encoding="utf-8"))
    except Exception:
        return []


def _lang_index_from_query() -> int:
    """URL 参数 ?lang=en 指定初始语言，便于分享与演示截图。"""
    raw = str(st.query_params.get("lang", "")).lower()
    return 1 if raw.startswith("en") else 0


def _qid_from_query(questions: list[dict]) -> dict | None:
    """URL 参数 ?qid=5 直达某道验收题，便于分享与演示截图。"""
    raw = st.query_params.get("qid")
    if not raw:
        return None
    try:
        qid = int(raw)
    except (TypeError, ValueError):
        return None
    return next((q for q in questions if q["id"] == qid), None)


def _resolve_figure(path_str: str) -> str | None:
    """解析图形图片路径。

    索引中的 image_path 是构建期写入的绝对路径，交付包换目录或换机器后会失效。
    故先按原路径查找，找不到再退回当前工程的 figures/ 目录按文件名查找
    （图形文件名形如「文档_页码_x_y.png」，全局唯一）。
    """
    if not path_str:
        return None
    p = Path(path_str)
    if p.exists():
        return str(p)
    alt = Path(config.FIGURE_DIR) / p.name
    return str(alt) if alt.exists() else None


def find_figure_image(source: str, page) -> str | None:
    """按 (来源文档, 页码) 找到命中的图形区域图片路径。"""
    try:
        idx = get_image_index()
        if not idx.ready:
            return None
        for m in idx.meta:
            if m.get("source") == source and int(m.get("page", -1)) == int(page):
                found = _resolve_figure(m.get("image_path", ""))
                if found:
                    return found
    except Exception:
        return None
    return None


def main() -> None:
    lang = st.sidebar.selectbox("Language / 语言", ["zh", "en"],
                                index=_lang_index_from_query(),
                                format_func=lambda x: "中文" if x == "zh" else "English")
    t = TEXT[lang]

    st.title(t["title"])
    st.caption(t["subtitle"])

    # ---- 侧边栏：知识库状态 ----
    st.sidebar.markdown(f"### {t['kb']}")
    try:
        stats = kb_stats(config.DB_DIR)
        img = get_image_index().stats()
        st.sidebar.metric("Text+Table+Figure chunks", stats.get("chunks", 0))
        st.sidebar.metric("Table KV chunks", stats.get("table_kv_chunks", 0))
        st.sidebar.metric("CLIP image vectors", img.get("images", 0))
        st.sidebar.write("Sources:", ", ".join(stats.get("sources", [])) or "—")
    except Exception as exc:
        st.sidebar.warning(f"{t['no_kb']}\n\n{exc}")

    mode = st.sidebar.selectbox(t["mode"], list(MODE_LABELS),
                                format_func=lambda m: MODE_LABELS[m])
    if mode != "optimized":
        st.sidebar.info("该链路用于对照实验；验收默认使用「优化链路」。")

    # ---- 验收问题快捷入口 ----
    questions = load_questions()
    st.markdown(f"**{t['samples']}**")
    cols = st.columns(4)
    preset = None
    for i, q in enumerate(questions):
        with cols[i % 4]:
            if st.button(f"id {q['id']} · {q.get('type', '')}", key=f"btn_{q['id']}",
                         use_container_width=True):
                preset = q["question"]

    st.divider()

    # ?qid=N 直达：自动填入并提交一次（用 session_state 去重，避免每次 rerun 重复执行）
    auto_q = _qid_from_query(questions)
    run_now = False
    if auto_q is not None:
        preset = preset or auto_q["question"]
        flag = f"_auto_{auto_q['id']}"
        if not st.session_state.get(flag):
            st.session_state[flag] = True
            run_now = True

    question = st.text_area(t["q_label"], value=preset or "", height=90,
                            placeholder=t["placeholder"])

    if st.button(t["ask"], type="primary") or run_now:
        q_text = (question or preset or "").strip()
        if not q_text:
            st.warning("请输入问题 / Please enter a question.")
        else:
            engine = get_engine()
            with st.spinner("检索中… / Retrieving…"):
                if mode == "optimized":
                    res = engine.answer_optimized(q_text)
                elif mode == "optimized_llm":
                    res = engine.answer_optimized_llm(q_text)
                else:
                    res = engine.answer_baseline(q_text)
            # 结果存入会话，避免后续 rerun（如图片加载、语言切换）把答案清空
            st.session_state["_last_qa"] = {"question": q_text, "res": res}

    last = st.session_state.get("_last_qa")
    if not last:
        return
    res = last["res"]
    question_used = last["question"]

    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader(t["answer"])
        st.success(res.answer or "（无答案）")
    with c2:
        st.metric(t["elapsed"], f"{res.elapsed:.2f}s",
                  delta="达标" if res.elapsed <= config.TARGET_RESPONSE_SECONDS else "超时",
                  delta_color="normal" if res.elapsed <= config.TARGET_RESPONSE_SECONDS else "inverse")

    # ---- 命中线索（检索精确度） ----
    matched_id = None
    for q in questions:
        if q["question"].strip() == question_used:
            matched_id = q["id"]
            break
    if matched_id is not None:
        tokens = KEY_TOKENS.get(matched_id, [])
        d = hit_detail(res.answer, tokens)
        st.markdown(f"**{t['hit']}**：{', '.join(d['matched']) or '—'}")
        st.write(t["passed"] if d["hit"] else t["failed"])

    # ---- 命中图形展示 ----
    if res.sources:
        fig_path = None
        for s in res.sources:
            if s.get("ctype") == "figure":
                fig_path = find_figure_image(s.get("source", ""), s.get("page"))
                if fig_path:
                    break
        if fig_path:
            st.subheader(t["figure"])
            st.image(fig_path, caption=f"{Path(fig_path).name}", use_container_width=True)

        st.subheader(t["sources"])
        st.dataframe(
            [{
                t["page"]: s.get("page"),
                t["ctype"]: s.get("ctype"),
                t["score"]: s.get("score"),
                "source": s.get("source"),
                t["preview"]: (s.get("text", "") or "").replace("\n", " ")[:120],
            } for s in res.sources],
            use_container_width=True,
        )


if __name__ == "__main__":
    main()