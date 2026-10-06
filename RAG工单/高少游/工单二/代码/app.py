# -*- coding: utf-8 -*-
"""RAG 问答系统优化版 · Streamlit 演示界面
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化

功能：
- 单题问答：优化后链路 / 优化前基线，展示答案、页码引用、检索片段与耗时；
- 前后对比：同一问题并排展示优化前/优化后答案与命中判定；
- 批量评估：一键跑通 data/questions.json 全部题目，实时展示准确率与耗时；
- 中英文提问均支持（中文为主）。

启动（项目根目录，激活 langchain2 环境，需已启动 Ollama）：
    streamlit run app.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

from src import config
from src.knowledge_base import _kv_file, build_kb
from src.qa_engine import QAEngine
from src.evaluate_keys import KEY_TOKENS, hit

st.set_page_config(
    page_title="RAG 招股说明书问答系统（优化版）",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
<style>
.main-title {font-size: 1.9rem; font-weight: 700; margin-bottom: .2rem;}
.sub-title {color: #6b7280; font-size: .95rem; margin-bottom: 1rem;}
.card {border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px 16px; margin-bottom: 12px;
       background: #ffffff; box-shadow: 0 1px 2px rgba(0,0,0,.04);}
.card-opt {border-left: 5px solid #16a34a;}
.card-base {border-left: 5px solid #f97316;}
.tag {display:inline-block; padding:2px 8px; border-radius:999px; font-size:.75rem;
      background:#eef2ff; color:#3730a3; margin-right:6px;}
.tag-ok {background:#dcfce7; color:#166534;}
.tag-bad {background:#fee2e2; color:#991b1b;}
.metric-big {font-size:1.6rem; font-weight:700;}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="正在加载问答引擎（首次较慢）...")
def get_engine() -> QAEngine:
    if not _kv_file(config.DB_DIR).exists():
        build_kb()
    return QAEngine()


@st.cache_data(show_spinner=False)
def load_questions() -> list[dict]:
    return json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]


def _sources_block(result) -> None:
    """展示检索到的来源片段（可折叠）。"""
    if not result.sources:
        return
    with st.expander(f"检索到的依据片段（{len(result.sources)} 条）", expanded=False):
        for s in result.sources:
            st.markdown(
                f"<span class='tag'>第 {s.get('page')} 页</span>"
                f"<span class='tag'>{s.get('ctype')}</span>"
                f"<span class='tag'>score {s.get('score')}</span>",
                unsafe_allow_html=True,
            )
            st.caption((s.get("section") or "—")[:80])
            st.code((s.get("text") or "")[:600], language=None)


def render_single(engine: QAEngine, question: str, mode: str) -> None:
    if mode == "优化后（结构分块 + 混合检索 + 重排 + 抽取式）":
        r = engine.answer_optimized(question)
        css, label = "card-opt", "优化后链路"
    else:
        r = engine.answer_baseline(question)
        css, label = "card-base", "优化前基线（01 工单）"

    st.markdown(
        f"<div class='card {css}'><span class='tag'>{label}</span>"
        f"<span class='tag'>耗时 {r.elapsed:.3f} s</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**回答：** {r.answer}")
    _sources_block(r)


def render_compare(engine: QAEngine, question: str, qid: int | None) -> None:
    c1, c2 = st.columns(2)
    with st.spinner("运行优化前 / 优化后两条链路..."):
        with c1:
            base = engine.answer_baseline(question)
        with c2:
            opt = engine.answer_optimized(question)

    tokens = KEY_TOKENS.get(qid or -1, [])
    with c1:
        ok = hit(base.answer, tokens) if tokens else None
        badge = "" if ok is None else (
            "<span class='tag tag-ok'>命中</span>" if ok
            else "<span class='tag tag-bad'>未命中</span>")
        st.markdown(
            f"<div class='card card-base'><b>优化前（基线）</b> "
            f"<span class='tag'>耗时 {base.elapsed:.2f} s</span>{badge}</div>",
            unsafe_allow_html=True)
        st.write(base.answer or "（无有效答案）")
    with c2:
        ok = hit(opt.answer, tokens) if tokens else None
        badge = "" if ok is None else (
            "<span class='tag tag-ok'>命中</span>" if ok
            else "<span class='tag tag-bad'>未命中</span>")
        st.markdown(
            f"<div class='card card-opt'><b>优化后</b> "
            f"<span class='tag'>耗时 {opt.elapsed:.3f} s</span>{badge}</div>",
            unsafe_allow_html=True)
        st.write(opt.answer)
        if opt.citations:
            st.caption("引用页码：" + "、".join(f"第{p}页" for p in opt.citations))

    if tokens:
        st.caption("参考答案关键线索：" + " / ".join(tokens))


def render_batch(engine: QAEngine) -> None:
    questions = load_questions()
    if st.button("开始批量评估（10 题）", type="primary"):
        rows, t_all = [], time.time()
        prog = st.progress(0.0, text="评估中...")
        for i, item in enumerate(questions, 1):
            qid, q = int(item["id"]), item["question"]
            tokens = KEY_TOKENS.get(qid, [])
            opt = engine.answer_optimized(q)
            base = engine.answer_baseline(q)
            rows.append({
                "题号": qid,
                "问题": q[:36] + "…",
                "优化前命中": "✅" if hit(base.answer, tokens) else "❌",
                "优化后命中": "✅" if hit(opt.answer, tokens) else "❌",
                "优化前耗时(s)": round(base.elapsed, 2),
                "优化后耗时(s)": round(opt.elapsed, 3),
            })
            prog.progress(i / len(questions), text=f"已完成 {i}/{len(questions)}")

        n = len(rows)
        b_ok = sum(r["优化前命中"] == "✅" for r in rows)
        o_ok = sum(r["优化后命中"] == "✅" for r in rows)
        st.dataframe(rows, use_container_width=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("优化前答案准确率", f"{b_ok / n * 100:.0f}%", f"{b_ok}/{n}")
        m2.metric("优化后答案准确率", f"{o_ok / n * 100:.0f}%", f"{o_ok}/{n}")
        m3.metric("优化后平均耗时", f"{sum(r['优化后耗时(s)'] for r in rows) / n:.3f} s",
                  f"总耗时 {time.time() - t_all:.0f}s")


def main() -> None:
    st.markdown("<div class='main-title'>📈 招股说明书 RAG 问答系统（优化版）</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化 ｜ "
        "PDF 解析 · 分块 · 检索三层优化 ｜ 目标：准确率 ≥ 90%，响应 ≤ 3s</div>",
        unsafe_allow_html=True)

    engine = get_engine()
    questions = load_questions()

    with st.sidebar:
        st.header("⚙️ 演示设置")
        mode = st.radio("问答模式", [
            "优化后（结构分块 + 混合检索 + 重排 + 抽取式）",
            "优化前（基线，01 工单）",
            "优化前后对比",
            "批量评估",
        ])
        st.divider()
        st.subheader("📋 预置问题")
        opts = [f"Q{q['id']}：{q['question'][:22]}…" for q in questions]
        picked = st.selectbox("从题目列表选择", ["（自定义输入）"] + opts)

    qid, preset_q = None, ""
    if picked != "（自定义输入）":
        idx = opts.index(picked)
        qid = int(questions[idx]["id"])
        preset_q = questions[idx]["question"]

    if mode == "批量评估":
        st.subheader("批量评估：10 题优化前后对比")
        render_batch(engine)
        return

    question = st.text_input("请输入问题（中文 / English）", value=preset_q,
                             placeholder="例如：武汉兴图新科电子股份有限公司注册资本是多少？")
    if not question.strip():
        st.info("请输入问题，或从左侧「预置问题」中选择。")
        return

    if mode == "优化前后对比":
        render_compare(engine, question, qid)
    else:
        render_single(engine, question, mode)


if __name__ == "__main__":
    main()