# -*- coding: utf-8 -*-
"""工单4 Streamlit 版中文 Web 界面（有依赖的环境使用）：``研发/app/ui/streamlit_app.py``。

工单编号：人工智能NLP-RAG-图像内容解析及检索优化

环境事实（设计/环境事实.md §2）：主解释器 **没有 streamlit**，但备用 GPU 解释器
``E:\\gao6gongdan\\工单1\\.gao6gongdan-src\\python.exe`` 自带 ``streamlit 1.37.1``。运行方式：

    E:\\gao6gongdan\\工单1\\.gao6gongdan-src\\python.exe -m streamlit run 研发/app/ui/streamlit_app.py

或（主解释器装了 streamlit 时）：``pwsh -NoProfile -File run_py.ps1 -m streamlit run ...``

**本文件与 ``serve_fallback.py`` 共用同一 ``QAEngine``**：不复制任何检索/生成逻辑；
缺失 streamlit 时给出明确的中文指引并以退出码 3 结束（不伪装成功）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"
TITLE = "招股说明书图像内容 RAG 问答系统（工单4）"


def build_engine() -> Any:
    """构造与 CLI/兜底 Web 完全一致的问答引擎（含运行时持久化）。"""
    from app.core.config import get_config
    from app.core.logging_conf import setup_logging
    from app.core.qa_engine import QAEngine
    from app.storage.sqlite_manager import SqliteManager

    cfg = get_config()
    log = setup_logging(cfg)
    storage = SqliteManager(Path(cfg.paths.index_dir) / "qa_runtime.sqlite3", logger=log)
    return QAEngine(cfg=cfg, logger=log, storage=storage)


def render(engine: Any) -> None:
    """渲染页面（答案 / 引用 / 检索精度 / 耗时 / 拒答态）。"""
    import streamlit as st

    from app.ui.serve_fallback import DEMO_QUESTIONS

    st.set_page_config(page_title=TITLE, page_icon="📄", layout="wide")
    st.title(TITLE)
    st.caption("双招股说明书（力源信息 / 兴图新科）· 文本 + 表格 + 图像三类证据 · 答案带 1-based 物理页码引用")
    if "engine" not in st.session_state:
        st.session_state.engine = engine
        st.session_state.session_id = "st-001"

    question = st.text_area("请输入问题（Ctrl+Enter 提交）", key="q", height=90)
    cols = st.columns([1, 1, 1, 4])
    lang = cols[0].selectbox("语言", ["auto", "zh", "en"], index=0)
    top_k = cols[1].number_input("Top-K", min_value=1, max_value=20, value=5)
    ask = cols[2].button("提问", type="primary")
    with st.expander("示例问题（含第 5、6 题图像题）", expanded=False):
        for demo in DEMO_QUESTIONS:
            if st.button(f"[{demo['tag']}] {demo['question'][:48]}…", key=f"demo-{demo['id']}"):
                st.session_state.q = demo["question"]
                question = demo["question"]

    if ask and question.strip():
        with st.spinner("检索与生成中…"):
            result = st.session_state.engine.answer(
                question, lang=lang, top_k=int(top_k),
                session_id=st.session_state.session_id, return_evidence=True)
        if result.refused:
            st.warning(f"不清楚（拒答原因：{result.refusal_reason}）")
        else:
            st.markdown(f"### 答案\n{result.answer}")
        if result.citations:
            st.markdown("### 引用")
            for cite in result.citations:
                st.markdown(f"- `{cite['corpus']}` 第 {cite['page']} 页 "
                            f"（{cite.get('modality')}"
                            + (f"/{cite.get('modality_sub')}" if cite.get("modality_sub") else "")
                            + f"，chunk `{cite.get('chunk_id')}`）")
        lat = result.latency_ms
        st.info(f"首字 {lat.get('first_token')} ms ｜ 总计 {lat.get('total')} ms"
                f"（检索 {lat.get('retrieve')} ms / 生成 {lat.get('generate')} ms）"
                f" ｜ 后端 {result.backend}:{result.model} ｜ 生成方式 {result.generation_method}"
                f" ｜ 缓存 {'命中' if result.cache.get('hit') else '未命中'}")
        st.markdown("### 检索精度（Top-K 命中块）")
        st.dataframe([{ "chunk_id": h.get("chunk_id"), "模态": h.get("type"),
                        "子类": h.get("modality_sub"), "页": h.get("page"),
                        "融合分": h.get("final_score"), "向量名次": h.get("vector_rank"),
                        "BM25 名次": h.get("bm25_rank")} for h in result.chunks], use_container_width=True)
        st.markdown("### 证据（生成阶段使用的块）")
        for hit in result.evidence:
            with st.expander(f"{hit.get('chunk_id')}｜{hit.get('type')}｜第 {hit.get('page')} 页"):
                st.text(str(hit.get("text") or "")[:2000])

    if st.button("健康检查"):
        st.json(st.session_state.engine.health())


def main() -> int:
    """入口：缺依赖时给出中文指引（不静默失败）。"""
    try:
        import streamlit  # noqa: F401
    except ModuleNotFoundError:
        print("[依赖缺失] 当前解释器没有 streamlit。请用备用解释器运行：", file=sys.stderr)
        print(r"  E:\gao6gongdan\工单1\.gao6gongdan-src\python.exe -m streamlit run 研发/app/ui/streamlit_app.py",
              file=sys.stderr)
        print("或用零依赖的纯标准库界面（推荐，任何解释器都能跑）：", file=sys.stderr)
        print("  pwsh -NoProfile -File run_py.ps1 研发/app/main.py --serve --port 8123", file=sys.stderr)
        return 3
    render(build_engine())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
