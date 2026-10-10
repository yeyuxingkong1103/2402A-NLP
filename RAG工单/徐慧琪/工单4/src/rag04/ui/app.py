# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""Streamlit 中英双语界面：上传 PDF、提问、查看答案/引用条数/引用来源/图像溯源。

设计约束（与 Task 19 的服务层保持一致）：
- ``import streamlit`` 只在 ``main()`` 内发生：导入本模块即可单测纯函数，
  且 ``python -m pytest`` 不会白白加载整个 Streamlit 运行时。
- 中英双语：界面可切中文/英文，答案语言一律原样取 ``Answer.lang`` 并在界面标注
  （评估区分 zh/en 两次运行，界面必须让人一眼看出答案回来的是哪种语言）。
- 时延诚实呈现：原样展示 ``Answer.latency_ms``，并按
  ``Settings.latency_budget_ms``（3 秒硬指标）判定是否达标；超预算不截断、
  不缓存、不重试、不美化（实测 p50 ≈ 3.9s，见 docs/reports/eval_full_04_rc2.json）。
  达标判定只有一份实现 —— ``rag04.config.latency_verdict``（先取 1 位小数再比较），
  与 ``rag04.api.server``、``benchmarks/loadtest.py`` 共用，避免同一测量三个结论。
- 上传即真入库且不误伤：``build_uploaded()`` 把消毒后的文件落盘并以 ``names``
  点名建库 —— 旧实现只调 ``build()``，而 ``build_all`` 只遍历 ``Settings.corpus``，
  新文件名永远不会被解析（界面却报成功）。新文档走**追加**（``names=corpus+[新文件]``、
  ``reset=False``，语料与上传件并列可检索）；只有覆盖语料源文件才 ``reset=True``
  清库重建（chunk_id 变了，不清会新旧共存）。
- 启动预热：``get_pipe()`` 里调 ``warm_up()``，让首条显示的问答不必付
  reranker/CLIP 的加载时延（失败只记录，界面照常可用）。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger("rag04.ui")

UI_TEXT: dict[str, dict[str, str]] = {
    "zh": {
        "title": "招股说明书智能问答（工单04 图像内容解析及检索优化）",
        "subtitle": "支持文本 / 表格 / 图像 三模态检索，中英文提问均可",
        "upload_label": "上传 PDF 文档",
        "upload_button": "上传并建库",
        "upload_hint": "支持 .pdf 格式；新文档会**追加**进索引（与语料并列可检索，"
                       "不清空现有索引）；若与语料同名，则覆盖源文件后全量重建索引",
        "question_label": "请输入问题（支持中文 / Chinese）",
        "question_ph": "例如：组织结构图中销售部有几个部门构成？",
        "ask_button": "提问",
        "answer_label": "答案",
        "citations_label": "引用来源",
        # 界面拿不到金标，算不出真精确度（precision）：标签只声明它确实是「条数」，
        # 不冒用「精确度」二字（precision@k 由评估报告给出，见 docs/reports/）。
        "metrics_label": "引用条数",
        "image_label": "图像溯源",
        "health_label": "系统状态",
        "latency": "响应时间",
        "backend": "生成后端",
        "refused": "该问题与文档无关，已拒答。",
        "error_empty": "请输入问题。",
        "no_pdf": "请先上传 PDF 或确认已建库。",
        # 答案语言标注与中文/英文可读名（中英双语评估对账用）
        "answer_lang": "答案语言",
        "lang_name_zh": "中文",
        "lang_name_en": "英文",
        # 3 秒硬指标：实测值与预算一并给出，超了就说超了
        "latency_budget": "时延预算",
        # 保留一位小数：判定由 latency_verdict 先把原值取到 1 位小数再比较，
        # 否则 3000.04 ms 会渲染成「3000.0 ms，超出 3000.0 ms 预算」的自相矛盾。
        "latency_within": "响应时间 {ms:.1f} ms，在 {budget:.1f} ms 预算内",
        "latency_over": "响应时间 {ms:.1f} ms，超出 {budget:.1f} ms 预算（如实上报，未截断）",
        # 上传建库（原先的 "Indexing..." / "OK" 是硬编码英文，现纳入双语文案；
        # 追加 vs 覆盖语料重建是两种语义，文案分开，不含糊成一句）
        "indexing": "正在建库：把这份文档追加进索引（不清空现有索引，耗时较长）…",
        "indexing_rebuild": "正在重建索引：覆盖语料文件后按语料全量重建"
                            "（先清空索引，耗时较长）…",
        "upload_ok": "建库完成：文档已追加进索引，可与语料一起检索。",
        "upload_ok_rebuild": "建库完成：已按语料全量重建索引（含本次覆盖的文件）。",
        "upload_error": "建库失败：{err}（索引可能只更新了一部分，请看日志后重试）",
        "upload_bad_name": "文件名为空或不是单段 .pdf 文件名，已拒绝（不会落盘）。",
        "overwrite_warn": "「{name}」已存在于项目根目录：继续会覆盖该文件（无法恢复），"
                          "并把新内容追加进索引。",
        "overwrite_warn_corpus": "「{name}」是语料文件：继续会覆盖源文件（未纳入版本"
                                 "管理，无法恢复），并按语料全量重建索引。",
        "overwrite_confirm": "我确认覆盖该文件（覆盖后无法恢复）",
        "overwrite_required": "未勾选覆盖确认，已取消建库（未写入、未动索引）。",
        # 预热状态：第一条问答的时延是否已含模型加载，用户有权知道
        "warmup_ok": "模型已预热（reranker / CLIP / 存储常驻，首次提问不再付加载时延）",
        "warmup_warn": "预热失败：{err}；服务可用，但首次提问会多付模型加载时延",
    },
    "en": {
        "title": "Prospectus Q&A (Work Order 04: Image Parsing & Retrieval)",
        "subtitle": "Text / Table / Image tri-modal retrieval. Ask in Chinese or English.",
        "upload_label": "Upload PDF",
        "upload_button": "Upload & Build Index",
        "upload_hint": "PDF only. New documents are **added** to the index "
                       "(searchable alongside the corpus, nothing cleared); a "
                       "corpus file name overwrites the source and rebuilds the index.",
        "question_label": "Your question (English / 中文 supported)",
        "question_ph": "e.g. How many departments make up the Sales Department?",
        "ask_button": "Ask",
        "answer_label": "Answer",
        "citations_label": "Citations",
        "metrics_label": "Cited Sources",
        "image_label": "Image Source",
        "health_label": "System Health",
        "latency": "Latency",
        "backend": "LLM Backend",
        "refused": "This question is unrelated to the documents; refused.",
        "error_empty": "Please enter a question.",
        "no_pdf": "Please upload a PDF or make sure the index is built.",
        "answer_lang": "Answer Language",
        "lang_name_zh": "Chinese",
        "lang_name_en": "English",
        "latency_budget": "Latency Budget",
        "latency_within": "Latency {ms:.1f} ms, within the {budget:.1f} ms budget",
        "latency_over": "Latency {ms:.1f} ms, over the {budget:.1f} ms budget "
                        "(reported as measured, not clamped)",
        "indexing": "Indexing: adding this document to the index (the existing "
                    "index is not cleared; this takes a while)…",
        "indexing_rebuild": "Rebuilding the index: the corpus file was replaced, "
                            "so the corpus is rebuilt from scratch (index cleared "
                            "first; this takes a while)…",
        "upload_ok": "Index built: the document was added and is searchable "
                     "alongside the corpus.",
        "upload_ok_rebuild": "Index built: the corpus was rebuilt from scratch "
                             "(including the file just overwritten).",
        "upload_error": "Index build failed: {err} (the index may be only partly "
                        "updated; check the logs and retry)",
        "upload_bad_name": "Rejected: the name is empty or not a single-segment "
                           ".pdf file name (nothing was written to disk).",
        "overwrite_warn": "「{name}」 already exists in the project root. "
                          "Continuing overwrites that file (it cannot be restored) "
                          "and adds the new content to the index.",
        "overwrite_warn_corpus": "「{name}」 is a corpus file. Continuing "
                                 "overwrites the source file (not under version "
                                 "control, cannot be restored) and rebuilds the "
                                 "index from the corpus.",
        "overwrite_confirm": "I confirm overwriting that file (it cannot be restored)",
        "overwrite_required": "Overwrite not confirmed; build cancelled "
                              "(nothing written, index untouched).",
        "warmup_ok": "Models preloaded (reranker / CLIP / store resident; the first "
                     "question pays no model-loading latency)",
        "warmup_warn": "Warmup failed: {err}; the UI still works, but the first "
                       "question pays model-loading latency",
    },
}


def t(key: str, lang: str) -> str:
    """取本地化文案。缺失时回退到 key，绝不抛错导致界面崩溃。"""
    return UI_TEXT.get(lang, UI_TEXT["zh"]).get(key, key)


def render_citations(citations: list[dict]) -> list[str]:
    """引用格式化为可展示行：页码 + 类型 + 来源ID + 相似度。"""
    out = []
    for c in citations:
        out.append(
            f"p{c.get('page')} | {c.get('block_type')} | {c.get('source_id')} "
            f"| score={c.get('score')}"
        )
    return out


def image_citations(citations: list[dict]) -> list[dict]:
    """选出「能展示原图」的引用：按 ``image_path`` 是否有值判定，而非 ``block_type``。

    RC-2 之后两张图题的头条引用是**图描述文本块**（``block_type="text"``，
    ``source_id=figuretext#39#fig4``），它同样带 ``image_path``；若只放行
    ``block_type == "image"``，图题将一张原图都显示不出来（交付文档承诺的
    「原图溯源」就无法演示）。同一张图可能同时被 image 块与描述块引用
    （如 ``image#153#fig38`` 与 ``figuretext#1#fig0``）：按路径去重，保留首次
    出现顺序（即排名最靠前的引用优先）。
    """
    out: list[dict] = []
    seen: set[str] = set()
    for c in citations or []:
        path = c.get("image_path")
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(c)
    return out


def format_latency(ms: float, budget_ms: float, lang: str = "zh") -> str:
    """时延如实格式化：实测值 + 预算值 + 达标判定，超预算不截断、不取整到预算内。

    判定与显示共用同一个数：``rag04.config.latency_verdict`` 先把原值取到 1 位
    小数再比较（接口 / 界面 / 压测报告只有这一份实现），因此 3000.04 ms 会显示
    「3000.0 ms，在 3000.0 ms 预算内」——同一份测量不会出现两个结论，也不会渲染
    成「3000.0 ms，超出 3000.0 ms 预算」。
    """
    # 函数内导入（不是模块级）：界面由 ``streamlit run <本文件>`` 以**脚本**方式
    # 执行，此时 sys.path 里没有 ``src``——模块级 ``import rag04`` 会直接
    # ModuleNotFoundError（路径兜底在 main() 里才补上）。延迟到调用时导入，
    # 任何启动方式都拿得到这份共享实现。
    from rag04.config import latency_verdict

    shown, within_budget = latency_verdict(ms, budget_ms)
    key = "latency_within" if within_budget else "latency_over"
    return t(key, lang).format(ms=shown, budget=float(budget_ms))


def format_answer_lang(ans_lang: str | None, ui_lang: str = "zh") -> str:
    """答案语言的可展示名（zh → 中文/Chinese，en → 英文/English）。

    未知取值（空串、None、混合语种等）原样回显（None 显示为空串），不臆造也不隐藏。
    """
    raw = "" if ans_lang is None else str(ans_lang)
    key = f"lang_name_{raw}"
    name = t(key, ui_lang)
    return raw if name == key else name


def safe_upload_name(name: str | None) -> str | None:
    """上传文件名消毒：只取末段文件名，且必须以 .pdf 结尾，否则返回 None。

    ``st.file_uploader(type=["pdf"])`` 只是前端过滤：``..\\..\\x.pdf`` 这类名字
    能穿越项目根目录（甚至弄脏仓库根目录），故落盘前必须再校验一次。
    """
    base = Path(str(name or "").replace("\\", "/")).name.strip()
    return base if base.lower().endswith(".pdf") else None


def needs_overwrite_confirm(fname: str, corpus: tuple[str, ...],
                            dest: Path) -> bool:
    """落盘前是否需要用户显式确认覆盖。

    两种情况都算破坏性写入：文件名就是既有语料（``Settings.corpus``），或目标
    路径上已经有同名文件（可能是上次上传的）。这些 PDF 都不在版本管理里，覆盖后
    无法恢复，故必须让操作者先勾选确认再写。
    """
    return fname in tuple(corpus) or Path(dest).exists()


def build_uploaded(pipe, dest: Path, data: bytes, fname: str) -> None:
    """把消毒后的上传文件落盘，并**把它并入索引**（两种分支，语义不同）。

    - **新文档**（不在 ``pipe.s.corpus`` 里）：追加式建库 ——
      ``names=corpus + [fname]``、``reset=False``。新文档带来新的 ``chunk_id``，
      不会与旧块冲突，因此无需清库；两份语料留在索引里，上传的文档与它们并列
      可检索（上传不再毁掉工单 16 题的问答能力）。
    - **与语料同名**（覆盖了语料源文件）：文件内容变了，``chunk_id`` 随之变化，
      必须 ``reset=True`` 先清空三库，再按语料全量重建（``names=corpus``，
      覆盖后的新内容随之入库）。这条分支在界面上已经过覆盖确认勾选。

    两种分支都必须点名 ``names``：``build_all`` 默认只遍历 ``Settings.corpus``，
    不点名就永远不会解析新上传的文件（旧实现「界面报成功、索引没变」的根因）。

    失败原样抛出（界面捕获后给本地化提示 + 原始错误），绝不吞掉。
    """
    corpus = list(pipe.s.corpus)
    Path(dest).write_bytes(data)
    if fname in corpus:
        pipe.build(names=corpus, reset=True)          # 覆盖语料：先清库再全量重建
    else:
        pipe.build(names=corpus + [fname], reset=False)   # 新文档：追加，不动语料


def warm_up(pipe) -> str | None:
    """界面进程内预热：预载 reranker / CLIP / 存储，抹掉首次显示的问答的加载时延。

    服务端 ``create_app()`` 启动时就做同一件事；界面此前没做，于是屏幕上第一条
    问答要多付约 8 秒模型加载。失败只记录并返回原因（成功返回 None），绝不抛出 ——
    预热是优化，不是界面可用性的前提。
    """
    try:
        pipe.warmup()
    except Exception as e:                     # noqa: BLE001 - 预热失败不得阻断界面
        logger.warning("界面预热失败（首次提问会付模型加载时延）：%s: %s",
                       type(e).__name__, e)
        return f"{type(e).__name__}: {e}"
    return None


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="工单04 RAG", layout="wide")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    lang = st.sidebar.selectbox(
        "Language / 语言", ["zh", "en"],
        format_func=lambda x: "中文" if x == "zh" else "English",
    )

    st.title(t("title", lang))
    st.caption(t("subtitle", lang))

    from rag04.config import get_settings
    from rag04.pipeline import RAGPipeline

    settings = get_settings()
    budget_ms = settings.latency_budget_ms

    @st.cache_resource
    def get_pipe():
        # 与 api/server.py 的启动预热对齐：不预热则第一条问答要付 reranker/CLIP
        # 的加载时延（约 8s），屏幕上的首答就被误读成链路慢。预热失败只记录。
        p = RAGPipeline(settings)
        return p, warm_up(p)

    pipe, warmup_error = get_pipe()

    # ---- 侧栏：上传与状态 ----
    with st.sidebar:
        st.subheader(t("upload_label", lang))
        up = st.file_uploader(t("upload_label", lang), type=["pdf"],
                              label_visibility="collapsed")
        st.caption(t("upload_hint", lang))
        # 消毒在按钮之前：非法名字当场提示，绝不落盘（safe_upload_name 只放行单段 .pdf）
        fname = safe_upload_name(up.name) if up is not None else None
        if up is not None and fname is None:
            st.warning(t("upload_bad_name", lang))
        dest = Path(settings.project_root) / fname if fname else None
        # 与语料同名 = 覆盖语料源文件：正文变了、chunk_id 随之变化，必须清库重建；
        # 其他情况都是追加（新文档不毁掉现有索引与语料问答能力）。
        replacing = bool(fname) and fname in settings.corpus
        # 覆盖既有文件是破坏性的（源 PDF 不在版本管理里，无法恢复）：先确认再写
        need_confirm = bool(fname) and needs_overwrite_confirm(
            fname, settings.corpus, dest)
        overwrite_ok = True
        if need_confirm:
            st.warning(t("overwrite_warn_corpus" if replacing
                         else "overwrite_warn", lang).format(name=fname))
            overwrite_ok = st.checkbox(t("overwrite_confirm", lang))
        if st.button(t("upload_button", lang)):
            if fname is None or dest is None:
                st.warning(t("no_pdf", lang))
            elif need_confirm and not overwrite_ok:
                st.error(t("overwrite_required", lang))
            else:
                with st.spinner(t("indexing_rebuild" if replacing else "indexing",
                                  lang)):
                    try:
                        build_uploaded(pipe, dest, up.getbuffer(), fname)
                    except Exception as e:      # noqa: BLE001 - 任何失败都要本地化
                        # 建库失败很常见（Ollama 未启动、索引被别的进程占着）：
                        # 给本地化提示 + 原始错误，不把 Streamlit 裸栈甩给用户
                        logger.warning("上传建库失败：%s: %s", type(e).__name__, e)
                        st.error(t("upload_error", lang).format(
                            err=f"{type(e).__name__}: {e}"))
                    else:
                        st.success(t("upload_ok_rebuild" if replacing
                                     else "upload_ok", lang))

        st.divider()
        st.subheader(t("health_label", lang))
        try:
            st.json(pipe.health())
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
        # 预热状态可见：失败时首答变慢是已知原因，不该让用户猜
        if warmup_error is None:
            st.caption(t("warmup_ok", lang))
        else:
            st.caption(t("warmup_warn", lang).format(err=warmup_error))
        # 3 秒硬指标就在界面上明示，避免只报时延不报预算
        st.caption(f"{t('latency_budget', lang)}: {budget_ms:.0f} ms")

    # ---- 主区：提问 ----
    st.subheader(t("question_label", lang))
    q = st.text_input(t("question_label", lang), placeholder=t("question_ph", lang),
                      label_visibility="collapsed")
    if st.button(t("ask_button", lang), type="primary"):
        if not q.strip():
            st.warning(t("error_empty", lang))
        else:
            with st.spinner("..."):
                try:
                    ans = pipe.ask(q)
                except Exception as e:
                    # 首次运行最常见的失败就是「还没建库」：给本地化指引，
                    # 原始异常照常附在后面（本地化提示不替代真实错误）。
                    st.error(t("no_pdf", lang))
                    st.caption(f"{type(e).__name__}: {e}")
                    return

            if ans.refused:
                st.info(t("refused", lang))
                return

            st.subheader(t("answer_label", lang))
            st.markdown(ans.answer)

            c1, c2, c3, c4 = st.columns(4)
            # 与下方 caption 同精度（一位小数），同一个量不出现两个不同读数
            c1.metric(t("latency", lang), f"{ans.latency_ms:.1f} ms")
            c2.metric(t("answer_lang", lang), format_answer_lang(ans.lang, lang))
            c3.metric(t("backend", lang), ans.llm_backend)
            c4.metric(t("metrics_label", lang), f"{len(ans.citations)}")
            # 实测时延与 3 秒预算的达标结论：超了如实说超，不缓存不美化
            st.caption(format_latency(ans.latency_ms, budget_ms, lang))

            st.subheader(t("citations_label", lang))
            for line in render_citations(ans.citations):
                st.text(line)

            # 按 image_path 选取（不是 block_type）：图描述文本块也要能带出原图
            imgs = image_citations(ans.citations)
            if imgs:
                st.subheader(t("image_label", lang))
                for c in imgs:
                    p = Path(c["image_path"])
                    if p.exists():
                        st.image(str(p), caption=f"p{c['page']} {c['source_id']}")


if __name__ == "__main__":
    main()
