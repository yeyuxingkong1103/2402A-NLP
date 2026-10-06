# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
脚本：Streamlit 交互界面

工单"产出物/系统功能"对应：
  1、问答界面   -> 「智能问答」页：文字输入、流式答案、引用出处（页码/章节）
  2、问答引擎   -> src/rag.py（Query 理解 → 混合检索 → 精排 → 生成 → 引用核验）
  3、PDF 解析   -> 「知识库管理」页：上传 PDF 并重建知识库
  4、知识库管理 -> 同上：查看向量库状态、重建索引、删除知识库

额外提供：
  · RAG vs 纯 LLM 对比页（工单"演示/验收"明确要求的对比分析）
  · 10 道必测题一键载入（工单给出的验收问题清单）
  · 用户反馈（好评/差评）落盘到 data/eval/feedback.jsonl，供后续迭代
  · 响应耗时面板（工单性能验收：提问到回答 ≤3 秒）

启动：streamlit run app.py
"""

from __future__ import annotations

from src import bootstrap  # 必须最先导入（离线 + 导入链预热 + 大栈线程）

import json
import os
import time
from datetime import datetime

import streamlit as st

from src import (asr, config, embedder, llm, rag, reranker, retriever,
                 vector_store)

st.set_page_config(page_title="招股说明书智能问答系统", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

FEEDBACK_FILE = os.path.join(config.EVAL_DIR, "feedback.jsonl")

# Query 理解三态（对应 rag.norm_analyze_mode）
_ANALYZE_LABELS = {
    "auto": "自动（推荐，省约 1.4s）",
    "always": "始终用 LLM（慢，质量最高）",
    "off": "关闭（最快）",
}


# ---------------------------------------------------------------------------
# 资源预热（进程内只做一次；模块级缓存跨 rerun 有效）
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _warmup() -> dict:
    """加载 embedding/精排模型、构建 BM25、预热 Ollama。

    放进大栈线程执行：sentence-transformers / transformers 的首次导入与
    from_pretrained() 在 Windows 上会走一条很深的 C 层调用链，而 Python 线程
    默认栈只有约 1MB，偶发 "Windows fatal exception: stack overflow"
    （进程直接消失，Python 层无法捕获）。见 src/bootstrap.py 的说明。
    """
    return bootstrap.run_with_large_stack(rag.warmup)


@st.cache_data(show_spinner=False)
def _load_questions() -> list[dict]:
    try:
        with open(config.EVAL_QUESTIONS_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return []


@st.cache_data(show_spinner=False, max_entries=20)
def _transcribe(audio_bytes: bytes) -> tuple[str, list]:
    """语音识别（结果按音频内容缓存）。

    Streamlit 每次交互都会重跑整个脚本，而 st.audio_input 会保留已录的音频；
    不缓存的话，用户每点一次界面都要重新识别一遍（每次约 1 秒）。
    """
    return asr.transcribe(audio_bytes)


def render_voice_input() -> str | None:
    """语音提问控件。返回用户确认后的问句；未确认时返回 None。

    识别不可靠（Windows 通用听写对专名会出同音错字），因此**强制回显**
    识别文本、允许用户修改后再提问，绝不静默提交。
    """
    if not config.ASR_ENABLED:
        return None
    av = asr.available()
    if not av["ok"]:
        return None  # 没有系统识别组件时静默隐藏，不打扰用户

    with st.expander("🎤 语音提问（中文）", expanded=False):
        st.caption(f"使用 Windows 自带离线识别引擎（{'、'.join(av['cultures'])}），"
                   "全程不联网、不下载模型。录音后**请先核对识别文字**再提问。")
        audio = st.audio_input("点击麦克风开始录音", key="voice_audio")
        if audio is None:
            return None

        try:
            text, fixes = _transcribe(audio.getvalue())
        except Exception as exc:
            st.error(f"语音识别失败：{exc}")
            return None

        if fixes:
            detail = "；".join(f"「{f['from']}」→「{f['to']}」" for f in fixes)
            st.caption(f"🔧 已按领域词表纠正同音错字：{detail}")

        if not text:
            st.warning("没有识别到内容。请靠近麦克风、放慢语速，或改用文字输入。")
            return None

        edited = st.text_input("识别结果（可直接修改）", value=text, key="voice_text")
        if st.button("✅ 用这句话提问", key="voice_send", type="primary"):
            if edited.strip():
                return edited.strip()
    return None


def _kb_status() -> dict:
    try:
        info = vector_store.info()
    except Exception as exc:
        info = {"mode": config.QDRANT_MODE, "collection": config.COLLECTION_NAME,
                "points": 0, "error": str(exc)}
    info["embedding"] = embedder.info()
    info["reranker"] = reranker.info()
    info["llm"] = llm.health()
    return info


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------
def render_sidebar() -> dict:
    st.sidebar.title("📈 招股说明书问答")
    st.sidebar.caption(f"工单编号：{config.WORKORDER_ID}")

    with st.sidebar.expander("⚙️ 检索参数", expanded=False):
        top_k = st.slider("送入 LLM 的片段数 top-k", 2, 10, config.FINAL_TOP_K)
        use_rerank = st.checkbox("启用 Cross-Encoder 精排", value=config.RERANKER_ENABLED)
        use_hybrid = st.checkbox("启用 BM25 混合召回", value=config.HYBRID_ENABLED)
        analyze_mode = st.selectbox(
            "Query 理解（意图/消歧/分解）",
            options=["auto", "always", "off"], index=0,
            format_func=lambda m: _ANALYZE_LABELS[m],
            help="自动：问题里已含公司全称时不调用 LLM（规则即可完成消歧/分解），"
                 "省约 1.4 秒；含指代的问题仍交给 LLM。"
                 "始终用 LLM：最慢但改写质量最高，适合做效果对照。")

    st.sidebar.divider()
    st.sidebar.subheader("📋 必测问题（工单给定 10 题）")
    questions = _load_questions()
    picked = None
    for q in questions:
        if st.sidebar.button(f"[{q['id']}] {q['question'][:26]}…",
                             key=f"q{q['id']}", use_container_width=True,
                             help=q["question"]):
            picked = q["question"]
    st.sidebar.divider()
    if st.sidebar.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    opts = {"top_k": top_k, "rerank": use_rerank, "hybrid": use_hybrid,
            "analyze": analyze_mode, "picked": picked}
    # 存进 session_state，供"RAG vs 纯 LLM"页共用同一套开关
    st.session_state.update(opts)
    return opts


# ---------------------------------------------------------------------------
# 引用 / 计时的通用渲染
# ---------------------------------------------------------------------------
def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    good = sum(1 for c in citations if c.get("verified"))
    st.caption(f"📎 引用出处（{len(citations)} 条，内容核验通过 {good} 条）："
               "「内容核验」= 系统把答案与片段做数字/词块比对，确认该编号确实支撑该句；"
               "标 ⚠️ 的是模型标注与核验结果不一致的编号。")
    for c in citations:
        head = (f"[{c['index']}] 第 {c['page']} 页 · "
                f"{'表格' if c.get('type') == 'table' else '正文'} · "
                f"{' > '.join(c.get('heading_path') or []) or '（无章节）'}")
        badge = "✅ 内容核验通过" if c.get("verified") else (
            f"⚠️ 疑似引用错位（更可能出自 [{c['suggested_index']}]）"
            if c.get("suggested_index") else "⚠️ 未通过内容核验")
        with st.expander(f"{head}　{badge}", expanded=False):
            if c.get("rerank_score") is not None:
                st.caption(f"精排相关性：{c['rerank_score']:.3f}")
            st.markdown(f"> {c.get('snippet', '')}…")


def render_repairs(repairs: list[dict]) -> None:
    """显示引用修正记录（有据可查才改，且必须告知用户）。"""
    if not repairs:
        return
    detail = "；".join(
        f"[{r['from']}](第{r['from_page']}页) → [{r['to']}](第{r['to_page']}页)" for r in repairs)
    st.info(f"🔧 系统按内容核验修正了 {len(repairs)} 处引用编号：{detail}。"
            f"（原片段与该句内容几乎无交集，另有片段高度重合，故纠正为后者）")


def render_timings(t: dict) -> None:
    parts = []
    for key, label in (("analyze_s", "Query理解"), ("retrieve_s", "检索"),
                       ("rerank_s", "精排"), ("generate_s", "生成")):
        if key in t:
            path = f"（{t['analyze_path']}）" if key == "analyze_s" and t.get("analyze_path") else ""
            parts.append(f"{label} {t[key]}s{path}")
    if "ttft_s" in t:
        parts.append(f"**首字 {t['ttft_s']}s**")
    total = t.get("total_s")
    if total is not None:
        parts.append(f"**总计 {total}s**")
    speed = "🟢 达标（≤3s）" if (total or 99) <= 3 else "🟡 超 3s（冷启动或上下文较长）"
    st.caption("⏱ " + " ｜ ".join(parts) + f" ｜ 工单指标：{speed}")


def save_feedback(question: str, answer: str, verdict: str, mode: str) -> None:
    os.makedirs(config.EVAL_DIR, exist_ok=True)
    row = {"time": datetime.now().isoformat(timespec="seconds"),
           "question": question, "answer": answer, "feedback": verdict, "mode": mode}
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 页面 1：智能问答
# ---------------------------------------------------------------------------
def page_chat(opts: dict) -> None:
    st.header("💬 智能问答")
    st.caption("基于《招股说明书1.pdf》检索增强生成；答案中的 [n] 对应下方引用出处的页码。")

    st.session_state.setdefault("messages", [])

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_repairs(msg.get("repairs") or [])
                render_citations(msg.get("citations") or [])
                if msg.get("timings"):
                    render_timings(msg["timings"])
                if msg.get("analysis"):
                    with st.expander("🧠 Query 理解结果", expanded=False):
                        st.json(msg["analysis"])

    voice_q = render_voice_input()

    question = st.chat_input("请输入关于招股说明书的问题（中文/English 均可）…")
    if opts.get("picked"):
        question = opts["picked"]
    if voice_q:
        question = voice_q

    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        holder: dict = {}
        buf = ""
        t0 = time.time()
        for event in rag.ask_stream(question, top_k=opts["top_k"],
                                    analyze=opts["analyze"],
                                    hybrid=opts["hybrid"],
                                    use_rerank=opts["rerank"]):
            if event["type"] == "meta":
                holder["analysis"] = event["analysis"]
                holder["contexts"] = event["contexts"]
                holder["timings"] = event["timings"]
                with st.expander(f"🔍 检索到 {len(event['contexts'])} 个片段",
                                 expanded=False):
                    for i, c in enumerate(event["contexts"], start=1):
                        st.markdown(f"**[{i}] 第 {c.get('page_idx', 0) + 1} 页** · "
                                    f"{' > '.join(c.get('heading_path') or [])}")
                        st.caption((c.get("raw_text") or c.get("text", ""))[:300])
            elif event["type"] == "delta":
                buf += event["text"]
                placeholder.markdown(buf + "▌")
            else:  # done
                holder["answer"] = event["answer"]
                holder["citations"] = event["citations"]
                holder["repairs"] = event.get("repairs") or []
                holder["truncated"] = bool(event.get("truncated"))
                holder["timings"] = event["timings"]
        placeholder.markdown(holder.get("answer", buf))
        if holder.get("truncated"):
            st.warning("⚠️ 本次生成被提前截断（检测到模型重复输出引用编号，或超过生成时限），"
                       "以上为已生成内容，建议换个问法重试。")
        render_repairs(holder.get("repairs") or [])
        render_citations(holder.get("citations") or [])
        holder["timings"]["wall_s"] = round(time.time() - t0, 3)
        render_timings(holder["timings"])
        if holder.get("analysis"):
            with st.expander("🧠 Query 理解结果（意图 / 消歧 / 分解）", expanded=False):
                st.json(holder["analysis"])

        c1, c2, _ = st.columns([1, 1, 6])
        if c1.button("👍 有用", key=f"up{len(st.session_state.messages)}"):
            save_feedback(question, holder.get("answer", ""), "up", "rag")
            st.toast("感谢反馈，已记录 ✅")
        if c2.button("👎 不准", key=f"down{len(st.session_state.messages)}"):
            save_feedback(question, holder.get("answer", ""), "down", "rag")
            st.toast("已记录，将用于后续优化 🙏")

    st.session_state.messages.append({
        "role": "assistant", "content": holder.get("answer", buf),
        "citations": holder.get("citations") or [],
        "repairs": holder.get("repairs") or [],
        "truncated": bool(holder.get("truncated")),
        "timings": holder.get("timings") or {},
        "analysis": holder.get("analysis") or {},
    })


# ---------------------------------------------------------------------------
# 页面 2：RAG vs 纯 LLM
# ---------------------------------------------------------------------------
def page_compare() -> None:
    st.header("⚖️ RAG 检索问答 vs 纯 LLM 回答")
    st.caption("工单验收要求：对比『基于 PDF 的返回结果』与『只使用 LLM 返回的答案』。")

    questions = _load_questions()
    options = ["（自定义问题）"] + [f"[{q['id']}] {q['question']}" for q in questions]
    choice = st.selectbox("选择要对比的问题", options, index=1 if questions else 0)
    default_q = questions[0]["question"] if questions else ""
    question = st.text_input("问题", value=default_q) if choice == "（自定义问题）" \
        else choice.split("] ", 1)[1]

    if not st.button("开始对比", type="primary"):
        return

    left, right = st.columns(2)
    with left:
        st.subheader("🔎 RAG（检索增强）")
        t0 = time.time()
        res = rag.ask(question, hybrid=st.session_state.get("hybrid", None),
                      use_rerank=st.session_state.get("rerank", None))
        st.markdown(res.answer or "（无回答）")
        render_repairs(res.repairs)
        render_citations(res.citations)
        res.timings["wall_s"] = round(time.time() - t0, 3)
        render_timings(res.timings)
    with right:
        st.subheader("🧠 纯 LLM（无检索）")
        t0 = time.time()
        pure = rag.ask_pure_llm(question)
        st.markdown(pure.answer or "（无回答）")
        st.caption("⚠️ 未检索文档，答案完全来自模型参数记忆，无法核对出处。")
        pure.timings["wall_s"] = round(time.time() - t0, 3)
        render_timings(pure.timings)

    c1, c2, _ = st.columns([1, 1, 6])
    if c1.button("👍 RAG 更好", key="cmp_up"):
        save_feedback(question, res.answer, "rag_better", "compare")
        st.toast("已记录 ✅")
    if c2.button("👎 纯 LLM 更好", key="cmp_down"):
        save_feedback(question, pure.answer, "llm_better", "compare")
        st.toast("已记录 ✅")


# ---------------------------------------------------------------------------
# 页面 3：知识库管理
# ---------------------------------------------------------------------------
def page_kb() -> None:
    st.header("📚 知识库管理")
    info = _kb_status()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("向量条数", f"{info.get('points', 0):,}")
    c2.metric("向量维度", info["embedding"]["dim"])
    c3.metric("Embedding 设备", info["embedding"]["device"])
    c4.metric("精排", "启用" if info["reranker"]["enabled"] else "未启用")

    st.markdown("#### 当前配置")
    st.json({
        "向量库": {"模式": info["mode"], "collection": info["collection"],
                   "路径": info.get("path", "")},
        "Embedding": info["embedding"],
        "Reranker": info["reranker"],
        "LLM": {k: v for k, v in info["llm"].items() if k != "models"},
        "检索": retriever.stats() if info.get("points") else {},
    })

    st.divider()
    st.markdown("#### 上传 PDF 并重建知识库")
    st.caption("上传后系统会：解析 PDF → 分块 → 向量化 → 重建向量库。"
               "**注意：会覆盖当前知识库**，548 页的招股书约需 3~5 分钟。")
    up = st.file_uploader("选择一个 PDF 文件", type=["pdf"])
    if up is not None and st.button("🚀 开始重建", type="primary"):
        _rebuild_from_upload(up)

    st.divider()
    st.markdown("#### 重新构建当前知识库")
    st.caption(f"对 {config.SOURCE_NAME} 重新执行 解析 → 分块 → 向量化 → 入库。")
    if st.button("♻️ 全量重建", help="等价于 python build_index.py --rebuild"):
        with st.spinner("正在重建，请勿关闭页面（约 3~5 分钟）…"):
            ok = _run_build()
        if ok:
            st.success("重建完成")
            retriever.refresh()
        else:
            st.error("重建失败，请查看上方日志")
        st.cache_resource.clear()

    with st.expander("🗑️ 危险操作：删除知识库"):
        st.caption("删除后需重新构建才能问答。")
        if st.button("确认删除 collection"):
            vector_store.ensure_collection(embedder.get_dim(), recreate=True)
            retriever.refresh()
            st.warning("已删除并重建空 collection")


def _run_build() -> bool:
    """调用 build_index 的全流程（复用其解析/分块/入库逻辑）。"""
    import subprocess
    import sys

    # 本地 Qdrant 是单进程文件锁：必须先释放本进程持有的客户端，
    # 否则子进程会因拿不到锁而失败。
    vector_store.close()

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable, "build_index.py", "--skip-parse", "--rebuild"],
                          cwd=config.ROOT, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    st.code(proc.stdout[-4000:] or "(无输出)")
    if proc.returncode != 0:
        st.code(proc.stderr[-2000:] or "(无错误输出)")
    return proc.returncode == 0


def _rebuild_from_upload(uploaded) -> None:
    """解析上传的 PDF 并重建知识库。"""
    import json as _json

    from src import chunker, pdf_parser

    tmp_path = os.path.join(config.PARSED_DIR, f"_upload_{uploaded.name}")
    with open(tmp_path, "wb") as fh:
        fh.write(uploaded.getbuffer())
    stem = os.path.splitext(uploaded.name)[0]

    progress = st.progress(0.0, text="解析 PDF…")
    try:
        content, markdown = pdf_parser.parse_pdf(tmp_path)
        with open(os.path.join(config.PARSED_DIR, f"{stem}.md"), "w",
                  encoding="utf-8") as fh:
            fh.write(markdown)
        with open(os.path.join(config.PARSED_DIR, f"{stem}_content_list.json"), "w",
                  encoding="utf-8") as fh:
            _json.dump(content, fh, ensure_ascii=False, indent=1)

        progress.progress(0.45, text="分块…")
        chunks = chunker.chunk_content_list(content, source=uploaded.name)

        progress.progress(0.55, text=f"向量化 {len(chunks)} 个片段…")
        vectors = embedder.embed_texts([c["text"] for c in chunks])

        progress.progress(0.9, text="写入向量库…")
        vector_store.ensure_collection(embedder.get_dim(), recreate=True)
        n = vector_store.upsert_chunks(chunks, vectors, progress=False)
        retriever.refresh()
        progress.progress(1.0, text="完成")
        st.success(f"知识库已重建：{n} 个片段（来源 {uploaded.name}）")
    except Exception as exc:
        st.error(f"重建失败：{exc}")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 页面 4：系统信息
# ---------------------------------------------------------------------------
def page_about() -> None:
    st.header("ℹ️ 系统信息")
    st.markdown(f"""
**工单编号**：{config.WORKORDER_ID}

| 环节 | 选型 | 本地路径 |
| --- | --- | --- |
| PDF 解析 | PyMuPDF + pdfplumber（MinerU 未安装，本机无权重，见技术文档） | — |
| 分块 | 自研：按标题层级 + 表格独立成块 | `src/chunker.py` |
| Embedding | BAAI/bge-m3（1024 维，fp16） | `{config.EMBEDDING_MODEL_PATH}` |
| 向量库 | Qdrant（本地嵌入式模式） | `{config.QDRANT_LOCAL_PATH}` |
| 精排 | BAAI/bge-reranker-v2-m3 | `{config.RERANKER_MODEL_PATH}` |
| LLM | Ollama {config.OLLAMA_MODEL} | `{config.OLLAMA_HOST}` |
| 框架 | LangChain（ChatOllama + LCEL） | — |
| 界面 | Streamlit | `app.py` |
| 语音输入 | Windows System.Speech（SAPI，系统自带离线引擎） | `src/asr.py` |

**硬约束**：全流程离线，`local_files_only=True` + `HF_HUB_OFFLINE=1`，
不调用任何会触发模型下载的接口（Ollama 只调 `/api/chat`，不调 `/api/pull`）。
""")
    st.markdown("#### 本地可用模型清单")
    st.write(llm.info().get("local_pulled", []))

    st.divider()
    st.markdown("#### 已知限制")
    st.markdown("""
- **语音输入仅支持中文**：语音识别走 Windows 自带的 System.Speech 离线引擎
  （本机无 whisper / faster-whisper / funasr / vosk，缓存中也没有任何 ASR 权重，
  硬约束又禁止下载，系统自带引擎是唯一可行路径）。本机只安装了 **zh-CN**
  识别器，因此**英文语音输入不可用**；英文**文字**问答不受影响。
  通用听写对专有名词可能出现同音错字（实测"兴图"→"信徒"），
  系统会按 `config.ASR_HOTWORDS` 领域词表自动纠正，并**强制回显识别文本**
  供用户确认后再提问。
- **本地向量库为单进程文件锁**：同一时刻只允许一个进程访问 `data/qdrant`，
  运行建库/评估脚本前请先关闭本页面。
- **上传 PDF 会覆盖当前知识库**：当前为单文档知识库设计（payload 中已带
  `source` 字段，多文档过滤能力已预留）。
""")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main() -> None:
    with st.spinner("正在加载本地模型（bge-m3 / reranker / Ollama）…"):
        warm = _warmup()
    if warm.get("llm_error"):
        st.error(f"LLM 不可用：{warm['llm_error']}\n请先启动 Ollama（ollama serve）")
    if warm.get("retriever_error"):
        st.error(f"检索不可用：{warm['retriever_error']}\n"
                 f"请先构建知识库：python build_index.py")

    opts = render_sidebar()
    tabs = st.tabs(["💬 智能问答", "⚖️ RAG vs 纯 LLM", "📚 知识库管理", "ℹ️ 系统信息"])
    with tabs[0]:
        page_chat(opts)
    with tabs[1]:
        page_compare()
    with tabs[2]:
        page_kb()
    with tabs[3]:
        page_about()


if __name__ == "__main__":
    main()
