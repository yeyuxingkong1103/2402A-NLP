# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-基于PDF文档的问答系统
app/streamlit_app.py — Streamlit 前端界面

功能：
  1. 上传 PDF 并一键入库（解析→分块→嵌入→Milvus+MySQL）
  2. 文字输入提问
  3. 语音输入（浏览器 Web Speech API，中文识别）
  4. 展示答案、引用来源、耗时
  5. 反馈：点赞/点踩/文字评论
  6. RAG / 纯 LLM 对比开关
  7. 所有注释包含工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""
import os
import sys
import time
import json
import requests
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# 工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 默认 API 地址（后端 FastAPI 运行在 8001 端口）
DEFAULT_API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8001")

# ============ 页面配置 ============
st.set_page_config(
    page_title="RAG PDF 问答系统 | 工单：人工智能NLP-RAG-基于PDF文档的问答系统",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 自定义样式
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 0; }
    .stTextArea textarea { font-size: 15px; }
    .ref-card { background:#f7f7f9; border-left:4px solid #4a90d9; padding:8px 12px;
                border-radius:4px; margin-bottom:6px; font-size:13px; }
    .latency-badge { background:#e8f5e9; color:#2e7d32; padding:2px 10px;
                     border-radius:12px; font-size:12px; font-weight:bold; }
    .metric-box { background:#fafafa; border:1px solid #e0e0e0; border-radius:8px;
                  padding:12px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ============ 会话状态 ============
# 工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 初始化会话状态
if "history" not in st.session_state:
    st.session_state.history = []
if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""

# ============ API 调用工具函数 ============
# 工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 封装后端 API 调用
def api_health(base):
    try:
        r = requests.get(f"{base}/api/health", timeout=5)
        return r.json()
    except Exception as e:
        return {"status": "error", "detail": str(e)}

def api_ask(base, question, top_k=5, use_rag=True):
    try:
        r = requests.post(f"{base}/api/ask", json={
            "question": question, "top_k": top_k, "use_rag": use_rag,
            "lang": st.session_state.get("answer_lang") or None,  # 工单二：回答语言透传（人工智能NLP-RAG-基于PDF文档的问答系统优化）
        }, timeout=60)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def api_feedback(base, qa_log_id, rating, is_correct=None, comment=None):
    try:
        r = requests.post(f"{base}/api/feedback", json={
            "qa_log_id": qa_log_id, "rating": rating,
            "is_correct": is_correct, "comment": comment
        }, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

# ============ PDF 入库（直接调用 src 模块） ============
# 工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 上传 PDF 后入库流水线
def ingest_pdf(pdf_path, progress_cb=None):
    """工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 解析→分块→嵌入→入库(Milvus+MySQL)"""
    from src.pdf_parser import parse_pdf
    from src.chunker import chunk_parsed_document
    from src.vector_store import VectorStore, import_chunks_json
    from src.db import get_session
    from src.models import Document, Chunk

    stats = {}
    # Step 1: 解析
    if progress_cb: progress_cb(0.1, "正在解析 PDF...")
    parsed = parse_pdf(pdf_path, extract_tables=True)
    out_json = Path("data/parsed") / (Path(pdf_path).stem + ".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    stats["pages"] = parsed.get("total_pages", len(parsed.get("pages", [])))
    stats["chars"] = len(parsed.get("text", ""))

    # Step 2: 分块
    if progress_cb: progress_cb(0.3, "正在分块...")
    chunked = chunk_parsed_document(parsed)
    chunks_json = Path("data/chunks") / (Path(pdf_path).stem + "_chunks.json")
    chunks_json.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_json, "w", encoding="utf-8") as f:
        json.dump(chunked, f, ensure_ascii=False, indent=2)
    chunks_list = chunked["chunks"]
    stats["chunks"] = len(chunks_list)

    # Step 3+4: 嵌入 + 入库 Milvus（import_chunks_json 内部自动嵌入）
    if progress_cb: progress_cb(0.5, "正在生成向量嵌入并写入 Milvus（首次加载模型较慢）...")
    vs = import_chunks_json(str(chunks_json))
    stats["milvus_mode"] = vs.get_stats().get("mode", "?")

    # Step 5: 入库 MySQL
    if progress_cb: progress_cb(0.9, "正在写入 MySQL 关系库...")
    try:
        import hashlib
        file_hash = hashlib.md5(open(pdf_path, "rb").read()).hexdigest()
        with get_session() as sess:
            doc = sess.query(Document).filter_by(file_hash=file_hash).first()
            if doc is None:
                doc = Document(filename=Path(pdf_path).name, file_path=str(pdf_path),
                               file_hash=file_hash, status="done",
                               total_pages=stats["pages"], total_chars=stats["chars"],
                               total_chunks=stats["chunks"])
                sess.add(doc); sess.commit(); sess.refresh(doc)
            for ck in chunks_list:
                sess.add(Chunk(doc_id=doc.id, chunk_id=ck["chunk_id"],
                               chunk_index=ck.get("chunk_index", 0),
                               global_index=ck.get("global_index", 0),
                               content=ck.get("text", ""), char_count=ck.get("char_count", 0),
                               page=ck.get("page", 0)))
            doc.status = "done"; sess.commit()
        stats["mysql_doc_id"] = doc.id
    except Exception as e:
        stats["mysql_error"] = str(e)

    if progress_cb: progress_cb(1.0, "入库完成！")
    return stats

# ============ 侧边栏 ============
with st.sidebar:
    st.header("⚙️ 设置")
    api_base = st.text_input("后端 API 地址", value=DEFAULT_API_BASE, key="api_base")

    st.divider()
    st.subheader("检索模式")
    compare_mode = st.toggle("🔄 对比模式（RAG vs 纯LLM）", value=False, help="同时调用 RAG 和纯 LLM，并排展示对比")
    if not compare_mode:
        use_rag = st.toggle("📚 使用 RAG 检索", value=True)
    else:
        use_rag = True  # 工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 对比模式下两者都调用
    top_k = st.slider("检索 Top-K", 1, 20, 5, help="从向量库检索的候选块数量")

    # 工单二：中英文问答语言切换（人工智能NLP-RAG-基于PDF文档的问答系统优化）
    st.divider()
    st.subheader("🌐 回答语言 / Language")
    lang_choice = st.radio("选择回答语言", ["自动检测", "中文", "English"], horizontal=True,
                           help="自动：按提问语言回答；英文问题将自动翻译为中文检索（bge-m3 多语言兜底）")
    st.session_state["answer_lang"] = {"自动检测": None, "中文": "zh", "English": "en"}[lang_choice]

    st.divider()
    st.subheader("📄 上传 PDF 入库")
    uploaded_file = st.file_uploader("选择 PDF 文件", type=["pdf"],
                                      help="上传后将自动执行：解析→分块→嵌入→入库")
    if uploaded_file is not None and st.button("🚀 开始入库", use_container_width=True):
        # 工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 保存上传文件并入库
        upload_dir = Path("附件")
        upload_dir.mkdir(parents=True, exist_ok=True)
        save_path = upload_dir / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.info(f"已保存: {save_path}")
        progress = st.progress(0.0, text="准备中...")
        try:
            stats = ingest_pdf(str(save_path),
                               progress_cb=lambda p, t: progress.progress(p, text=t))
            progress.empty()
            st.success("✅ 入库完成！")
            st.json(stats)
        except Exception as e:
            progress.empty()
            st.error(f"❌ 入库失败: {e}")

    st.divider()
    st.caption("工单编号：人工智能NLP-RAG-基于PDF文档的问答系统")

# ============ 主区域 ============
st.title("📄 基于 PDF 文档的 RAG 问答系统")
st.caption("工单：人工智能NLP-RAG-基于PDF文档的问答系统 ｜ 后端 FastAPI + Milvus + MySQL + DeepSeek LLM")

# ---- 健康状态 ----
health = api_health(api_base)
hcols = st.columns(4)
with hcols[0]:
    color = "🟢" if health.get("status") == "ok" else "🔴"
    st.metric("API 状态", f"{color} {health.get('status', '?')}")
with hcols[1]:
    st.metric("MySQL", health.get("mysql", "unknown"))
with hcols[2]:
    st.metric("Milvus", health.get("milvus", "unknown"))
with hcols[3]:
    st.metric("LLM 模型", health.get("llm_model", "?"))

st.divider()

# ---- 示例问题 ----
st.subheader("💡 示例问题")
try:
    eq_resp = requests.get(f"{api_base}/api/questions", timeout=5)
    example_questions = eq_resp.json() if eq_resp.ok else []
except Exception:
    example_questions = []
if example_questions:
    eq_cols = st.columns(min(len(example_questions), 4))
    for i, eq in enumerate(example_questions[:4]):
        with eq_cols[i]:
            if st.button(eq["text"][:20] + "...", key=f"eq_{eq['id']}", help=eq["text"]):
                st.session_state["question_input"] = eq["text"]

st.divider()

# ---- 提问区 ----
st.subheader("💬 提问")

# 工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 语音输入（浏览器 Web Speech API）
voice_html = """
<div style="margin-bottom:10px;">
  <button id="voiceBtn" onclick="toggleVoice()"
    style="background:#4a90d9;color:white;border:none;border-radius:8px;
           padding:6px 16px;font-size:14px;cursor:pointer;">
    🎤 开始语音输入
  </button>
  <span id="voiceStatus" style="margin-left:8px;color:#888;font-size:13px;"></span>
  <div id="voiceResult" style="margin-top:6px;color:#333;font-size:14px;"></div>
</div>
<script>
// 工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— Web Speech API 中文语音识别
let recognition = null;
let isRecording = false;
function toggleVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { alert("当前浏览器不支持 Web Speech API，请使用 Chrome/Edge。"); return; }
  if (isRecording) { recognition.stop(); return; }
  recognition = new SR();
  recognition.lang = "zh-CN";
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.onstart = function() {
    isRecording = true;
    document.getElementById("voiceBtn").textContent = "⏹️ 停止录音";
    document.getElementById("voiceStatus").textContent = "正在聆听...";
  };
  recognition.onresult = function(event) {
    let finalText = ""; let interimText = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      if (event.results[i].isFinal) finalText += event.results[i][0].transcript;
      else interimText += event.results[i][0].transcript;
    }
    document.getElementById("voiceResult").textContent = finalText + interimText;
    if (finalText) {
      // 工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 回填到 Streamlit 文本框
      const ta = window.parent.document.querySelector('textarea[data-testid="stTextArea"]');
      if (ta) {
        const nativeSetter = Object.getOwnPropertyDescriptor(
          window.parent.HTMLTextAreaElement.prototype, 'value').set;
        nativeSetter.call(ta, finalText);
        ta.dispatchEvent(new window.parent.Event('input', {bubbles:true}));
      }
    }
  };
  recognition.onerror = function(e) {
    document.getElementById("voiceStatus").textContent = "错误: " + e.error;
  };
  recognition.onend = function() {
    isRecording = false;
    document.getElementById("voiceBtn").textContent = "🎤 开始语音输入";
    document.getElementById("voiceStatus").textContent = "识别结束";
  };
  recognition.start();
}
</script>
"""
import streamlit.components.v1 as components
components.html(voice_html, height=120)

# 工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 文字输入
default_q = st.session_state.get("question_input", "")
question = st.text_area("输入你的问题", value=default_q, height=80,
                         placeholder="例如：报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
                         key="question_input")

ask_clicked = st.button("🚀 提问", type="primary", use_container_width=True)


# ---- 工单二：SSE 流式问答（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----
def _sse_stream(base, question_text, top_k_n, use_rag_flag):
    """调用 /api/ask_stream SSE 端点，逐步产出 (type, data) 事件"""
    import json as _json
    try:
        r = requests.post(f"{base}/api/ask_stream",
                          json={"question": question_text, "top_k": top_k_n,
                                "use_rag": use_rag_flag,
                                "lang": st.session_state.get("answer_lang") or None},
                          stream=True, timeout=120)
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                yield _json.loads(line[6:])
    except Exception as e:
        yield {"type": "error", "detail": str(e)}


def stream_answer_ui(base, question_text, top_k_n, use_rag_flag):
    """流式渲染：检索引用即时展示 + LLM 答案逐字显示（工单二前端逐步显示）"""
    import json as _json
    holder = st.empty()      # 答案逐步刷新区
    refs_box = st.container()
    answer_parts, final = [], None
    for ev in _sse_stream(base, question_text, top_k_n, use_rag_flag):
        etype = ev.get("type")
        if etype == "retrieved":
            with refs_box:
                st.caption(f"🔍 已检索 {ev.get('count', 0)} 个知识块"
                           f"（{ev.get('latency_ms', 0):.0f} ms），正在生成答案...")
                for ref in ev.get("references", [])[:5]:
                    st.caption(f"　· 第{ref.get('page')}页 score={ref.get('score')}")
        elif etype == "delta":
            answer_parts.append(ev.get("text", ""))
            holder.markdown("".join(answer_parts) + " ▌")
        elif etype == "done":
            final = ev
        elif etype == "error":
            holder.error(f"流式请求失败: {ev.get('detail')}")
    if answer_parts:
        holder.markdown("".join(answer_parts))
    return final


# ---- 提问处理 ----
if ask_clicked and question.strip():
    # 工单二：RAG 模式默认走 SSE 流式（逐步显示）；对比模式保留同步接口（人工智能NLP-RAG-基于PDF文档的问答系统优化）
    if use_rag and not compare_mode:
        st.subheader("📝 RAG 模式（流式）")
        done = stream_answer_ui(api_base, question, top_k, use_rag=True)
        if done:
            mcols = st.columns(3)
            with mcols[0]:
                st.metric("总耗时", f"{done.get('latency_ms', 0):.0f} ms")
            with mcols[1]:
                tu = done.get("token_usage", {})
                st.metric("Token 用量", tu.get("total_tokens", 0))
            with mcols[2]:
                st.metric("缓存", "命中" if done.get("cache_hit") else "未命中")
    else:
        with st.spinner("正在生成回答..."):
            results = []
            # 工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 对比模式同时调用 RAG 和纯 LLM
            if compare_mode:
                rag_resp, rag_err = api_ask(api_base, question, top_k, use_rag=True)
                llm_resp, llm_err = api_ask(api_base, question, top_k, use_rag=False)
                results.append(("RAG 模式", rag_resp, rag_err))
                results.append(("纯 LLM 模式", llm_resp, llm_err))
            else:
                resp, err = api_ask(api_base, question, top_k, use_rag=use_rag)
                label = "RAG 模式" if use_rag else "纯 LLM 模式"
                results.append((label, resp, err))

    # ---- 展示回答 ----
    for label, resp, err in results:
        st.subheader(f"📝 {label}")
        if err:
            st.error(f"请求失败: {err}")
            continue
        if not resp:
            st.warning("未获得响应")
            continue

        # 答案
        st.markdown(resp.get("answer", "(空回答)"))

        # 耗时 & 指标
        mcols = st.columns(4)
        with mcols[0]:
            st.metric("总耗时", f"{resp.get('latency_ms', 0):.0f} ms")
        with mcols[1]:
            tu = resp.get("token_usage", {})
            st.metric("Token 用量", tu.get("total_tokens", 0))
        with mcols[2]:
            qu = resp.get("query_understanding")
            st.metric("意图", qu.get("intent", "-") if qu else "-")
        with mcols[3]:
            bk = resp.get("breakdown")
            st.metric("检索耗时", f"{bk.get('retrieval_ms', 0):.0f} ms" if bk else "-")

        # 引用来源
        refs = resp.get("references", [])
        if refs:
            with st.expander(f"📚 引用来源（{len(refs)} 条）", expanded=True):
                for i, ref in enumerate(refs):
                    page = ref.get("page", "?")
                    score = ref.get("score")
                    score_str = f" | 相似度: {score:.3f}" if score else ""
                    preview = ref.get("preview", "")[:200]
                    st.markdown(
                        f'<div class="ref-card"><b>[{i+1}] 第 {page} 页</b>{score_str}<br>'
                        f'<span style="color:#555;">{preview}...</span></div>',
                        unsafe_allow_html=True)

        # 工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 反馈区域
        qa_log_id = resp.get("qa_log_id")
        if qa_log_id:
            st.markdown("**反馈**")
            fb_cols = st.columns([1, 1, 4])
            with fb_cols[0]:
                if st.button("👍 点赞", key=f"like_{label}_{qa_log_id}"):
                    fb, fe = api_feedback(api_base, qa_log_id, rating=5, is_correct=1)
                    st.success("已记录点赞！" if not fe else f"失败: {fe}")
            with fb_cols[1]:
                if st.button("👎 点踩", key=f"dislike_{label}_{qa_log_id}"):
                    fb, fe = api_feedback(api_base, qa_log_id, rating=1, is_correct=0)
                    st.success("已记录点踩！" if not fe else f"失败: {fe}")
            with fb_cols[2]:
                comment = st.text_input("文字评论", key=f"comment_{label}_{qa_log_id}",
                                         placeholder="可填写改进建议...")
                if st.button("提交评论", key=f"submit_comment_{label}_{qa_log_id}"):
                    if comment.strip():
                        fb, fe = api_feedback(api_base, qa_log_id, rating=3, comment=comment)
                        st.success("评论已提交！" if not fe else f"失败: {fe}")
        else:
            st.caption("（未记录 qa_log_id，反馈不可用）")

        # 查询理解详情
        qu = resp.get("query_understanding")
        if qu:
            with st.expander("🔍 查询理解详情"):
                st.json(qu)

        st.divider()

    # 工单：人工智能NLP-RAG-基于PDF文档的问答系统 —— 记录历史
    st.session_state.history.insert(0, {
        "question": question, "time": time.strftime("%H:%M:%S"),
        "mode": "对比" if compare_mode else ("RAG" if use_rag else "LLM"),
    })

# ---- 历史记录 ----
if st.session_state.history:
    st.subheader("📜 最近提问")
    for h in st.session_state.history[:10]:
        st.markdown(f"- `[{h['time']}]` **[{h['mode']}]** {h['question']}")

st.caption("工单：人工智能NLP-RAG-基于PDF文档的问答系统 ｜ 启动: `streamlit run app/streamlit_app.py --server.port 8501`")
