# -*- coding: utf-8 -*-
"""
RAG 问答服务（FastAPI + 内置 Web 界面）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
          人工智能NLP-RAG-金融问答系统部署

对外接口：
  GET  /                    问答界面（内置 HTML，无需前端构建）
  POST /api/ask             单轮问答
  POST /api/chat            多轮对话（维护 session 上下文）
  POST /api/feedback        用户反馈（点赞/点踩，驱动自适应重排）
  POST /api/upload          上传 PDF 并增量入库
  GET  /api/health          健康检查（Docker HEALTHCHECK 用）
  GET  /api/metrics         性能指标（工单13 监控用）
  GET  /api/graph           知识图谱数据（前端可视化用）
  GET  /api/documents       已入库文档列表

启动：
    uvicorn rag_core.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from . import config, llm, query_understand
from .pipeline import PRESETS, Pipeline

app = FastAPI(
    title="招股说明书智能问答系统",
    description="基于 RAG 的金融文档问答服务（工单01~13）",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# 全局状态
# ---------------------------------------------------------------------------
_PIPELINE: Pipeline | None = None
_SESSIONS: dict[str, list[query_understand.Turn]] = defaultdict(list)
_METRICS: dict[str, Any] = {
    "requests": 0, "errors": 0, "latencies": [], "started_at": time.time(),
}
MAX_LATENCY_SAMPLES = 1000


def get_pipeline() -> Pipeline:
    """惰性初始化流水线（首次请求时才加载索引，加快容器启动）。"""
    global _PIPELINE
    if _PIPELINE is None:
        cfg = PRESETS.get("wo06_hybrid", PRESETS["wo01_baseline"])
        _PIPELINE = Pipeline(cfg, collection="prospectus")
        try:
            _PIPELINE.load_index()
        except Exception as e:
            print(f"[warn] 索引未就绪（请先运行建索引脚本）：{e}")
    return _PIPELINE


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    top_k: int = Field(5, ge=1, le=20)
    strategy: str | None = Field(None, description="vector / fulltext / hybrid")
    reranker: str | None = Field(None, description="none / llm / tfidf / adaptive / cascade")
    return_trace: bool = Field(False, description="是否返回检索与耗时明细")


class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class FeedbackRequest(BaseModel):
    session_id: str
    question: str
    chunk_id: str
    helpful: bool


# ---------------------------------------------------------------------------
# 接口实现
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    """健康检查：容器编排与负载均衡依赖此接口。"""
    p = get_pipeline()
    return {
        "status": "ok",
        "vector_count": p.retriever.vs.count(),
        "uptime_seconds": round(time.time() - _METRICS["started_at"], 1),
        "llm_model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL_NAME,
    }


@app.get("/api/metrics")
def metrics():
    """性能指标：为工单13 的性能分析提供运行时数据。"""
    lat = _METRICS["latencies"]
    out: dict[str, Any] = {
        "requests_total": _METRICS["requests"],
        "errors_total": _METRICS["errors"],
        "uptime_seconds": round(time.time() - _METRICS["started_at"], 1),
    }
    if lat:
        import statistics
        s = sorted(lat)
        out.update({
            "latency_avg_ms": round(statistics.mean(s) * 1000, 1),
            "latency_p50_ms": round(s[len(s) // 2] * 1000, 1),
            "latency_p95_ms": round(s[int(len(s) * 0.95)] * 1000, 1),
            "latency_max_ms": round(s[-1] * 1000, 1),
            "samples": len(s),
        })
    out["llm_usage"] = llm.get_usage()
    return out


@app.post("/api/ask")
def ask(req: AskRequest):
    p = get_pipeline()
    if req.strategy:
        p.cfg.strategy = req.strategy
    if req.reranker:
        p.cfg.reranker = req.reranker

    t0 = time.perf_counter()
    _METRICS["requests"] += 1
    try:
        result = p.ask(req.question, top_k=req.top_k, return_trace=True)
    except Exception as e:
        _METRICS["errors"] += 1
        raise HTTPException(status_code=500, detail=f"问答失败：{e}")

    elapsed = time.perf_counter() - t0
    _METRICS["latencies"].append(elapsed)
    if len(_METRICS["latencies"]) > MAX_LATENCY_SAMPLES:
        _METRICS["latencies"] = _METRICS["latencies"][-MAX_LATENCY_SAMPLES:]

    payload = {
        "question": req.question,
        "answer": result["answer"],
        "citations": result.get("citations", []),
        "latency_ms": round(elapsed * 1000, 1),
    }
    if req.return_trace:
        payload["trace"] = {
            "understanding": result.get("understanding"),
            "timings": {k: round(v, 4) for k, v in result.get("timings", {}).items()},
            "docs": [
                {"chunk_id": d.get("chunk_id"), "doc": d.get("doc"),
                 "page": d.get("page"), "type": d.get("type"),
                 "score": round(float(d.get("final_score", d.get("score", 0))), 4),
                 "snippet": (d.get("text") or "")[:200]}
                for d in result.get("docs", [])
            ],
        }
    return payload


@app.post("/api/chat")
def chat(req: ChatRequest):
    """多轮对话：按 session_id 维护上下文，自动做指代消解。"""
    p = get_pipeline()
    history = _SESSIONS[req.session_id]

    t0 = time.perf_counter()
    _METRICS["requests"] += 1
    try:
        result = p.ask(req.question, history=history, return_trace=True)
    except Exception as e:
        _METRICS["errors"] += 1
        raise HTTPException(status_code=500, detail=f"对话失败：{e}")
    elapsed = time.perf_counter() - t0
    _METRICS["latencies"].append(elapsed)

    history.append(query_understand.Turn(
        question=req.question, answer=result["answer"],
        docs=result.get("docs", [])))

    return {
        "session_id": req.session_id,
        "question": req.question,
        "rewritten": result.get("understanding", {}).get("改写后", req.question),
        "answer": result["answer"],
        "citations": result.get("citations", []),
        "latency_ms": round(elapsed * 1000, 1),
        "turn": len(history),
    }


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    """用户反馈：写入自适应重排器，实现「越用越准」。"""
    from .rerank import AdaptiveReranker

    rr = AdaptiveReranker()
    rr.record(req.question, {"chunk_id": req.chunk_id, "text": "", "type": "text"},
              req.helpful)
    return {"ok": True, "stats": rr.stats()}


@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    """上传 PDF 并增量入库（工单01 用户手册要求的「上传 PDF 文档」）。"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    dst = config.CACHE_DIR / "uploads" / file.filename
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(file.file.read())

    from .chunk import chunk_blocks
    from .pdf_parse import parse_pdf
    from .vectorstore import VectorStore

    p = get_pipeline()
    parsed = parse_pdf(dst, dst.stem, with_tables=p.cfg.with_tables,
                       with_images=p.cfg.with_images, use_cache=False)
    chunks = chunk_blocks(parsed.blocks, strategy=p.cfg.chunk_strategy,
                          size=p.cfg.chunk_size, overlap=p.cfg.chunk_overlap)
    n = VectorStore(p.collection).add_chunks(chunks)
    return {"ok": True, "file": file.filename, "pages": parsed.n_pages,
            "chunks_added": n}


@app.get("/api/documents")
def documents():
    p = get_pipeline()
    return {"collection": p.collection,
            "vector_count": p.retriever.vs.count(),
            "docs": [{"name": d.name, "pages": d.n_pages} for d in p.docs]}


@app.get("/api/graph")
def graph_data(max_nodes: int = 200):
    """输出知识图谱数据，供前端 ECharts / D3 可视化。"""
    import json

    kg_path = config.GRAPH_DIR / "kg.json"
    if not kg_path.exists():
        return {"nodes": [], "links": [], "msg": "知识图谱尚未构建"}

    data = json.loads(kg_path.read_text(encoding="utf-8"))
    ents = sorted(data.get("entities", []), key=lambda e: -e.get("degree", 0))
    keep = {e["name"] for e in ents[:max_nodes]}
    nodes = [{"id": e["name"], "name": e["name"], "type": e["type"],
              "desc": e.get("description", "")[:120],
              "size": min(10 + e.get("degree", 0), 50)}
             for e in ents[:max_nodes]]
    links = [{"source": r["source"], "target": r["target"],
              "type": r["type"], "desc": r.get("description", "")[:120]}
             for r in data.get("relations", [])
             if r["source"] in keep and r["target"] in keep]
    return {"nodes": nodes, "links": links, "stats": data.get("stats", {})}


# ---------------------------------------------------------------------------
# 内置 Web 界面（零构建依赖，直接返回 HTML）
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_INDEX_HTML)


_INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>招股说明书智能问答系统</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
         background: #f5f7fa; color: #1f2937; height: 100vh; display: flex; flex-direction: column; }
  header { background: #1e3a5f; color: #fff; padding: 14px 24px;
           display: flex; align-items: center; justify-content: space-between; }
  header h1 { font-size: 17px; margin: 0; font-weight: 600; }
  header .meta { font-size: 12px; opacity: .75; }
  #chat { flex: 1; overflow-y: auto; padding: 20px 24px; }
  .msg { max-width: 820px; margin: 0 auto 18px; display: flex; gap: 10px; }
  .msg.user { flex-direction: row-reverse; }
  .avatar { width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
            display: flex; align-items: center; justify-content: center;
            font-size: 13px; color: #fff; }
  .user .avatar { background: #2563eb; }
  .bot .avatar { background: #059669; }
  .bubble { background: #fff; border-radius: 10px; padding: 12px 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,.08); white-space: pre-wrap;
            line-height: 1.7; font-size: 14px; flex: 1; }
  .user .bubble { background: #dbeafe; }
  .cites { margin-top: 10px; padding-top: 8px; border-top: 1px dashed #e5e7eb;
           font-size: 12px; color: #6b7280; }
  .cite { display: inline-block; background: #f3f4f6; border-radius: 4px;
          padding: 2px 7px; margin: 3px 4px 0 0; }
  .feedback { margin-top: 8px; font-size: 13px; }
  .feedback button { border: 1px solid #d1d5db; background: #fff; border-radius: 6px;
                     padding: 3px 12px; cursor: pointer; margin-right: 6px; }
  .feedback button:hover { background: #f9fafb; }
  footer { padding: 14px 24px; background: #fff; border-top: 1px solid #e5e7eb; }
  .inputbar { max-width: 820px; margin: 0 auto; display: flex; gap: 10px; }
  input[type=text] { flex: 1; padding: 11px 14px; border: 1px solid #d1d5db;
                     border-radius: 8px; font-size: 14px; outline: none; }
  input[type=text]:focus { border-color: #2563eb; }
  button.send { background: #2563eb; color: #fff; border: none; border-radius: 8px;
                padding: 11px 24px; font-size: 14px; cursor: pointer; }
  button.send:disabled { background: #9ca3af; cursor: not-allowed; }
  .hint { max-width: 820px; margin: 8px auto 0; font-size: 12px; color: #9ca3af; }
  .typing { color: #6b7280; font-style: italic; }
</style>
</head>
<body>
<header>
  <h1>招股说明书智能问答系统</h1>
  <span class="meta" id="meta">加载中…</span>
</header>
<div id="chat">
  <div class="msg bot">
    <div class="avatar">AI</div>
    <div class="bubble">您好，我是招股说明书问答助手。您可以询问《招股说明书1》《招股说明书2》中的公司信息、财务数据、行业情况等问题。支持多轮对话，例如先问「兴图新科的军用领域收入是多少」，再追问「这个公司的法定代表人是谁」。</div>
  </div>
</div>
<footer>
  <div class="inputbar">
    <input type="text" id="q" placeholder="请输入您的问题，回车发送…" autocomplete="off">
    <button class="send" id="send">发送</button>
  </div>
  <div class="hint">支持多轮对话 · 答案均标注来源页码 · Enter 发送</div>
</footer>
<script>
const chat = document.getElementById('chat');
const q = document.getElementById('q');
const send = document.getElementById('send');
const SESSION = 'sess-' + Math.random().toString(36).slice(2);

fetch('/api/health').then(r => r.json()).then(d => {
  document.getElementById('meta').textContent =
    `向量库 ${d.vector_count} 条 · 模型 ${d.llm_model} · 嵌入 ${d.embed_model}`;
}).catch(() => document.getElementById('meta').textContent = '服务未就绪');

function addMsg(role, text, cites, qid) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + (role === 'user' ? 'user' : 'bot');
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? '我' : 'AI';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  if (cites && cites.length) {
    const c = document.createElement('div');
    c.className = 'cites';
    c.innerHTML = '来源：' + cites.map(x =>
      `<span class="cite">《${x.doc}》第${x.page}页</span>`).join('');
    bubble.appendChild(c);
  }
  if (role === 'bot' && qid) {
    const fb = document.createElement('div');
    fb.className = 'feedback';
    fb.innerHTML = '<button data-v="1">有帮助</button><button data-v="0">没帮助</button>';
    fb.querySelectorAll('button').forEach(b => b.onclick = () => {
      fetch('/api/feedback', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({session_id: SESSION, question: qid,
                              chunk_id: (cites[0]||{}).chunk_id || '', helpful: b.dataset.v === '1'})});
      fb.innerHTML = '<span style="color:#059669">感谢您的反馈，系统将据此优化检索排序。</span>';
    });
    bubble.appendChild(fb);
  }
  wrap.appendChild(avatar); wrap.appendChild(bubble);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
  return bubble;
}

async function ask() {
  const text = q.value.trim();
  if (!text) return;
  q.value = ''; send.disabled = true;
  addMsg('user', text);
  const loading = addMsg('bot', '正在检索文档并生成答案…');
  loading.classList.add('typing');
  const t0 = Date.now();
  try {
    const r = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: SESSION, question: text})
    });
    const d = await r.json();
    loading.classList.remove('typing');
    if (d.detail) { loading.textContent = '出错了：' + d.detail; }
    else {
      let ans = d.answer;
      if (d.rewritten && d.rewritten !== text) ans = `（已将问题理解为：${d.rewritten}）\\n\\n` + ans;
      loading.textContent = ans;
      if (d.citations && d.citations.length) {
        const c = document.createElement('div');
        c.className = 'cites';
        c.innerHTML = '来源：' + d.citations.map(x =>
          `<span class="cite">《${x.doc}》第${x.page}页</span>`).join('');
        loading.appendChild(c);
      }
      const t = document.createElement('div');
      t.className = 'cites';
      t.textContent = `耗时 ${d.latency_ms} ms · 第 ${d.turn} 轮`;
      loading.appendChild(t);
    }
  } catch (e) {
    loading.classList.remove('typing');
    loading.textContent = '请求失败：' + e;
  }
  send.disabled = false;
  q.focus();
}

send.onclick = ask;
q.addEventListener('keydown', e => { if (e.key === 'Enter') ask(); });
q.focus();
</script>
</body>
</html>"""
