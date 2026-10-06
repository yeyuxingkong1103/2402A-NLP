# -*- coding: utf-8 -*-
"""
Graph RAG 问答 Web 服务（问答界面 + 图谱可视化页面）
工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

对应产出物「一、系统功能：1、问答界面（支持文字和语音输入）2、问答引擎
3、PDF解析模块 4、知识图谱管理（可视化展示）」的 Web 侧实现。

接口一览：
  GET  /                 问答界面（文字输入 + 浏览器语音输入，零前端构建）
  GET  /graph            知识图谱交互式可视化页面（拖拽/过滤/点击看属性）
  POST /api/ask          Graph RAG 问答（返回答案 + 检索到的子图 + 耗时）
  GET  /api/graph        图谱数据（nodes/links/categories/stats），供前端渲染
  GET  /api/communities  社区列表与社区摘要
  GET  /api/report       返回 research_report.py 生成的金融研报
  POST /api/upload       上传 PDF → 解析 → 分块；可选增量抽取并入图谱
  GET  /api/stats        图谱统计
  GET  /api/health       健康检查
  GET  /api/documents    已纳入语料的文档列表

启动：
    python 工单08-GraphRAG金融问答/src/serve.py
    python .../serve.py --port 8008 --reload
或：
    uvicorn serve:app --host 0.0.0.0 --port 8008     # 在 src 目录下执行
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, File, HTTPException, UploadFile     # noqa: E402
from fastapi.responses import HTMLResponse, PlainTextResponse    # noqa: E402
from pydantic import BaseModel, Field                            # noqa: E402

from rag_core.chunk import chunk_blocks                          # noqa: E402
from rag_core.graph_rag import GraphExtractor, GraphRAG          # noqa: E402
from rag_core.pdf_parse import parse_pdf                         # noqa: E402

from build_graph import EXTRACT_CACHE, GRAPH_PATH                # noqa: E402
from prepare_corpus import (                                     # noqa: E402
    RESULTS_DIR, load_graph, save_upload_chunks,
)
from visualize_graph import build_html, build_payload            # noqa: E402

REPORT_MD = RESULTS_DIR / "financial_report.md"
UPLOAD_PDF_DIR = Path(__file__).resolve().parents[1] / "data" / "uploads"
UPLOAD_PDF_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="金融知识图谱问答系统（Graph RAG）",
    description="人工智能NLP-RAG-基于Graph RAG 实现金融问答 · 工单08",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# 全局状态（惰性加载 + mtime 热更新：重建图谱后无需重启服务）
# ---------------------------------------------------------------------------
_STATE: dict[str, Any] = {"kg": None, "mtime": 0.0, "rag": None, "retriever": None}
_METRICS: dict[str, Any] = {"requests": 0, "errors": 0, "latencies": [],
                            "started_at": time.time()}


def get_kg(force: bool = False):
    """载入知识图谱；文件更新后自动重新载入。"""
    path = Path(GRAPH_PATH)
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"知识图谱尚未构建（缺少 {path}）。"
                   f"请先运行 src/build_graph.py 与 src/visualize_graph.py。")
    mtime = path.stat().st_mtime
    if force or _STATE["kg"] is None or mtime > _STATE["mtime"]:
        _STATE["kg"] = load_graph(path)
        _STATE["mtime"] = mtime
        for k in [k for k in _STATE if k.startswith("rag::")]:
            del _STATE[k]                          # 图谱变了，缓存的问答器全部失效
    return _STATE["kg"]


def get_rag(mode: str = "hybrid") -> GraphRAG:
    """构造（并缓存）GraphRAG 问答器。"""
    kg = get_kg()
    key = f"rag::{mode}"
    if _STATE.get(key) is None:
        _STATE[key] = GraphRAG(kg, retriever=_get_retriever(), mode=mode)
    return _STATE[key]


def _get_retriever():
    """可选的向量检索兜底通道；索引不存在时静默降级为纯图谱检索。"""
    if _STATE.get("retriever_checked") is None:
        try:
            from rag_core.retriever import Retriever
            r = Retriever("ccf_finance")
            r.load_bm25()
            _STATE["retriever"] = r if r.vs.count() else None
        except Exception:
            _STATE["retriever"] = None
        _STATE["retriever_checked"] = True
    return _STATE["retriever"]


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    mode: str = Field("hybrid", description="local | global | hybrid")
    top_k: int = Field(5, ge=1, le=20)
    with_subgraph: bool = Field(True, description="是否返回检索到的子图结构")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    kg = None
    try:
        kg = get_kg()
    except HTTPException:
        pass
    return {
        "status": "ok" if kg is not None else "graph_missing",
        "graph_loaded": kg is not None,
        "n_entities": len(kg.entities) if kg else 0,
        "n_relations": len(getattr(kg, "relations_merged", [])) if kg else 0,
        "n_communities": len(kg.communities) if kg else 0,
        "uptime_seconds": round(time.time() - _METRICS["started_at"], 1),
    }


@app.get("/api/stats")
def stats():
    kg = get_kg()
    return {"stats": kg.stats(),
            "graph_file": str(GRAPH_PATH),
            "communities": len(kg.communities),
            "has_summaries": bool(getattr(kg, "community_summaries", None))}


@app.get("/api/documents")
def documents():
    """列出图谱中出现的文档（从实体来源 chunk_id 前缀与社区摘要无法直接得到，
    这里以抽取缓存的 chunk_id 前缀为准）。"""
    docs: dict[str, int] = {}
    if Path(EXTRACT_CACHE).exists():
        for line in Path(EXTRACT_CACHE).read_text(encoding="utf-8").splitlines():
            try:
                cid = json.loads(line).get("chunk_id", "")
            except Exception:
                continue
            doc = cid.rsplit("-", 2)[0] if cid.count("-") >= 2 else cid
            docs[doc] = docs.get(doc, 0) + 1
    return {"n_docs": len(docs),
            "docs": [{"doc": d, "chunks": n}
                     for d, n in sorted(docs.items(), key=lambda x: -x[1])]}


@app.get("/api/communities")
def communities(limit: int = 30):
    kg = get_kg()
    summaries = getattr(kg, "community_summaries", {}) or {}
    items = sorted(kg.communities.items(), key=lambda x: -len(x[1]))[:limit]
    return {"n_communities": len(kg.communities),
            "communities": [{"id": cid, "size": len(members),
                             "members": members[:10],
                             "summary": summaries.get(cid, "")}
                            for cid, members in items]}


@app.get("/api/graph")
def graph_data(max_nodes: int = 300):
    """输出知识图谱数据，供前端 ECharts / 内置渲染器可视化。"""
    kg = get_kg()
    if max_nodes == 300:
        payload = build_payload(kg, max_nodes=max_nodes)
    else:
        payload = build_payload(kg, max_nodes=max(10, min(max_nodes, 3000)))
    return {"nodes": payload["nodes"], "links": payload["links"],
            "categories": payload["categories"], "colors": payload["colors"],
            "stats": payload["stats"]}


@app.post("/api/ask")
def ask(req: AskRequest):
    """Graph RAG 问答：返回答案、检索到的子图（实体+关系）、耗时。"""
    t0 = time.perf_counter()
    _METRICS["requests"] += 1
    kg = get_kg()
    mode = req.mode if req.mode in ("local", "global", "hybrid") else "hybrid"
    try:
        rag = get_rag(mode)
        res = rag.answer(req.question, top_k=req.top_k)
    except Exception as e:
        _METRICS["errors"] += 1
        raise HTTPException(status_code=500, detail=f"问答失败：{e}")

    out: dict[str, Any] = {
        "question": req.question, "answer": res["answer"], "mode": mode,
        "latency": round(time.perf_counter() - t0, 3), "trace": res.get("trace", {}),
    }
    if req.with_subgraph:
        local = kg.local_search(req.question)
        out["subgraph"] = {
            "seeds": local["seeds"],
            "n_nodes": len(local["nodes"]),
            "n_edges": len(local["edges"]),
            "nodes": [{"name": n, "type": kg.entities[n].type,
                       "degree": kg.entities[n].degree,
                       "description": (kg.entities[n].description or "")[:160]}
                      for n in sorted(local["nodes"],
                                      key=lambda x: -kg.entities[x].degree)[:25]
                      if n in kg.entities],
            "edges": [{"source": s, "target": t, "type": d.get("type", "相关"),
                       "description": (d.get("description") or "")[:160]}
                      for s, t, d in local["edges"][:40]],
        }
    lat = out["latency"]
    _METRICS["latencies"].append(lat)
    _METRICS["latencies"] = _METRICS["latencies"][-500:]
    return out


@app.get("/api/report", response_class=PlainTextResponse)
def report():
    """返回 research_report.py 生成的金融研报（Markdown 原文）。"""
    if not REPORT_MD.exists():
        return ("尚未生成研报。请先运行：\n"
                "python 工单08-GraphRAG金融问答/src/research_report.py")
    return REPORT_MD.read_text(encoding="utf-8")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...),
                 rebuild: int = 0,
                 max_chunks: int = 30):
    """
    PDF 解析模块：上传 PDF → 解析（文字+表格）→ 结构分块 → 落盘待并入图谱。

    Args:
        rebuild:    1 = 立即对新文档做 LLM 实体关系抽取并增量并入当前图谱
                    （调用较贵，默认 0：只解析分块，等下次 build_graph.py 统一构建）
        max_chunks: rebuild=1 时最多抽取多少块，防止误传大文件把额度打满
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")
    safe_name = Path(file.filename).name
    dst = UPLOAD_PDF_DIR / safe_name
    dst.write_bytes(await file.read())

    try:
        parsed = parse_pdf(dst, doc_name=Path(safe_name).stem,
                           with_tables=True, with_images=False, use_cache=False)
        chunks = chunk_blocks(parsed.blocks, strategy="structure",
                              size=900, overlap=120)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 解析失败：{e}")

    saved = save_upload_chunks(Path(safe_name).stem, chunks)
    result: dict[str, Any] = {
        "doc": Path(safe_name).stem, "pdf_saved": str(dst),
        "n_pages": parsed.n_pages, "n_blocks": len(parsed.blocks),
        "n_chunks": len(chunks), "chunks_saved": str(saved),
        "table_chunks": sum(1 for c in chunks if c.type == "table"),
        "message": "已解析并落盘。运行 src/build_graph.py 即可把该文档并入图谱。",
    }

    if rebuild:
        try:
            kg = get_kg()
            extractor = GraphExtractor("optimized")
            added_e = added_r = 0
            todo = chunks[:max_chunks]
            with Path(EXTRACT_CACHE).open("a", encoding="utf-8") as f:
                for c in todo:
                    data = extractor.extract(c.text, c.chunk_id)
                    delta = kg.incremental_add(c.chunk_id, data)
                    added_e += delta["new_entities"]
                    added_r += delta["new_relations"]
                    f.write(json.dumps({"chunk_id": c.chunk_id, "data": data},
                                       ensure_ascii=False) + "\n")
                    f.flush()
            kg.detect_communities()
            kg.save(GRAPH_PATH)
            _STATE["kg"] = None                     # 触发下一次请求重新载入
            result.update({
                "rebuild": True, "extracted_chunks": len(todo),
                "new_entities": added_e, "new_relations": added_r,
                "message": f"已增量抽取 {len(todo)} 块并并入图谱"
                           f"（新增 {added_e} 实体 / {added_r} 关系）。",
            })
        except Exception as e:
            result.update({"rebuild": False, "error": f"增量抽取失败：{e}"})
    return result


@app.get("/api/metrics")
def metrics():
    lat = _METRICS["latencies"]
    return {
        "requests": _METRICS["requests"], "errors": _METRICS["errors"],
        "latency_avg": round(sum(lat) / len(lat), 3) if lat else 0,
        "latency_max": round(max(lat), 3) if lat else 0,
        "uptime_seconds": round(time.time() - _METRICS["started_at"], 1),
    }


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------
@app.get("/graph", response_class=HTMLResponse)
def graph_page(max_nodes: int = 300):
    """知识图谱交互式可视化页面（布局/交互由 visualize_graph.build_html 提供）。"""
    kg = get_kg()
    return HTMLResponse(build_html(build_payload(kg, max_nodes=max_nodes)))


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_INDEX_HTML)


_INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>金融知识图谱问答 · Graph RAG（工单08）</title>
<style>
  *{box-sizing:border-box}
  body{margin:0;font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:#f1f5f9;
       color:#0f172a;height:100vh;display:flex;flex-direction:column}
  header{background:#0f2c4d;color:#fff;padding:12px 22px;display:flex;align-items:center;
         gap:14px;box-shadow:0 2px 8px rgba(0,0,0,.15)}
  header h1{font-size:16px;margin:0;font-weight:600}
  header .sub{font-size:12px;opacity:.72}
  header .sp{margin-left:auto;display:flex;gap:10px;align-items:center}
  header a{color:#93c5fd;text-decoration:none;font-size:13px;
           border:1px solid #1e4b7a;padding:5px 12px;border-radius:6px}
  header a:hover{background:#1e4b7a}
  select{background:#1e4b7a;color:#fff;border:1px solid #2c5f94;border-radius:6px;
         padding:5px 8px;font-size:13px}
  #chat{flex:1;overflow-y:auto;padding:22px}
  .msg{max-width:860px;margin:0 auto 20px;display:flex;gap:10px}
  .msg.user{flex-direction:row-reverse}
  .av{width:32px;height:32px;border-radius:50%;flex-shrink:0;display:flex;
      align-items:center;justify-content:center;font-size:12.5px;color:#fff}
  .user .av{background:#2563eb}.bot .av{background:#0d9488}
  .bub{background:#fff;border-radius:10px;padding:13px 17px;line-height:1.78;
       font-size:14px;box-shadow:0 1px 3px rgba(0,0,0,.07);flex:1;white-space:pre-wrap}
  .user .bub{background:#dbeafe}
  .meta{font-size:11.5px;color:#64748b;margin-top:8px;border-top:1px dashed #e2e8f0;
        padding-top:7px}
  details{margin-top:8px;font-size:12.5px}
  summary{cursor:pointer;color:#2563eb;outline:none}
  table{border-collapse:collapse;width:100%;margin-top:8px;font-size:12px}
  th,td{border:1px solid #e2e8f0;padding:4px 7px;text-align:left}
  th{background:#f8fafc;color:#475569;font-weight:600}
  #bottom{border-top:1px solid #e2e8f0;background:#fff;padding:14px 22px}
  #row{max-width:860px;margin:0 auto;display:flex;gap:10px;align-items:flex-end}
  textarea{flex:1;resize:none;height:52px;padding:10px 12px;border:1px solid #cbd5e1;
           border-radius:8px;font-size:14px;font-family:inherit;outline:none;line-height:1.5}
  textarea:focus{border-color:#2563eb}
  button{border:0;border-radius:8px;cursor:pointer;font-size:14px;padding:0 18px;
         height:52px;color:#fff;background:#2563eb;font-weight:600}
  button:hover{background:#1d4ed8}
  button:disabled{background:#94a3b8;cursor:not-allowed}
  #mic{width:52px;padding:0;background:#0d9488;font-size:19px}
  #mic:hover{background:#0f766e}
  #mic.rec{background:#dc2626;animation:pulse 1.1s infinite}
  @keyframes pulse{50%{opacity:.55}}
  #tip{max-width:860px;margin:8px auto 0;font-size:11.5px;color:#94a3b8;text-align:center}
  .ex{max-width:860px;margin:0 auto 14px;display:flex;gap:8px;flex-wrap:wrap}
  .ex span{font-size:12px;background:#e2e8f0;color:#334155;padding:5px 11px;
           border-radius:14px;cursor:pointer}
  .ex span:hover{background:#cbd5e1}
</style>
</head>
<body>
<header>
  <h1>金融知识图谱问答系统</h1>
  <span class="sub">Graph RAG · 工单08</span>
  <div class="sp">
    <span class="sub">检索模式</span>
    <select id="mode">
      <option value="hybrid" selected>hybrid 局部+全局</option>
      <option value="local">local 子图检索</option>
      <option value="global">global 社区摘要</option>
    </select>
    <a href="/graph" target="_blank">知识图谱可视化</a>
  </div>
</header>
<div id="chat">
  <div class="msg bot"><div class="av">AI</div><div class="bub">
您好，我是基于 Graph RAG 的金融问答助手。
我先从知识图谱中定位问题涉及的实体、沿关系扩展出子图（局部检索），
再在社区摘要上做全局检索，最后据此生成答案。回答下方可展开查看检索到的子图结构。
<details><summary>系统状态</summary><div id="stat">加载中…</div></details>
  </div></div>
</div>
<div id="bottom">
  <div class="ex" id="examples"></div>
  <div id="row">
    <textarea id="q" placeholder="请输入问题，例如：招商银行2019年的营业收入和不良贷款率分别是多少？"></textarea>
    <button id="mic" title="语音输入（需浏览器支持）">🎤</button>
    <button id="send">发送</button>
  </div>
  <div id="tip">Enter 发送 ｜ Shift+Enter 换行 ｜ 语音输入使用浏览器 Web Speech API（Chrome/Edge）</div>
</div>
<script>
const chat=document.getElementById('chat'), qEl=document.getElementById('q');
const sendBtn=document.getElementById('send'), micBtn=document.getElementById('mic');
const esc=s=>String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

fetch('/api/health').then(r=>r.json()).then(d=>{
  document.getElementById('stat').innerHTML =
    '图谱：'+d.n_entities+' 实体 / '+d.n_relations+' 关系 / '+d.n_communities+' 社区　'+
    '状态：'+(d.graph_loaded?'就绪':'图谱未构建');
}).catch(()=>{document.getElementById('stat').textContent='服务未就绪';});

const EXAMPLES=[
 '平安银行在2019年的董事长致辞中，提到其盈利增长的关键因素有哪些？',
 '招商银行2019年的营业收入、归属于本行股东的净利润和不良贷款率分别是多少？',
 '中国邮政储蓄银行2019年末的不良贷款率和拨备覆盖率分别是多少？',
 '分析这些银行和保险公司在面对经济周期波动时的共同策略与差异化策略。'
];
const exBox=document.getElementById('examples');
EXAMPLES.forEach(t=>{const s=document.createElement('span');s.textContent=t;
  s.onclick=()=>{qEl.value=t;qEl.focus();};exBox.appendChild(s);});

function bubble(cls,html){const d=document.createElement('div');d.className='msg '+cls;
  d.innerHTML=(cls==='user'?'<div class="av">我</div>':'<div class="av">AI</div>')+
              '<div class="bub">'+html+'</div>';
  chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d;}

function subgraphHtml(sg){
  if(!sg||!sg.n_nodes) return '<div class="meta">未在图谱中命中实体（已走全局检索/原文兜底）</div>';
  let h='<details><summary>查看检索到的知识图谱子图（'+sg.n_nodes+' 实体 / '+sg.n_edges+' 关系）</summary>';
  h+='<div class="meta">种子实体：'+(sg.seeds.join('、')||'—')+'</div>';
  if(sg.nodes.length){
    h+='<table><tr><th>实体</th><th>类型</th><th>度数</th><th>属性</th></tr>';
    sg.nodes.slice(0,12).forEach(n=>{h+='<tr><td>'+esc(n.name)+'</td><td>'+esc(n.type)+
      '</td><td>'+n.degree+'</td><td>'+esc(n.description||'—')+'</td></tr>';});
    h+='</table>';
  }
  if(sg.edges.length){
    h+='<table><tr><th>头实体</th><th>关系</th><th>尾实体</th><th>原文依据</th></tr>';
    sg.edges.slice(0,12).forEach(e=>{h+='<tr><td>'+esc(e.source)+'</td><td>'+esc(e.type)+
      '</td><td>'+esc(e.target)+'</td><td>'+esc(e.description||'—')+'</td></tr>';});
    h+='</table>';
  }
  return h+'</details>';
}

async function ask(){
  const question=qEl.value.trim(); if(!question) return;
  bubble('user',esc(question)); qEl.value=''; sendBtn.disabled=true;
  const loading=bubble('bot','<i>图谱检索中…</i>');
  try{
    const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:question,mode:document.getElementById('mode').value,
                           top_k:5,with_subgraph:true})});
    const d=await r.json();
    if(!r.ok) throw new Error(d.detail||('HTTP '+r.status));
    const tr=d.trace||{}, loc=tr.local||{}, glo=tr.global||{};
    loading.querySelector('.bub').innerHTML =
      esc(d.answer) + subgraphHtml(d.subgraph) +
      '<div class="meta">模式：'+d.mode+'　｜　耗时：'+d.latency+'s　｜　'+
      '局部检索：'+(loc.n_nodes||0)+' 实体/'+(loc.n_edges||0)+' 关系　｜　'+
      '全局检索：命中 '+(glo.n_communities||0)+' 个社区'+
      (tr.vector?('　｜　原文兜底：'+tr.vector.n_docs+' 片段'):'')+'</div>';
  }catch(e){
    loading.querySelector('.bub').innerHTML='<span style="color:#dc2626">请求失败：'+esc(e.message)+'</span>';
  }finally{sendBtn.disabled=false;chat.scrollTop=chat.scrollHeight;}
}
sendBtn.onclick=ask;
qEl.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();ask();}});

// ---- 语音输入（浏览器 Web Speech API，无需后端）----
const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
if(!SR){ micBtn.title='当前浏览器不支持语音输入（建议 Chrome / Edge）'; }
else{
  const rec=new SR(); rec.lang='zh-CN'; rec.interimResults=true; rec.continuous=false;
  let recording=false;
  rec.onresult=e=>{ let t='';
    for(let i=e.resultIndex;i<e.results.length;i++) t+=e.results[i][0].transcript;
    qEl.value=t; };
  rec.onerror=e=>{ recording=false; micBtn.classList.remove('rec');
    if(e.error!=='aborted') bubble('bot','语音识别失败：'+esc(e.error)+'（请检查麦克风权限）'); };
  rec.onend=()=>{ recording=false; micBtn.classList.remove('rec'); };
  micBtn.onclick=()=>{ if(recording){ rec.stop(); return; }
    try{ rec.start(); recording=true; micBtn.classList.add('rec');
         micBtn.title='正在录音，点击停止'; }
    catch(err){ bubble('bot','无法启动语音识别：'+esc(err.message)); } };
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="工单08 Graph RAG Web 服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = ap.parse_args()

    import uvicorn
    print(f"问答界面：      http://{args.host}:{args.port}/")
    print(f"图谱可视化：    http://{args.host}:{args.port}/graph")
    print(f"接口文档：      http://{args.host}:{args.port}/docs")
    if args.reload:
        uvicorn.run("serve:app", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
