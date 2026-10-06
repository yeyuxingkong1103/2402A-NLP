# -*- coding: utf-8 -*-
"""
工单04 Web 演示服务（多模态问答，支持中文/英文提问）
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

在工单01/02/03 问答服务的基础上，把**图像语义块**接进问答链路：
检索命中的图像会连同缩略图、语义描述、图页来源一起返回给前端展示，
「检索精确度」也可按需（with_precision=true）计算并返回。

接口：
    GET  /                    问答界面（内置 HTML，含图像卡片展示）
    POST /api/ask             单轮问答 {question, top_k, with_precision}
    GET  /api/health          健康检查
    GET  /api/image_inventory 图像清单（页码/类型/CLIP 分类）
    GET  /api/image_questions 内置图像问题（id5/id6）
    GET  /images/<path>       抽取出的图像文件（静态）

启动：
    python src/serve.py                    # 默认 0.0.0.0:8000
    python src/serve.py --port 8010
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_extractor import IMAGES_DIR, RESULTS_DIR, WORKDIR  # noqa: E402
from rag_core import config, evaluate, generator  # noqa: E402
from run_evaluation import GROUND_TRUTH, judge_keywords  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

COLLECTION = "wo04_multimodal"

app = FastAPI(
    title="招股说明书多模态问答（工单04）",
    description="PDF 图像内容解析及检索优化 —— 文本/表格/图像三路知识库",
    version="1.0.0",
)

_RETRIEVER = None
_METRICS = {"requests": 0, "errors": 0, "latencies": []}


def get_retriever():
    """惰性装载索引（首次请求时初始化，加快启动）。"""
    global _RETRIEVER
    if _RETRIEVER is None:
        from build_multimodal_index import load_retriever
        _RETRIEVER = load_retriever(COLLECTION)
    return _RETRIEVER


# ---------------------------------------------------------------------------
# 图像元信息（供前端展示）
# ---------------------------------------------------------------------------
def _descriptions() -> dict:
    p = RESULTS_DIR / "image_descriptions.json"
    if not p.exists():
        return {"images": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"images": []}


def image_meta_for(doc: str, page: int) -> list[dict]:
    """取某页图像的展示信息（文件、来源、描述、层级事实）。"""
    out = []
    for r in _descriptions().get("images", []):
        if r.get("doc") == doc and int(r.get("page", -1)) == page:
            # 清单里的 file 形如 results/images/招股说明书2/p39_fullpage.png，
            # 前端通过 /images/<文档名>/<文件名> 访问（见 get_image 路由）
            fp = (r.get("file") or "").replace("\\", "/")
            rel = fp.split("results/images/", 1)[1] if "results/images/" in fp else ""
            out.append({
                "file": r.get("file"),
                "url": f"/images/{rel}" if rel else "",
                "source": r.get("source"),
                "chart_type": r.get("chart_type"),
                "vlm_description": (r.get("vlm_description") or "")[:800],
                "hierarchy_facts": (r.get("hierarchy_facts") or [])[:10],
                "chart_data": r.get("chart_data") or {},
            })
    return out


def _public_chunk(d: dict) -> dict:
    """把检索片段整理成前端友好的结构。"""
    return {
        "chunk_id": d.get("chunk_id"), "doc": d.get("doc"), "page": d.get("page"),
        "type": d.get("type"), "section": d.get("section", ""),
        "score": round(float(d.get("final_score", d.get("score", 0) or 0)), 4),
        "snippet": (d.get("text") or "")[:520],
        "images": (image_meta_for(str(d.get("doc", "")), int(d.get("page", 0)))
                   if d.get("type") == "image" else []),
    }


# ---------------------------------------------------------------------------
# 请求/响应模型
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., description="用户问题（中文/英文均可）")
    top_k: int = Field(5, ge=1, le=20)
    with_precision: bool = Field(False, description="是否计算 Context Precision（额外 LLM 开销）")
    question_id: int | None = Field(None, description="内置问题 id（5/6 时可返回要点判定）")


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    try:
        r = get_retriever()
        n_vec = r.vs.count()
        n_bm25 = len(r.load_bm25().doc_ids)
    except Exception as e:
        return JSONResponse({"status": "degraded", "error": str(e)}, status_code=200)
    return {
        "status": "ok", "collection": COLLECTION,
        "vector_count": n_vec, "bm25_docs": n_bm25,
        "embed_model": config.EMBED_MODEL_NAME, "llm_model": config.LLM_MODEL,
        "n_images": len(_descriptions().get("images", [])),
        "uptime_seconds": round(time.time() - _STARTED, 1),
    }


@app.get("/api/image_inventory")
def image_inventory():
    p = RESULTS_DIR / "image_inventory.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="尚未生成图片清单，请先运行 image_extractor.py")
    data = json.loads(p.read_text(encoding="utf-8"))
    return {"n_images": data.get("n_images"),
            "n_page_render": data.get("n_page_render"),
            "images": data.get("images", [])}


@app.get("/api/image_questions")
def image_questions():
    return {"questions": config.QUESTIONS_IMAGE}


@app.post("/api/ask")
def ask(req: AskRequest):
    r = get_retriever()
    t0 = time.perf_counter()
    _METRICS["requests"] += 1
    try:
        res = r.retrieve(req.question, strategy="hybrid", top_k=req.top_k,
                         recall_k=config.TOP_K_RECALL, reranker="cascade",
                         fusion="rrf", alpha=config.HYBRID_ALPHA)
        docs = res.docs
        ctx = r.format_context(docs)
        gen = generator.generate_rag(req.question, docs, ctx)
    except Exception as e:
        _METRICS["errors"] += 1
        raise HTTPException(status_code=500, detail=f"问答失败：{e}")

    elapsed = time.perf_counter() - t0
    _METRICS["latencies"].append(elapsed)

    payload = {
        "question": req.question,
        "answer": gen.answer,
        "citations": gen.citations,
        "chunks": [_public_chunk(d) for d in docs],
        "hit_images": [im for d in docs if d.get("type") == "image"
                       for im in image_meta_for(str(d.get("doc", "")),
                                                int(d.get("page", 0)))],
        "timings": {
            "retrieve_s": round(res.timings.get("total", 0), 3),
            "total_s": round(elapsed, 3),
            "detail": {k: round(v, 4) for k, v in res.timings.items()},
        },
        "under_3s": elapsed <= 3.0,
    }

    if req.with_precision:
        gt = GROUND_TRUTH.get(req.question_id or -1, "")
        try:
            payload["precision"] = {
                "context_precision": round(
                    evaluate.context_precision(
                        req.question, [d.get("text", "") for d in docs], gt), 4),
                "keyword": judge_keywords(gen.answer, req.question_id or -1),
            }
        except Exception as e:
            payload["precision"] = {"error": str(e)}
    return payload


@app.get("/api/metrics")
def metrics():
    lat = _METRICS["latencies"]
    out = {"requests_total": _METRICS["requests"],
           "errors_total": _METRICS["errors"]}
    if lat:
        s = sorted(lat)
        out.update({
            "latency_avg_ms": round(sum(s) / len(s) * 1000, 1),
            "latency_p95_ms": round(s[int(len(s) * 0.95)] * 1000, 1),
            "latency_max_ms": round(s[-1] * 1000, 1), "samples": len(s)})
    return out


@app.get("/images/{rel_path:path}")
def get_image(rel_path: str):
    """静态返回抽出的图像（用于前端缩略图与人工核验）。"""
    p = (IMAGES_DIR / rel_path).resolve()
    if not str(p).startswith(str(IMAGES_DIR.resolve())) or not p.exists():
        raise HTTPException(status_code=404, detail="图像不存在")
    return FileResponse(p)


# ---------------------------------------------------------------------------
# 内置 Web 界面
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_INDEX_HTML)


_INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>招股说明书多模态问答（工单04）</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; font-family:"Microsoft YaHei","PingFang SC",sans-serif;
         background:#f5f7fa; color:#1f2937; height:100vh; display:flex; flex-direction:column; }
  header { background:#1e3a5f; color:#fff; padding:12px 22px; display:flex;
           justify-content:space-between; align-items:center; }
  header h1 { font-size:16px; margin:0; }
  header .meta { font-size:12px; opacity:.8; }
  #chat { flex:1; overflow-y:auto; padding:18px 22px; }
  .msg { max-width:900px; margin:0 auto 16px; display:flex; gap:10px; }
  .msg.user { flex-direction:row-reverse; }
  .avatar { width:30px; height:30px; border-radius:50%; flex-shrink:0; color:#fff;
            display:flex; align-items:center; justify-content:center; font-size:12px; }
  .user .avatar { background:#2563eb; } .bot .avatar { background:#059669; }
  .bubble { background:#fff; border-radius:10px; padding:12px 16px; flex:1;
            box-shadow:0 1px 3px rgba(0,0,0,.08); white-space:pre-wrap;
            line-height:1.7; font-size:14px; }
  .user .bubble { background:#dbeafe; }
  .cites { margin-top:9px; padding-top:7px; border-top:1px dashed #e5e7eb;
           font-size:12px; color:#6b7280; }
  .cite { display:inline-block; background:#f3f4f6; border-radius:4px;
          padding:2px 7px; margin:3px 4px 0 0; }
  .imgcard { margin-top:10px; border:1px solid #e5e7eb; border-radius:8px;
             padding:8px; display:flex; gap:10px; background:#fafafa; }
  .imgcard img { width:120px; height:auto; border:1px solid #ddd; border-radius:4px;
                 background:#fff; object-fit:contain; }
  .imgcard .desc { font-size:12px; color:#374151; max-height:150px; overflow:auto;
                   white-space:pre-wrap; }
  .idx { font-size:12px; color:#9ca3af; margin-top:6px; }
  footer { padding:12px 22px; background:#fff; border-top:1px solid #e5e7eb; }
  .inputbar { max-width:900px; margin:0 auto; display:flex; gap:9px; }
  input[type=text] { flex:1; padding:10px 13px; border:1px solid #d1d5db;
                     border-radius:8px; font-size:14px; outline:none; }
  button { background:#2563eb; color:#fff; border:none; border-radius:8px;
           padding:10px 20px; font-size:14px; cursor:pointer; }
  button.ghost { background:#fff; color:#2563eb; border:1px solid #2563eb; }
  button:disabled { background:#9ca3af; cursor:not-allowed; }
  .hint { max-width:900px; margin:7px auto 0; font-size:12px; color:#9ca3af; }
</style>
</head>
<body>
<header>
  <h1>招股说明书多模态问答系统 · 工单04（图像内容解析及检索优化）</h1>
  <span class="meta" id="meta">加载中…</span>
</header>
<div id="chat">
  <div class="msg bot"><div class="avatar">AI</div><div class="bubble">
    您好，我是多模态招股书问答助手。除文本与表格外，我已解析《招股说明书2》中的<b>图表</b>
    （组织结构图、IC 市场结构图等），可回答「销售部有几个下属部门」「2008 年 IC 市场
    增长率最快/负增长的行业」这类答案只在图里的问题。支持中英文提问。</div></div>
</div>
<footer>
  <div class="inputbar">
    <input type="text" id="q" placeholder="请输入问题，例如：武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成？" autocomplete="off">
    <button class="ghost" id="demo5">图像问题5</button>
    <button class="ghost" id="demo6">图像问题6</button>
    <button id="send">发送</button>
  </div>
  <div class="hint">检索命中图像时会展示图像卡片与语义描述 · 答案均标注来源页码 · Enter 发送</div>
</footer>
<script>
const chat=document.getElementById('chat'), q=document.getElementById('q'),
      send=document.getElementById('send');
const Q5="武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？";
const Q6="武汉力源信息技术股份有限公司招股意向书中，从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？";
fetch('/api/health').then(r=>r.json()).then(d=>{
  document.getElementById('meta').textContent =
    `索引 ${d.collection||'-'} · 向量 ${d.vector_count??'-'} · 图像 ${d.n_images??0} 张 · 嵌入 ${d.embed_model||'-'}`;
}).catch(()=>document.getElementById('meta').textContent='服务未就绪（请先建索引）');

function addMsg(role,text){
  const w=document.createElement('div'); w.className='msg '+role;
  const a=document.createElement('div'); a.className='avatar';
  a.textContent = role==='user'?'我':'AI';
  const b=document.createElement('div'); b.className='bubble'; b.textContent=text;
  w.appendChild(a); w.appendChild(b); chat.appendChild(w);
  chat.scrollTop=chat.scrollHeight; return b;
}
async function ask(text){
  text = (text||q.value).trim(); if(!text) return;
  q.value=''; send.disabled=true; addMsg('user',text);
  const b=addMsg('bot','正在检索文本/表格/图像并生成答案…');
  const t0=Date.now();
  try{
    const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:text,top_k:5,with_precision:false})});
    const d=await r.json();
    if(d.detail){ b.textContent='出错了：'+d.detail; }
    else{
      b.textContent=d.answer;
      if(d.citations&&d.citations.length){
        const c=document.createElement('div'); c.className='cites';
        c.innerHTML='来源：'+d.citations.map(x=>`<span class="cite">《${x.doc}》第${x.page}页</span>`).join('');
        b.appendChild(c);
      }
      (d.hit_images||[]).forEach(im=>{
        const card=document.createElement('div'); card.className='imgcard';
        const img=document.createElement('img');
        img.src = im.url || ('/images/'+encodeURIComponent((im.file||'').split('/').pop()));
        img.alt='命中图像';
        const desc=document.createElement('div'); desc.className='desc';
        desc.textContent='【图像来源】'+(im.source==='page_render'?'矢量图整页渲染':'内嵌位图')
          +'\\n'+(im.vlm_description||'');
        card.appendChild(img); card.appendChild(desc); b.appendChild(card);
      });
      const t=document.createElement('div'); t.className='cites';
      t.textContent=`检索 ${d.timings.retrieve_s}s · 端到端 ${d.timings.total_s}s · `
        + (d.under_3s?'满足≤3s':'超过3s（LLM 网络耗时）');
      b.appendChild(t);
    }
  }catch(e){ b.textContent='请求失败：'+e; }
  send.disabled=false; q.focus();
}
send.onclick=()=>ask();
document.getElementById('demo5').onclick=()=>ask(Q5);
document.getElementById('demo6').onclick=()=>ask(Q6);
q.addEventListener('keydown',e=>{ if(e.key==='Enter') ask(); });
q.focus();
</script>
</body>
</html>"""


_STARTED = time.time()


def main() -> None:
    ap = argparse.ArgumentParser(description="工单04 Web 服务")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = ap.parse_args()

    import uvicorn
    print(f"[启动] http://127.0.0.1:{args.port}  （索引 {COLLECTION}）")
    if args.reload:
        uvicorn.run("serve:app", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
