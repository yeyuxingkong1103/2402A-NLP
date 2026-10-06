# -*- coding: utf-8 -*-
"""
Web 服务（工单03 演示与验收入口）
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

在浏览器里演示「表格解析 + 检索优化」的效果，页面上同时给出：
    · 生成的答案
    · 检索精确度（片段精度 / 要点召回 / 表格命中）
    · 检索到的表格原文（Markdown 原样渲染，能直接看到表头与数据行的对应关系）
    · 来源页码与各阶段耗时

接口：
    GET  /                 问答界面（内置 HTML，零前端构建）
    POST /api/ask          单轮问答（返回答案 + 检索精确度 + 表格原文）
    GET  /api/health       健康检查
    GET  /api/documents    已入库文档与索引统计
    GET  /api/table_inventory  表格清单（前 N 张，含页码/行列数/表头）

启动：
    python serve.py                       # 默认 http://127.0.0.1:8013
    python serve.py --port 8000 --reload
    首次使用请先建索引：python build_index_with_tables.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_core import config                       # noqa: E402
import build_index_with_tables as bi              # noqa: E402
import table_qa as tq                             # noqa: E402

RESULTS_DIR = ROOT / "工单03-表格解析及检索优化" / "results"

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field
    _HAS_FASTAPI = True
except ImportError:                               # pragma: no cover
    _HAS_FASTAPI = False
    FastAPI = object                              # 占位，保证模块可被 import
    BaseModel = object

    class Field:                                  # type: ignore
        def __init__(self, *a, **kw): pass


# ---------------------------------------------------------------------------
# 演示问题集：把 14 问的答案要点做成速查表，页面上可直接点选并即时算精确度
# ---------------------------------------------------------------------------
def _demo_questions() -> list[dict]:
    """14 个演示问题 + 判分要点（表格类 4 + 文本类 10）。"""
    try:
        import run_evaluation as reval
        text_gt = reval.TEXT_GROUND_TRUTH
    except Exception:                             # pragma: no cover
        text_gt = {}

    out = []
    for q in tq.TABLE_QUESTIONS:
        gt = tq.TABLE_GROUND_TRUTH[q["id"]]
        out.append({"id": q["id"], "group": "表格类", "question": q["question"],
                    "must": gt["must"], "pages": gt["pages"]})
    for q in config.QUESTIONS_XINGTU:
        gt = text_gt.get(q["id"])
        if not gt:
            continue
        out.append({"id": q["id"], "group": "文本类", "question": q["question"],
                    "must": gt["must"], "pages": gt.get("pages", [])})
    return out


_DEMO = _demo_questions()
_GT_BY_QUESTION = {d["question"]: d for d in _DEMO}


# ---------------------------------------------------------------------------
# 服务状态
# ---------------------------------------------------------------------------
class State:
    retriever = None
    collection = bi.COLLECTION_WITH_TABLES
    use_llm = True


STATE = State()


def get_retriever():
    """惰性装载检索器（首次请求时才加载模型与索引，加快启动）。"""
    if STATE.retriever is None:
        try:
            STATE.retriever = bi.load_retriever(STATE.collection)
        except Exception as e:
            raise RuntimeError(
                f"索引未就绪（{e}）。请先运行：python build_index_with_tables.py"
            )
    return STATE.retriever


def check_llm_available() -> bool:
    """是否配置了生成模型 API Key。"""
    return bool(config.DEEPSEEK_API_KEY)


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
if _HAS_FASTAPI:
    app = FastAPI(
        title="招股说明书表格问答系统（工单03）",
        description="PDF 文档的表格解析及检索优化 —— 表格类问题检索问答服务",
        version="1.0.0",
    )

    class AskRequest(BaseModel):
        question: str = Field(..., description="用户问题")
        top_k: int = Field(5, ge=1, le=20)
        use_llm: bool = Field(True, description="是否调用生成模型（关闭则抽取式作答）")
        use_table_boost: bool = Field(True, description="是否启用表格块优先级提升")

    @app.get("/api/health")
    def health():
        try:
            r = get_retriever()
            n_vec = r.vs.count()
        except Exception as e:
            return {"status": "index_missing", "detail": str(e)}
        return {
            "status": "ok",
            "collection": STATE.collection,
            "vector_count": n_vec,
            "embed_model": config.EMBED_MODEL_NAME,
            "llm_model": config.LLM_MODEL,
            "llm_available": check_llm_available(),
        }

    @app.get("/api/documents")
    def documents():
        """已入库文档 + 索引统计（读 index_stats.json）。"""
        stats_file = RESULTS_DIR / "index_stats.json"
        stats = {}
        if stats_file.exists():
            stats = json.loads(stats_file.read_text(encoding="utf-8"))
        try:
            n_vec = get_retriever().vs.count()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))
        return {
            "collection": STATE.collection,
            "vector_count": n_vec,
            "docs": stats.get("docs", []),
            "chunk_types": stats.get("chunk_types", {}),
            "n_table_records": stats.get("n_table_records"),
            "n_merged_tables": stats.get("n_merged_tables"),
        }

    @app.get("/api/table_inventory")
    def table_inventory(limit: int = 50):
        """表格清单（页码/行列数/表头/字符数），供界面侧栏浏览。"""
        f = RESULTS_DIR / "table_inventory.json"
        if not f.exists():
            return {"tables": [], "msg": "表格清单未生成，请先运行 table_extractor.py"}
        data = json.loads(f.read_text(encoding="utf-8"))
        return {"summary": data.get("summary", {}),
                "tables": data.get("tables", [])[:limit]}

    @app.get("/api/demo_questions")
    def demo_questions():
        return {"questions": _DEMO}

    @app.post("/api/ask")
    def ask(req: AskRequest):
        try:
            retriever = get_retriever()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))

        use_llm = req.use_llm and check_llm_available()
        t0 = time.perf_counter()
        try:
            res = bi.answer_with_tables(
                req.question, retriever, top_k=req.top_k,
                use_table_boost=req.use_table_boost, use_llm=use_llm)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"问答失败：{e}")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 若该问题是 14 问之一，直接算检索精确度；否则只给检索分数
        gt = _GT_BY_QUESTION.get(req.question.strip())
        precision = None
        if gt:
            rs = tq.score_retrieval({"must": gt["must"]}, res["docs"])
            ans = tq.score_answer(res["answer"], {"must": gt["must"], "bonus": []})
            precision = {**rs, **ans, "grounding_pages": gt["pages"],
                         "group": gt["group"], "qid": gt["id"]}

        docs = [{
            "chunk_id": d.get("chunk_id"), "doc": d.get("doc"), "page": d.get("page"),
            "type": d.get("type"), "caption": d.get("caption", ""),
            "section": d.get("section", ""),
            "score": round(float(d.get("final_score", d.get("score", 0)) or 0), 4),
            "table_boost": d.get("table_boost"),
            "text": d.get("text", ""),
        } for d in res["docs"]]

        return {
            "question": req.question,
            "answer": res["answer"],
            "answer_mode": res["mode"],
            "precision": precision,
            "docs": docs,
            "timings": res["timings"],
            "latency_ms": round(elapsed_ms, 1),
        }

    # -- 界面 ---------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index():
        demo = json.dumps(_DEMO, ensure_ascii=False)
        return HTMLResponse(_HTML.replace("__DEMO_QUESTIONS__", demo))


_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>招股说明书表格问答系统 · 工单03</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; font-family:"Microsoft YaHei","PingFang SC",sans-serif;
         background:#f5f7fa; color:#1f2937; }
  header { background:#1e3a5f; color:#fff; padding:12px 22px; display:flex;
           justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  header .meta { font-size:12px; opacity:.8; }
  .wrap { display:flex; gap:16px; padding:16px 22px; align-items:flex-start; }
  .main { flex:1; min-width:0; }
  .side { width:330px; flex-shrink:0; }
  .card { background:#fff; border-radius:10px; padding:14px 16px;
          box-shadow:0 1px 3px rgba(0,0,0,.08); margin-bottom:14px; }
  .card h3 { margin:0 0 10px; font-size:13px; color:#374151; }
  .q { display:block; width:100%; text-align:left; border:1px solid #e5e7eb;
       background:#fafafa; border-radius:6px; padding:7px 10px; margin-bottom:6px;
       font-size:12.5px; cursor:pointer; line-height:1.5; }
  .q:hover { background:#eff6ff; border-color:#93c5fd; }
  .q .tag { display:inline-block; font-size:11px; border-radius:3px;
            padding:0 5px; margin-right:6px; color:#fff; }
  .tag.table { background:#b45309; } .tag.text { background:#0369a1; }
  .inputbar { display:flex; gap:10px; }
  input[type=text] { flex:1; padding:11px 14px; border:1px solid #d1d5db;
                     border-radius:8px; font-size:14px; outline:none; }
  input[type=text]:focus { border-color:#2563eb; }
  button.send { background:#2563eb; color:#fff; border:none; border-radius:8px;
                padding:11px 24px; font-size:14px; cursor:pointer; }
  button.send:disabled { background:#9ca3af; cursor:not-allowed; }
  .answer { white-space:pre-wrap; line-height:1.75; font-size:14px; }
  .prec { display:flex; gap:10px; flex-wrap:wrap; margin-top:10px; }
  .pill { background:#f3f4f6; border-radius:6px; padding:4px 10px; font-size:12px; }
  .pill b { color:#1d4ed8; }
  .pill.ok { background:#ecfdf5; } .pill.bad { background:#fef2f2; }
  .doc { border:1px solid #e5e7eb; border-radius:8px; margin-bottom:10px; overflow:hidden; }
  .doc .hd { background:#f9fafb; padding:6px 10px; font-size:12px; color:#4b5563;
             display:flex; justify-content:space-between; gap:8px; }
  .doc .bd { padding:10px; overflow-x:auto; font-size:12.5px; }
  .doc table { border-collapse:collapse; font-size:12px; }
  .doc table td, .doc table th { border:1px solid #d1d5db; padding:3px 7px;
                                 white-space:nowrap; }
  .doc table th { background:#f3f4f6; }
  .doc pre { margin:0; white-space:pre-wrap; word-break:break-all; font-size:12px;
             font-family:inherit; line-height:1.7; }
  .badge { font-size:11px; border-radius:3px; padding:1px 6px; color:#fff; }
  .badge.table { background:#b45309; } .badge.text { background:#6b7280; }
  .muted { color:#9ca3af; font-size:12px; }
  .typing { color:#6b7280; font-style:italic; }
</style>
</head>
<body>
<header>
  <h1>招股说明书表格问答系统 · 工单03 表格解析及检索优化</h1>
  <span class="meta" id="meta">加载中…</span>
</header>
<div class="wrap">
  <div class="main">
    <div class="card">
      <div class="inputbar">
        <input type="text" id="q" placeholder="请输入问题，例如：武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？" autocomplete="off">
        <button class="send" id="send">发送</button>
      </div>
      <div class="muted" style="margin-top:8px">
        提示：页面会同时给出「检索精确度」与「检索到的表格原文」，便于核对答案是否直接从表格读出。
      </div>
    </div>
    <div id="out"></div>
  </div>
  <div class="side">
    <div class="card">
      <h3>演示问题（14 问）</h3>
      <div id="qlist"></div>
    </div>
    <div class="card">
      <h3>索引概况</h3>
      <div id="stats" class="muted">—</div>
    </div>
  </div>
</div>
<script>
const DEMO = __DEMO_QUESTIONS__;
const out = document.getElementById('out');
const qEl = document.getElementById('q');
const sendEl = document.getElementById('send');

fetch('/api/health').then(r=>r.json()).then(d=>{
  document.getElementById('meta').textContent =
    d.status==='ok' ? `集合 ${d.collection} · 向量 ${d.vector_count} 条 · 生成模型 ${d.llm_available?'已配置':'未配置(抽取式)'}`
                    : '索引未就绪：请先运行 build_index_with_tables.py';
}).catch(()=>document.getElementById('meta').textContent='服务未就绪');

fetch('/api/documents').then(r=>r.json()).then(d=>{
  document.getElementById('stats').innerHTML =
    `集合：${d.collection}<br>向量：${d.vector_count}<br>`+
    `chunk 类型：${JSON.stringify(d.chunk_types||{})}<br>`+
    `表格记录：${d.n_table_records ?? '-'} 张（跨页合并 ${d.n_merged_tables ?? '-'} 张）<br>`+
    `文档：${(d.docs||[]).map(x=>`${x.name}(${x.pages}页/表${x.table_blocks})`).join('、')}`;
}).catch(()=>{});

const qlist = document.getElementById('qlist');
DEMO.forEach(item=>{
  const b = document.createElement('button');
  b.className = 'q';
  b.innerHTML = `<span class="tag ${item.group==='表格类'?'table':'text'}">${item.group}</span>`+
                `id ${item.id} · ${item.question}`;
  b.onclick = ()=>{ qEl.value = item.question; ask(); };
  qlist.appendChild(b);
});

function mdTable(md){
  // 把 Markdown 表格渲染成 HTML 表格；非表格内容按纯文本展示
  const lines = md.split('\n').filter(x=>x.trim().startsWith('|'));
  if (lines.length < 2) return null;
  const cut = l => l.trim().replace(/^\||\|$/g,'').split('|').map(s=>s.trim());
  const head = cut(lines[0]);
  const isSep = l => cut(l).every(c=>/^:?-{2,}:?$/.test(c));
  const body = lines.slice(1).filter(l=>!isSep(l)).map(cut);
  let h = '<table><thead><tr>' + head.map(c=>`<th>${c}</th>`).join('') + '</tr></thead><tbody>';
  body.forEach(r=>{ h += '<tr>' + r.map(c=>`<td>${c}</td>`).join('') + '</tr>'; });
  return h + '</tbody></table>';
}

function render(r){
  const box = document.createElement('div');
  box.className = 'card';
  let html = `<div class="answer">${r.answer}</div>`;
  html += `<div class="muted" style="margin-top:8px">作答模式：${r.answer_mode} ·
           检索 ${Math.round((r.timings.retrieve||0)*1000)} ms ·
           总耗时 ${r.latency_ms} ms</div>`;
  if (r.precision){
    const p = r.precision;
    html += `<div class="prec">
      <span class="pill">检索精确度 <b>${p.retrieval_precision.toFixed(3)}</b></span>
      <span class="pill">片段精度@k <b>${p.precision_at_k.toFixed(2)}</b></span>
      <span class="pill">要点召回 <b>${p.keypoint_recall.toFixed(2)}</b></span>
      <span class="pill ${p.table_hit?'ok':'bad'}">表格命中 <b>${p.table_hit?'是':'否'}</b></span>
      <span class="pill ${p.correct?'ok':'bad'}">答案要点全中 <b>${p.correct?'是':'否'}</b></span>
    </div>`;
    if (p.missing_points && p.missing_points.length)
      html += `<div class="muted" style="margin-top:6px">答案缺失要点：${p.missing_points.join('、')}</div>`;
  }
  html += `<div class="muted" style="margin-top:12px">检索到的片段（${r.docs.length} 条）：</div>`;
  r.docs.forEach((d,i)=>{
    const t = mdTable(d.text);
    html += `<div class="doc"><div class="hd">
      <span>[片段${i+1}] 《${d.doc}》第${d.page}页
        <span class="badge ${d.type==='table'?'table':'text'}">${d.type==='table'?'表格':'正文'}</span>
        ${d.caption?' · '+d.caption:''}</span>
      <span>分数 ${d.score}${d.table_boost?` · 表格加权×${d.table_boost}`:''}</span>
    </div><div class="bd">${ t || '<pre>'+d.text+'</pre>' }</div></div>`;
  });
  box.innerHTML = html;
  out.prepend(box);
}

async function ask(){
  const text = qEl.value.trim();
  if(!text) return;
  sendEl.disabled = true;
  const loading = document.createElement('div');
  loading.className = 'card typing';
  loading.textContent = '正在检索文档并生成答案…';
  out.prepend(loading);
  try {
    const res = await fetch('/api/ask', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question:text, top_k:5})
    });
    const d = await res.json();
    loading.remove();
    if (d.detail){ alert('出错了：'+d.detail); }
    else render(d);
  } catch(e){ loading.textContent = '请求失败：'+e; }
  sendEl.disabled = false;
  qEl.focus();
}
sendEl.onclick = ask;
qEl.addEventListener('keydown', e=>{ if(e.key==='Enter') ask(); });
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="工单03 Web 问答服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8013)
    ap.add_argument("--collection", default=bi.COLLECTION_WITH_TABLES)
    ap.add_argument("--reload", action="store_true")
    ap.add_argument("--build-if-missing", action="store_true",
                    help="索引不存在时先自动建索引（耗时较长）")
    ap.add_argument("--no-llm", action="store_true",
                    help="强制关闭生成模型，全部走抽取式作答")
    args = ap.parse_args()

    if not _HAS_FASTAPI:
        print("未安装 fastapi/uvicorn，请先执行：\n"
              "  pip install fastapi uvicorn[standard] -i https://pypi.tuna.tsinghua.edu.cn/simple")
        sys.exit(1)

    STATE.collection = args.collection
    STATE.use_llm = not args.no_llm

    from rag_core.vectorstore import VectorStore
    if VectorStore(args.collection).count() == 0:
        if args.build_if_missing:
            print(f"[serve] 集合 `{args.collection}` 为空，开始建索引…")
            bi.build_index([config.PDF_PROSPECTUS_1, config.PDF_PROSPECTUS_2],
                           ["招股说明书1", "招股说明书2"],
                           collection=args.collection, with_tables=True)
        else:
            print(f"[serve] 警告：集合 `{args.collection}` 尚无索引。\n"
                  f"        请先运行：python build_index_with_tables.py")

    import uvicorn
    print(f"[serve] 问答界面： http://{args.host}:{args.port}")
    print(f"[serve] 集合：{args.collection} | 生成模型："
          f"{'已配置' if check_llm_available() else '未配置（抽取式作答）'}")
    uvicorn.run("serve:app" if args.reload else app,
                host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
