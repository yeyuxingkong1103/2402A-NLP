# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-功能测试及评估

功能测试及评估的 Web 演示服务。

把本工单的四个产物（测试用例 / 测试结果 / 评估结果 / 问题分析）在一个页面里展示，
并内置实时问答框，方便演示视频里「跑一遍 RAG、看结果、看评估、看问题」的完整流程。

接口：
    GET  /                     评估看板（内置 HTML，零前端依赖）
    GET  /api/health           健康检查
    GET  /api/corpus           语料统计（corpus_stats.json）
    GET  /api/testcases        测试用例（test_cases.json）
    GET  /api/results          RAG 测试结果（rag_test_results.json）
    GET  /api/evaluation       评估结果（evaluation.json）
    GET  /api/problems         问题分析（problem_analysis.json / .md）
    POST /api/ask              实时问答（走 01-06 的 wo06_hybrid 流水线）

启动：
    python serve.py                     # http://127.0.0.1:8007
    python serve.py --port 8010 --reload
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

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import HTMLResponse, PlainTextResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from rag_core import config  # noqa: E402
from prepare_corpus import COLLECTION, PRESET, WO_NO  # noqa: E402

WO_DIR = Path(__file__).resolve().parents[1]
RESULTS = WO_DIR / "results"

app = FastAPI(
    title="RAG 功能测试及评估看板",
    description=f"{WO_NO} —— 10 个问题 × 检索结果 × 评估指标 × 问题分析",
    version="1.0.0",
)

_PIPELINE = None
_METRICS: dict[str, Any] = {"requests": 0, "errors": 0, "started_at": time.time()}


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------
def _read_json(name: str) -> dict:
    p = RESULTS / name
    if not p.exists():
        raise HTTPException(
            status_code=404,
            detail=f"缺少 {name}，请先运行：python prepare_corpus.py → "
                   f"build_questions.py → run_rag_test.py → run_evaluation.py → "
                   f"analyze_problems.py",
        )
    return json.loads(p.read_text(encoding="utf-8"))


def get_pipeline():
    """惰性加载 01-06 的 RAG 流水线（首次问答请求时才加载索引）。"""
    global _PIPELINE
    if _PIPELINE is None:
        from prepare_corpus import load_pipeline
        _PIPELINE = load_pipeline(COLLECTION)
    return _PIPELINE


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    counts = {k: (RESULTS / k).exists() for k in (
        "corpus_stats.json", "test_cases.json", "rag_test_results.json",
        "evaluation.json", "problem_analysis.json")}
    return {
        "status": "ok",
        "wo_no": WO_NO,
        "preset": PRESET,
        "collection": COLLECTION,
        "llm_model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL_NAME,
        "wo07_artifacts": counts,
        "uptime_seconds": round(time.time() - _METRICS["started_at"], 1),
    }


@app.get("/api/corpus")
def corpus():
    return _read_json("corpus_stats.json")


@app.get("/api/testcases")
def testcases():
    return _read_json("test_cases.json")


@app.get("/api/results")
def results():
    return _read_json("rag_test_results.json")


@app.get("/api/evaluation")
def evaluation():
    return _read_json("evaluation.json")


@app.get("/api/problems", response_class=PlainTextResponse)
def problems():
    p = RESULTS / "problem_analysis.md"
    if not p.exists():
        raise HTTPException(status_code=404,
                            detail="缺少 problem_analysis.md，请先运行 analyze_problems.py")
    return p.read_text(encoding="utf-8")


class AskRequest(BaseModel):
    question: str = Field(..., description="问题")
    top_k: int = Field(5, ge=1, le=20)


@app.post("/api/ask")
def ask(req: AskRequest):
    """实时问答：用 01-06 的 RAG 系统现场演示检索与生成。"""
    _METRICS["requests"] += 1
    t0 = time.perf_counter()
    try:
        p = get_pipeline()
        trace = p.ask(req.question, top_k=req.top_k, return_trace=True)
    except Exception as e:
        _METRICS["errors"] += 1
        raise HTTPException(status_code=500, detail=f"问答失败：{e}")
    return {
        "question": req.question,
        "answer": trace.get("answer", ""),
        "citations": trace.get("citations", []),
        "understanding": trace.get("understanding", {}),
        "latency": round(time.perf_counter() - t0, 3),
        "timings": {k: round(v, 4) for k, v in (trace.get("timings") or {}).items()},
        "retrieved": [
            {"doc": d.get("doc"), "page": d.get("page"), "type": d.get("type"),
             "score": round(float(d.get("final_score", d.get("score", 0))), 4),
             "snippet": (d.get("text") or "")[:200]}
            for d in trace.get("docs", [])
        ],
    }


# ---------------------------------------------------------------------------
# 看板页面
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_INDEX_HTML)


_INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG 功能测试及评估看板</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; font-family:"Microsoft YaHei","PingFang SC",sans-serif;
         background:#f5f7fa; color:#1f2937; }
  header { background:#1e3a5f; color:#fff; padding:14px 24px; }
  header h1 { font-size:17px; margin:0; }
  header .meta { font-size:12px; opacity:.8; margin-top:4px; }
  nav { display:flex; gap:6px; background:#fff; padding:10px 24px;
        border-bottom:1px solid #e5e7eb; flex-wrap:wrap; }
  nav button { border:1px solid #d1d5db; background:#fff; border-radius:6px;
               padding:7px 16px; cursor:pointer; font-size:13px; }
  nav button.active { background:#1e3a5f; color:#fff; border-color:#1e3a5f; }
  main { padding:18px 24px 60px; max-width:1280px; margin:0 auto; }
  .card { background:#fff; border-radius:10px; padding:16px 20px; margin-bottom:14px;
          box-shadow:0 1px 3px rgba(0,0,0,.07); }
  .card h3 { margin:0 0 10px; font-size:15px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { border-bottom:1px solid #eef1f5; padding:7px 8px; text-align:left;
           vertical-align:top; }
  th { background:#f8fafc; font-weight:600; color:#475569; position:sticky; top:0; }
  tr:hover td { background:#fbfdff; }
  .ok { color:#059669; font-weight:600; }
  .bad { color:#dc2626; font-weight:600; }
  .pill { display:inline-block; background:#eef2ff; color:#3730a3; border-radius:10px;
          padding:1px 9px; font-size:12px; margin-right:4px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; }
  .kpi { background:#f8fafc; border:1px solid #eef1f5; border-radius:8px; padding:12px; }
  .kpi .v { font-size:22px; font-weight:700; color:#1e3a5f; }
  .kpi .k { font-size:12px; color:#64748b; margin-top:2px; }
  pre { white-space:pre-wrap; word-break:break-word; font-size:12.5px;
        background:#f8fafc; padding:12px; border-radius:8px; max-height:640px;
        overflow:auto; }
  .qbox { display:flex; gap:8px; }
  .qbox input { flex:1; padding:10px 12px; border:1px solid #d1d5db;
                border-radius:8px; font-size:14px; }
  .qbox button { background:#2563eb; color:#fff; border:none; border-radius:8px;
                 padding:10px 22px; cursor:pointer; }
  .muted { color:#94a3b8; font-size:12px; }
  .detail { font-size:12.5px; line-height:1.7; }
</style>
</head>
<body>
<header>
  <h1>RAG 功能测试及评估看板</h1>
  <div class="meta" id="meta">工单编号：人工智能NLP-RAG-功能测试及评估 · 加载中…</div>
</header>
<nav>
  <button data-tab="overview" class="active">总览</button>
  <button data-tab="cases">测试用例</button>
  <button data-tab="results">检索结果</button>
  <button data-tab="evaluation">评估指标</button>
  <button data-tab="problems">问题分析</button>
  <button data-tab="live">实时问答</button>
</nav>
<main id="main"><div class="card">加载中…</div></main>

<script>
const main = document.getElementById('main');
let DATA = {};

async function jget(u, raw) {
  const r = await fetch(u);
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  return raw ? r.text() : r.json();
}

function kpi(v, k) { return `<div class="kpi"><div class="v">${v}</div><div class="k">${k}</div></div>`; }
const f3 = x => (x === null || x === undefined) ? '—' : Number(x).toFixed(3);
const pc = x => (x === null || x === undefined) ? '—' : (Number(x) * 100).toFixed(1) + '%';

async function loadAll() {
  DATA.health = await jget('/api/health').catch(e => ({error: e.message}));
  DATA.corpus = await jget('/api/corpus').catch(() => null);
  DATA.cases  = await jget('/api/testcases').catch(() => null);
  DATA.res    = await jget('/api/results').catch(() => null);
  DATA.eval   = await jget('/api/evaluation').catch(() => null);
  document.getElementById('meta').textContent =
    `工单编号：人工智能NLP-RAG-功能测试及评估 · 预设 ${DATA.health.preset || '-'} · ` +
    `集合 ${DATA.health.collection || '-'} · 嵌入 ${DATA.health.embed_model || '-'}`;
  render('overview');
}

function render(tab) {
  document.querySelectorAll('nav button').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === tab));
  if (tab === 'overview') return renderOverview();
  if (tab === 'cases')    return renderCases();
  if (tab === 'results')  return renderResults();
  if (tab === 'evaluation') return renderEval();
  if (tab === 'problems') return renderProblems();
  if (tab === 'live')     return renderLive();
}

function renderOverview() {
  const c = DATA.corpus, s = DATA.res && DATA.res.summary, p = DATA.eval && DATA.eval.pass;
  let html = '<div class="card"><h3>语料</h3><div class="grid">';
  if (c) {
    html += kpi(c.n_docs, '年报份数')
          + kpi(c.total_pages, '总页数')
          + kpi(c.total_chunks, '检索块数')
          + kpi(c.total_table_chunks, '表格块数')
          + kpi(c.n_docs_filename_broken + '/' + c.n_docs, '文件名乱码份数');
  }
  html += '</div></div>';
  html += '<div class="card"><h3>测试与评估</h3><div class="grid">';
  if (s) {
    html += kpi(pc(s.retrieval_hit_rate), '检索命中率')
          + kpi(pc(s.avg_page_hit_rate), '期望页码命中率')
          + kpi(s.refused_count, '拒答题数')
          + kpi(s.avg_latency.toFixed(2) + 's', '平均耗时');
  }
  if (p) html += kpi(pc(p.pass_rate), '综合判定通过率');
  if (DATA.eval && DATA.eval.keyword_accuracy)
    html += kpi(pc(DATA.eval.keyword_accuracy.accuracy), '关键词判准准确率');
  html += '</div></div>';
  main.innerHTML = html;
}

function renderCases() {
  const cases = DATA.cases && DATA.cases.cases || [];
  let html = '<div class="card"><h3>测试用例（' + cases.length + ' 个问题）</h3>' +
    '<table><tr><th>编号</th><th>题型</th><th>难度</th><th>问题</th>' +
    '<th>主责文档</th><th>期望页码</th><th>来自示例</th></tr>';
  for (const c of cases) {
    html += `<tr><td>${c.id}</td><td><span class="pill">${c.question_type}</span></td>` +
      `<td>${c.difficulty}</td><td class="detail">${c.question}</td>` +
      `<td>${(c.reference_docs||[]).join('、')}</td>` +
      `<td>${(c.expected_pages||[]).join('、') || '跨多页'}</td>` +
      `<td>${c.sample_source ? '<span class="ok">是</span>' : '扩充'}</td></tr>`;
  }
  main.innerHTML = html + '</table></div>';
}

function renderResults() {
  const rs = DATA.res && DATA.res.results || [];
  let html = '<div class="card"><h3>逐题检索结果</h3>' +
    '<table><tr><th>编号</th><th>命中</th><th>首次排名</th><th>页码命中</th>' +
    '<th>表格块</th><th>耗时</th><th>生成答案</th></tr>';
  for (const r of rs) {
    html += `<tr><td>${r.id}</td>` +
      `<td class="${r.hit_reference_doc ? 'ok' : 'bad'}">${r.hit_reference_doc ? '✓' : '✗'}</td>` +
      `<td>${r.rank_of_reference_doc || '—'}</td>` +
      `<td>${(r.page_hits||[]).join('、') || '—'} / ${(r.expected_pages||[]).join('、') || '不限'}</td>` +
      `<td>${r.n_table_chunks}</td><td>${r.latency.toFixed(2)}s</td>` +
      `<td class="detail">${(r.answer||'').slice(0,180)}…</td></tr>` +
      `<tr><td colspan="7" class="detail"><b>检索片段：</b>` +
      (r.retrieved||[]).map(d => `<span class="pill">《${d.doc}》p${d.page} ${d.type} ${d.score}</span>`).join('') +
      `</td></tr>`;
  }
  main.innerHTML = html + '</table></div>';
}

function renderEval() {
  const ev = DATA.eval;
  if (!ev) { main.innerHTML = '<div class="card">暂无 evaluation.json，请先运行 run_evaluation.py</div>'; return; }
  const s = ev.summary, rs = ev.records || [];
  let html = '<div class="card"><h3>评估指标汇总</h3><div class="grid">' +
    kpi(f3(s.faithfulness), '忠实度') + kpi(f3(s.answer_relevancy), '答案相关性') +
    kpi(f3(s.context_precision), '上下文精度') + kpi(f3(s.context_recall), '上下文召回') +
    kpi(f3(s.answer_correctness), '答案正确性') + kpi(f3(s.hit_rate), 'Hit Rate') +
    kpi(f3(s.mrr), 'MRR') + kpi(pc(ev.pass.pass_rate), '通过率') + '</div>' +
    `<div class="muted" style="margin-top:8px">评估器：${s.evaluator}；判定规则：${
      Object.entries(ev.pass_rules).map(([k,v])=>k+'='+v).join('，')}</div></div>`;
  html += '<div class="card"><h3>逐题评估</h3><table>' +
    '<tr><th>编号</th><th>题型</th><th>Faith.</th><th>Ans.Rel.</th><th>Ctx.Prec.</th>' +
    '<th>Ctx.Rec.</th><th>Ans.Corr.</th><th>关键词</th><th>判定</th></tr>';
  for (const r of rs) {
    const m = r.metrics || {}, k = r.keywords || {};
    html += `<tr><td>${r.id}</td><td>${r.question_type}</td>` +
      `<td>${f3(m.faithfulness)}</td><td>${f3(m.answer_relevancy)}</td>` +
      `<td>${f3(m.context_precision)}</td><td>${f3(m.context_recall)}</td>` +
      `<td>${f3(m.answer_correctness)}</td>` +
      `<td>${(k.命中||[]).length}/${(k.命中||[]).length + (k.漏答||[]).length}</td>` +
      `<td class="${r.passed ? 'ok' : 'bad'}">${r.passed ? '通过' : '未通过'}</td></tr>`;
    if (!r.passed) html += `<tr><td colspan="9" class="muted">未通过原因：${(r.fail_reasons||[]).join('；')}</td></tr>`;
  }
  main.innerHTML = html + '</table></div>';
}

async function renderProblems() {
  main.innerHTML = '<div class="card">加载问题分析…</div>';
  try {
    const txt = await jget('/api/problems', true);
    main.innerHTML = '<div class="card"><h3>检索结果问题分析</h3><pre>' +
      txt.replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c])) + '</pre></div>';
  } catch (e) {
    main.innerHTML = '<div class="card">' + e.message + '</div>';
  }
}

function renderLive() {
  main.innerHTML = `<div class="card"><h3>实时问答（01-06 RAG 系统）</h3>
    <div class="qbox"><input id="q" placeholder="输入问题，例如：平安银行2019年末拨备覆盖率是多少？">
    <button id="go">提问</button></div>
    <div id="out" style="margin-top:12px"></div></div>`;
  const q = document.getElementById('q'), out = document.getElementById('out');
  async function go() {
    if (!q.value.trim()) return;
    out.innerHTML = '<div class="muted">检索并生成中…</div>';
    try {
      const r = await fetch('/api/ask', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({question: q.value, top_k: 5})});
      const d = await r.json();
      if (d.detail) { out.innerHTML = '<div class="bad">' + d.detail + '</div>'; return; }
      out.innerHTML = `<div class="card"><b>答案</b><div class="detail">${
        (d.answer||'').replace(/\n/g,'<br>')}</div>
        <div class="muted" style="margin-top:8px">耗时 ${d.latency}s · 意图 ${
          (d.understanding||{})['意图']||'-'}</div>
        <div style="margin-top:8px"><b>检索片段</b><br>${
          (d.retrieved||[]).map(x=>`<span class="pill">《${x.doc}》p${x.page} ${x.type} ${x.score}</span>`).join('')}</div>
        </div>`;
    } catch (e) { out.innerHTML = '<div class="bad">请求失败：' + e + '</div>'; }
  }
  document.getElementById('go').onclick = go;
  q.addEventListener('keydown', e => { if (e.key === 'Enter') go(); });
  q.focus();
}

document.querySelectorAll('nav button').forEach(b =>
  b.onclick = () => render(b.dataset.tab));
loadAll();
</script>
</body>
</html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=f"评估看板服务（{WO_NO}）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8007)
    ap.add_argument("--reload", action="store_true", help="代码热重载（开发用）")
    args = ap.parse_args()

    import uvicorn
    print(f"===== {WO_NO} · 评估看板 =====")
    print(f"http://{args.host}:{args.port}  （Ctrl+C 退出）")
    if args.reload:
        uvicorn.run("serve:app", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
