#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web 问答服务（serve.py）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

功能
    1. 单模式问答：优化后（工单02：structure + 混合检索 + 级联重排 + 优化 Prompt）
       / 优化前（工单01 基线：fixed + 纯向量 + 不重排 + 朴素 Prompt）；
    2. 对比模式：同一个问题，左右并排返回「优化前答案 vs 优化后答案」，
       并附检索页码、耗时与要点命中核对 —— 直接支撑工单演示环节
       「针对问题进行检索，对比优化前的检索答案」；
    3. 健康检查 / 指标 / 预置问题接口，便于演示与长时间运行的稳定性观察；
    4. 接口层容错：任何异常都转成结构化错误返回，服务不中断。

启动
    python src/serve.py                     # 默认 0.0.0.0:8000
    python src/serve.py --port 8080 --no-build   # 跳过索引构建（索引已存在时）

接口
    GET  /                 对比问答界面（内置 HTML，零前端构建）
    GET  /api/health       健康检查（含向量库条数、索引状态）
    GET  /api/metrics      运行时指标（请求数、错误数、延迟分位）
    GET  /api/questions    10 个预置问题（演示用）
    POST /api/ask          单模式问答 {question, mode, top_k}
    POST /api/compare      优化前后对比 {question, top_k}
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (CHUNK_OVERLAP, CHUNK_SIZE, QUESTIONS, TOP_K,   # noqa: E402
                    WO_ID, QAConfig, answer_question, build_index, check_answer,
                    get_retriever, require_llm_key)

# 两种模式：优化前 = 工单01 基线；优化后 = 工单02 三优化叠加
MODES: dict[str, QAConfig] = {
    "baseline": QAConfig(name="优化前（工单01 基线）", collection="wo02_fixed",
                         strategy="vector", reranker="none", prompt="naive"),
    "optimized": QAConfig(name="优化后（工单02）", collection="wo02_structure",
                          strategy="hybrid", fusion="rrf", reranker="cascade",
                          prompt="optimized"),
}

_METRICS = {"requests": 0, "errors": 0, "latencies": [], "started_at": time.time()}
_RETRIEVERS: dict[str, object] = {}


def get_engine(mode: str):
    """惰性装载检索器（首次请求或启动预热时构建），避免重复加载索引。"""
    if mode not in MODES:
        raise KeyError(f"未知模式：{mode}，可选 {list(MODES)}")
    if mode not in _RETRIEVERS:
        cfg = MODES[mode]
        _RETRIEVERS[mode] = get_retriever(cfg.collection, cfg.reranker, verbose=False)
    return _RETRIEVERS[mode]


def ask_once(mode: str, question: str, top_k: int = TOP_K) -> dict:
    """执行一次问答并做要点核对（判定不调用 LLM，纯关键词比对）。"""
    cfg = MODES[mode]
    retriever = get_engine(mode)
    res = answer_question(cfg, question, retriever=retriever)
    out = {
        "mode": mode,
        "mode_label": cfg.name,
        "question": question,
        "answer": res["answer"],
        "retrieval_seconds": res.get("retrieval_seconds", 0.0),
        "rerank_seconds": res.get("timings", {}).get("rerank", 0.0),
        "generate_seconds": res.get("generate_seconds", 0.0),
        "total_seconds": res.get("total_seconds", 0.0),
        "pages": [d.get("page") for d in res["docs"][:top_k]],
        "sources": [
            {"page": d.get("page"), "section": d.get("section", ""),
             "type": d.get("type", "text"),
             "snippet": (d.get("text") or "")[:120]}
            for d in res["docs"][:top_k]
        ],
        "error": res.get("error", ""),
    }
    # 只有命中预置 10 问之一才做要点核对（自由提问没有标准要点）
    qid = next((q["id"] for q in QUESTIONS if q["question"] == question), None)
    out["check"] = check_answer(res["answer"], qid) if qid else None
    return out


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
def create_app():
    """构造 FastAPI 应用；未安装 fastapi 时给出安装提示。"""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
        from pydantic import BaseModel, Field
    except ImportError as e:                                  # pragma: no cover
        raise SystemExit(f"缺少 Web 依赖（{e}），请先执行：pip install fastapi uvicorn")

    app = FastAPI(title="招股说明书智能问答系统（工单02 优化版）",
                  description="基于 RAG 的金融文档问答服务 · " + WO_ID,
                  version="2.0.0")

    class AskRequest(BaseModel):
        question: str = Field(..., description="用户问题（中英文均可）")
        mode: str = Field("optimized", description="optimized | baseline")
        top_k: int = Field(TOP_K, ge=1, le=20)

    class CompareRequest(BaseModel):
        question: str = Field(..., description="用户问题")
        top_k: int = Field(TOP_K, ge=1, le=20)

    @app.get("/api/health")
    def health():
        """健康检查：演示长时间运行时用于确认服务与索引状态。"""
        info = {"status": "ok", "wo_id": WO_ID,
                "uptime_seconds": round(time.time() - _METRICS["started_at"], 1)}
        for mode, cfg in MODES.items():
            try:
                vs = get_engine(mode)
                info[f"{mode}_vectors"] = vs.vs.count() if hasattr(vs, "vs") else -1
            except Exception as e:                            # 容错：单库异常不影响健康检查
                info[f"{mode}_error"] = str(e)
        return info

    @app.get("/api/metrics")
    def metrics():
        lat = sorted(_METRICS["latencies"])

        def _pct(q):
            return round(lat[min(int(len(lat) * q), len(lat) - 1)], 3) if lat else 0.0

        return {
            "requests_total": _METRICS["requests"],
            "errors_total": _METRICS["errors"],
            "uptime_seconds": round(time.time() - _METRICS["started_at"], 1),
            "latency_avg_s": round(sum(lat) / len(lat), 3) if lat else 0.0,
            "latency_p50_s": _pct(0.50), "latency_p95_s": _pct(0.95),
            "latency_max_s": round(lat[-1], 3) if lat else 0.0,
            "samples": len(lat),
        }

    @app.get("/api/questions")
    def questions():
        """演示用预置问题（工单指定的 10 问）。"""
        return {"questions": QUESTIONS}

    @app.post("/api/ask")
    def ask(req: AskRequest):
        _METRICS["requests"] += 1
        t0 = time.perf_counter()
        try:
            return ask_once(req.mode, req.question, req.top_k)
        except Exception as e:                                # 容错：结构化错误返回
            _METRICS["errors"] += 1
            raise HTTPException(status_code=500, detail=f"问答失败：{e}")
        finally:
            _METRICS["latencies"].append(time.perf_counter() - t0)
            if len(_METRICS["latencies"]) > 1000:
                _METRICS["latencies"] = _METRICS["latencies"][-1000:]

    @app.post("/api/compare")
    def compare(req: CompareRequest):
        """优化前后对比：同一问题跑两套配置，并排返回。"""
        _METRICS["requests"] += 1
        t0 = time.perf_counter()
        try:
            return {
                "question": req.question,
                "baseline": ask_once("baseline", req.question, req.top_k),
                "optimized": ask_once("optimized", req.question, req.top_k),
            }
        except Exception as e:
            _METRICS["errors"] += 1
            raise HTTPException(status_code=500, detail=f"对比失败：{e}")
        finally:
            _METRICS["latencies"].append(time.perf_counter() - t0)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(_INDEX_HTML)

    return app


_INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>招股说明书智能问答系统 · 工单02 优化版</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
         background: #f5f7fa; color: #1f2937; height: 100vh; display: flex; flex-direction: column; }
  header { background: #1e3a5f; color: #fff; padding: 12px 22px; display: flex;
           align-items: center; justify-content: space-between; }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  header .meta { font-size: 12px; opacity: .8; }
  .toolbar { padding: 10px 22px; background: #fff; border-bottom: 1px solid #e5e7eb;
             display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .toolbar button { border: 1px solid #d1d5db; background: #fff; border-radius: 6px;
                    padding: 5px 12px; font-size: 13px; cursor: pointer; }
  .toolbar button.on { background: #2563eb; color: #fff; border-color: #2563eb; }
  #chat { flex: 1; overflow-y: auto; padding: 18px 22px; }
  .msg { max-width: 1100px; margin: 0 auto 16px; display: flex; gap: 10px; }
  .msg.user { flex-direction: row-reverse; }
  .avatar { width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0; color: #fff;
            display: flex; align-items: center; justify-content: center; font-size: 12px; }
  .user .avatar { background: #2563eb; } .bot .avatar { background: #059669; }
  .bubble { background: #fff; border-radius: 10px; padding: 12px 15px; flex: 1;
            box-shadow: 0 1px 3px rgba(0,0,0,.08); line-height: 1.7; font-size: 14px;
            white-space: pre-wrap; }
  .user .bubble { background: #dbeafe; }
  .cmp { display: flex; gap: 12px; max-width: 1100px; margin: 0 auto 16px; }
  .col { flex: 1; background: #fff; border-radius: 10px; padding: 12px 15px;
         box-shadow: 0 1px 3px rgba(0,0,0,.08); font-size: 14px; line-height: 1.7;
         white-space: pre-wrap; }
  .col h3 { margin: 0 0 8px; font-size: 13px; padding-bottom: 6px; border-bottom: 1px solid #e5e7eb; }
  .col.old h3 { color: #b45309; } .col.new h3 { color: #047857; }
  .kpis { font-size: 12px; color: #6b7280; margin-top: 8px; padding-top: 6px;
          border-top: 1px dashed #e5e7eb; }
  .cite { display: inline-block; background: #f3f4f6; border-radius: 4px;
          padding: 1px 6px; margin: 2px 4px 0 0; font-size: 12px; color: #4b5563; }
  .ok { color: #047857; } .bad { color: #b91c1c; }
  footer { padding: 12px 22px; background: #fff; border-top: 1px solid #e5e7eb; }
  .inputbar { max-width: 1100px; margin: 0 auto; display: flex; gap: 10px; }
  input[type=text] { flex: 1; padding: 10px 14px; border: 1px solid #d1d5db;
                     border-radius: 8px; font-size: 14px; outline: none; }
  button.send { background: #2563eb; color: #fff; border: none; border-radius: 8px;
                padding: 10px 22px; font-size: 14px; cursor: pointer; }
  .hint { max-width: 1100px; margin: 6px auto 0; font-size: 12px; color: #9ca3af; }
  .preset { padding: 8px 22px 0; max-width: 1100px; margin: 0 auto; }
  .preset span { display: inline-block; background: #eef2ff; color: #3730a3; font-size: 12px;
                 border-radius: 12px; padding: 3px 10px; margin: 0 6px 6px 0; cursor: pointer; }
</style>
</head>
<body>
<header>
  <h1>招股说明书智能问答系统 · 工单02 检索优化版</h1>
  <span class="meta" id="meta">加载中…</span>
</header>
<div class="toolbar">
  <span style="font-size:13px;color:#6b7280">模式：</span>
  <button id="m-compare" class="on">对比（优化前 vs 优化后）</button>
  <button id="m-optimized">仅优化后</button>
  <button id="m-baseline">仅优化前</button>
  <span style="font-size:12px;color:#9ca3af">对比模式直接演示工单验收项「对比优化前的检索答案」</span>
</div>
<div class="preset" id="preset"></div>
<div id="chat"></div>
<footer>
  <div class="inputbar">
    <input type="text" id="q" placeholder="请输入问题（支持中文/英文），回车发送…" autocomplete="off">
    <button class="send" id="send">发送</button>
  </div>
  <div class="hint">答案均标注来源页码 · 对比模式同时展示优化前后两版答案 · Enter 发送</div>
</footer>
<script>
const chat = document.getElementById('chat');
const q = document.getElementById('q');
const send = document.getElementById('send');
let MODE = 'compare';

fetch('/api/health').then(r => r.json()).then(d => {
  document.getElementById('meta').textContent =
    `向量库 基线${d.baseline_vectors ?? '-'} / 优化${d.optimized_vectors ?? '-'} 条 · 运行 ${d.uptime_seconds}s`;
}).catch(() => document.getElementById('meta').textContent = '服务未就绪');

fetch('/api/questions').then(r => r.json()).then(d => {
  const box = document.getElementById('preset');
  d.questions.forEach(x => {
    const s = document.createElement('span');
    s.textContent = 'id' + x.id + ' · ' + x.question.slice(0, 16) + '…';
    s.title = x.question;
    s.onclick = () => { q.value = x.question; ask(); };
    box.appendChild(s);
  });
}).catch(() => {});

document.getElementById('m-compare').onclick = () => setMode('compare');
document.getElementById('m-optimized').onclick = () => setMode('optimized');
document.getElementById('m-baseline').onclick = () => setMode('baseline');
function setMode(m) {
  MODE = m;
  for (const k of ['compare', 'optimized', 'baseline'])
    document.getElementById('m-' + k).classList.toggle('on', k === m);
}

function addUser(text) {
  const w = document.createElement('div');
  w.className = 'msg user';
  w.innerHTML = '<div class="avatar">我</div><div class="bubble"></div>';
  w.querySelector('.bubble').textContent = text;
  chat.appendChild(w); chat.scrollTop = chat.scrollHeight;
}
function kpis(r) {
  const c = r.check;
  let s = `耗时 ${(r.total_seconds || 0).toFixed(2)}s（检索 ${(r.retrieval_seconds*1000|0)}ms`
        + ` / 重排 ${(r.rerank_seconds*1000|0)}ms / 生成 ${(r.generate_seconds*1000|0)}ms）`;
  s += ` · 检索页码 ${(r.pages || []).join(',') || '-'}`;
  if (c) s += ` · 要点核对 ${c['正确'] ? '<span class="ok">√ 命中</span>' : '<span class="bad">× 漏答 ' + (c['漏答']||[]).join('、') + '</span>'}`;
  return s;
}
function renderOne(r) {
  const w = document.createElement('div');
  w.className = 'msg bot';
  w.innerHTML = '<div class="avatar">AI</div><div class="bubble"></div>';
  const b = w.querySelector('.bubble');
  b.textContent = r.answer || '（无答案）';
  const k = document.createElement('div');
  k.className = 'kpis'; k.innerHTML = kpis(r);
  b.appendChild(k);
  chat.appendChild(w); chat.scrollTop = chat.scrollHeight;
}
function renderCompare(d) {
  const wrap = document.createElement('div');
  wrap.className = 'cmp';
  for (const [key, cls] of [['baseline', 'old'], ['optimized', 'new']]) {
    const r = d[key];
    const col = document.createElement('div');
    col.className = 'col ' + cls;
    col.innerHTML = `<h3>${r.mode_label}</h3>`;
    const body = document.createElement('div');
    body.textContent = r.answer || '（无答案）';
    col.appendChild(body);
    const k = document.createElement('div');
    k.className = 'kpis'; k.innerHTML = kpis(r);
    col.appendChild(k);
    wrap.appendChild(col);
  }
  chat.appendChild(wrap); chat.scrollTop = chat.scrollHeight;
}

async function ask() {
  const text = q.value.trim();
  if (!text) return;
  q.value = ''; send.disabled = true;
  addUser(text);
  const loading = document.createElement('div');
  loading.className = 'msg bot';
  loading.innerHTML = '<div class="avatar">AI</div><div class="bubble">正在检索文档并生成答案…</div>';
  chat.appendChild(loading); chat.scrollTop = chat.scrollHeight;
  try {
    const url = MODE === 'compare' ? '/api/compare' : '/api/ask';
    const body = MODE === 'compare' ? {question: text} : {question: text, mode: MODE};
    const r = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify(body)});
    const d = await r.json();
    loading.remove();
    if (d.detail) { renderOne({answer: '出错了：' + d.detail}); }
    else if (MODE === 'compare') renderCompare(d);
    else renderOne(d);
  } catch (e) {
    loading.remove();
    renderOne({answer: '请求失败：' + e});
  }
  send.disabled = false; q.focus();
}
send.onclick = ask;
q.addEventListener('keydown', e => { if (e.key === 'Enter') ask(); });
q.focus();
</script>
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="工单02 Web 问答服务")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-build", action="store_true",
                    help="跳过索引构建（索引已存在时加快启动）")
    ap.add_argument("--force-build", action="store_true", help="强制重建两套索引")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 72)
    print(f"  工单02 Web 问答服务 | {WO_ID}")
    print("=" * 72)
    require_llm_key()

    if not args.no_build:
        print("[1] 准备索引…")
        build_index(MODES["baseline"].collection, "fixed",
                    size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, force=args.force_build)
        build_index(MODES["optimized"].collection, "structure",
                    size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, force=args.force_build)

    print("[2] 预热检索器…")
    for mode in MODES:
        try:
            get_engine(mode)
            print(f"    {mode} 就绪")
        except Exception as e:                     # 容错：单模式失败不影响另一模式
            print(f"    [warn] {mode} 装载失败：{e}")

    app = create_app()
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    print(f"[3] 服务启动：{url}    （Ctrl+C 停止）")
    print("    对比模式即为工单演示要求的「优化前后检索答案对比」")
    try:
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except OSError as e:                           # 容错：端口占用给出可操作提示
        print(f"[error] 端口 {args.port} 启动失败：{e}\n"
              f"        可换端口重试：python src/serve.py --port {args.port + 1}")
        return 1
    except KeyboardInterrupt:
        print("\n服务已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
