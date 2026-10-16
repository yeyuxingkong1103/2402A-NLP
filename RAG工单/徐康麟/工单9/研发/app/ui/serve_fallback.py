# -*- coding: utf-8 -*-
"""工单9 Graph RAG 优化版 Web 界面（**纯标准库兜底实现**，零第三方依赖）。

工单编号：人工智能NLP-RAG-Graph RAG 优化任务

为什么用标准库：本机缺失包清单里没有 gradio/streamlit（`_共享约定.md` §2），
且纪律禁止裸 pip install。为保证「任何解释器都能起界面」，这里用 ``http.server`` 实现。

**默认端口 8125**（工单9 指定；工单8 用 8124、工单7 用 8123，互不冲突）。

页面必须展示的四块（工单要求「页面能展示 问题 / 检索到的上下文（含图谱三元组）/ 答案 / 引用」）：

    ① 问题输入（含中英文示例问题快捷按钮）
    ② **答案**（逐字来自证据，带 [块N] 引用；拒答时显示护栏文案）
    ③ **引用**（每条：公司 / 页码 / chunk_id）
    ④ **检索到的上下文** —— 每条显示块号、通道（base / base+graph / graph_only / page_expand /
       field_probe）、页码、融合分、图谱实体、正文；并单独一块显示 **【图谱三元组】**
       （head --关系--> tail，置信度 + 来源块）

HTTP 契约（与 设计/接口设计.md §2 对齐）：

    GET  /                 问答页
    GET  /healthz          健康检查（未就绪项如实说明原因）
    GET  /api/demos        示例问题清单
    GET  /api/stats        图谱规模 / 配置指纹 / 端口
    POST /api/ask (== /ask) 问答 → {ok, answer, refused, generation_method, latency_ms,
                                    contexts[], citations[], triples[], graph, diagnostics}

设计纪律：
    * 端口只从配置读（``--port`` / ``RAG_SERVER__OPT_PORT``），默认 **8125**；
    * 「不清楚 / 拒答」是**正确响应**（HTTP 200 + ``refused=true``），不是错误；
    * 任何异常都返回**结构化 JSON 错误**，绝不 500 裸崩；
    * 单进程 + ``ThreadingHTTPServer``（每请求一线程，I/O 型并发，符合资源纪律）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

WORK_ORDER = "人工智能NLP-RAG-Graph RAG 优化任务"

_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_ROOT / "研发"), str(_ROOT / "研发" / "评估")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_PORT = 8125
TITLE = "Graph RAG 金融问答（工单9 · 优化版）"

#: 示例问题（中英文各若干，直接对应工单「多语言支持」验收项）
DEMO_QUESTIONS: list[dict[str, str]] = [
    {"id": "Q07", "tag": "中文·数值多指标",
     "question": "平安银行2019年年度报告中，该行2019年实现营业收入和净利润分别是多少？同比增速分别是多少？年末不良贷款率和拨备覆盖率分别是多少？"},
    {"id": "Q02", "tag": "中文·定性枚举",
     "question": "招商银行2019年年度报告中提到的创新商业模式有哪些？"},
    {"id": "Q17", "tag": "中文·定性多举措",
     "question": "中国太保2021年年度报告中，公司如何通过“保险产品+养老社区+专业服务”的新业务模式与“太保服务”建设提升客户经营能力？"},
    {"id": "Q09", "tag": "中文·跨文档对比",
     "question": "对比平安银行与招商银行2019年年度报告：两家银行2019年营业收入分别是多少？同比增速分别是多少？"},
    {"id": "Q13", "tag": "英文·数值",
     "question": "According to Postal Savings Bank of China's 2019 Annual Report, what were the bank's net profit and its year-on-year growth rate?"},
    {"id": "en2", "tag": "英文·数值",
     "question": "What were China Pacific Insurance Group's total revenue and net profit in 2021?"},
]

_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TITLE__</title>
<style>
  :root { --bg:#f5f7fb; --card:#fff; --line:#e2e6f0; --ink:#1b2333; --muted:#6b7386;
          --accent:#2f6fed; --chip:#eef3ff; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif;
         background:var(--bg); color:var(--ink); }
  header { background:linear-gradient(90deg,#2f6fed,#6f9cf7); color:#fff; padding:16px 22px; }
  header h1 { margin:0; font-size:19px; }
  header p { margin:6px 0 0; font-size:12.5px; opacity:.93; }
  main { max-width:1120px; margin:16px auto 70px; padding:0 14px; display:grid; gap:12px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px;
          box-shadow:0 1px 3px rgba(20,30,60,.05); }
  .card h2 { margin:0 0 10px; font-size:14px; color:var(--muted); font-weight:700; letter-spacing:.4px; }
  textarea { width:100%; min-height:70px; border:1px solid var(--line); border-radius:8px;
             padding:10px; font-size:15px; font-family:inherit; resize:vertical; }
  .row { display:flex; gap:9px; align-items:center; flex-wrap:wrap; margin-top:9px; }
  button { border:0; border-radius:8px; padding:9px 16px; font-size:14px; cursor:pointer;
           background:var(--accent); color:#fff; }
  button.ghost { background:#eef2fb; color:#274a9a; }
  button:disabled { opacity:.55; cursor:not-allowed; }
  .chip { display:inline-block; background:var(--chip); color:#274a9a; border-radius:20px;
          padding:2px 10px; font-size:12px; margin:0 5px 5px 0; }
  .q-btn { background:#f3f6fc; color:#2b3a5c; font-size:12.5px; padding:6px 10px; border:1px solid var(--line); }
  .answer { white-space:pre-wrap; line-height:1.75; font-size:15px; }
  .ctx { border:1px solid var(--line); border-radius:8px; padding:9px 11px; margin-bottom:8px;
         background:#fcfdff; }
  .ctx .meta { font-size:12px; color:var(--muted); margin-bottom:5px; }
  .ctx .text { font-size:13px; line-height:1.6; white-space:pre-wrap; color:#2a3348; }
  .kv { font-size:12.5px; color:var(--muted); }
  .badge { display:inline-block; border-radius:5px; padding:1px 7px; font-size:11.5px;
           background:#eef2fb; color:#274a9a; margin-right:5px; }
  .badge.g { background:#e8f7ee; color:#1f7a45; }
  .badge.p { background:#fff4e5; color:#a3620a; }
  .badge.e { background:#fdecea; color:#b3261e; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th,td { border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }
  th { color:var(--muted); font-weight:600; font-size:12px; }
  .muted { color:var(--muted); font-size:13px; }
  .spin { display:inline-block; width:13px; height:13px; border:2px solid #cfd8ea;
          border-top-color:var(--accent); border-radius:50%; animation:sp .8s linear infinite; }
  @keyframes sp { to { transform:rotate(360deg); } }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <p>三层面优化（实体/关系构建层 · 检索层 · 生成层）｜端口 8125｜支持中文与英文提问</p>
</header>
<main>
  <section class="card">
    <h2>① 问题</h2>
    <textarea id="q" placeholder="请输入中文或英文问题，例如：平安银行2019年营业收入和净利润分别是多少？"></textarea>
    <div class="row">
      <button id="go">提问</button>
      <label class="kv"><input type="checkbox" id="usellm" checked /> 允许 LLM 润色（不勾选则只走模板快路径）</label>
      <span id="status" class="kv"></span>
    </div>
    <div class="row" id="demos"></div>
  </section>

  <section class="card">
    <h2>② 答案</h2>
    <div id="answer" class="answer muted">尚未提问。</div>
    <div class="row" id="metrics"></div>
  </section>

  <section class="card">
    <h2>③ 引用</h2>
    <div id="cites" class="muted">—</div>
  </section>

  <section class="card">
    <h2>④ 检索到的上下文（含图谱三元组）</h2>
    <div id="triples" class="muted">—</div>
    <div id="ctx" class="muted" style="margin-top:10px">—</div>
  </section>

  <section class="card">
    <h2>⑤ 图谱结构（问句命中的局部子图）</h2>
    <div id="graph" class="muted">—</div>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

async function ask(q) {
  q = (q || $('q').value || '').trim();
  $('status').innerHTML = '<span class="spin"></span> 检索中…';
  $('go').disabled = true;
  try {
    const r = await fetch('/api/ask', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q, use_llm: $('usellm').checked})
    });
    render(await r.json());
  } catch (e) {
    $('answer').innerHTML = '<span class="badge e">请求失败</span> ' + esc(e);
  } finally {
    $('go').disabled = false;
    $('status').textContent = '';
  }
}

function render(d) {
  if (!d.ok) {
    $('answer').innerHTML = '<span class="badge e">错误</span> ' + esc(d.error || '未知错误') +
                            '\n' + esc(d.answer || '');
    $('metrics').innerHTML = ''; $('cites').innerHTML = '—';
    $('ctx').innerHTML = '—'; $('triples').innerHTML = '—'; $('graph').innerHTML = '—';
    return;
  }
  $('answer').textContent = d.answer || '';
  if (d.refused) { $('answer').innerHTML = '<span class="badge p">拒答护栏</span>\n' + esc(d.answer); }
  const L = d.latency_ms || {};
  $('metrics').innerHTML = [
    `<span class="badge ${d.refused ? 'p' : 'g'}">${esc(d.generation_method)}</span>`,
    `<span class="badge">总耗时 ${(+L.total || 0).toFixed(0)} ms</span>`,
    `<span class="badge">检索 ${(+L.retrieve || 0).toFixed(0)} ms</span>`,
    `<span class="badge">图谱 ${(+L.graph || 0).toFixed(0)} ms</span>`,
    `<span class="badge">生成 ${(+L.generate || 0).toFixed(0)} ms</span>`,
    `<span class="badge">语言 ${esc(d.language)}</span>`
  ].join('');
  const cites = d.citations || [];
  $('cites').innerHTML = cites.length ? cites.map((c, i) =>
    `<div class="kv">[块${i + 1}] <b>${esc(c.company)}</b> 第 ${c.page} 页 · ` +
    `<span class="muted">${esc(c.chunk_id)}</span></div>`).join('') : '—';

  const tr = d.triples || [];
  $('triples').innerHTML = tr.length
    ? '<div class="kv" style="margin-bottom:6px"><b>【知识图谱三元组】</b>（head --关系--&gt; tail，置信度；来源块）</div>' +
      tr.map(t => `<div class="kv">${esc(t.head)} <span class="badge g">${esc(t.relation)}</span>&rarr; ` +
                  `${esc(t.tail)} <span class="muted">（${(+t.confidence || 0).toFixed(2)}；${esc(t.chunk_id)}）</span></div>`).join('')
    : '<div class="kv">本次检索未命中图谱三元组。</div>';

  const ctx = d.contexts || [];
  $('ctx').innerHTML = ctx.map(c => {
    const ch = esc(c.retrieval_channel || 'base');
    const cls = ch === 'graph_only' ? 'g' : (ch === 'field_probe' ? 'p' : '');
    const ents = (c.graph_entities || []).slice(0, 5).map(e => `<span class="chip">${esc(e)}</span>`).join('');
    const fn = String(c.file_name || '');
    const company = fn.split('__').length >= 4 ? fn.split('__')[3] : fn;
    return `<div class="ctx"><div class="meta">` +
      `<span class="badge">#${c.rank}</span>` +
      `<span class="badge ${cls}">${ch}</span>` +
      `<span class="badge">第 ${c.page} 页</span>` +
      `<span class="badge">${esc(c.type)}</span>` +
      `<span class="kv">融合分 ${(+c.score || 0).toFixed(4)} · ${esc(company)}</span>` +
      `</div>${ents ? '<div style="margin-bottom:5px">' + ents + '</div>' : ''}` +
      `<div class="text">${esc((c.text || '').slice(0, 700))}</div></div>`;
  }).join('') || '<div class="kv">无上下文。</div>';

  const g = d.graph || {};
  const seeds = (g.seed_names || []).map(s => `<span class="chip">${esc(s)}</span>`).join('');
  $('graph').innerHTML =
    `<div class="kv">命中实体：${seeds || '（无）'}</div>` +
    `<div class="kv">子图：${(g.nodes || []).length} 节点 / ${(g.edges || []).length} 边</div>` +
    ((g.edges || []).length ? `<table><tr><th>head</th><th>关系</th><th>tail</th><th>置信度</th></tr>` +
      g.edges.slice(0, 12).map(e => `<tr><td>${esc(e.head)}</td><td>${esc(e.relation)}</td>` +
        `<td>${esc(e.tail)}</td><td>${(+e.confidence || 0).toFixed(2)}</td></tr>`).join('') + '</table>' : '');
}

async function boot() {
  const box = $('demos');
  try {
    const d = await (await fetch('/api/demos')).json();
    box.innerHTML = (d.questions || []).map(q =>
      `<button class="ghost q-btn" data-q="${esc(q.question)}">${esc(q.tag)}</button>`).join('');
    box.querySelectorAll('button').forEach(b => b.onclick = () => {
      $('q').value = b.getAttribute('data-q'); ask();
    });
  } catch (e) { box.innerHTML = ''; }
  try {
    const s = await (await fetch('/api/stats')).json();
    $('status').textContent = `图谱 ${s.graph_nodes} 节点 / ${s.graph_edges} 边 · 索引 ${s.index_chunks} 块`;
  } catch (e) {}
}
$('go').onclick = () => ask();
$('q').addEventListener('keydown', e => { if (e.ctrlKey && e.key === 'Enter') ask(); });
boot();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 引擎单例（进程内复用，避免每次请求重新加载索引/图谱）
# ---------------------------------------------------------------------------
class _EngineHolder:
    """引擎懒加载单例（首请求加载，之后复用）。"""

    def __init__(self) -> None:
        self._engine: Any = None
        self._lock = threading.Lock()
        self._error: str | None = None

    def get(self) -> Any:
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is None:
                from app.graphrag_opt.配置 import get_config
                from app.graphrag_opt.引擎 import OptimizedEngine

                try:
                    cfg = get_config()
                    os.environ.setdefault("RAG_LOG__DIR", str(cfg.log_dir))
                    self._engine = OptimizedEngine(cfg=cfg, profile="after", logger=_safe_logger())
                except Exception as exc:  # noqa: BLE001 —— 引擎构造失败必须留痕，不静默
                    self._error = f"{type(exc).__name__}: {exc}"
                    raise
        return self._engine

    @property
    def error(self) -> str | None:
        return self._error


def _safe_logger() -> Any:
    try:
        from app.core.logging_conf import setup_logging

        return setup_logging()
    except Exception:  # noqa: BLE001
        class _Null:
            def log_event(self, *a: Any, **k: Any) -> None:
                return None

            def bind(self, **k: Any) -> "_Null":
                return self

            def enter(self, *a: Any, **k: Any) -> "_Null":
                return self

        return _Null()


HOLDER = _EngineHolder()


def build_response(question: str, *, top_k: int = 10, use_llm: bool = True,
                   language: str = "auto") -> dict[str, Any]:
    """跑一次问答并组装**页面所需的完整结构**（答案 / 引用 / 上下文 / 三元组 / 图谱 / 计时）。"""
    text = str(question or "").strip()
    if not text:
        return {"ok": False, "refused": True, "error": "问题不能为空",
                "answer": "请输入一个具体的问题（支持中文或英文）。",
                "latency_ms": {"total": 0.0}, "contexts": [], "citations": [], "triples": [],
                "graph": {}, "generation_method": "input_guard", "language": "zh"}
    started = time.perf_counter()
    try:
        engine = HOLDER.get()
    except Exception as exc:  # noqa: BLE001 —— 引擎起不来要返回结构化错误（不 500）
        return {"ok": False, "refused": True, "error": f"引擎初始化失败：{type(exc).__name__}: {exc}",
                "answer": "系统当前不可用（引擎初始化失败），请查看 部署/日志/app.log。",
                "latency_ms": {"total": round((time.perf_counter() - started) * 1000, 2)},
                "contexts": [], "citations": [], "triples": [], "graph": {},
                "generation_method": "engine_unavailable", "language": "zh"}
    try:
        ans = engine.answer(text, top_k=int(top_k), use_llm=bool(use_llm), language=language)
    except Exception as exc:  # noqa: BLE001 —— 单次问答异常也必须结构化返回
        return {"ok": False, "refused": True, "error": f"{type(exc).__name__}: {exc}",
                "answer": "本次问答出现异常，已记录日志。请重试或换一个问法。",
                "latency_ms": {"total": round((time.perf_counter() - started) * 1000, 2)},
                "contexts": [], "citations": [], "triples": [], "graph": {},
                "generation_method": "error", "language": "zh"}

    diag = ans.diagnostics or {}
    contexts = []
    for idx, item in enumerate(ans.evidence_chunks, 1):
        contexts.append({
            "rank": idx, "chunk_id": item.get("chunk_id"), "page": item.get("page"),
            "file_name": item.get("file_name"), "type": item.get("type"),
            "retrieval_channel": item.get("retrieval_channel"),
            "score": item.get("score"), "graph_entities": item.get("graph_entities") or [],
            "graph_distance": item.get("graph_distance"), "text": item.get("text") or "",
        })
    citations = [{"chunk_id": c["chunk_id"], "page": c["page"],
                  "company": (str(c["file_name"]).split("__")[3] if str(c["file_name"]).count("__") >= 3
                              else str(c["file_name"])[:18])} for c in contexts[:6]]
    triples: list[dict[str, Any]] = []
    for edge in ((ans.graph or {}).get("edges") or [])[:12]:
        triples.append({"head": edge.get("head"), "relation": edge.get("relation"),
                        "tail": edge.get("tail"), "confidence": edge.get("confidence"),
                        "chunk_id": edge.get("chunk_id") or ""})
    return {
        "ok": True, "question": text, "answer": ans.answer, "refused": bool(ans.refused),
        "language": ans.language, "generation_method": ans.generation_method,
        "backend": ans.backend, "model": ans.model, "latency_ms": ans.latency_ms,
        "contexts": contexts, "citations": citations, "triples": triples,
        "graph": {"seed_names": (ans.graph or {}).get("seed_names") or [],
                  "nodes": (ans.graph or {}).get("nodes") or [],
                  "edges": (ans.graph or {}).get("edges") or []},
        "diagnostics": {k: v for k, v in diag.items() if k != "linked_entities"},
    }


class _Handler(BaseHTTPRequestHandler):
    server_version = "GraphRAGOpt/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 —— 静默默认访问日志
        return

    # -- 输出工具 ---------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):  # 客户端提前断开是正常现象
            pass

    def _json(self, payload: dict[str, Any], code: int = 200) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    # -- 路由 -------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, _PAGE_HTML.replace("__TITLE__", TITLE).encode("utf-8"),
                       "text/html; charset=utf-8")
            return
        if path in ("/healthz", "/health"):
            try:
                self._json(HOLDER.get().health())
            except Exception as exc:  # noqa: BLE001
                self._json({"status": "degraded", "error": f"{type(exc).__name__}: {exc}"})
            return
        if path == "/api/demos":
            self._json({"questions": DEMO_QUESTIONS})
            return
        if path == "/api/stats":
            try:
                engine = HOLDER.get()
                self._json({"graph_nodes": engine.graph.node_count,
                            "graph_edges": engine.graph.edge_count,
                            "index_chunks": (engine.health().get("checks", {})
                                             .get("index", {}).get("chunks")),
                            "config_fingerprint": engine.cfg.fingerprint(),
                            "port": getattr(self.server, "server_port", DEFAULT_PORT),
                            "work_order": WORK_ORDER})
            except Exception as exc:  # noqa: BLE001
                self._json({"error": f"{type(exc).__name__}: {exc}"})
            return
        self._json({"error": "not found", "path": path}, code=404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in ("/api/ask", "/ask"):
            self._json({"error": "not found", "path": path}, code=404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8", errors="replace") or "{}")
            question = str(payload.get("question") or payload.get("q") or "")
            top_k = int(payload.get("top_k") or 10)
            use_llm = bool(payload.get("use_llm", True))
            language = str(payload.get("language") or "auto")
        except Exception as exc:  # noqa: BLE001 —— 请求体坏了也要返回结构化错误
            self._json({"ok": False, "error": f"请求体解析失败：{type(exc).__name__}: {exc}",
                        "answer": "请求格式不正确（应为 JSON：{\"question\": \"…\"}）。",
                        "refused": True, "contexts": [], "citations": [], "triples": [],
                        "graph": {}, "latency_ms": {"total": 0.0}})
            return
        started = time.perf_counter()
        result = build_response(question, top_k=top_k, use_llm=use_llm, language=language)
        result["server_ms"] = round((time.perf_counter() - started) * 1000, 2)
        self._json(result)


def build_server(port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """构造 HTTP 服务（供 ``main()``、``serve()`` 与单元测试共用）。

    工单编号：人工智能NLP-RAG-Graph RAG 优化任务

    与工单8 的差异：工单9 用模块级 ``HOLDER`` 惰性持有引擎，因此**不需要**外部传入
    ``engine``；这里与 ``serve()`` 共用同一条装配路径，避免两处各写一套 Handler 绑定。

    修复记录（2026-10-13）：``app/ui/__init__.py`` 一直 `from .serve_fallback import
    build_server`，但本模块此前**并未定义该函数** → `import app.ui` 必然 ImportError。
    直接以脚本方式运行 ``serve_fallback.py`` 不会触发父包 ``__init__``，所以该缺陷被掩盖
    了很久（界面能起，但作为包导入即崩）。现补齐。
    """
    httpd = ThreadingHTTPServer((host, int(port)), _Handler)
    httpd.daemon_threads = True
    return httpd


def serve(port: int = DEFAULT_PORT, host: str = "127.0.0.1", *, warmup: bool = True) -> int:
    """启动 Web 界面。``warmup=True`` 时先加载引擎（首次请求不会卡住）。"""
    if warmup:
        try:
            print("[启动] 正在预热引擎（加载索引与图谱）…")
            t0 = time.perf_counter()
            HOLDER.get().warmup(with_llm=False)
            print(f"[启动] 预热完成，耗时 {round((time.perf_counter() - t0), 2)} s")
        except Exception as exc:  # noqa: BLE001 —— 预热失败不阻止启动（页面会显示降级原因）
            print(f"[启动] 预热失败（{type(exc).__name__}: {exc}），界面仍会启动并如实报错")
    httpd = build_server(port, host)
    print(f"[启动] {TITLE}")
    print(f"[启动] 请访问 http://{host}:{int(port)}/  （Ctrl+C 停止）")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[停止] 收到中断信号，正在关闭…")
    finally:
        httpd.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{WORK_ORDER} —— Web 界面（端口 {DEFAULT_PORT}）")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("RAG_SERVER__OPT_PORT") or DEFAULT_PORT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-warmup", action="store_true", help="跳过预热（首请求会慢一些）")
    args = parser.parse_args(argv)
    return serve(port=int(args.port), host=str(args.host), warmup=not args.no_warmup)


if __name__ == "__main__":
    raise SystemExit(main())
