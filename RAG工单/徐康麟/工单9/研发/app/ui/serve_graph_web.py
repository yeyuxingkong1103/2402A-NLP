# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 说明：本文件复用自工单8（工单9 未修改其业务逻辑），仅补齐工单9 的编号头注释。
"""工单8 Web 问答界面（纯标准库，端口 8124）：问题 / 图谱结构 / 检索结果 / 答案。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

为什么纯标准库（设计取舍，见 ``设计/技术选型.md``）：
    本机**未安装** gradio / streamlit，依赖红线又不允许裸 ``pip install``；
    而本界面只需表单提交 + JSON 返回，``http.server.ThreadingHTTPServer`` 完全够用，
    且**零外网依赖**（页面不引用任何 CDN／字体／图片）。

界面（一屏四段，对应工单原文「输出检索结果、以及解析出来的知识图谱结构」）：
    ① 问题输入（含中英文示例一键填充）
    ② 图谱结构：命中实体 + 一跳/两跳邻居 + 途经关系（JSON 缩略 + 可视化标签云）
    ③ 检索结果：Top-K 命中块（chunk_id / 页码 / 文件名 / 通道 / 分数 / 文本摘要）
    ④ 答案：Ollama qwen2.5:3b 依据证据块生成，标注 [块N]

端点：
    GET  /                 界面
    POST /api/ask          问答（Graph RAG）
    POST /api/retrieve     只检索
    GET  /api/graph        图谱概览
    GET  /api/questions    示例问题
    GET  /health           健康检查

用法：
    pwsh -NoProfile -File run_py.ps1 研发/app/ui/serve_graph_web.py --port 8124
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.graphrag.config import WORK_ORDER  # noqa: E402
from app.graphrag.engine import GraphRAGEngine  # noqa: E402

TITLE = "工单8 Graph RAG 金融问答"

#: 示例问题（中文 4 道 + 英文 1 道，全部取自 工单7 的 17 题测试集，便于一键复现实测结果）
DEMO_QUESTIONS: list[dict[str, str]] = [
    {"lang": "zh", "q": "平安银行2019年年度报告中，该行2019年实现营业收入和净利润分别是多少？"},
    {"lang": "zh", "q": "招商银行2019年年度报告中提到的创新商业模式有哪些？"},
    {"lang": "zh", "q": "中信证券2020年年度报告中，公司负责人是谁？"},
    {"lang": "zh", "q": "对比平安银行与招商银行2019年年度报告：两家银行2019年营业收入分别是多少？"},
    {"lang": "en", "q": "According to Postal Savings Bank of China's 2019 Annual Report, what were the "
                        "bank's net profit and its year-on-year growth rate, the NPL ratio, and the "
                        "provision coverage ratio in 2019?"},
]


class _Handler(BaseHTTPRequestHandler):
    """单请求处理：所有端点都在这里（无框架）。"""

    server_version = "Gongdan8GraphRAG/1.0"
    engine: Any = None
    log: Any = None
    digest_cache: dict[str, Any] = {}

    # -- 工具 ------------------------------------------------------------
    def _send(self, body: bytes, *, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"),
                   content_type="application/json; charset=utf-8", status=status)

    def _html(self, text: str, status: int = 200) -> None:
        self._send(text.encode("utf-8"), content_type="text/html; charset=utf-8", status=status)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8", errors="replace") or "{}")
        except json.JSONDecodeError as exc:
            self._warn("web.bad_json", error=f"{type(exc).__name__}: {exc}")
            return {}
        return data if isinstance(data, dict) else {}

    def _warn(self, event: str, **fields: Any) -> None:
        try:
            if self.log is not None:
                self.log.log_event(event, level="WARNING", **fields)
        except Exception:  # noqa: BLE001 —— 日志失败不得影响响应
            pass

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 —— 覆盖默认 stderr 输出
        try:
            if self.log is not None:
                self.log.log_event("web.access", client=self.client_address[0], message=fmt % args)
        except Exception:  # noqa: BLE001
            pass

    # -- 路由 ------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 —— BaseHTTPRequestHandler 约定
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            self._html(_PAGE.replace("__TITLE__", TITLE)
                       .replace("__DEMOS__", json.dumps(DEMO_QUESTIONS, ensure_ascii=False))
                       .replace("__WORK_ORDER__", WORK_ORDER))
            return
        if path in ("/health", "/api/health"):
            self._json(self.engine.health())
            return
        if path in ("/api/questions", "/questions"):
            self._json({"questions": DEMO_QUESTIONS, "count": len(DEMO_QUESTIONS)})
            return
        if path in ("/api/graph", "/graph"):
            query = parse_qs(parsed.query or "")
            top = _as_int(query.get("top", ["120"])[0], 120)
            edges = _as_int(query.get("edges", ["240"])[0], 240)
            key = f"{top}:{edges}:{self.engine.graph.node_count}"
            cached = self.digest_cache.get(key)
            if cached is None:
                cached = self.engine.graph.to_public_dict(max_nodes=top, max_edges=edges)
                self.digest_cache.clear()
                self.digest_cache[key] = cached
            self._json(cached)
            return
        self._json({"error": {"code": "E_NOT_FOUND", "message": f"未知路径 {path}"}}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        started = time.perf_counter()
        payload = self._body()
        question = str(payload.get("question") or "").strip()
        if not question or len(question) > 500:
            self._json({"error": {"code": "E_PARAM_QUESTION_EMPTY",
                                  "message": "question 不得为空且长度 ≤500"}}, 400)
            return
        top_k = _as_int(payload.get("top_k", payload.get("topK")), 0)
        lang = str(payload.get("lang") or "auto")
        use_graph = bool(payload.get("use_graph", True))
        try:
            if path in ("/api/ask", "/ask", "/api/query", "/query"):
                answer = self.engine.answer(question, top_k=(top_k or None), use_graph=use_graph,
                                            language=lang,
                                            use_llm=bool(payload.get("use_llm", True)))
                body = answer.to_dict()
                body["cited_pages"] = sorted({int(c.get("page") or 0)
                                              for c in answer.evidence_chunks if c.get("page")})
                body["server_ms"] = round((time.perf_counter() - started) * 1000, 2)
                self._json(body)
                return
            if path in ("/api/retrieve", "/retrieve"):
                self._json(self.engine.retrieve(question, top_k=(top_k or 10), use_graph=use_graph))
                return
        except Exception as exc:  # noqa: BLE001 —— 未预期异常必须带类型返回，不裸崩
            self._warn("web.error", path=path, error=f"{type(exc).__name__}: {exc}")
            self._json({"error": {"code": "E_INTERNAL",
                                  "message": f"{type(exc).__name__}: {exc}"}}, 500)
            return
        self._json({"error": {"code": "E_NOT_FOUND", "message": f"未知路径 {path}"}}, status=404)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def build_server(engine: Any, *, host: str = "127.0.0.1", port: int = 8124, logger: Any = None) -> ThreadingHTTPServer:
    """构造 HTTP 服务（供 ``--serve`` 与测试共用）。"""
    handler = type("BoundHandler", (_Handler,), {"engine": engine, "log": logger, "digest_cache": {}})
    server = ThreadingHTTPServer((host, int(port)), handler)
    server.daemon_threads = True
    if logger is not None:
        try:
            logger.log_event("web.serving", host=host, port=int(port), title=TITLE,
                             endpoints=["/", "/health", "/api/questions", "/api/graph",
                                        "/api/ask", "/api/retrieve"])
        except Exception:  # noqa: BLE001
            pass
    return server


def serve(engine: Any, *, host: str = "127.0.0.1", port: int = 8124, logger: Any = None) -> None:
    """阻塞式启动（Ctrl+C 停止）。"""
    server = build_server(engine, host=host, port=port, logger=logger)
    print(f"[OK] {TITLE} 已启动：http://{host}:{int(port)}/  （Ctrl+C 停止）")
    try:
        server.serve_forever()
    finally:
        server.shutdown()
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"工单8 Graph RAG Web 界面（纯标准库）：{WORK_ORDER}")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8124, help="默认 8124（工单7 用 8123，避免串台）")
    parser.add_argument("--no-warmup", action="store_true", help="跳过预热（首问会慢几秒）")
    args = parser.parse_args(argv)
    from app.core.logging_conf import setup_logging

    log = setup_logging()
    engine = GraphRAGEngine(logger=log)
    if not args.no_warmup:
        warm = engine.warmup(with_llm=True)
        print(f"[预热] 索引 {warm.get('index_chunks')} 块 · 图谱 "
              f"{(warm.get('graph') or {}).get('nodes')} 节点 / {(warm.get('graph') or {}).get('edges')} 边 · "
              f"Ollama 可用={warm.get('llm_available')} · 耗时 {warm.get('elapsed_ms')} ms")
    serve(engine, host=args.host, port=int(args.port), logger=log)
    return 0


_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TITLE__</title>
<style>
  :root { --ink:#1b2230; --line:#c9cfda; --brand:#1f69c8; --bg:#f5f7fa; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font-family:"Microsoft YaHei",SimHei,"PingFang SC",sans-serif; font-size:14px; }
  header { background:#182234; color:#fff; padding:14px 22px; }
  header h1 { margin:0; font-size:19px; font-weight:600; }
  header p { margin:6px 0 0; font-size:12px; color:#c3d2e8; }
  main { max-width:1280px; margin:18px auto 60px; padding:0 16px; }
  .card { background:#fff; border:1px solid var(--line); border-radius:8px; padding:16px 18px; margin-bottom:16px; }
  .card h2 { margin:0 0 12px; font-size:15px; color:var(--brand); border-left:4px solid var(--brand); padding-left:8px; }
  textarea { width:100%; min-height:64px; padding:10px; border:1px solid var(--line); border-radius:6px;
             font-family:inherit; font-size:14px; resize:vertical; }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:10px; }
  button { background:var(--brand); color:#fff; border:0; border-radius:6px; padding:9px 18px;
           font-size:14px; cursor:pointer; }
  button.ghost { background:#eef2f8; color:#33415c; }
  button:disabled { opacity:.5; cursor:progress; }
  select,input[type=number] { padding:7px 9px; border:1px solid var(--line); border-radius:6px; font-family:inherit; }
  .demo { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
  .demo button { background:#f0f4fa; color:#2a3a55; font-size:12px; padding:6px 10px; text-align:left; }
  .answer { white-space:pre-wrap; line-height:1.75; background:#f7fbff; border:1px solid #cfe0f7;
            border-radius:6px; padding:14px; min-height:56px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th,td { border-bottom:1px solid #e6eaf1; padding:8px 6px; text-align:left; vertical-align:top; }
  th { background:#f2f5fa; font-weight:600; }
  .tag { display:inline-block; padding:1px 7px; border-radius:10px; font-size:12px; margin-right:4px; }
  .t-base { background:#e8eefb; color:#1f4fa8; }
  .t-base-graph { background:#dff3e6; color:#1c7a45; }
  .t-graph-only { background:#fde8dd; color:#a8410f; }
  .mono { font-family:Consolas,monospace; font-size:12px; color:#4a5568; }
  .muted { color:#6b7688; font-size:12px; }
  .chips span { display:inline-block; background:#eef3fb; color:#2b3d5e; border-radius:12px;
                padding:2px 10px; margin:3px 4px 3px 0; font-size:12px; }
  .edges div { font-size:12px; color:#3c4a63; padding:2px 0; }
  pre { background:#fbfcfe; border:1px solid #e3e8f0; border-radius:6px; padding:10px; overflow:auto;
        max-height:280px; font-size:12px; }
  .stat { display:inline-block; margin-right:16px; font-size:12px; color:#4a5568; }
  #status { font-size:12px; color:#6b7688; margin-left:8px; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <p>工单编号：__WORK_ORDER__ ｜ 检索：bge-m3 向量 + BM25 + <b>知识图谱一跳/两跳扩展</b> ｜ 生成：Ollama qwen2.5:3b ｜ 端口 8124</p>
</header>
<main>
  <div class="card">
    <h2>① 问题</h2>
    <textarea id="q" placeholder="请输入问题（中文或英文）…"></textarea>
    <div class="row">
      <label>语言
        <select id="lang"><option value="auto">自动</option><option value="zh">中文</option><option value="en">英文</option></select>
      </label>
      <label>top_k <input type="number" id="topk" value="10" min="1" max="30" style="width:72px" /></label>
      <label><input type="checkbox" id="usegraph" checked /> 启用图谱扩展</label>
      <label><input type="checkbox" id="usellm" checked /> 启用模型生成</label>
      <button id="go">提问</button>
      <button class="ghost" id="retr">只检索</button>
      <span id="status"></span>
    </div>
    <div class="demo" id="demos"></div>
  </div>

  <div class="card" id="c-answer" style="display:none">
    <h2>④ 答案（依据证据块生成，[块N] 对应下方检索结果）</h2>
    <div class="answer" id="answer"></div>
    <div class="row muted" id="meta"></div>
  </div>

  <div class="card" id="c-graph" style="display:none">
    <h2>② 图谱结构（本次问句命中的实体 + 一跳/两跳邻居 + 途经关系）</h2>
    <div class="row muted" id="graph-stat"></div>
    <div class="chips" id="seeds"></div>
    <div class="chips" id="nodes"></div>
    <div class="edges" id="edges"></div>
    <details><summary class="muted">查看图谱结构 JSON 缩略</summary><pre id="graph-json"></pre></details>
  </div>

  <div class="card" id="c-hits" style="display:none">
    <h2>③ 检索结果（答案所在的文本块）</h2>
    <table>
      <thead><tr><th>#</th><th>chunk_id</th><th>页码</th><th>文件名</th><th>通道</th><th>分数</th><th>图谱</th><th>文本</th></tr></thead>
      <tbody id="hits"></tbody>
    </table>
  </div>
</main>
<script>
const DEMOS = __DEMOS__;
const $ = id => document.getElementById(id);
const demos = $('demos');
DEMOS.forEach(d => {
  const b = document.createElement('button');
  b.textContent = (d.lang === 'en' ? '[EN] ' : '[中文] ') + d.q.slice(0, 42) + (d.q.length > 42 ? '…' : '');
  b.onclick = () => { $('q').value = d.q; $('lang').value = d.lang; };
  demos.appendChild(b);
});
function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function tag(ch) {
  const cls = ch === 'graph_only' ? 't-graph-only' : (ch === 'base+graph' ? 't-base-graph' : 't-base');
  return '<span class="tag ' + cls + '">' + esc(ch) + '</span>';
}
async function callApi(path) {
  const body = { question: $('q').value.trim(), lang: $('lang').value,
                 top_k: parseInt($('topk').value || '10', 10),
                 use_graph: $('usegraph').checked, use_llm: $('usellm').checked };
  if (!body.question) { $('status').textContent = '请先输入问题'; return; }
  $('go').disabled = $('retr').disabled = true;
  $('status').textContent = path.includes('retrieve') ? '检索中…' : '检索并生成中…';
  const t0 = performance.now();
  try {
    const resp = await fetch(path, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
    const data = await resp.json();
    if (data.error) { $('status').textContent = '错误：' + data.error.message; return; }
    $('status').textContent = '完成，用时 ' + Math.round(performance.now() - t0) + ' ms';
    if (!path.includes('retrieve')) renderAnswer(data);
    renderGraph(data.graph || {}, data.diagnostics || {}, data.graph_used);
    renderHits(data.chunks || data.hits || []);
  } catch (e) {
    $('status').textContent = '请求失败：' + e;
  } finally {
    $('go').disabled = $('retr').disabled = false;
  }
}
function renderAnswer(d) {
  $('c-answer').style.display = 'block';
  $('answer').textContent = d.answer || '（无）';
  const lm = d.latency_ms || {};
  $('meta').innerHTML = '生成方式 ' + esc(d.generation_method) + ' ｜ 模型 ' + esc(d.model) +
    ' ｜ 语言 ' + esc(d.language) +
    ' ｜ 检索 ' + Math.round(lm.retrieve || 0) + ' ms（图谱 ' + Math.round(lm.graph || 0) + ' ms）' +
    ' ｜ 生成 ' + Math.round(lm.generate || 0) + ' ms ｜ 合计 ' + Math.round(lm.total || 0) + ' ms';
}
function renderGraph(g, diag, used) {
  $('c-graph').style.display = 'block';
  $('graph-stat').innerHTML =
    '<span class="stat">命中实体 ' + (g.seed_names || []).length + '</span>' +
    '<span class="stat">邻域节点 ' + ((g.nodes || []).length) + '</span>' +
    '<span class="stat">途经关系 ' + ((g.edges || []).length) + '</span>' +
    '<span class="stat">图谱独有块进 Top-K ' + (diag.graph_only_in_topk || 0) + '</span>' +
    '<span class="stat">图谱启用 ' + (used ? '是' : '否') + '</span>';
  $('seeds').innerHTML = '<b>命中实体：</b>' + ((g.seed_names || []).map(s => '<span>' + esc(s) + '</span>').join('') || '<span class="muted">（无）</span>');
  const nodes = (g.nodes || []).slice(0, 40);
  $('nodes').innerHTML = '<b>邻域节点：</b>' + nodes.map(n =>
    '<span>' + esc(n.name) + '·' + esc(n.type_zh) + '·' + (n.distance === 0 ? '命中' : n.distance + '跳') + '</span>').join('');
  const edges = (g.edges || []).slice(0, 30);
  $('edges').innerHTML = '<b>途经关系：</b><br/>' + (edges.map(e =>
    esc(e.head) + ' —<b>' + esc(e.relation) + '</b>→ ' + esc(e.tail) +
    ' <span class="muted">(置信度 ' + (e.confidence == null ? '-' : Number(e.confidence).toFixed(2)) + ')</span>').join('<br/>') || '<span class="muted">（无）</span>');
  $('graph-json').textContent = JSON.stringify({ seed_names: g.seed_names, nodes: nodes,
      edges: edges, distances: g.distances_named || {},
      graph_only_in_topk: diag.graph_only_in_topk, graph_displaced: diag.graph_displaced,
      graph_blocked_by_net_gain: diag.graph_blocked_by_net_gain }, null, 1);
}
function renderHits(hits) {
  $('c-hits').style.display = 'block';
  const tb = $('hits'); tb.innerHTML = '';
  hits.slice(0, 12).forEach(h => {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td>' + (h.rank || '') + '</td>' +
      '<td class="mono">' + esc(h.chunk_id) + '</td>' +
      '<td>' + (h.page || '') + '</td>' +
      '<td class="mono">' + esc(String(h.file_name || '').slice(0, 46)) + '</td>' +
      '<td>' + tag(h.retrieval_channel || 'base') + '</td>' +
      '<td>' + (h.score == null ? '' : Number(h.score).toFixed(4)) + '</td>' +
      '<td class="muted">' + esc((h.graph_entities || []).join('、')) +
        (h.graph_distance == null ? '' : '（' + h.graph_distance + '跳）') + '</td>' +
      '<td>' + esc(String(h.text || '').slice(0, 220)) + '…</td>';
    tb.appendChild(tr);
  });
}
$('go').onclick = () => callApi('/api/ask');
$('retr').onclick = () => callApi('/api/retrieve');
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
