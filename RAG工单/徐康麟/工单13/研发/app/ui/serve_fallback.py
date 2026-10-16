# -*- coding: utf-8 -*-
"""工单13 Web 界面（**纯标准库兜底实现**，零第三方依赖）：``研发/app/ui/serve_fallback.py``。

工单编号：人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化

为什么用标准库：本机 pip 通道不通（设计/环境事实.md §3），``fastapi``/``uvicorn`` 虽在
工单1 venv 中可用，但为了保证「任何解释器都能起 Web 界面」，这里用 ``http.server`` 实现；
真 Streamlit 版本见 ``streamlit_app.py``（有依赖的环境用）。**两者共用同一个 QAEngine**，
不复制任何业务逻辑（设计/接口设计.md §3.6）。

HTTP 契约（与 设计/接口设计.md §3 对齐，供 T6 评测与浏览器共用）：

    GET  /                    中文问答页
    GET  /health              健康检查（含依赖探测，未就绪项如实说明原因）
    GET  /api/questions       示例问题清单（含第 5、6 题）
    POST /api/ask  (== /ask)  单次问答 → {answer, citations, cited_pages, refused, latency_ms,
                                         backend, cache, evidence, chunks, ...}
    POST /api/retrieve        只检索不生成（检索精度留痕用）

设计纪律：
    * 端口默认 **8123**（``RAG_SERVER__FALLBACK_PORT``），只杀自己的端口（设计 §1）；
    * 「不清楚」是**正确响应**（HTTP 200 + ``refused=true``），不是错误；
    * 每个请求都写结构化日志（``web.request`` / ``web.response``，含函数名/输入摘要/耗时）。
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

WORK_ORDER = "人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化"

TITLE = "RAG 性能剖析与优化问答系统（工单13）"

#: 示例问题（含图像题：第 5 题组织结构图 / 第 6 题 IC 市场图表）
DEMO_QUESTIONS: list[dict[str, str]] = [
    {"id": "5", "tag": "图像·组织结构图",
     "question": "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？"},
    {"id": "6", "tag": "图像·图表",
     "question": "武汉力源信息技术股份有限公司招股意向书中，从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？"},
    {"id": "1", "tag": "表格·数值",
     "question": "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？"},
    {"id": "4", "tag": "枚举·关联方",
     "question": "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？"},
    {"id": "260", "tag": "多点数值",
     "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"},
    {"id": "543", "tag": "单值",
     "question": "武汉兴图新科电子股份有限公司注册资本是多少？"},
    {"id": "en", "tag": "英文提问",
     "question": "What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?"},
]

_PAGE_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TITLE__</title>
<style>
  :root { --bg:#f6f7fb; --card:#fff; --line:#e3e6ef; --ink:#1c2333; --muted:#697089;
          --accent:#2f6fed; --warn:#b7791f; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif;
         background:var(--bg); color:var(--ink); }
  header { background:linear-gradient(90deg,#2f6fed,#5b8def); color:#fff; padding:18px 24px; }
  header h1 { margin:0; font-size:20px; }
  header p { margin:6px 0 0; font-size:13px; opacity:.92; }
  main { max-width:1080px; margin:18px auto 60px; padding:0 16px; display:grid; gap:14px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:16px; box-shadow:0 1px 3px rgba(20,30,60,.05); }
  .card h2 { margin:0 0 10px; font-size:15px; color:var(--muted); font-weight:600; }
  textarea { width:100%; min-height:74px; border:1px solid var(--line); border-radius:8px;
             padding:10px; font-size:15px; font-family:inherit; resize:vertical; }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:10px; }
  button { border:0; border-radius:8px; padding:10px 18px; font-size:15px; cursor:pointer;
           background:var(--accent); color:#fff; }
  button.ghost { background:#eef2fb; color:var(--accent); font-size:13px; padding:7px 12px; }
  button:disabled { opacity:.55; cursor:progress; }
  .answer { font-size:16px; line-height:1.75; white-space:pre-wrap; }
  .refused { color:var(--warn); font-weight:600; }
  .cites { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
  .cite { background:#eef2fb; color:#1f4fd8; border-radius:6px; padding:4px 8px; font-size:13px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th,td { border-bottom:1px solid var(--line); text-align:left; padding:6px 8px; vertical-align:top; }
  .tag { display:inline-block; border-radius:4px; padding:1px 6px; font-size:12px; margin-right:6px; }
  .tag.image { background:#e8f5e9; color:#1b5e20; }
  .tag.table { background:#fff8e1; color:#8d6e00; }
  .tag.text  { background:#e8eefc; color:#1f4fd8; }
  .muted { color:var(--muted); font-size:13px; }
  pre { margin:0; white-space:pre-wrap; word-break:break-all; font-size:13px; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <p>4 份 A 股年报（招商银行 / 中国平安 / 平安银行 / 邮储银行）· 文本 + 表格 + 图像三类证据 · 答案带页级引用 · 支持中英文提问</p>
</header>
<main>
  <section class="card">
    <h2>提问</h2>
    <textarea id="q" placeholder="请输入问题，例如：武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成？"></textarea>
    <div class="row">
      <button id="ask">提问</button>
      <label class="muted">语言
        <select id="lang">
          <option value="auto">自动</option>
          <option value="zh">中文</option>
          <option value="en">English</option>
        </select>
      </label>
      <span class="muted">Top-K <input id="topk" type="number" value="5" min="1" max="20" style="width:64px" /></span>
      <span id="status" class="muted"></span>
    </div>
    <div class="row" id="demos"></div>
  </section>

  <section class="card">
    <h2>答案</h2>
    <div id="answer" class="answer muted">尚未提问。</div>
    <div id="cites" class="cites"></div>
    <div id="meta" class="muted" style="margin-top:10px"></div>
  </section>

  <section class="card">
    <h2>检索精度（Top-K 命中块）</h2>
    <div id="hits" class="muted">尚无检索结果。</div>
  </section>

  <section class="card">
    <h2>证据（生成阶段使用的块）</h2>
    <div id="evidence" class="muted">尚无证据。</div>
  </section>
</main>
<script>
const demos = __DEMOS__;
function renderDemos(){
  const box = document.getElementById('demos');
  demos.forEach(d => {
    const b = document.createElement('button');
    b.className = 'ghost'; b.textContent = '[' + d.tag + '] ' + d.question.slice(0, 34) + '…';
    b.title = d.question;
    b.onclick = () => { document.getElementById('q').value = d.question; };
    box.appendChild(b);
  });
}
function tagHtml(m){
  const cls = (m === 'image' || m === 'table') ? m : 'text';
  const label = m === 'image' ? '图像' : (m === 'table' ? '表格' : '正文');
  return '<span class="tag ' + cls + '">' + label + '</span>';
}
function esc(s){ return (s === null || s === undefined) ? '' : String(s)
  .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
async function ask(){
  const question = document.getElementById('q').value.trim();
  if(!question){ return; }
  const btn = document.getElementById('ask');
  btn.disabled = true;
  document.getElementById('status').textContent = '检索与生成中…';
  const t0 = performance.now();
  try {
    const resp = await fetch('/api/ask', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question: question, lang: document.getElementById('lang').value,
                            top_k: parseInt(document.getElementById('topk').value || '5', 10)})
    });
    const data = await resp.json();
    const wall = Math.round(performance.now() - t0);
    const ansBox = document.getElementById('answer');
    ansBox.classList.remove('muted');
    ansBox.innerHTML = esc(data.answer);
    if (data.refused){
      ansBox.classList.add('refused');
      ansBox.innerHTML = '不清楚<span class="muted">（拒答原因：' + esc(data.refusal_reason) + '）</span>';
    } else { ansBox.classList.remove('refused'); }
    const cites = document.getElementById('cites');
    cites.innerHTML = '';
    (data.citations || []).forEach(c => {
      const s = document.createElement('span');
      s.className = 'cite';
      s.textContent = c.corpus + '：第' + c.page + '页' +
        (c.modality === 'image' ? '（' + (c.modality_sub === 'diagram' ? '组织结构图' : (c.modality_sub === 'chart' ? '图表' : '插图')) + '）'
                                : (c.modality === 'table' ? '（表格）' : ''));
      s.title = c.quote || '';
      cites.appendChild(s);
    });
    const lat = data.latency_ms || {};
    document.getElementById('meta').textContent =
      '首字 ' + (lat.first_token ?? '-') + ' ms ｜ 总计 ' + (lat.total ?? wall) + ' ms' +
      '（检索 ' + (lat.retrieve ?? '-') + ' ms / 生成 ' + (lat.generate ?? '-') + ' ms）' +
      ' ｜ 后端 ' + (data.backend || '-') + ':' + (data.model || '-') +
      ' ｜ 生成方式 ' + (data.generation_method || '-') +
      ' ｜ 缓存 ' + ((data.cache && data.cache.hit) ? '命中' : '未命中') +
      ' ｜ trace ' + (data.trace_id || '-');
    const hits = document.getElementById('hits');
    let html = '<table><tr><th>#</th><th>块</th><th>模态</th><th>页</th><th>融合分</th><th>向量/BM25 名次</th></tr>';
    (data.chunks || []).forEach((h, i) => {
      html += '<tr><td>' + (i+1) + '</td><td>' + esc(h.chunk_id) + '</td><td>' + tagHtml(h.type) +
              ((h.modality_sub) ? '<span class="muted">' + esc(h.modality_sub) + '</span>' : '') +
              '</td><td>' + esc(h.page) + '</td><td>' + (h.final_score ?? h.score ?? '-') +
              '</td><td>' + (h.vector_rank ?? '-') + ' / ' + (h.bm25_rank ?? '-') + '</td></tr>';
    });
    html += '</table>';
    hits.innerHTML = html;
    const ev = document.getElementById('evidence');
    ev.innerHTML = (data.evidence || []).map(h =>
      '<div style="margin-bottom:8px">' + tagHtml(h.type) + '<b>' + esc(h.chunk_id) + '</b> ' +
      '<span class="muted">' + esc(h.file_name) + ' 第' + esc(h.page) + '页</span><pre>' +
      esc((h.text || '').slice(0, 600)) + '</pre></div>').join('') || '<span class="muted">无</span>';
  } catch (err) {
    document.getElementById('answer').textContent = '请求失败：' + err;
  } finally {
    btn.disabled = false;
    document.getElementById('status').textContent = '';
  }
}
renderDemos();
document.getElementById('ask').onclick = ask;
document.getElementById('q').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { ask(); }
});
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    """单请求处理：所有端点都在这里（无框架）。"""

    server_version = "Gongdan4RAGFallback/1.0"
    engine: Any = None
    log: Any = None

    # -- 工具 ------------------------------------------------------------
    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8", errors="replace") or "{}")
        except json.JSONDecodeError as exc:
            self.log.log_event("web.bad_json", level="WARNING", error=f"{type(exc).__name__}: {exc}",
                               raw=raw[:120].decode("utf-8", errors="replace"))
            return {}
        return data if isinstance(data, dict) else {}

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 —— 覆盖默认 stderr 输出
        self.log.log_event("web.access", client=self.client_address[0], message=fmt % args)

    # -- 路由 ------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 —— BaseHTTPRequestHandler 约定
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._html(_PAGE_HTML.replace("__TITLE__", TITLE)
                       .replace("__DEMOS__", json.dumps(DEMO_QUESTIONS, ensure_ascii=False)))
            return
        if path in ("/health", "/api/health"):
            self._json(self.engine.health())
            return
        if path in ("/api/questions", "/questions"):
            self._json({"questions": DEMO_QUESTIONS, "count": len(DEMO_QUESTIONS)})
            return
        self._json({"error": {"code": "E_NOT_FOUND", "message": f"未知路径 {path}"}}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        started = time.perf_counter()
        if path in ("/api/ask", "/ask", "/api/query", "/query"):
            payload = self._body()
            question = str(payload.get("question") or "").strip()
            lang = str(payload.get("lang") or "auto")
            top_k = payload.get("top_k") or payload.get("topK")
            try:
                top_k_int = int(top_k) if top_k is not None else None
            except (TypeError, ValueError):
                self._json({"error": {"code": "E_PARAM_TOP_K_RANGE", "message": "top_k 必须是整数"}}, 400)
                return
            if not question or len(question) > 500:
                self._json({"error": {"code": "E_PARAM_QUESTION_EMPTY",
                                      "message": "question 不得为空且长度 ≤500"}}, 400)
                return
            self.log.log_event("web.request", path=path, lang=lang, top_k=top_k_int,
                               question_digest=question[:80])
            try:
                result = self.engine.answer(question, lang=lang, top_k=top_k_int,
                                            session_id=payload.get("session_id"),
                                            return_evidence=bool(payload.get("return_evidence", True)),
                                            use_cache=bool(payload.get("use_cache", True)))
            except Exception as exc:  # noqa: BLE001 —— 未预期异常必须带 trace 返回
                self.log.log_event("web.error", level="ERROR", path=path,
                                   error=f"{type(exc).__name__}: {exc}")
                self._json({"error": {"code": "E_INTERNAL", "message": f"{type(exc).__name__}: {exc}"}}, 500)
                return
            body = result.to_dict()
            # 兼容评测端字段名（测试/评测骨架/evaluate_16q_skeleton.py：cited_pages / chunks / latency_ms）
            body["cited_pages"] = result.cited_pages
            body["chunks"] = result.chunks
            self.log.log_event("web.response", path=path, refused=result.refused,
                               total_ms=round((time.perf_counter() - started) * 1000, 2),
                               chars=len(result.answer), trace_id=result.trace_id)
            self._json(body)
            return
        if path in ("/api/retrieve", "/retrieve"):
            payload = self._body()
            question = str(payload.get("question") or "").strip()
            if not question:
                self._json({"error": {"code": "E_PARAM_QUESTION_EMPTY", "message": "question 不得为空"}}, 400)
                return
            try:
                top_k = int(payload.get("top_k") or 10)
            except (TypeError, ValueError):
                top_k = 10
            self.log.log_event("web.request", path=path, top_k=top_k, question_digest=question[:80])
            self._json(self.engine.retrieve_only(question, top_k=top_k,
                                                 modality=payload.get("modality")))
            return
        self._json({"error": {"code": "E_NOT_FOUND", "message": f"未知路径 {path}"}}, status=404)


def build_server(engine: Any, *, host: str = "127.0.0.1", port: int = 8123, logger: Any = None) -> ThreadingHTTPServer:
    """构造 HTTP 服务（供 ``--serve`` 与测试共用）。"""
    handler = type("BoundHandler", (_Handler,), {"engine": engine, "log": logger})
    server = ThreadingHTTPServer((host, int(port)), handler)
    server.daemon_threads = True
    if logger is not None:
        logger.log_event("web.serving", host=host, port=int(port), title=TITLE,
                         endpoints=["/", "/health", "/api/questions", "/api/ask", "/api/retrieve"])
    return server


def serve(engine: Any, *, host: str = "127.0.0.1", port: int = 8123, logger: Any = None) -> None:
    """阻塞式启动 Web 服务（Ctrl+C 停止）。"""
    log = logger if logger is not None else getattr(engine, "log", None)
    server = build_server(engine, host=host, port=port, logger=log)
    print(f"[OK] {TITLE} 已启动：http://{host}:{int(port)}/  （Ctrl+C 停止）")
    try:
        server.serve_forever()
    finally:
        server.shutdown()
        server.server_close()
        if log is not None:
            log.log_event("web.stopped", host=host, port=int(port))


def main(argv: list[str] | None = None) -> int:
    """独立启动入口：``python 研发/app/ui/serve_fallback.py --port 8123``。"""
    import argparse
    import sys
    from pathlib import Path

    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.core.config import get_config
    from app.core.logging_conf import setup_logging
    from app.core.qa_engine import QAEngine
    from app.storage.sqlite_manager import SqliteManager

    parser = argparse.ArgumentParser(description=f"工单13 Web 界面（纯标准库）：{WORK_ORDER}")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args(argv)
    cfg = get_config()
    log = setup_logging(cfg)
    storage = SqliteManager(Path(cfg.paths.index_dir) / "qa_runtime.sqlite3", logger=log)
    engine = QAEngine(cfg=cfg, logger=log, storage=storage)
    if not args.no_warmup:
        engine.warmup(with_generate=True)
    serve(engine, host=(args.host or cfg.server_host),
          port=int(args.port or getattr(cfg, "fallback_port", 8123)), logger=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
