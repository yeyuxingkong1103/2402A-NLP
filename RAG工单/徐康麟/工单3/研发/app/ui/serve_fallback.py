# -*- coding: utf-8 -*-
"""工单3 纯标准库备用界面（设计/接口设计.md §3.25 冻结）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

* 只用 ``http.server``（``ThreadingHTTPServer``）+ 内联 HTML/CSS/JS，**无任何外部静态依赖**；
* 业务逻辑一律走 ``app.core.qa_engine``（与 Streamlit 界面同一套），本文件不实现检索/生成/引用；
* 路由：``GET /``、``GET /api/health``、``GET /api/files``、``POST /api/ask``、
  ``POST /api/ask/stream``（SSE）、``GET /api/sessions``、``POST /api/feedback``、``POST /api/clear``；
* 默认端口取 ``cfg.server_port``（``RAG_SERVER__PORT`` 可覆盖），主机取 ``cfg.server_host``。

用法（工作目录 = 工单3）：
    pwsh -NoProfile -File run_py.ps1 研发/app/ui/serve_fallback.py            # 默认 127.0.0.1:8600
    pwsh -NoProfile -File run_py.ps1 研发/app/ui/serve_fallback.py --port 8501
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core.config import get_config  # noqa: E402
from app.core.errors import RagError  # noqa: E402
from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402
from app.core.qa_engine import QAEngine, get_engine  # noqa: E402

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

PAGE_HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>工单3 · 招股说明书问答（标准库备用界面）</title>
<style>
 body{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;margin:0;background:#f5f6f8;color:#1c1f23}
 header{background:#11324f;color:#fff;padding:14px 20px;font-size:16px}
 main{max-width:980px;margin:0 auto;padding:18px}
 .card{background:#fff;border:1px solid #e2e5ea;border-radius:10px;padding:14px;margin-bottom:14px}
 .row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
 select,input,button{font-size:14px;padding:7px 10px;border:1px solid #c8cdd6;border-radius:8px;background:#fff}
 button{background:#11324f;color:#fff;border-color:#11324f;cursor:pointer}
 button.ghost{background:#fff;color:#11324f}
 .ans{white-space:pre-wrap;line-height:1.65}
 .cite{display:inline-block;background:#eef4fb;border:1px solid #cfe0f3;border-radius:6px;padding:2px 8px;margin:2px 4px 2px 0;font-size:13px}
 .meta{color:#6b7280;font-size:12.5px;margin-top:6px}
 details{margin-top:6px} summary{cursor:pointer;color:#11324f;font-size:13px}
 pre{background:#f8f9fb;border:1px solid #e6e9ee;border-radius:8px;padding:8px;white-space:pre-wrap;font-size:12.5px}
 .chunk{border-left:3px solid #cfe0f3;padding-left:8px;margin:6px 0}
 .fb{font-size:18px;background:none;border:none;cursor:pointer;padding:2px 6px}
 .hist{border-top:1px dashed #e2e5ea;padding-top:8px;margin-top:8px}
</style></head>
<body>
<header>工单3 · 人工智能NLP-RAG-PDF文档的表格解析及检索优化 —— 问答界面（纯标准库备用界面）</header>
<main>
  <div class="card">
    <div class="row">
      <label>PDF 范围</label>
      <select id="files" multiple size="1" style="min-width:260px"></select>
      <label>top-k</label><input id="topk" type="number" value="5" min="1" max="20" style="width:70px"/>
      <button id="ask">提问</button>
      <button id="clear" class="ghost">清空对话</button>
      <span id="health" class="meta"></span>
    </div>
    <div class="row" style="margin-top:10px">
      <input id="q" placeholder="例如：武汉兴图新科电子股份有限公司的注册资本是多少？" style="flex:1;min-width:320px"/>
    </div>
  </div>
  <div class="card"><div id="thread" class="ans"></div></div>
  <div class="card">
    <div class="meta">最近一次检索片段（含页码，可与答案引用对照）</div>
    <div id="chunks"></div>
  </div>
</main>
<script>
let sessionId = null, lastAnswer = null, lastMessageId = null;
async function jget(u){const r=await fetch(u);return await r.json();}
async function jpost(u,b){const r=await fetch(u,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b||{})});return await r.json();}
async function boot(){
  const h = await jget("/api/health");
  document.getElementById("health").textContent =
    `检索块 ${h.retriever.chunks||0} · 模型 ${h.llm.model} · 后端 ${h.llm.backend}`;
  const f = await jget("/api/files");
  const sel = document.getElementById("files");
  sel.innerHTML = "";
  (f.files||[]).forEach(x=>{const o=document.createElement("option");
    o.value=x.file_name; o.textContent=`${x.file_name}（${x.page_count||"?"} 页 / ${x.chunk_count} 块）`; sel.appendChild(o);});
}
function cites(list){
  if(!list||!list.length) return "";
  return list.map(c=>`<span class="cite">[${c.file_name}: ${c.page}]</span>`).join("");
}
document.getElementById("ask").onclick = async ()=>{
  const q = document.getElementById("q").value.trim(); if(!q) return;
  const files = Array.from(document.getElementById("files").selectedOptions).map(o=>o.value);
  const topk = parseInt(document.getElementById("topk").value||"5",10);
  const thread = document.getElementById("thread");
  thread.innerHTML += `<div class="hist"><b>你：</b>${q}</div><div class="hist"><b>助手：</b>检索中…</div>`;
  const holder = thread.querySelectorAll(".hist b + *");
  const target = thread.lastElementChild;
  const t0 = performance.now();
  try{
    const res = await jpost("/api/ask",{question:q, session_id:sessionId, file_names:files.length?files:null, top_k:topk});
    sessionId = res.session_id || sessionId;
    lastAnswer = res; lastMessageId = res.message_id;
    const wall = Math.round(performance.now()-t0);
    target.innerHTML = `<b>助手：</b><span class="ans">${res.text||"不清楚"}</span><div>${cites(res.citations)}</div>`+
      `<div class="meta">首字 ${res.first_token_ms} ms · 总 ${res.total_ms} ms · 端到端 ${wall} ms · trace ${res.trace_id}`+
      `${res.is_unknown?`（不清楚：${res.unknown_reason}）`:""}</div>`+
      `<div><button class="fb" onclick="fb('up')">👍</button><button class="fb" onclick="fb('down')">👎</button></div>`;
    renderChunks(res);
  }catch(e){ target.innerHTML = `<b>助手：</b>出错：${e}`; }
};
function renderChunks(r){
  const box = document.getElementById("chunks"); box.innerHTML="";
  if(!r||!r.chunks) return;
  (r.support_chunks||[]).forEach(c=>{const head=((c.content_digest||{}).head)||"";box.innerHTML += `<div class="chunk"><b>[支持块] ${c.file_name}: ${c.page}</b><pre>${head}</pre></div>`;});
  r.chunks.forEach(c=>{const head=((c.content_digest||{}).head)||"";box.innerHTML += `<div class="chunk"><b>#${c.rank} ${c.file_name}: ${c.page} · ${c.type} · score ${c.score}</b><pre>${head}</pre></div>`;});
}
async function fb(rating){
  if(!lastAnswer) return;
  await jpost("/api/feedback",{rating:rating, session_id:sessionId, message_id:lastMessageId, answer_id:lastAnswer.answer_id, trace_id:lastAnswer.trace_id});
  alert(rating==="up"?"已记录点赞":"已记录点踩");
}
document.getElementById("clear").onclick = async ()=>{
  if(sessionId) await jpost("/api/clear",{session_id:sessionId});
  sessionId=null; document.getElementById("thread").innerHTML=""; document.getElementById("chunks").innerHTML="";
};
boot();
</script>
</body></html>
"""


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    """序列化响应体（UTF-8，不转义中文）。"""
    return json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")


def _answer_payload(engine: QAEngine, question: str, *, session_id: str | None,
                    file_names: list[str] | None, top_k: int) -> dict[str, Any]:
    """执行一次问答并组装 HTTP 响应（**复用** ``QAEngine.answer_payload``，§7 契约字段）。

    t14 契约对齐：字段名 = ``text``（不是 ``answer``）、``chunks``（不是 ``retrieval``）；
    ``file_names`` 含不存在的文件 → 抛 ``RagError``（HTTP 层转 400，§7 状态码表）。
    """
    log = get_logger("serve_fallback")
    started = time.perf_counter()
    with log.enter("serve_fallback._answer_payload",
                   {"question": question[:60], "file_names": file_names, "top_k": top_k}) as span:
        checked = engine.validate_files(file_names)
        before = engine.store.message_count(session_id) if session_id else 0
        answer = engine.ask(question, session_id=session_id, file_names=checked, top_k=top_k, logger=log)
        sid = session_id or (answer.conversation_id or "")
        if not sid:
            sessions = engine.sessions(limit=1)
            sid = sessions[0]["session_id"] if sessions else ""
        after = engine.store.message_count(sid) if sid else before
        payload = engine.answer_payload(answer, session_id=sid,
                                        wall_ms=round((time.perf_counter() - started) * 1000, 2))
        payload["message_id"] = after
        span.set_output({"unknown": answer.is_unknown, "citations": len(answer.citations),
                         "wall_ms": payload["wall_ms"]})
        return payload


class FallbackHandler(BaseHTTPRequestHandler):
    """标准库 HTTP 处理器（所有业务逻辑委托 ``QAEngine``）。"""

    engine: QAEngine
    server_version = "RAGFallback/1.0"

    # -- 工具 ------------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 —— 覆写基类签名
        """把 http.server 的访问日志转成结构化日志（禁止默认 stderr 噪声）。"""
        get_logger("serve_fallback").log_event("ui.http_access", level="DEBUG", client=self.client_address[0],
                                               message=fmt % args)

    def _send(self, code: int, body: bytes, *, content_type: str = "application/json; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            get_logger("serve_fallback").log_event("ui.bad_json", level="WARNING",
                                                   error_type=type(exc).__name__, message=str(exc),
                                                   raw=raw[:120].decode("utf-8", errors="replace"))
            raise RagError("请求体不是合法 JSON", code="RAG-6001", stage="ui") from exc
        return data if isinstance(data, dict) else {}

    # -- 路由 ------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 —— http.server 约定
        """GET 路由：/、/api/health、/api/files、/api/sessions。"""
        log = get_logger("serve_fallback")
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                self._send(200, PAGE_HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
            elif path == "/api/health":
                self._send(200, _json_bytes(self.engine.health()))
            elif path == "/api/files":
                self._send(200, _json_bytes({"files": self.engine.files()}))
            elif path == "/api/sessions":
                self._send(200, _json_bytes({"sessions": self.engine.sessions()}))
            else:
                self._send(404, _json_bytes({"error": f"未知路径：{path}"}))
        except Exception as exc:  # noqa: BLE001 —— 统一 500 + 日志（不静默）
            log.log_event("ui.get_failed", level="ERROR", path=path,
                          error_type=type(exc).__name__, message=str(exc))
            self._send(500, _json_bytes({"error": f"{type(exc).__name__}: {exc}"}))

    def do_POST(self) -> None:  # noqa: N802
        """POST 路由：/api/ask、/api/ask/stream、/api/feedback、/api/clear。"""
        log = get_logger("serve_fallback")
        path = urlparse(self.path).path
        try:
            body = self._read_json()
            if path == "/api/ask":
                question = str(body.get("question") or "").strip()
                if not question:
                    self._send(400, _json_bytes({"error": "question 不能为空"}))
                    return
                payload = _answer_payload(self.engine, question,
                                          session_id=body.get("session_id"),
                                          file_names=list(body.get("file_names") or []) or None,
                                          top_k=int(body.get("top_k") or self.engine.cfg.retrieval.top_k))
                self._send(200, _json_bytes(payload))
            elif path == "/api/ask/stream":
                self._send_stream(body)
            elif path == "/api/feedback":
                fid = self.engine.feedback(rating=str(body.get("rating") or ""),
                                           session_id=body.get("session_id"),
                                           message_id=body.get("message_id"),
                                           answer_id=body.get("answer_id"), trace_id=body.get("trace_id"),
                                           comment=body.get("comment"))
                self._send(200, _json_bytes({"feedback_id": fid}))
            elif path == "/api/clear":
                removed = self.engine.clear_conversation(str(body.get("session_id") or ""))
                self._send(200, _json_bytes({"removed": removed}))
            else:
                self._send(404, _json_bytes({"error": f"未知路径：{path}"}))
        except RagError as exc:
            # §7 状态码表：400 = 入参非法（空问题、file_names 含不存在的文件）
            status = 400 if str(getattr(exc, "code", "")) == "RAG-6001" else 500
            log.log_event("ui.post_rejected" if status == 400 else "ui.post_failed",
                          level="WARNING" if status == 400 else "ERROR", path=path, code=exc.code,
                          error_type=type(exc).__name__, message=str(exc))
            self._send(status, _json_bytes({"error": str(exc), "code": getattr(exc, "code", "")}))
        except Exception as exc:  # noqa: BLE001
            log.log_event("ui.post_failed", level="ERROR", path=path,
                          error_type=type(exc).__name__, message=str(exc))
            self._send(500, _json_bytes({"error": f"{type(exc).__name__}: {exc}"}))

    def _send_stream(self, body: Mapping[str, Any]) -> None:
        """SSE 流式作答：逐块推送 delta，最后推一条带完整答案的事件。"""
        question = str(body.get("question") or "").strip()
        if not question:
            self._send(400, _json_bytes({"error": "question 不能为空"}))
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        log = get_logger("serve_fallback")
        with log.enter("serve_fallback._send_stream", {"question": question[:60]}) as span:
            deltas = self.engine.ask(question, session_id=body.get("session_id"),
                                     file_names=list(body.get("file_names") or []) or None,
                                     top_k=int(body.get("top_k") or self.engine.cfg.retrieval.top_k),
                                     stream=True, logger=log)
            first_ms = None
            index = 0
            for delta in deltas:
                if delta.is_first:
                    first_ms = delta.first_token_ms
                event = self.engine.stream_event(delta, index=index)   # §7 契约事件体（唯一实现）
                index += 1
                self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()
            span.set_output({"first_token_ms": first_ms})


def serve(*, host: str | None = None, port: int | None = None, engine: QAEngine | None = None) -> ThreadingHTTPServer:
    """启动备用界面（返回已绑定的 server，便于测试里 ``serve_forever`` 于线程中）。"""
    cfg = get_config()
    log = get_logger("serve_fallback")
    host = host or cfg.server_host
    port = int(port or cfg.server_port)
    eng = engine or get_engine()
    handler = type("BoundHandler", (FallbackHandler,), {"engine": eng})
    httpd = ThreadingHTTPServer((host, port), handler)
    log.log_event("ui.serve_start", host=host, port=port, backend=getattr(eng.generator.llm, "backend", None)
                  and eng.generator.llm.backend.name)
    return httpd


def main(argv: list[str] | None = None) -> int:
    """命令行入口（``--host/--port/--no-warmup``）。"""
    parser = argparse.ArgumentParser(description="工单3 标准库备用界面")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-warmup", action="store_true", help="跳过预热（调试用，首题会慢约 1 s）")
    args = parser.parse_args(argv)
    cfg = get_config()
    setup_logging(cfg, force=True)
    log = get_logger("serve_fallback")
    engine = QAEngine(cfg=cfg, logger=log)
    if not args.no_warmup:
        info = engine.warmup()
        log.log_event("ui.warmup_done", **{k: v for k, v in info.items() if k != "backends"})
    httpd = serve(host=args.host, port=args.port, engine=engine)
    print(f"备用界面已启动：http://{httpd.server_address[0]}:{httpd.server_address[1]}  (Ctrl+C 退出)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.log_event("ui.serve_stop", reason="KeyboardInterrupt")
    finally:
        httpd.server_close()
        shutdown_logging()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 —— 顶层兜底
        import traceback

        traceback.print_exc()
        shutdown_logging()
        raise SystemExit(1)
