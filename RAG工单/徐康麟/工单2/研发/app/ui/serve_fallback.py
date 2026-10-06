"""备用界面（纯标准库 http.server）：本机演示与在线测试用，不依赖 streamlit。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 接口层（对应 设计/接口设计.md §5）

为什么需要它：本机 **streamlit 不可用且断网无法安装**（环境事实 2.2），
`app/ui/streamlit_app.py` 是给算力云准备的**真正的 Streamlit 应用**；本文件提供
本机可运行的同构界面，二者共用同一套 ``app/core`` 业务逻辑与同一份事件契约。

HTTP 契约（§5）：
- ``GET  /``                     单页界面（原生 JS fetch + 流式渲染）
- ``GET  /api/health``           ``{ok, health: QAEngine.health()}``（**嵌套**，与 设计/接口设计.md 一致）
- ``GET  /api/stats``            ``{ok, tables: QAEngine.stats()}``（**嵌套**，与 设计/接口设计.md 一致）
- ``GET  /api/conversations``    会话列表
- ``POST /api/conversations``    新建会话
- ``GET  /api/messages``         会话消息
- ``POST /api/ask``              问答（``stream=true`` 走 SSE，逐事件 ``data: {...}``）
- ``POST /api/feedback``         反馈
- ``POST /api/index/build``      显式重建索引

错误响应统一 ``{"ok": false, "code": ..., "message": ..., "trace_id": ...}``。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.errors import ERROR_MESSAGES, RAGError  # noqa: E402
from app.core.logging_conf import flush_logs, logger, new_trace_id, setup_logging  # noqa: E402

#: 问答请求串行化（Ollama 单实例，避免并发排队把首字延迟拖爆）
_ASK_LOCK = threading.Lock()


def _engine():
    """惰性获取 QAEngine（导入期不加载模型，保证服务秒起）。"""
    from app.core.qa_engine import get_qa_engine

    return get_qa_engine()


INDEX_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>RAG 问答系统（工单2 · 本机备用界面）</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: "Microsoft YaHei", system-ui, sans-serif; margin: 0; display: flex; height: 100vh; }
  aside { width: 320px; padding: 16px; border-right: 1px solid #8884; overflow-y: auto; }
  main { flex: 1; display: flex; flex-direction: column; }
  #chat { flex: 1; overflow-y: auto; padding: 16px; }
  .msg { margin-bottom: 14px; padding: 10px 12px; border-radius: 8px; max-width: 78%; white-space: pre-wrap; }
  .user { background: #2f6feb; color: #fff; margin-left: auto; }
  .bot { background: #8882; }
  .meta { font-size: 12px; color: #888; margin-top: 6px; }
  .cite { font-size: 13px; color: #2f6feb; }
  form { display: flex; gap: 8px; padding: 12px 16px; border-top: 1px solid #8884; }
  input[type=text] { flex: 1; padding: 10px; font-size: 15px; }
  button { padding: 10px 16px; font-size: 15px; cursor: pointer; }
  .pill { display: inline-block; font-size: 12px; padding: 2px 8px; border-radius: 10px; background: #8882; margin: 2px 4px 2px 0; }
  section { margin-bottom: 18px; }
  h3 { font-size: 14px; margin: 10px 0 6px; }
</style>
</head>
<body>
<aside>
  <h2 style="font-size:16px">RAG 文档问答</h2>
  <div class="meta">工单2：人工智能NLP-RAG-基于PDF文档的问答系统优化</div>
  <section>
    <h3>会话</h3>
    <button onclick="newConversation()">新建会话</button>
    <div id="conversations"></div>
  </section>
  <section>
    <h3>健康状态</h3>
    <div id="health">加载中…</div>
  </section>
  <section>
    <h3>统计</h3>
    <div id="stats">—</div>
  </section>
  <section>
    <h3>示例问题</h3>
    <div id="samples"></div>
  </section>
</aside>
<main>
  <div id="chat"></div>
  <form onsubmit="ask(event)">
    <input type="text" id="question" placeholder="请输入问题（中文或英文）" autocomplete="off" />
    <button type="submit">提问</button>
  </form>
</main>
<script>
const SAMPLES = [
  "武汉兴图新科电子股份有限公司的注册资本是多少？",
  "公司的法定代表人是谁？",
  "报告期内军工收入占比分别是多少？",
  "公司的主要客户有哪些？",
  "募集资金用途有哪些？",
  "公司产品需要符合哪些技术标准？",
  "公司获得过哪些科技进步奖？",
  "公司的上游和下游分别是什么行业？",
  "What is the registered capital of the company?",
  "Who is the legal representative?"
];
let conversationId = null;

function el(id) { return document.getElementById(id); }

async function api(path, options) {
  const res = await fetch(path, options);
  const data = await res.json();
  if (!data.ok && data.code) { throw new Error("[" + data.code + "] " + data.message); }
  return data;
}

async function loadHealth() {
  try {
    const data = await api("/api/health");
    const h = data.health || {};
    el("health").innerHTML =
      '<span class="pill">嵌入 ' + (h.embedder?.model || "-") + " / " + (h.embedder?.dimension || "-") + "维</span>" +
      '<span class="pill">向量 ' + (h.index?.count ?? "-") + "</span>" +
      '<span class="pill">BM25 ' + (h.bm25_docs ?? "-") + "</span>" +
      '<span class="pill">LLM ' + (h.llm?.backend?.name || "-") + "</span>" +
      '<span class="pill">重排 ' + (h.reranker?.mode || "-") + "</span>" +
      '<span class="pill">索引 ' + (h.ready ? "就绪" : "未就绪") + "</span>";
  } catch (err) { el("health").textContent = "健康检查失败：" + err.message; }
}

async function loadStats() {
  try {
    const data = await api("/api/stats");
    const t = data.tables || {};
    el("stats").innerHTML = Object.keys(t).map(k => '<span class="pill">' + k + " " + t[k] + "</span>").join("");
  } catch (err) { el("stats").textContent = "统计失败：" + err.message; }
}

async function loadConversations() {
  try {
    const data = await api("/api/conversations");
    el("conversations").innerHTML = (data.items || []).map(c =>
      '<div style="margin:6px 0"><a href="#" onclick="switchConversation(\\'' + c.conversation_id + '\\');return false;">' +
      c.title + "</a> <span class=\\"meta\\">" + c.message_count + "条</span></div>").join("") || "<div class=meta>暂无会话</div>";
  } catch (err) { el("conversations").textContent = err.message; }
}

function switchConversation(id) { conversationId = id; el("chat").innerHTML = ""; loadMessages(); }

async function loadMessages() {
  if (!conversationId) return;
  const data = await api("/api/messages?conversation_id=" + encodeURIComponent(conversationId));
  (data.items || []).forEach(m => addBubble(m.role === "user" ? "user" : "bot", m.content, ""));
}

async function newConversation() {
  const data = await api("/api/conversations", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
  conversationId = data.conversation_id;
  el("chat").innerHTML = "";
  loadConversations();
}

function addBubble(kind, text, meta, citations) {
  const div = document.createElement("div");
  div.className = "msg " + kind;
  div.textContent = text;
  if (citations && citations.length) {
    const c = document.createElement("div");
    c.className = "cite";
    c.textContent = citations.map(x => "[" + (x.page ? "页码: " + x.page : "页码: ?") + "] " + x.chunk_id).join("  ");
    div.appendChild(c);
  }
  if (meta) {
    const m = document.createElement("div");
    m.className = "meta";
    m.textContent = meta;
    div.appendChild(m);
  }
  el("chat").appendChild(div);
  el("chat").scrollTop = el("chat").scrollHeight;
  return div;
}

async function ask(ev) {
  ev.preventDefault();
  const question = el("question").value.trim();
  if (!question) return;
  el("question").value = "";
  addBubble("user", question, "");
  const bubble = addBubble("bot", "", "检索中…");
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question, conversation_id: conversationId, stream: true })
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "", text = "", meta = "", citations = [];
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      const parts = buffer.split("\\n\\n");
      buffer = parts.pop();
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        const payload = JSON.parse(line.slice(5));
        if (payload.event === "status") { meta = payload.data.msg || ""; }
        else if (payload.event === "first_token") { meta = "首字响应 " + Math.round(payload.data.first_token_ms) + " ms"; }
        else if (payload.event === "delta") { text += payload.data.text; bubble.textContent = text; }
        else if (payload.event === "citations") { citations = payload.data.citations || []; }
        else if (payload.event === "error") { text = "⚠ " + payload.data.message; bubble.textContent = text; }
        else if (payload.event === "done") {
          const a = payload.data.answer;
          text = a.answer;
          meta = "模式 " + a.mode + " · 首字 " + Math.round(a.first_token_ms) + " ms · 端到端 " + Math.round(a.total_ms) + " ms";
          citations = a.citations || [];
          if (payload.data.trace_id) { conversationId = conversationId || null; }
        }
        bubble.textContent = text;
        if (meta) {
          let m = bubble.querySelector(".meta");
          if (!m) { m = document.createElement("div"); m.className = "meta"; bubble.appendChild(m); }
          m.textContent = meta;
        }
      }
      el("chat").scrollTop = el("chat").scrollHeight;
    }
    if (citations.length) {
      const c = document.createElement("div");
      c.className = "cite";
      c.textContent = citations.map(x => "[" + "页码: " + x.page + "] " + x.chunk_id).join("  ");
      bubble.appendChild(c);
    }
    loadConversations(); loadStats();
  } catch (err) {
    bubble.textContent = "请求失败：" + err.message;
  }
}

el("samples").innerHTML = SAMPLES.map((q, i) =>
  '<div style="margin:4px 0"><a href="#" onclick="document.getElementById(\\'question\\').value=\\'' + q.replace(/'/g, "") + '\\';return false;">' + (i + 1) + ". " + q + "</a></div>").join("");
loadHealth(); loadStats(); loadConversations();
</script>
</body>
</html>
"""


class RAGHTTPServer(ThreadingHTTPServer):
    """HTTP 服务器。

    显式关闭 ``allow_reuse_address``：Windows 下 ``SO_REUSEADDR`` 允许**两个进程同时绑定同一端口**，
    导致新进程抢走端口、旧进程仍显示 LISTENING，请求随机失败（实测踩坑）。
    """

    allow_reuse_address = False
    daemon_threads = True


class RAGRequestHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理器（标准库）。"""

    server_version = "RAGFallbackUI/2.0"
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        """把访问日志交给结构化 logger（不打印到 stderr 之外的地方）。"""
        logger.debug("app.ui.serve_fallback", "HTTP 访问", client=self.address_string(), detail=fmt % args)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        """返回 JSON。"""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, code: str, message: str, status: int, trace_id: str = "") -> None:
        """统一错误结构。"""
        self._send_json(
            {"ok": False, "code": code, "message": message, "trace_id": trace_id or new_trace_id()}, status=status
        )

    def _read_json(self) -> dict[str, Any]:
        """读取请求体 JSON（容错：空体返回 {}）。"""
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            logger.warning("app.ui.serve_fallback", "请求体不是合法 JSON", body=raw[:200])
            return {}

    # ------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        """GET 路由。"""
        path = urlparse(self.path)
        query = parse_qs(path.query)
        try:
            if path.path in {"/", "/index.html", "/ui"}:
                body = INDEX_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path.path == "/api/health":
                self._send_json({"ok": True, "health": _engine().health()})
                return
            if path.path == "/api/stats":
                self._send_json({"ok": True, "tables": _engine().stats()})
                return
            if path.path == "/api/conversations":
                items = [item.model_dump(mode="json") for item in _engine().list_conversations()]
                self._send_json({"ok": True, "items": items})
                return
            if path.path == "/api/messages":
                conversation_id = (query.get("conversation_id") or [""])[0]
                if not conversation_id:
                    self._send_error_json("CONFIG_ERROR", "缺少 conversation_id", 400)
                    return
                items = [item.model_dump(mode="json") for item in _engine().get_messages(conversation_id)]
                self._send_json({"ok": True, "items": items})
                return
            self._send_error_json("CONFIG_ERROR", f"未知路径: {path.path}", 404)
        except RAGError as exc:
            logger.exception("app.ui.serve_fallback", "GET 业务失败", path=path.path, code=exc.code)
            self._send_error_json(exc.code, exc.user_message, 500)
        except Exception as exc:
            logger.exception("app.ui.serve_fallback", "GET 未预期异常", path=path.path)
            self._send_error_json("RAG_ERROR", str(exc), 500)

    def do_POST(self) -> None:  # noqa: N802
        """POST 路由。"""
        path = urlparse(self.path)
        try:
            if path.path == "/api/conversations":
                self._read_json()
                conversation_id = _engine().new_conversation()
                self._send_json({"ok": True, "conversation_id": conversation_id})
                return
            if path.path == "/api/ask":
                payload = self._read_json()
                question = str(payload.get("question") or "").strip()
                if not question:
                    self._send_error_json("CONFIG_ERROR", "问题不能为空", 400)
                    return
                conversation_id = payload.get("conversation_id") or None
                if payload.get("stream", True):
                    self._stream_ask(question, conversation_id)
                else:
                    with _ASK_LOCK:
                        answer = _engine().ask(question, conversation_id=conversation_id)
                    self._send_json({"ok": True, "answer": answer.model_dump(mode="json")})
                return
            if path.path == "/api/feedback":
                payload = self._read_json()
                feedback_id = _engine().submit_feedback(
                    conversation_id=str(payload.get("conversation_id") or ""),
                    message_id=payload.get("message_id"),
                    rating=str(payload.get("rating") or "up"),
                    comment=str(payload.get("comment") or ""),
                    question=str(payload.get("question") or ""),
                )
                self._send_json({"ok": True, "feedback_id": feedback_id})
                return
            if path.path == "/api/index/build":
                payload = self._read_json()
                pdf = payload.get("pdf") or ""
                with _ASK_LOCK:
                    stats = _engine().build_index(Path(pdf) if pdf else None, reset=bool(payload.get("rebuild", True)))
                self._send_json({"ok": True, "stats": stats})
                return
            self._send_error_json("CONFIG_ERROR", f"未知路径: {path.path}", 404)
        except RAGError as exc:
            logger.exception("app.ui.serve_fallback", "POST 业务失败", path=path.path, code=exc.code)
            self._send_error_json(exc.code, exc.user_message, 500)
        except Exception as exc:
            logger.exception("app.ui.serve_fallback", "POST 未预期异常", path=path.path)
            self._send_error_json("RAG_ERROR", str(exc), 500)

    # ------------------------------------------------------------------
    def _stream_ask(self, question: str, conversation_id: str | None) -> None:
        """SSE 流式问答（每行 ``data: {event, data}\\n\\n``）。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            with _ASK_LOCK:
                for event, payload in _engine().stream(question, conversation_id=conversation_id):
                    data = payload
                    if event == "done":
                        data = {
                            "answer": payload["answer"].model_dump(mode="json"),
                            "trace_id": payload.get("trace_id", ""),
                        }
                    elif event == "citations":
                        data = {"citations": payload.get("citations", [])}
                    line = json.dumps({"event": event, "data": data}, ensure_ascii=False)
                    self.wfile.write(f"data: {line}\n\n".encode("utf-8"))
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.warning("app.ui.serve_fallback", "客户端提前断开连接", question=question[:60])
        except Exception as exc:
            logger.exception("app.ui.serve_fallback", "SSE 流式问答异常")
            try:
                line = json.dumps(
                    {"event": "error", "data": {"code": "RAG_ERROR", "message": str(exc)}}, ensure_ascii=False
                )
                self.wfile.write(f"data: {line}\n\n".encode("utf-8"))
            except Exception:
                logger.exception("app.ui.serve_fallback", "错误事件回写失败")


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""
    parser = argparse.ArgumentParser(description="工单2 备用界面（标准库 http.server）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8600, help="监听端口（0=随机未占用端口）")
    parser.add_argument("--no-warmup", action="store_true", help="跳过预热（默认预热嵌入与 LLM）")
    return parser


def main(argv: list[str] | None = None) -> int:
    """启动服务。"""
    args = build_parser().parse_args(argv)
    setup_logging()
    settings = get_settings()
    engine = _engine()
    loaded = engine.load_index()
    if not args.no_warmup:
        try:
            engine.warmup()
        except Exception:
            logger.exception("app.ui.serve_fallback", "预热失败（不影响服务启动）")
    try:
        httpd = RAGHTTPServer((args.host, args.port), RAGRequestHandler)
    except OSError as exc:
        logger.exception("app.ui.serve_fallback", "端口不可用", host=args.host, port=args.port)
        print(f"启动失败：{exc}", file=sys.stderr)
        return 2
    host, port = httpd.server_address[0], httpd.server_address[1]
    print("=" * 70)
    print("RAG 备用界面已启动（工单2：人工智能NLP-RAG-基于PDF文档的问答系统优化）")
    print(f"地址        : http://{host}:{port}/")
    print(f"索引就绪    : {loaded}")
    print(f"日志目录    : {settings.paths.logs}")
    print("按 Ctrl+C 停止服务")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n收到中断信号，正在停止…")
    finally:
        httpd.server_close()
        flush_logs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
