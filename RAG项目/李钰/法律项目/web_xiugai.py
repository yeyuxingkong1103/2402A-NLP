# -*- coding: utf-8 -*-
"""
Web启动模块（多角色版 + 对话历史持久化）
运行：python web_xiugai.py  然后访问 http://localhost:8000/
"""
import json
import threading
import time
import traceback
import webbrowser
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import config as C
from logger import get_logger, tail_logs

log = get_logger("web")

# ======================== 配置 ========================
HOST = getattr(C, "WEB_HOST", "127.0.0.1")
PORT = getattr(C, "WEB_PORT", 8000)
SESSION_ID = getattr(C, "WEB_SESSION_ID", "default")
HTML_FILE = Path(__file__).parent / "rag_hy.html"
DATA_DIR = Path(__file__).parent / "data" / "conversations"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 兜底角色（当 role_manager 缺失时使用）
DEFAULT_ROLES = [
    {"id": "lawyer_friend", "name": "金牌律师", "type": "律师朋友",
     "greeting": "嘿，我在。先别急，把情况慢慢说给我听。"},
    {"id": "legal_qa", "name": "法律知识库", "type": "全能法律知识库",
     "greeting": "请说出你想要了解的法律知识吧！"},
    {"id": "case_analyst", "name": "案例分析师", "type": "类案检索与裁判观点分析",
     "greeting": "你好，我是案例分析师。请把案情、争议焦点和证据情况告诉我。"},
]


# ======================== 对话存储 ========================
class ConvStore:
    """文件型对话存储：data/conversations/<sid>.json"""

    def __init__(self, d: Path):
        self.dir = Path(d)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock() # 创建可重入锁，用于多线程下安全读写文件

    def _p(self, sid: str) -> Path:
        safe = "".join(c for c in str(sid) if c.isalnum() or c in "-_")[:80] or "default"
        return self.dir / f"{safe}.json"

    def load(self, sid):
        with self.lock:
            p = self._p(sid)  # 获取会话文件路径
            if not p.exists():
                return None
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None

    def save(self, c): # 保存会话字典
        with self.lock:
            self._p(c["session_id"]).write_text(
                json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
                            # ensure_ascii=False 保留中文；indent=2 格式化缩进
    def delete(self, sid):  # 删除指定会话
        with self.lock:
            p = self._p(sid)
            if p.exists():
                p.unlink() # 删除文件
                return True
            return False
                                                           # 附加元数据
    def append(self, sid, role_id, user_id, role, content, meta=None):
        with self.lock:
            c = self.load(sid) or {
                "session_id": sid, "role_id": role_id, "user_id": user_id,
                "title": "", "created_at": datetime.now().isoformat(timespec="seconds"),
                "updated_at": "", "messages": [],
            }
            if role_id: c["role_id"] = role_id
            if user_id: c["user_id"] = user_id
            msg = {"role": role, "content": content,
                   "ts": datetime.now().isoformat(timespec="seconds")}
            if meta: msg.update(meta) # 如果存在元数据，则合并到消息字典中
            c["messages"].append(msg)
            c["updated_at"] = msg["ts"]
            if not c.get("title") and role == "user":
                t = content.strip().replace("\n", " ")
                c["title"] = (t[:24] + ("…" if len(t) > 24 else "")) or "新对话"
            self.save(c)
            return c

    def list_all(self):
        with self.lock:
            items = []
            for p in self.dir.glob("*.json"): # 匹配 JSON 文件
                if p.name.startswith("_"): continue
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                items.append({
                    "session_id": d.get("session_id", p.stem),
                    "title": d.get("title") or "新对话",
                    "role_id": d.get("role_id", ""),
                    "updated_at": d.get("updated_at", ""),
                    "created_at": d.get("created_at", ""),
                    "message_count": len(d.get("messages", [])),
                })
            items.sort(key=lambda x: x.get("updated_at") or x.get("created_at", ""), reverse=True)
            return items


store = ConvStore(DATA_DIR)

# ======================== 全局状态 ========================
rag_chain = None
role_manager = None
rag_lock = threading.Lock()
init_status = {"ready": False, "error": None, "step": "等待初始化"}


def _roles():
    if role_manager is not None:
        try:
            rs = role_manager.list_roles()
            if rs: return rs
        except Exception as e:
            log.warning("[Web] role_manager.list_roles 失败: %s", e)
    return DEFAULT_ROLES


def init_rag():
    global rag_chain, role_manager
    try:
        init_status["step"] = "加载角色模块..."
        try:
            from role_manager import RoleManager
            role_manager = RoleManager()
            log.info("[Web] 已加载角色: %s", list(role_manager.roles.keys()))
        except Exception as e:
            log.warning("[Web] role_manager 未加载: %s", e)
            role_manager = None

        init_status["step"] = "加载RAG..."
        from rag_chain import RAGChain
        init_status["step"] = "连接 Milvus / Ollama..."
        rag_chain = RAGChain()
        init_status["ready"] = True
        init_status["step"] = "就绪"
        log.info("[Web] RAG初始化完成")
    except Exception as e:
        init_status["error"] = str(e)
        init_status["step"] = f"初始化失败: {e}"
        log.exception("[Web] RAG初始化失败: %s", e)


def _invoke(q, sid, rid, uid):
    """兼容旧/新 RAGChain.invoke 签名"""
    try:
        return rag_chain.invoke(q, sid, rid, uid)
    except TypeError:
        return rag_chain.invoke(q, sid)


# ======================== HTTP 处理 ========================
class Handler(BaseHTTPRequestHandler):
    # 关闭默认 stderr 噪声日志，统一走我们的 logger
    def log_message(self, fmt, *args):
        pass

    # -------- 请求计时与日志 --------
    def _start(self):
        self._t0 = time.time()

    def _end(self, code, extra=""):
        dt = (time.time() - getattr(self, "_t0", time.time())) * 1000
        log.info("[HTTP] %s %s -> %d %.1fms%s",
                 self.command, self.path, code, dt,
                 (" " + extra) if extra else "")

    # -------- 输出 --------
    def _json(self, code, data):
        b = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)
        self._end(code)

    def _html(self, html):
        b = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
        self._end(200)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    # -------- GET --------
    def do_GET(self):
        self._start()
        u = urlparse(self.path)
        p = u.path
        q = parse_qs(u.query)
        try:
            if p in ("/", ""):
                try:
                    self._html(HTML_FILE.read_text(encoding="utf-8"))
                except FileNotFoundError:
                    self._html(f"<h1>{HTML_FILE.name} 缺失</h1>")
            elif p == "/status":
                self._json(200, init_status)
            elif p == "/roles":
                self._json(200, _roles())
            elif p == "/conversations":
                self._json(200, store.list_all())
            elif p.startswith("/conversations/"):
                c = store.load(p[len("/conversations/"):])
                if c:
                    self._json(200, c)
                else:
                    self._json(404, {"error": "未找到"})
            elif p == "/logs":
                n = int((q.get("n") or [C.LOG_TAIL_DEFAULT])[0])
                lvl = (q.get("level") or [None])[0]
                lines = tail_logs(n=n, level=lvl)
                self._json(200, {
                    "file": str(C.LOG_FILE),
                    "count": len(lines),
                    "level": lvl,
                    "lines": lines,
                })
            elif p == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                self._end(204)
            else:
                self.send_error(404)
                self._end(404)
        except Exception as e:
            log.exception("[HTTP] GET %s 异常: %s", p, e)
            self._json(500, {"error": str(e)})

    # -------- POST --------
    def do_POST(self):
        self._start()
        p = self.path.split("?")[0]
        if p != "/ask":
            self.send_error(404)
            self._end(404)
            return
        if not init_status["ready"] or rag_chain is None:
            self._json(503, {"error": "系统未就绪", "step": init_status["step"]})
            return
        try:
            data = self._body()
            q = (data.get("question") or "").strip()
            if not q:
                self._json(400, {"error": "问题不能为空"})
                return
            rid = data.get("role_id") or getattr(C, "DEFAULT_ROLE", "lawyer_friend")
            sid = data.get("session_id") or SESSION_ID
            uid = data.get("user_id") or "anonymous"
            log.info("[/ask] uid=%s role=%s sid=%s q=%s", uid, rid, sid, q[:80])

            try:
                store.append(sid, rid, uid, "user", q)
            except Exception as e:
                log.warning("[Store] 存用户消息失败: %s", e)

            with rag_lock:
                result = _invoke(q, sid, rid, uid)

            result.setdefault("role", {"id": rid, "name": rid})
            result.setdefault("sources", [])

            try:
                srcs = [{
                    "source": s.get("source", ""),
                    "text": (s.get("text") or "")[:200],
                    "score": s.get("score", 0),
                } for s in (result.get("sources") or [])[:5]]
                store.append(sid, rid, uid, "assistant",
                             result.get("answer", ""), meta={"sources": srcs})
            except Exception as e:
                log.warning("[Store] 存助手消息失败: %s", e)

            result["session_id"] = sid
            self._json(200, result)
        except Exception as e:
            log.exception("[/ask] 处理异常: %s", e)
            self._json(500, {"error": str(e), "answer": f"处理出错: {e}"})

    # -------- DELETE --------
    def do_DELETE(self):
        self._start()
        p = self.path.split("?")[0]
        try:
            if p.startswith("/conversations/"):
                sid = p[len("/conversations/"):]
                ok = store.delete(sid)
                log.info("[DELETE] /conversations/%s ok=%s", sid, ok)
                self._json(200, {"ok": ok})
            else:
                self.send_error(404)
                self._end(404)
        except Exception as e:
            log.exception("[DELETE] %s 异常: %s", p, e)
            self._json(500, {"error": str(e)})


# ======================== 入口 ========================
def main():
    log.info("=" * 56)
    log.info("  法律RAG问答系统 · Web 服务")
    log.info("  http://%s:%s/", HOST, PORT)
    log.info("  对话存储: %s", DATA_DIR)
    log.info("  日志文件: %s", C.LOG_FILE)
    log.info("=" * 56)
    print(f"法律RAG问答系统启动中... 访问 http://{HOST}:{PORT}/  日志: {C.LOG_FILE}")

    threading.Thread(target=init_rag, daemon=True).start()
    server = HTTPServer((HOST, PORT), Handler)
    log.info("[Web] 服务器已启动: http://%s:%s/", HOST, PORT)

    threading.Thread(target=webbrowser.open,
                     args=(f"http://{HOST}:{PORT}/",), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("[Web] 服务器已停止")
        server.shutdown()


if __name__ == "__main__":
    main()