# 工单编号：人工智能NLP-RAG-LightRAG优化
"""工单编号：人工智能NLP-RAG-基于PDF 文档的问答系统。仅本机使用的 HTTP 服务。"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .core import DOCS, ROOT, KnowledgeBase, build_index, llm_configured


class AppState:
    def __init__(self):
        self.kb = KnowledgeBase()
        self.lock = threading.RLock()
        self.rebuilding = False
        self.error = None
        self.log_lock = threading.Lock()

    def rebuild(self):
        try:
            # 单个后台构建任务；在交换知识库前旧索引继续提供查询。
            build_index(self.kb.embedder)
            new_kb = KnowledgeBase()
            with self.lock:
                self.kb = new_kb
                self.error = None
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.rebuilding = False


def serve(host="127.0.0.1", port=8765):
    state = AppState()

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, payload, code=200):
            content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/health":
                return self.send_json({"ok": True, "rebuilding": state.rebuilding, "error": state.error,
                    "llm_configured": llm_configured(), "manifest": state.kb.manifest,
                    "active_backend": "BGE + TF-IDF" if state.kb.dense is not None else "TF-IDF",
                    "embedding_status": state.kb.embedder.reason})
            if path == "/api/graph":
                return self.send_json(state.kb.graph)
            if path == "/api/documents":
                return self.send_json(state.kb.manifest["documents"])
            if path.startswith("/pdf/"):
                name = unquote(path[5:])
                candidate = (DOCS / name).resolve()
                if candidate.parent != DOCS.resolve() or not candidate.exists() or candidate.suffix.lower() != ".pdf":
                    return self.send_json({"error": "文档不存在"}, 404)
                content, content_type = candidate.read_bytes(), "application/pdf"
            elif path in {"/", "/index.html"}:
                content, content_type = (ROOT / "static/index.html").read_bytes(), "text/html; charset=utf-8"
            else:
                return self.send_json({"error": "接口不存在"}, 404)
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):
            try:
                path = urlparse(self.path).path
                # 本机 Web 应用拒绝跨站脚本写入。
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
                    return self.send_json({"error": "不允许跨站请求"}, 403)
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > 40 * 1024 * 1024:
                    return self.send_json({"error": "请求体必须为 1 字节至 40 MB"}, 413)
                raw = self.rfile.read(size)
                if path == "/api/upload":
                    name = parse_qs(urlparse(self.path).query).get("name", [""])[0]
                    if not name or Path(name).name != name or "/" in name or "\\" in name or not name.lower().endswith(".pdf"):
                        return self.send_json({"error": "请选择名称合法的 PDF 文件"}, 400)
                    if not raw.startswith(b"%PDF-"):
                        return self.send_json({"error": "文件内容不是 PDF"}, 400)
                    import pymupdf
                    try:
                        with pymupdf.open(stream=raw, filetype="pdf") as doc:
                            if doc.needs_pass or not len(doc) or len(doc) > 1200:
                                return self.send_json({"error": "PDF 已加密、为空或超过页数上限"}, 400)
                    except Exception:
                        return self.send_json({"error": "PDF 无法解析"}, 400)
                    with state.lock:
                        if state.rebuilding:
                            return self.send_json({"error": "已有知识库构建任务，请稍后"}, 409)
                        if (DOCS / name).exists():
                            return self.send_json({"error": "同名文件已存在，请更名后上传"}, 409)
                        (DOCS / name).write_bytes(raw)
                        state.rebuilding = True
                        state.error = None
                    threading.Thread(target=state.rebuild, daemon=True).start()
                    return self.send_json({"message": "PDF 已上传，正在后台构建知识库"}, 202)
                data = json.loads(raw)
                if path == "/api/report":
                    result = state.kb.ask("根据平安银行和招商银行2019年年报，按经营概况、风险管理、数字化创新三部分生成金融研报摘要。仅依据检索证据，未知内容明确指出。", strategy="graph_optimized", use_cache=False)
                    return self.send_json(result)
                if path == "/api/settings":
                    state.kb.query_settings.update({k:v for k,v in data.items() if k in {"strategy","reranker","weight","field","fuzzy"}})
                    state.kb.cache.clear()
                    return self.send_json(state.kb.query_settings)
                if path == "/api/ask":
                    with state.lock:
                        kb = state.kb
                    result = kb.ask(data.get("question", ""), data.get("mode", "rag"), strategy=data.get("strategy", "hybrid"), session_id=data.get("session_id", "browser"))
                    return self.send_json(result)
                if path == "/api/feedback":
                    if data.get("rating") not in {"up", "down"}:
                        raise ValueError("反馈必须为 up 或 down")
                    entry = {"time": time.time(), "rating": data["rating"], "question": str(data.get("question", ""))[:1200],
                             "comment": str(data.get("comment", ""))[:2000]}
                    log = ROOT / "data/logs/feedback.jsonl"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    with state.log_lock, log.open("a", encoding="utf-8") as out:
                        out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    for h in state.kb.chunks:
                        if str(data.get("source_id", "")) == h["id"]:
                            state.kb.rerank_feedback[h["id"]] += 1 if data["rating"] == "up" else -1
                    return self.send_json({"message": "反馈已记录"})
                return self.send_json({"error": "接口不存在"}, 404)
            except (ValueError, TypeError, AttributeError) as exc:
                return self.send_json({"error": str(exc)}, 400)
            except Exception as exc:
                print(f"服务错误：{type(exc).__name__}: {exc}", flush=True)
                return self.send_json({"error": "服务内部异常，请查看终端日志"}, 500)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"系统就绪：http://{host}:{port}；模型生成：{'已配置' if llm_configured() else '未配置，当前展示检索证据'}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
