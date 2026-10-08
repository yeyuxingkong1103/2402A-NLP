# -*- coding: utf-8 -*-
# 工单17：模拟API服务（有bug版+优化版）
"""
模拟RAGFlow API服务的两个版本：
- BuggyServer: 有并发瓶颈和资源泄漏（每请求重复初始化、无连接池、无限流）
- OptimizedServer: 优化后（单例模式、连接池、信号量限流、异步队列）

用Python http.server实现，无需额外依赖。
"""
import gc
import json
import random
import threading
import time
import weakref
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs


# ====== 模拟组件 ======

class MockModel:
    """模拟重量级模型（DeepDoc/VLM），初始化耗时。"""
    def __init__(self, name="model"):
        self.name = name
        time.sleep(0.05)  # 模拟初始化耗时50ms
        self.weights = [0.0] * 10000  # 模拟内存占用

    def infer(self, text):
        time.sleep(0.01)  # 模拟推理10ms
        return f"answer_for_{text[:20]}"


class MockDBPool:
    """模拟数据库连接池。"""
    def __init__(self, max_conn=10):
        self.max_conn = max_conn
        self.active = 0
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            self.active += 1
            return self.active

    def release(self):
        with self._lock:
            self.active = max(0, self.active - 1)


# ====== 有Bug的服务器（基线）======

class BuggyServer:
    """问题：每请求重复初始化模型、无连接池管理、无限流。"""
    def __init__(self):
        self.request_count = 0
        self._leaked_objects = []  # 模拟内存泄漏

    def handle_query(self, question):
        # Bug1: 每次请求都创建新模型实例
        model = MockModel("buggy_model")
        # Bug2: 结果缓存到全局列表不释放（泄漏）
        self._leaked_objects.append({"q": question, "result": model.infer(question), "model": model})
        # Bug3: 模拟DB连接未归还
        # 无连接池管理
        time.sleep(0.02)  # 模拟检索耗时
        self.request_count += 1
        return {"answer": model.infer(question), "count": self.request_count}

    def handle_upload(self, filename):
        # Bug: 文档解析阻塞API线程
        model = MockModel("deepdoc_parser")
        time.sleep(0.1)  # 模拟解析耗时100ms
        self._leaked_objects.append({"file": filename, "model": model})
        return {"status": "parsed", "file": filename}

    def mem_usage_mb(self):
        return 50.0 + len(self._leaked_objects) * 0.08  # 基础50MB + 每个泄漏对象80KB


# ====== 优化后的服务器 ======

class OptimizedServer:
    """优化：模型单例、连接池、信号量限流、异步队列、GC。"""
    def __init__(self):
        self.request_count = 0
        self._model = MockModel("singleton_model")  # 单例初始化一次
        self._parser = MockModel("singleton_deepdoc")  # 解析器单例
        self._db_pool = MockDBPool(max_conn=20)  # 连接池
        self._semaphore = threading.Semaphore(10)  # 限流：最多10并发
        self._cache = {}  # 查询缓存
        self._cache_max = 100  # 缓存上限（防泄漏）
        self._lock = threading.Lock()

    def handle_query(self, question):
        with self._semaphore:  # 限流
            # 缓存命中
            cache_key = hash(question)
            if cache_key in self._cache:
                self.request_count += 1
                return {"answer": self._cache[cache_key], "count": self.request_count, "cached": True}

            conn_id = self._db_pool.acquire()
            try:
                result = self._model.infer(question)
                time.sleep(0.015)  # 检索耗时
                # 缓存管理（有上限，LRU淘汰）
                with self._lock:
                    if len(self._cache) >= self._cache_max:
                        self._cache.pop(next(iter(self._cache)))
                    self._cache[cache_key] = result
                self.request_count += 1
                return {"answer": result, "count": self.request_count, "conn": conn_id}
            finally:
                self._db_pool.release()  # 确保连接归还

    def handle_upload(self, filename):
        with self._semaphore:
            conn_id = self._db_pool.acquire()
            try:
                result = self._parser.infer(filename)
                time.sleep(0.08)
                self.request_count += 1
                return {"status": "parsed", "file": filename, "conn": conn_id}
            finally:
                self._db_pool.release()

    def mem_usage_mb(self):
        gc.collect()
        return 10.0 + len(self._cache) * 0.001  # 基础10MB + 缓存极小占用


# ====== HTTP Handler 工厂 ======

def make_handler(server_instance, server_type="buggy"):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # 静默日志

        def do_POST(self):
            path = urlparse(self.path).path
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8") if content_len else "{}"
            data = json.loads(body) if body else {}

            t0 = time.perf_counter()
            if "chat" in path or "completion" in path:
                question = data.get("question", data.get("message", "test"))
                result = server_instance.handle_query(question)
            elif "upload" in path:
                result = server_instance.handle_upload(data.get("file", "test.pdf"))
            else:
                result = {"error": "unknown endpoint"}

            elapsed = time.perf_counter() - t0
            result["elapsed_sec"] = round(elapsed, 4)
            result["server_type"] = server_type

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/health":
                result = {"status": "ok", "mem_mb": round(server_instance.mem_usage_mb(), 2),
                          "requests": server_instance.request_count}
            else:
                result = {"error": "not found"}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
    return Handler


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def run_server(server_type="buggy", port=0):
    """运行服务器，返回端口。port=0自动分配。"""
    if server_type == "buggy":
        instance = BuggyServer()
    else:
        instance = OptimizedServer()
    server = ThreadedHTTPServer(("127.0.0.1", port), make_handler(instance, server_type))
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, instance, port


if __name__ == "__main__":
    import sys
    stype = sys.argv[1] if len(sys.argv) > 1 else "buggy"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    server, instance, actual_port = run_server(stype, port)
    print(f"{stype} server running on port {actual_port}")
    print(f"Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.shutdown()
