"""工单10：金融问答系统部署的可运行核心代码。"""
from http.server import BaseHTTPRequestHandler
import json

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        data = {"status": "ok", "service": "financial-qa"}
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

if __name__ == "__main__":
    from http.server import ThreadingHTTPServer
    print("服务运行在 http://127.0.0.1:8000")
    ThreadingHTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
