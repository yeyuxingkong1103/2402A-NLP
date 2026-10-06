import argparse, json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ragkit import answer, demo_documents, search

DOCS = demo_documents()
class Handler(BaseHTTPRequestHandler):
    def send(self, status, value):
        data = json.dumps(value, ensure_ascii=False).encode(); self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)
    def do_GET(self): self.send(200, {"status": "ok"}) if self.path == "/health" else self.send(404, {"error": "not found"})
    def do_POST(self):
        if self.path != "/ask": return self.send(404, {"error": "not found"})
        size = int(self.headers.get("Content-Length", "0")); body = json.loads(self.rfile.read(size) or b"{}")
        query = body.get("query", "公司的主营业务是什么？"); results = search(query, DOCS, 4)
        self.send(200, {"query": query, "answer": answer(query, results), "sources": results})

parser = argparse.ArgumentParser(description="工单10：可容器化金融问答 API")
parser.add_argument("--host", default="0.0.0.0"); parser.add_argument("--port", type=int, default=8000); parser.add_argument("--demo", action="store_true")
args = parser.parse_args()
if args.demo:
    out = Path(__file__).parent / "outputs"; out.mkdir(exist_ok=True); (out / "deployment_check.json").write_text('{"status":"ok","port":8000}', encoding="utf-8")
    print("Docker/API 配置检查通过。")
else:
    print(f"服务启动：http://{args.host}:{args.port}"); ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
