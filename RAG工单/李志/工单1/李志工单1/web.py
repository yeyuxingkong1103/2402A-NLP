import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs
from ragkit import answer, demo_documents, load_documents, search

docs = load_documents(sys.argv[1:]) if len(sys.argv) > 1 else demo_documents()
page = """<!doctype html><meta charset=utf-8><title>PDF RAG</title><style>body{max-width:900px;margin:50px auto;font:16px sans-serif;background:#f4f7fb;color:#14213d}main{background:white;padding:30px;border-radius:16px}input{width:75%;padding:12px}button{padding:12px 20px;background:#246bfd;color:white;border:0}pre{white-space:pre-wrap;background:#edf3ff;padding:20px}</style><main><h1>PDF 文档问答</h1><form method=post><input name=q autofocus><button>提问</button></form>__RESULT__</main>"""
class Handler(BaseHTTPRequestHandler):
    def render(self, result=""):
        data=page.replace("__RESULT__",result).encode(); self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.end_headers(); self.wfile.write(data)
    def do_GET(self): self.render()
    def do_POST(self):
        query=parse_qs(self.rfile.read(int(self.headers.get("Content-Length",0))).decode()).get("q",[""])[0]; rows=search(query,docs,5)
        sources="\n".join(f"[{x['source']} 第{x['page']}页]" for x in rows); self.render(f"<pre>{answer(query,rows)}\n\n{sources}</pre>")
print("http://127.0.0.1:8001"); ThreadingHTTPServer(("127.0.0.1",8001),Handler).serve_forever()
