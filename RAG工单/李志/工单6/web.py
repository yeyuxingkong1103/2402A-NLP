import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs
from ragkit import demo_documents, load_documents, search
docs=load_documents(sys.argv[1:]) if len(sys.argv)>1 else demo_documents()
class H(BaseHTTPRequestHandler):
 def show(self,text=""): data=("<meta charset=utf-8><h1>BM25 / 向量 / 混合检索</h1><form method=post><input name=q style='width:60%;padding:12px'><select name=mode><option>hybrid</option><option>bm25</option><option>vector</option></select><button>检索</button></form><pre>"+text+"</pre>").encode();self.send_response(200);self.send_header("Content-Type","text/html;charset=utf-8");self.end_headers();self.wfile.write(data)
 def do_GET(self): self.show()
 def do_POST(self):
  form=parse_qs(self.rfile.read(int(self.headers.get("Content-Length",0))).decode());q=form.get("q",[""])[0];mode=form.get("mode",["hybrid"])[0];self.show("\n\n".join(f"{x['score']} {x['source']} 第{x['page']}页\n{x['text']}" for x in search(q,docs,5,mode)))
print("http://127.0.0.1:8006");ThreadingHTTPServer(("127.0.0.1",8006),H).serve_forever()
