import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs
from ragkit import answer, demo_documents, load_documents, search
docs=load_documents(sys.argv[1:]) if len(sys.argv)>1 else demo_documents()
class H(BaseHTTPRequestHandler):
 def show(self,text=""): data=("<meta charset=utf-8><h1>Query 理解与多轮问答</h1><form method=post><input name=q style='width:70%;padding:12px'><button>分析并检索</button></form><pre>"+text+"</pre>").encode();self.send_response(200);self.send_header("Content-Type","text/html;charset=utf-8");self.end_headers();self.wfile.write(data)
 def do_GET(self): self.show()
 def do_POST(self):
  q=parse_qs(self.rfile.read(int(self.headers.get("Content-Length",0))).decode()).get("q",[""])[0];self.show(answer(q,search(q,docs,5)))
print("http://127.0.0.1:8005");ThreadingHTTPServer(("127.0.0.1",8005),H).serve_forever()
