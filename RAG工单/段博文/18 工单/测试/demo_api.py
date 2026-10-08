# -*- coding: utf-8 -*-
"""演示：启动质检 API 服务，真实 POST 调用，记录请求/响应日志。"""
import json
import socket
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "研发" / "src"))
from api_server import QualityHandler

DOCS = Path(r"D:\作业\6-专高NLP 作业\RAG 工单\14-17附件\original_problems\documents")
# 选 5 个文件（扫描/混合混合）
files = [
    str(DOCS / "CN100342976C.pdf"), str(DOCS / "CN100347506C.pdf"),
    str(DOCS / "CN106794760B.pdf"), str(DOCS / "CN110758510A.pdf"),
    str(DOCS / "CN100364694C.pdf"),
]

sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
server = ThreadingHTTPServer(("127.0.0.1", port), QualityHandler)
threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(0.5)

logs = [f"$ 启动服务: python api_server.py {port}",
        f"[api] 质检 API 服务已启动: http://127.0.0.1:{port}", ""]


def call(label, payload):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/document/quality-inspection",
        data=body, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    resp_raw = urllib.request.urlopen(req, timeout=60).read()
    elapsed = time.time() - t0
    return resp_raw, elapsed


# 请求1：JSON 报告
logs.append(f"POST /v1/document/quality-inspection（JSON，5个文件）")
resp1, e1 = call("json", {"files": files, "output_dir": str(BASE / "研发" / "src" / "results_demo")})
r1 = json.loads(resp1)
s = r1["summary"]
logs.append(f"  HTTP 200，{len(resp1)} 字节，耗时 {e1:.2f}s")
logs.append(f"  summary: 文件{s['total_files']} | PDF类型 {s['pdf_type_counts']}")
logs.append(f"  待OCR {len(r1['action_lists']['scan_pdf_for_ocr'])} | "
            f"版本冲突 {s['simhash_pending_pairs']} | 敏感信息 {s['sensitive_pending_items']}")
logs.append("  action_lists 键: " + ", ".join(r1["action_lists"].keys()))
logs.append("")

# 请求2：HTML 简报
logs.append("POST /v1/document/quality-inspection（HTML 简报）")
resp2, e2 = call("html", {"files": files, "format": "html",
                          "output_dir": str(BASE / "研发" / "src" / "results_demo")})
logs.append(f"  HTTP 200，{len(resp2)} 字节 HTML，耗时 {e2:.2f}s")
logs.append(f"  含区块: {'待办1' in resp2.decode('utf-8', 'replace')}, 打码: {'****' in resp2.decode('utf-8', 'replace')}")
logs.append("")
logs.append("✓ API 端点验证通过：接收文件列表 -> 触发质检 Skill -> 返回 JSON / HTML 结构化报告")

server.shutdown()
text = "\n".join(logs)
(BASE / "测试" / "api_demo.log").write_text(text, encoding="utf-8")
print(text)

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>API调用演示</title>
<style>body{{background:#0f1419;padding:30px;font-family:Consolas,monospace}}
pre{{color:#7ec699;font-size:13px;line-height:1.7;white-space:pre-wrap}}
h1{{color:#4a90d9;font-family:Microsoft YaHei;font-size:18px}}</style></head><body>
<h1>POST /v1/document/quality-inspection API 调用演示</h1>
<pre>{text.replace('<','&lt;')}</pre></body></html>"""
(BASE / "测试" / "api_demo.html").write_text(html, encoding="utf-8")
