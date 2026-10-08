# -*- coding: utf-8 -*-
"""工单18：质检 API 服务。

新增端点：POST /v1/document/quality-inspection
  请求体：{"path": "文件夹路径"} 或 {"files": ["文件1", ...]}
          可选 "format": "html" 返回 HTML 简报，默认 JSON
  返回：结构化质检报告（JSON 及 HTML 简报），含全部"待确认/待审核"列表
零 Web 框架依赖（http.server），生产中可平移到 RAGFlow Flask/FastAPI 蓝图。
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from document_quality_assessment import DocumentQualityAssessment
from document_quality_assessment.report import render_html

DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"
DEFAULT_PORT = 8380


class QualityHandler(BaseHTTPRequestHandler):
    def _send(self, code: int, body, content_type: str):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            self._send(200, json.dumps({"status": "ok", "skill": "document-quality-assessment"}),
                       "application/json; charset=utf-8")
        else:
            self._send(404, '{"error":"not found"}', "application/json; charset=utf-8")

    def do_POST(self):
        if self.path.split("?")[0] != "/v1/document/quality-inspection":
            self._send(404, '{"error":"not found"}', "application/json; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except Exception as e:
            self._send(400, json.dumps({"error": f"请求体解析失败: {e}"}),
                       "application/json; charset=utf-8")
            return

        want_html = payload.get("format") == "html"
        try:
            skill = DocumentQualityAssessment(payload.get("config"))
            result = skill.assess(
                target=payload.get("path"),
                files=payload.get("files"),
                output_dir=str(payload.get("output_dir") or DEFAULT_RESULTS),
                resume=bool(payload.get("resume", True)),
            )
        except Exception as e:
            self._send(500, json.dumps({"error": f"质检执行失败: {e}"}),
                       "application/json; charset=utf-8")
            return

        if want_html:
            self._send(200, render_html(result.to_dict()), "text/html; charset=utf-8")
        else:
            self._send(200, json.dumps(result.to_dict(), ensure_ascii=False),
                       "application/json; charset=utf-8")

    def log_message(self, fmt, *args):  # 精简日志
        sys.stderr.write("[api] " + fmt % args + "\n")


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    server = ThreadingHTTPServer(("0.0.0.0", port), QualityHandler)
    print(f"质检 API 服务已启动: http://0.0.0.0:{port}")
    print("端点: POST /v1/document/quality-inspection")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
