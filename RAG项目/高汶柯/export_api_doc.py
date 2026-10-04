"""导出后端 OpenAPI 文档。

输出到独立文件，不覆盖手写的《接口文档.md》：
    api_doc_generated.json
    api_doc_generated.md
"""
import json
import os

import requests

BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

data = requests.get(f"{BASE}/openapi.json", timeout=30).json()

with open("api_doc_generated.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

md = ["# RAG 多角色扮演系统 - 接口文档（自动生成）\n"]
md.append(f"**版本**: {data['info'].get('version', '1.0.0')}\n")
md.append(f"**标题**: {data['info'].get('title', '')}\n")
md.append("> 本文件由 export_api_doc.py 自动生成；手写维护版见《接口文档.md》。\n")

for path, methods in data["paths"].items():
    for method, detail in methods.items():
        md.append(f"\n## {method.upper()} {path}\n")
        md.append(f"**说明**: {detail.get('summary', '')}\n")
        if "requestBody" in detail:
            md.append("**请求体**:\n```json\n")
            schema = detail["requestBody"]["content"]["application/json"]["schema"]
            md.append(json.dumps(schema, ensure_ascii=False, indent=2))
            md.append("\n```\n")
        md.append("**响应**:\n```json\n")
        md.append('{"reply": "回答内容"}\n')
        md.append("```\n")

with open("api_doc_generated.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print("✅ 已生成 api_doc_generated.json 和 api_doc_generated.md（未改动 接口文档.md）")
