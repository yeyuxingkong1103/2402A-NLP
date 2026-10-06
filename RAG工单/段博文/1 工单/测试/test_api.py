# -*- coding: utf-8 -*-
"""临时测试脚本：验证 /api/chat 问答是否正常"""
import httpx
import json

QUESTION = "武汉兴图新科电子股份有限公司注册资本是多少？"

body = {"query": QUESTION, "top_k": 5, "stream": False}
resp = httpx.post("http://localhost:8000/api/chat", json=body, timeout=120)
data = resp.json()

print("问题:", data.get("query", ""))
print("回答:", data.get("answer", "")[:300])
print("参考条数:", len(data.get("references", [])))
for i, r in enumerate(data.get("references", [])[:5], 1):
    score = r.get("score", 0)
    src = r.get("source", "")
    pg = r.get("page_number", 0)
    content = r.get("page_content", "")[:80]
    print(f"  [{i}] score={score:.4f} source={src} page={pg}")
    print(f"      {content}")
