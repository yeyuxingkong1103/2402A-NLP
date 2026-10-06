#!/bin/bash
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# scripts/ask_demo_step3.sh —— 工单四 Step3 验收：id5/id6 问答演示
PY=/home/dabaie/code/my_project/.venv/bin/python
cd /home/dabaie/code/工单/工单四

$PY - <<'EOF'
import json, urllib.request

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"
QUESTIONS = [
    (5, "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，"
        "其中大客户销售部有几个销售处构成？", "招股说明书2"),
    (6, "从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？"
        "负增长的是哪个行业？", "招股说明书2"),
]
for qid, q, doc in QUESTIONS:
    body = json.dumps({"question": q, "doc_id": doc,
                       "use_image": True, "use_rag": True}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8004/api/v4/ask", data=body,
        headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=120).read())
    imgs = [(r.get("image_id"), r.get("page"), r.get("ref_id"))
            for r in d.get("references", []) if r.get("type") == "image"]
    tabs = [r.get("ref_id") for r in d.get("references", [])
            if r.get("type") == "table"]
    print(f"\n===== id {qid} =====")
    print("route:", d["route"].get("mode"), round(d["route"].get("confidence", 0), 2))
    print("answer:", d["answer"][:280])
    print("image_refs:", imgs, "| table_refs:", tabs)
    print("latency_ms:", d["latency_ms"], "| breakdown:", d["breakdown"])
    with open(f"/tmp/ask_id{qid}.json", "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
EOF
