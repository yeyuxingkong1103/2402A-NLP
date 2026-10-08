# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# scripts/probe_rag_tables_step5.py —— 探查 rag_tables 行数与检索行为（临时诊断）
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")
st = client.describe_collection("rag_tables")
print("num_entities:", st.get("num_entities"), "| fields:", [f["name"] for f in st.get("fields", [])])

# 无过滤检索测试
try:
    res = client.search(collection_name="rag_tables", data=[[0.0] * 1024],
                        limit=3, output_fields=["doc_id", "table_id"])
    print("random-vec search hits:", len(res[0]) if res else 0)
    for hit in (res[0] if res else [])[:3]:
        print("  ", hit.get("distance"), hit.get("entity", {}).get("doc_id"))
except Exception as e:
    print("search error:", type(e).__name__, str(e)[:200])

# query 全量抽样
try:
    q = client.query(collection_name="rag_tables", filter="", limit=3,
                     output_fields=["doc_id", "table_id", "page"])
    print("query sample:", len(q))
    for row in q[:3]:
        print("  ", {k: row.get(k) for k in ("doc_id", "table_id", "page")})
except Exception as e:
    print("query error:", type(e).__name__, str(e)[:200])
