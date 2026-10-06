# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# scripts/load_milvus_collections_step5.py —— 检查并加载 Milvus 全部 collection（运维脚本）
# 背景：WSL 崩溃后 Milvus 重启，collection 处于卸载状态，table_only 路由检索空结果
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")
names = client.list_collections()
print("collections:", names)
for name in names:
    state = client.get_load_state(name)
    try:
        rows = client.get_collection_stats(name).get("row_count")
    except Exception:
        rows = "?"
    print(f"  {name}: {state.get('state')} rows={rows}")
    if str(state.get("state")) != "Loaded":
        client.load_collection(name)
        print(f"    -> loaded: {client.get_load_state(name).get('state')}")
print("DONE")
