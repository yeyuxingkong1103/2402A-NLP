# -*- coding:utf-8 -*-
"""
工单编号：人工智能NLP‑RAG‑图像内容解析及检索优化
Milvus集合创建、清理脚本
集合名称：rag_waterwork4
CLIP输出向量维度：512
"""
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")
COLLECTION_NAME = "rag_waterwork4"

if client.has_collection(collection_name=COLLECTION_NAME):
    client.drop_collection(collection_name=COLLECTION_NAME)
    print(f"✅ 旧集合 {COLLECTION_NAME} 已删除")

client.create_collection(
    collection_name=COLLECTION_NAME,
    dimension=512,
    auto_id=True
)
print(f"✅ 集合 {COLLECTION_NAME} 创建完成！auto_id开启")
