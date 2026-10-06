# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 重建生产向量索引缓存 + 验证 Q2/Q9 答案（用后删除）
from dotenv import load_dotenv
load_dotenv()
from loguru import logger
logger.remove()
logger.add(__import__("sys").stderr, level="WARNING")
from src.rag_engine import get_hybrid_retriever, OptimizedEngine

get_hybrid_retriever()  # 触发全量编码并写缓存
eng = OptimizedEngine(top_k=8)
for q in ["公司的法定代表人是谁？", "Who is the legal representative of the company?"]:
    out = eng.ask(q, use_cache=False)
    print(f"Q: {q}")
    print(f"A: {out['answer'][:100]}")
    print(f"   kv在引用中: {any(x.get('kv') for x in out['references'])} | {out['latency_ms']}ms")
