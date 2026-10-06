#!/usr/bin/env python3
"""验证恢复"""
import os, sys
sys.path.insert(0, "/home/dabaie/code/工单/工单一")
os.chdir("/home/dabaie/code/工单/工单一")

from dotenv import load_dotenv; load_dotenv()

for mod in ["pdf_parser", "chunker", "embedding", "vector_store", "db", "models",
            "query_understanding", "llm_client", "retriever", "rag_engine", "schemas", "api"]:
    try:
        m = __import__(f"src.{mod}", fromlist=["*"])
        print(f"✅ src.{mod}")
    except Exception as e:
        print(f"❌ src.{mod}: {type(e).__name__} {str(e)[:80]}")

print("\n--- 端到端测试 ---")
from src.rag_engine import RAGEngine
engine = RAGEngine(top_k=5)
r = engine.ask_rag("武汉兴图新科电子股份有限公司法定代表人是谁？")
print(f"\n⏱️  总: {r['latency_ms']:.0f}ms  检索: {r['breakdown']['retrieve_ms']:.0f}ms  LLM: {r['breakdown']['llm_ms']:.0f}ms")
print(f"\n💡 答案: {r['answer'][:200]}")
print(f"\n📚 引用: {len(r['references'])} 条")
for ref in r["references"][:3]:
    print(f"   第{ref['page']}页  score={ref['score']:.4f}")
