# 临时：检查注册资本 chunk 金额可见性 + v3 实答（工单四）
from dotenv import load_dotenv

load_dotenv()
from src.rag_engine_v3 import RAGEngineV3

q = "武汉兴图新科电子股份有限公司的注册资本是多少？"
eng = RAGEngineV3()
eng.ask_rag("本次发行的发行股数是多少？", doc_id="招股说明书1")
res = eng.table_retriever.retrieve(q, top_k=5, doc_id="招股说明书1")
for h in res["fused"][:3]:
    c = h.get("content") or ""
    idx = c.find("注册资本")
    while idx != -1:
        print(f"p{h.get('page')} ...{c[max(0,idx-30):idx+80]}...".replace("\n", " "))
        idx = c.find("注册资本", idx + 1)
    print("chunk 长度:", len(c))
    print("-" * 70)
print("=== v3 实答 ===")
out = eng.ask_rag(q, doc_id="招股说明书1")
print(out["answer"][:400])
