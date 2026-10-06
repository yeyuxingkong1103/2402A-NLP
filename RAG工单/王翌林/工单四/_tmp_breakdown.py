# 临时：检索/LLM 耗时分解（人工智能NLP-RAG-图像内容解析及检索优化）
import time

from dotenv import load_dotenv

load_dotenv()

from src.rag_engine_v3 import RAGEngineV3
from src.rag_engine_v4 import RAGEngineV4

v3 = RAGEngineV3(top_k=5)
v4 = RAGEngineV4(v3_engine=v3, top_k=5)

CASES = [
    ("招股说明书1", "公司主营业务是什么"),
    ("招股说明书2", "前五大客户占营业收入的比例？"),
    ("招股说明书2", "本次募集资金总额是多少"),
    ("招股说明书2", "从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？"),
]
# 预热
v3.ask_llm("你好")
v3.ask_rag("公司主营业务是什么", doc_id="招股说明书1")
v4.ask("公司主营业务是什么", doc_id="招股说明书1")

for doc, q in CASES:
    r = v4.ask(q, doc_id=doc, use_image=True)
    bd = r.get("breakdown", {})
    print(f"\nQ: {q[:30]}")
    print(f"  total={r['latency_ms']:.0f}ms retrieve={bd.get('retrieve_ms',0):.0f} "
          f"llm={bd.get('llm_ms',0):.0f} nimg={len(r['retrieved_images'])} "
          f"ntxt={len(r['retrieved_text_chunks'])} ntbl={len(r['retrieved_tables'])}")
