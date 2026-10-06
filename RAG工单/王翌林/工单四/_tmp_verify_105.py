# 临时：单题验证 id105/id106（人工智能NLP-RAG-图像内容解析及检索优化）
import time

from dotenv import load_dotenv

load_dotenv()

from src.rag_engine_v3 import RAGEngineV3
from src.rag_engine_v4 import RAGEngineV4

v3 = RAGEngineV3(top_k=5)
v4 = RAGEngineV4(v3_engine=v3, top_k=5)

QUESTIONS = [
    (105, "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？"),
    (106, "从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？"),
]
for qid, q in QUESTIONS:
    t0 = time.perf_counter()
    r = v4.ask(q, doc_id="招股说明书2", use_image=True)
    print("=" * 70)
    print(f"id{qid} latency={(time.perf_counter()-t0)*1000:.0f}ms n_images={len(r['retrieved_images'])}")
    print("ANSWER:", r["answer"])
