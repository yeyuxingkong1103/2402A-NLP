# 临时：验证 VLM 入库后图像检索（人工智能NLP-RAG-图像内容解析及检索优化）
from src.image_parser.image_retriever import ImageRetriever

ir = ImageRetriever()
for q in ("公司销售部下设有哪些销售部门和销售处", "2008年中国IC市场哪个领域增长率最快，哪个负增长"):
    print("=" * 70)
    print("Q:", q)
    hits = ir.retrieve(q, top_k=3)
    for h in hits:
        print(f"  {h['image_id']} p{h['page']} score={h['final_score']} kw={h['kw_hits']}")
        print(f"    caption: {h.get('caption','')[:80]}")
        print(f"    vqa: {h.get('vqa_text','')[:120].replace(chr(10),' / ')}")
