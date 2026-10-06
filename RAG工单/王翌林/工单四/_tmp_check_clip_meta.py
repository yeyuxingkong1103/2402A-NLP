# 临时：检查 rag_images 的 CLIP 向量入库（人工智能NLP-RAG-图像内容解析及检索优化）
from src.image_parser.image_store import ImageStore

s = ImageStore()
rows = s.client.query(s.collection, filter="doc_id != ''",
                      output_fields=["image_id", "caption", "metadata"], limit=10)
print("rows:", len(rows))
for r in rows:
    cv = (r.get("metadata") or {}).get("clip_embedding")
    print(r["image_id"], "caption:", bool((r.get("caption") or "").strip()),
          "clip_dim:", None if not cv else len(cv))
