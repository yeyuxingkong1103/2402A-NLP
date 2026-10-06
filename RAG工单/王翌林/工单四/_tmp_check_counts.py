# 临时：检查 Milvus 各 collection 行数（人工智能NLP-RAG-图像内容解析及检索优化）
from src.table_parser.table_store import TableStore
from src.image_parser.image_store import ImageStore

s = TableStore()
print("table mode:", s.mode)

im = ImageStore()
print("rag_images rows:", im.count())

# 分 doc_id 统计 rag_tables
try:
    rows = s.client.query(
        collection_name=s.collection,
        filter="doc_id != ''",
        output_fields=["doc_id"],
        limit=2000,
    )
    from collections import Counter
    print("rag_tables total:", len(rows), "by doc:", dict(Counter(r["doc_id"] for r in rows)))
except Exception as e:
    print("query rag_tables failed:", e)

try:
    rows = im.client.query(
        collection_name=im.collection,
        filter="doc_id != ''",
        output_fields=["doc_id", "caption"],
        limit=2000,
    )
    from collections import Counter
    print("rag_images by doc:", dict(Counter(r["doc_id"] for r in rows)))
    print("non-empty captions:", sum(1 for r in rows if (r.get("caption") or "").strip()))
except Exception as e:
    print("query rag_images failed:", e)
