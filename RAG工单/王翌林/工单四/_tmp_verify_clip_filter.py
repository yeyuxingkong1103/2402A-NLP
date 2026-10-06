# 临时：验证 CLIP 空/单文档过滤表达式（工单四）
from dotenv import load_dotenv

load_dotenv()
from src.image_parser.image_store import ImageStore
from src.image_parser.image_embedding import ImageClipEmbedder

s = ImageStore()
print("filter none:", repr(s._doc_filter(None)))
print("filter one:", repr(s._doc_filter(["招股说明书2"])))

rows = s.client.query(collection_name=s.collection, filter="",
                      output_fields=["image_id"], limit=1000)
print("empty-filter query rows:", len(rows))

clip = ImageClipEmbedder()
qv = clip.embed_query_text("组织结构图 销售部有几个部门")
h_all = s.search_clip(qv, top_k=3)
h_doc = s.search_clip(qv, top_k=3, doc_ids=["招股说明书2"])
print("clip all:", [(h["image_id"], round(h["score"], 3)) for h in h_all])
print("clip doc2:", [(h["image_id"], round(h["score"], 3)) for h in h_doc])
