# 临时脚本：检查图像清单与 rag_images 库状态（人工智能NLP-RAG-图像内容解析及检索优化）
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from dotenv import load_dotenv
load_dotenv()

# 1) 图像清单字段
man = json.load(open("data/images/招股说明书2_images.json", encoding="utf-8"))
print("manifest keys:", list(man.keys()))
for im in man.get("images", []):
    print(im.get("image_id"), "p" + str(im.get("page")),
          "image_type=", im.get("image_type"), "| keys:", sorted(im.keys()))

# 2) rag_images collection 状态
from src.image_parser.image_store import ImageStore
store = ImageStore()
print("\nrag_images rows =", store.count())
rows = store.client.query(collection_name=store.collection,
                          filter='doc_id == "招股说明书2"',
                          output_fields=["image_id", "page", "caption", "ocr_text",
                                         "vqa_text"], limit=10)
for r in rows:
    print(f"\n{r.get('image_id')} p{r.get('page')}")
    print("  caption:", repr((r.get("caption") or "")[:100]))
    print("  vqa:", repr((r.get("vqa_text") or "")[:150]))
    print("  ocr:", repr((r.get("ocr_text") or "")[:100]))
