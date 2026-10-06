# 临时：Qwen2-VL 冒烟（人工智能NLP-RAG-图像内容解析及检索优化）
import json
import sys
import time

from PIL import Image

from src.image_parser.image_captioner import Qwen2VLEngine

manifest = json.load(open("data/images/招股说明书2_images.json", encoding="utf-8"))
metas = {m["image_id"]: m for m in manifest["images"]}
m = metas[sys.argv[1] if len(sys.argv) > 1 else "img_008"]
print("image:", m["image_id"], "page:", m.get("page"), "path:", m["path"])

eng = Qwen2VLEngine()
t0 = time.time()
eng._ensure_loaded()
print(f"model loaded in {time.time()-t0:.1f}s")

t0 = time.time()
img = Image.open(m["path"])
cap = eng.caption(img)
print(f"caption ({time.time()-t0:.1f}s):\n{cap}")
