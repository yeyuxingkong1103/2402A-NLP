# 临时：仅重解析 img_008 并合并回 parsed JSON（人工智能NLP-RAG-图像内容解析及检索优化）
import json

from src.image_parser.image_captioner import Qwen2VLEngine, ImageCaptioner

MANIFEST = "data/images/招股说明书2_images.json"
OUT = "data/image_descriptions/招股说明书2_images_parsed.json"

manifest = json.load(open(MANIFEST, encoding="utf-8"))
meta = next(x for x in manifest["images"] if x["image_id"] == "img_008")

cap = ImageCaptioner(vlm_engine=Qwen2VLEngine())
new_one = cap.parse_one(meta)

data = json.load(open(OUT, encoding="utf-8"))
for i, im in enumerate(data["images"]):
    if im["image_id"] == "img_008":
        data["images"][i] = new_one
json.dump(data, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("status:", new_one["parse_status"], new_one["parse_steps"])
for qa in new_one["vqa_qa"]:
    print("=" * 60)
    print("Q:", qa["q"])
    print("A:", qa["a"][:500])
