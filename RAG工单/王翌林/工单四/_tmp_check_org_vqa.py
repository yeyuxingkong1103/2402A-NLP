# 临时：检查 img_008 新 VQA（人工智能NLP-RAG-图像内容解析及检索优化）
import json

d = json.load(open("data/image_descriptions/招股说明书2_images_parsed.json", encoding="utf-8"))
im = [x for x in d["images"] if x["image_id"] == "img_008"][0]
print("status:", im["parse_status"], im["parse_steps"])
for qa in im["vqa_qa"]:
    print("=" * 60)
    print("Q:", qa["q"])
    print("A:", qa["a"])
