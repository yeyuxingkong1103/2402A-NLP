# 临时：检查 VLM 重解析结果（人工智能NLP-RAG-图像内容解析及检索优化）
import json

d = json.load(open("data/image_descriptions/招股说明书2_images_parsed.json", encoding="utf-8"))
print("vlm_engine:", d.get("vlm_engine"), "stats:", d.get("stats"))
for im in d["images"]:
    print("=" * 70)
    print(im["image_id"], "p", im.get("page"), "status:", im.get("parse_status"),
          "steps:", im.get("parse_steps"), "ocr_engine:", im.get("ocr_engine"))
    print("CAPTION:", (im.get("caption") or "")[:400])
    for qa in im.get("vqa_qa", []):
        print(f"  Q: {qa['q']}")
        print(f"  A: {qa['a'][:220]}")
    print("OCR:", (im.get("ocr_text") or "")[:150].replace("\n", " / "))
