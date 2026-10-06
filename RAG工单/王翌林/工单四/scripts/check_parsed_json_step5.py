# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# scripts/check_parsed_json_step5.py —— 核对图像解析 JSON 字段与重点图路径（临时辅助）
import json
from pathlib import Path

p = Path("data/image_descriptions/招股说明书2_images_parsed.json")
d = json.loads(p.read_text(encoding="utf-8"))
items = d if isinstance(d, list) else d.get("images", d.get("items", []))
print("type:", type(d).__name__, "n:", len(items))
for im in items:
    if str(im.get("image_id")) in ("img_008", "img_011", "img_012"):
        print(im["image_id"], "| path:", im.get("path"), "| keys:", sorted(im.keys()))
        vqa = im.get("vqa_qa") or []
        print("  vqa_qa[0]:", vqa[0] if vqa else None)
        print("  ocr[:80]:", (im.get("ocr_text") or "")[:80])
