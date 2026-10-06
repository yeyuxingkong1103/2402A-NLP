# -*- coding: utf-8 -*-
# 工单四 Step 2 验收辅助：筛选疑似图表的大尺寸图像（人工智能NLP-RAG-图像内容解析及检索优化）
import json

for doc in ["招股说明书1", "招股说明书2"]:
    d = json.load(open(f"data/images/{doc}_images.json", encoding="utf-8"))
    big = [i for i in d["images"] if i["width"] >= 350 and i["height"] >= 280]
    print(doc, "大图候选数:", len(big))
    for i in big[:12]:
        print(" ", i["image_id"], "p", i["page"], f"{i['width']}x{i['height']}", i["bbox"])
