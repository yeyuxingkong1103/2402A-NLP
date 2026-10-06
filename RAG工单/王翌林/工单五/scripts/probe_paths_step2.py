# -*- coding: utf-8 -*-
# 工单四 Step 2 验收辅助：导出大图候选路径（人工智能NLP-RAG-图像内容解析及检索优化）
import json

d = json.load(open("data/images/招股说明书2_images.json", encoding="utf-8"))
big = [i for i in d["images"] if i["width"] >= 350 and i["height"] >= 280]
for i in big[:25]:
    print(i["image_id"], i["path"])
