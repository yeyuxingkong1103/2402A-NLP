# -*- coding: utf-8 -*-
# 工单四 Step 2 验收辅助：汇总两册 PDF 提取结果并复制关键证据截图
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
import json
import shutil
from pathlib import Path

root = Path(".")
ss = root / "docs" / "screenshots"
ss.mkdir(parents=True, exist_ok=True)

# 工单四：关键图证据截图（重点题依赖）
evidence = [
    ("data/images/招股说明书2/page_039_draw_1.png", "step2_组织结构图_L2渲染.png"),
    ("data/images/招股说明书2/page_072_img_1.png", "step2_IC市场应用结构饼图.png"),
    ("data/images/招股说明书2/page_072_img_2.png", "step2_IC市场增长率柱状图.png"),
]
for src, dst in evidence:
    if Path(src).exists():
        shutil.copy(src, ss / dst)
        print("截图已存:", ss / dst)

for doc in ["招股说明书1", "招股说明书2"]:
    d = json.load(open(f"data/images/{doc}_images.json", encoding="utf-8"))
    s = d["stats"]
    l2 = [i for i in d["images"] if i["extract_source"] == "L2_vector"]
    print(f"{doc}: 保留 {s['extracted']} 张 | L1位图 {s['extracted']-len(l2)} | "
          f"L2矢量渲染 {len(l2)} | 过滤小图 {s['filtered_small']} | "
          f"去重 {s['duplicates']} | 失败 {s['failed']}")
    for i in l2:
        print("   L2:", i["image_id"], i["path"], f"{i['width']}x{i['height']}")
