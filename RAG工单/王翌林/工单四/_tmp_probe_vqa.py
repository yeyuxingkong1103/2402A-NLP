# 临时：探测 VLM 对销售部并列子部门的读图（人工智能NLP-RAG-图像内容解析及检索优化）
import json
from PIL import Image
from src.image_parser.image_captioner import Qwen2VLEngine

m = [x for x in json.load(open("data/images/招股说明书2_images.json", encoding="utf-8"))["images"]
     if x["image_id"] == "img_008"][0]
eng = Qwen2VLEngine()
eng._ensure_loaded()
img = Image.open(m["path"])
questions = [
    "请只根据图中连线结构回答：标注为“销售部”的方框，向下连接了哪几个平级的子部门方框？请按从左到右逐一完整列出方框内文字，并给出数量。",
    "图中“销售部”的直接下属部门共有几个？分别叫什么名字？",
]
for i, q in enumerate(questions, 1):
    print(f"--- Q{i}: {q}")
    print(eng.generate(img, f"请仔细看图并回答问题，输出为中文，严格依据图中可见的方框与连线：\n{q}",
                       max_new_tokens=300))
