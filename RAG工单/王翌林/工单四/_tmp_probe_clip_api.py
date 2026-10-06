# 临时：探测 ChineseCLIP API（人工智能NLP-RAG-图像内容解析及检索优化）
import inspect

import transformers
print("transformers:", transformers.__version__)

from transformers import ChineseCLIPModel, ChineseCLIPProcessor
mdl = ChineseCLIPModel.from_pretrained(
    "/home/dabaie/models/chinese-clip-vit-base-patch16")
proc = ChineseCLIPProcessor.from_pretrained(
    "/home/dabaie/models/chinese-clip-vit-base-patch16")
print("get_text_features sig:", inspect.signature(mdl.get_text_features))
print("get_image_features sig:", inspect.signature(mdl.get_image_features))

from PIL import Image
import json
m = json.load(open("data/images/招股说明书2_images.json", encoding="utf-8"))["images"][0]
img = Image.open(m["path"]).convert("RGB")
inp = proc(text=["组织结构图"], images=img, return_tensors="pt", padding=True)
out = mdl(**inp)
print("model output type:", type(out).__name__)
print("attrs:", [a for a in dir(out) if "text" in a or "image" in a])
print("text_embeds shape:", None if out.text_embeds is None else out.text_embeds.shape)
print("image_embeds shape:", None if out.image_embeds is None else out.image_embeds.shape)
