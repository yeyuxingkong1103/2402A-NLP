# 临时：探测 ChineseCLIP 特征字段（人工智能NLP-RAG-图像内容解析及检索优化））
import torch
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

mdl = ChineseCLIPModel.from_pretrained("/home/dabaie/models/chinese-clip-vit-base-patch16")
proc = ChineseCLIPProcessor.from_pretrained("/home/dabaie/models/chinese-clip-vit-base-patch16")

ti = proc(text=["组织结构图 销售部"], return_tensors="pt", padding=True)
to = mdl.get_text_features(**ti)
print("text out:", type(to).__name__)
if hasattr(to, "keys"):
    for k, v in to.items():
        print("  ", k, None if v is None else tuple(v.shape) if hasattr(v, "shape") else type(v))
tt = mdl.get_text_features(**ti, return_dict=False)
print("text tuple lens:", len(tt), [tuple(x.shape) if hasattr(x, "shape") else x for x in tt])

from PIL import Image
import json
mp = json.load(open("data/images/招股说明书2_images.json", encoding="utf-8"))["images"][0]
ii = proc(images=Image.open(mp["path"]).convert("RGB"), return_tensors="pt")
io = mdl.get_image_features(**ii)
print("image out:", type(io).__name__)
if hasattr(io, "keys"):
    for k, v in io.items():
        print("  ", k, None if v is None else tuple(v.shape) if hasattr(v, "shape") else type(v))
