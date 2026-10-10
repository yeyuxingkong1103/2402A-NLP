# -*- coding: utf-8 -*-
"""探明 transformers 5.x 下 ChineseCLIP 的取特征 API。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import ChineseCLIPModel, ChineseCLIPProcessor
from PIL import Image

MODEL = r"G:/Trae-work产出/6ac4dd3ca72b335420a91488/_models/chinese-clip-vit-base-patch16"
m = ChineseCLIPModel.from_pretrained(MODEL).eval()
p = ChineseCLIPProcessor.from_pretrained(MODEL)

img = Image.new("RGB", (224, 224), "white")
inp = p(images=[img], return_tensors="pt")
out = m.get_image_features(**inp)
print("image_features type:", type(out))
if hasattr(out, "__dict__"):
    for k, v in out.items() if hasattr(out, "items") else []:
        print("   key:", k, type(v), getattr(v, "shape", None))

with torch.no_grad():
    full = m(**inp)
print("full keys:", list(full.keys()) if hasattr(full, "keys") else type(full))
for k in (full.keys() if hasattr(full, "keys") else []):
    v = full[k]
    print("   ", k, type(v), getattr(v, "shape", None))

tinp = p(text=["组织结构图", "柱状图"], return_tensors="pt", padding=True, truncation=True, max_length=52)
tout = m.get_text_features(**tinp)
print("text_features type:", type(tout))