# -*- coding: utf-8 -*-
"""
工单编号：人工智能 NLP-RAG-PDF 文档的图像内容解析及检索优化
步骤3：图像语义描述 —— BLIP 图像描述模型（本机 HF 缓存，零下载）生成图像描述
运行：python caption_images.py
"""
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

MODEL_ID = "Salesforce/blip-image-captioning-base"  # 已在 ~/.cache/huggingface/hub
OUT = Path("images") / "blip_captions.json"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID, local_files_only=True,
                                                         torch_dtype=torch.float32).to(device)
    model.eval()
    print(f"BLIP 加载完成（device={device}）")

    caps = {}
    for png in sorted(Path("images").glob("page*.png")):
        image = Image.open(png).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=40)
        caps[png.stem] = processor.decode(out[0], skip_special_tokens=True)
        print(f"{png.stem}: {caps[png.stem]}")

    OUT.write_text(json.dumps(caps, ensure_ascii=False), encoding="utf-8")
    print(f"描述完成 -> {OUT}")


if __name__ == "__main__":
    main()
