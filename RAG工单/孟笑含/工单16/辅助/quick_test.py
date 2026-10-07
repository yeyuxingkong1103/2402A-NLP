# -*- coding: utf-8 -*-
"""快速验证：base vs base+LoRA 各推理 1 条"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from PIL import Image

BASE = "/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"
LORA = "./output/qwen2.5-vl-3b-lora/v3-20261007-165343/checkpoint-13"

QUESTION = "根据文本信息，该静电除尘器的发明人是："
OPTIONS = ["A. P·吉特勒", "B. 张三", "C. 李四", "D. 王五"]

print("=" * 60)
print("1. 加载 processor + base 模型")
print("=" * 60)
processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True,
)
print("✅ base 模型加载完成")

def ask(model, question, options):
    prompt = question + "\n" + "\n".join(options) + "\n请直接回答选项字母。"
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(gen, skip_special_tokens=True)

print("\n" + "=" * 60)
print("2. base 模型推理")
print("=" * 60)
print("Q:", QUESTION)
print("A(base):", ask(base_model, QUESTION, OPTIONS))

print("\n" + "=" * 60)
print("3. 加载 LoRA")
print("=" * 60)
lora_model = PeftModel.from_pretrained(base_model, LORA)
lora_model.eval()
print("✅ LoRA 加载完成")

print("\n" + "=" * 60)
print("4. base+LoRA 推理")
print("=" * 60)
print("Q:", QUESTION)
print("A(lora):", ask(lora_model, QUESTION, OPTIONS))

print("\n✅ 快速验证完成")
