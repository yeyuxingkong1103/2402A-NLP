"""工单16：从磁盘加载真实适配器推理；不读取测试答案。"""

import argparse, json, threading
from functools import lru_cache
from pathlib import Path
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image

ROOT = Path(__file__).resolve().parent
lock = threading.Lock()


@lru_cache(maxsize=1)
def load():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        ROOT / "models/MinerU2.5-1.2B",
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, ROOT / "models/industrial-vlm-lora")
    model.eval()
    processor = AutoProcessor.from_pretrained(
        ROOT / "models/industrial-vlm-lora",
        local_files_only=True,
        min_pixels=56 * 56,
        max_pixels=224 * 224,
    )
    return model, processor


def infer(image, question):
    candidate = (ROOT / image).resolve()
    if not candidate.is_relative_to(ROOT) or not candidate.is_file():
        raise ValueError("图像必须位于本工单研发目录中")
    with lock:
        model, processor = load()
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": question}],
            }
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[prompt],
            images=[Image.open(candidate).convert("RGB")],
            return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        return processor.tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "answer": infer(args.image, args.question),
                "model": "本项目MinerU基座+industrial-vlm-lora",
            },
            ensure_ascii=False,
        )
    )
