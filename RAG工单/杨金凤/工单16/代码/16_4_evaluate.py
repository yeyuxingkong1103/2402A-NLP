"""
评估微调前后模型在 IMDR 视觉 QA 上的表现。
指标：
  - 答案精确匹配率（EM, 选项字母匹配）
  - ROUGE-L
  - 工业术语准确率
"""
import os
import json
import re
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from rouge_score import rouge_scorer
from tqdm import tqdm

MODEL_DIR = "/root/autodl-tmp/models/Qwen2-VL-2B-Instruct"
LORA_DIR = "/root/autodl-tmp/ragflow/finetune_output/lora_final"
DATA_DIR = "/root/autodl-tmp/ragflow/finetune_data"
VAL_FILE = os.path.join(DATA_DIR, "val.jsonl")

# 工业术语表
INDUSTRY_TERMS = [
    "淬火", "公差", "配合", "配气带孔盘", "链条", "静电除尘器",
    "分配装置", "散料", "块状散料", "圆锥形", "圆柱形", "外壳",
    "气流方向", "部件", "编号", "保护管", "分散装置",
]

MAX_IMAGE_SIZE = 768


def resize_image(image, max_size=MAX_IMAGE_SIZE):
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)


def extract_option_letter(text: str) -> str:
    """从模型输出里提取选项字母 A/B/C/D"""
    # 匹配 "A."、"A、"、"选A"、"答案：A" 等
    m = re.search(r'\b([ABCD])\b', text)
    if m:
        return m.group(1)
    m = re.search(r'^\s*([ABCD])', text)
    if m:
        return m.group(1)
    return text.strip()[:1]


def infer(model, processor, image, question):
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": question}],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    generated = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return generated.strip()


def compute_term_acc(preds, refs):
    """工业术语准确率"""
    scores = []
    for p, r in zip(preds, refs):
        ref_terms = {t for t in INDUSTRY_TERMS if t in r}
        pred_terms = {t for t in INDUSTRY_TERMS if t in p}
        if not ref_terms:
            scores.append(1.0)
        else:
            scores.append(len(ref_terms & pred_terms) / len(ref_terms))
    return sum(scores) / len(scores)


def main():
    # 加载验证集
    samples = []
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"验证集 {len(samples)} 条")

    # 加载 processor
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)

    # ========== 评估基线模型 ==========
    print("\n>>> 加载基线模型...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    base_model.eval()

    preds_base = []
    for s in tqdm(samples, desc="Baseline"):
        image = resize_image(Image.open(os.path.join(DATA_DIR, s["images"][0])).convert("RGB"))
        q = s["messages"][0]["content"].replace("<image>", "").strip()
        preds_base.append(infer(base_model, processor, image, q))

    del base_model
    torch.cuda.empty_cache()

    # ========== 评估微调模型 ==========
    print("\n>>> 加载微调模型...")
    ft_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    ft_model = PeftModel.from_pretrained(ft_model, LORA_DIR)
    ft_model.eval()

    preds_ft = []
    for s in tqdm(samples, desc="Finetuned"):
        image = resize_image(Image.open(os.path.join(DATA_DIR, s["images"][0])).convert("RGB"))
        q = s["messages"][0]["content"].replace("<image>", "").strip()
        preds_ft.append(infer(ft_model, processor, image, q))

    # ========== 计算指标 ==========
    refs = [s["messages"][1]["content"] for s in samples]

    # 选项字母精确匹配
    em_base = sum(extract_option_letter(p) == extract_option_letter(r)
                  for p, r in zip(preds_base, refs)) / len(refs)
    em_ft = sum(extract_option_letter(p) == extract_option_letter(r)
                for p, r in zip(preds_ft, refs)) / len(refs)

    # ROUGE-L
    rouger = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_base = sum(rouger.score(r, p)["rougeL"].fmeasure
                     for p, r in zip(preds_base, refs)) / len(refs)
    rouge_ft = sum(rouger.score(r, p)["rougeL"].fmeasure
                   for p, r in zip(preds_ft, refs)) / len(refs)

    # 术语准确率
    term_base = compute_term_acc(preds_base, refs)
    term_ft = compute_term_acc(preds_ft, refs)

    # ========== 报告 ==========
    print("\n" + "=" * 60)
    print(f"{'指标':<20}{'基线':>12}{'微调后':>12}{'提升':>12}")
    print("-" * 60)
    print(f"{'精确匹配率(EM)':<20}{em_base:>12.4f}{em_ft:>12.4f}{em_ft-em_base:>+12.4f}")
    print(f"{'ROUGE-L':<20}{rouge_base:>12.4f}{rouge_ft:>12.4f}{rouge_ft-rouge_base:>+12.4f}")
    print(f"{'术语准确率':<20}{term_base:>12.4f}{term_ft:>12.4f}{term_ft-term_base:>+12.4f}")
    print("=" * 60)

    # 保存报告
    report = {
        "metrics": {
            "baseline": {"EM": em_base, "ROUGE-L": rouge_base, "term_acc": term_base},
            "finetuned": {"EM": em_ft, "ROUGE-L": rouge_ft, "term_acc": term_ft},
            "improvement": {
                "EM": em_ft - em_base,
                "ROUGE-L": rouge_ft - rouge_base,
                "term_acc": term_ft - term_base,
            },
        },
        "samples": [
            {"question": s["messages"][0]["content"].replace("<image>", "").strip(),
             "reference": r, "pred_base": pb, "pred_ft": pf}
            for s, r, pb, pf in zip(samples, refs, preds_base, preds_ft)
        ],
    }
    out_path = "/root/autodl-tmp/ragflow/finetune_output/eval_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 详细报告 -> {out_path}")


if __name__ == "__main__":
    main()
