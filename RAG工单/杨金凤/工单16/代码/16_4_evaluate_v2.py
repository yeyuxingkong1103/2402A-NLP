"""
评估 v2：不依赖选项字母，用语义/内容匹配
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
MAX_IMAGE_SIZE = 768

INDUSTRY_TERMS = [
    "淬火", "公差", "配合", "配气带孔盘", "链条", "静电除尘器",
    "分配装置", "散料", "块状散料", "圆锥形", "圆柱形", "外壳",
    "气流方向", "部件", "编号", "保护管", "分散装置",
]


def resize_image(image, max_size=MAX_IMAGE_SIZE):
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)


def infer(model, processor, image, question):
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": question}],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    generated = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return generated.strip()


def semantic_match(pred: str, ref: str) -> bool:
    """
    语义匹配：判断预测是否命中参考答案
    两种命中方式：
      1. 预测中出现参考答案的选项字母
      2. 预测中包含参考答案的选项内容（去除选项字母后的文本，长度>4）
    """
    # 提取参考答案的字母和内容
    m = re.match(r'^\s*([ABCD])[\.、\s]\s*(.+)$', ref)
    if not m:
        # 参考答案不含字母，直接做包含判断
        ref_content = ref.strip()
        return ref_content in pred or (len(ref_content) > 5 and ref_content[:8] in pred)

    ref_letter = m.group(1)
    ref_content = m.group(2).strip()

    # 方式1：字母匹配（要求字母独立出现，且附近没有其他字母）
    # 排除 "ABCD" 这种列举
    letters_in_pred = re.findall(r'([ABCD])(?=[\.、\s：:]|$)', pred)
    if letters_in_pred:
        # 预测中有明确的选择
        return ref_letter in letters_in_pred

    # 方式2：内容匹配
    # 去掉标点、空格后比较
    clean = lambda s: re.sub(r'[\s\.，,。、；;：:]+', '', s)
    ref_clean = clean(ref_content)
    pred_clean = clean(pred)

    # 参考内容前 6 个字在预测里出现就算命中
    key = ref_clean[:6]
    if len(key) >= 4 and key in pred_clean:
        return True

    # 完整内容匹配
    if ref_clean in pred_clean:
        return True

    return False


def compute_metrics(preds, refs):
    em = sum(semantic_match(p, r) for p, r in zip(preds, refs)) / len(refs)
    rouger = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge = sum(rouger.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)) / len(refs)

    term_scores = []
    for p, r in zip(preds, refs):
        ref_terms = {t for t in INDUSTRY_TERMS if t in r}
        pred_terms = {t for t in INDUSTRY_TERMS if t in p}
        if not ref_terms:
            term_scores.append(1.0)
        else:
            term_scores.append(len(ref_terms & pred_terms) / len(ref_terms))
    term_acc = sum(term_scores) / len(term_scores)

    return {"EM": em, "ROUGE-L": rouge, "term_acc": term_acc}


def main():
    samples = []
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"验证集 {len(samples)} 条")

    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)

    # === 基线 ===
    print("\n>>> 基线模型")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    base_model.eval()
    preds_base = []
    for s in tqdm(samples, desc="Baseline"):
        img = resize_image(Image.open(os.path.join(DATA_DIR, s["images"][0])).convert("RGB"))
        q = s["messages"][0]["content"].replace("<image>", "").strip()
        preds_base.append(infer(base_model, processor, img, q))
    del base_model
    torch.cuda.empty_cache()

    # === 微调 ===
    print("\n>>> 微调模型")
    ft_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    ft_model = PeftModel.from_pretrained(ft_model, LORA_DIR)
    ft_model.eval()
    preds_ft = []
    for s in tqdm(samples, desc="Finetuned"):
        img = resize_image(Image.open(os.path.join(DATA_DIR, s["images"][0])).convert("RGB"))
        q = s["messages"][0]["content"].replace("<image>", "").strip()
        preds_ft.append(infer(ft_model, processor, img, q))

    refs = [s["messages"][1]["content"] for s in samples]

    m_base = compute_metrics(preds_base, refs)
    m_ft = compute_metrics(preds_ft, refs)

    print("\n" + "=" * 70)
    print(f"{'指标':<20}{'基线':>15}{'微调后':>15}{'提升':>15}")
    print("-" * 70)
    for k in ["EM", "ROUGE-L", "term_acc"]:
        b, f = m_base[k], m_ft[k]
        print(f"{k:<20}{b:>15.4f}{f:>15.4f}{f-b:>+15.4f}")
    print("=" * 70)

    report = {
        "metrics": {
            "baseline": m_base,
            "finetuned": m_ft,
            "improvement": {k: m_ft[k] - m_base[k] for k in m_base},
        },
        "samples": [
            {"question": s["messages"][0]["content"].replace("<image>", "").strip(),
             "reference": r, "pred_base": pb, "pred_ft": pf,
             "match_base": semantic_match(pb, r),
             "match_ft": semantic_match(pf, r)}
            for s, r, pb, pf in zip(samples, refs, preds_base, preds_ft)
        ],
    }
    with open("/root/autodl-tmp/ragflow/finetune_output/eval_report_v2.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 报告 -> /root/autodl-tmp/ragflow/finetune_output/eval_report_v2.json")


if __name__ == "__main__":
    main()
