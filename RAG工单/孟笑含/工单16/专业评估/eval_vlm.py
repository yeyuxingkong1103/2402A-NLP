# -*- coding: utf-8 -*-
"""工单 16 评估脚本：base vs base+LoRA 对比
指标：BLEU-4 / ROUGE-L / 术语准确率 / 选项准确率
"""
import os, json, re, time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

BASE = "/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"
LORA = "./output/qwen2.5-vl-3b-lora/v3-20261007-165343/checkpoint-13"
DATA = "./finetune_data/train.jsonl"
MAX_NEW = 64

# 工业术语词表（工单要求：淬火、公差配合等）
TERMS = ["淬火", "公差", "配合", "公差配合", "静电除尘", "管状", "圆锥", "外壳",
         "台阶", "入口", "直径", "专利", "发明", "部件", "编号", "图"]

rouge = Rouge()
smooth = SmoothingFunction().method1

print("=" * 70)
print("1. 加载数据")
print("=" * 70)
with open(DATA, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
print(f"样本数：{len(data)}")

def extract_qa(item):
    """从 messages 提取 question / answer / image"""
    msgs = item["messages"]
    q_msg = msgs[0]
    a_msg = msgs[1]
    # 问题文本
    q_text = ""
    img_path = None
    for c in q_msg["content"]:
        if c.get("type") == "text":
            q_text = c["text"]
        elif c.get("type") == "image":
            img_path = c.get("image")
    # 答案
    a_content = a_msg["content"]
    if isinstance(a_content, list):
        ans = a_content[0]["text"]
    else:
        ans = a_content
    return q_text, ans, img_path

print("\n" + "=" * 70)
print("2. 加载 processor + base 模型")
print("=" * 70)
processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
)
base_model.eval()
print("✅ base 加载完成")

def build_inputs(question, img_path):
    content = []
    if img_path and os.path.exists(img_path):
        content.append({"type": "image", "image": img_path})
    content.append({"type": "text", "text": question + "\n请直接回答。"})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if img_path and os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB")
        inputs = processor(text=[text], images=[img], return_tensors="pt").to("cuda:0")
    else:
        inputs = processor(text=[text], return_tensors="pt").to("cuda:0")
    return inputs

def infer(model, question, img_path):
    inputs = build_inputs(question, img_path)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip()

def term_hit(text, terms):
    """命中术语数"""
    return sum(1 for t in terms if t in text)

def eval_model(model, tag):
    """跑全量评估"""
    bleu_scores, rouge_scores, term_acc, opt_acc = [], [], 0, 0
    n = len(data)
    print(f"\n===== 评估 {tag}（{n} 条）=====")
    t0 = time.time()
    for i, item in enumerate(data):
        q, gold, img = extract_qa(item)
        pred = infer(model, q, img)
        # BLEU
        ref = [list(gold)]
        hyp = list(pred)
        if hyp:
            bleu_scores.append(sentence_bleu(ref, hyp, smoothing_function=smooth))
        # ROUGE
        try:
            r = rouge.get_scores(pred if pred else " ", gold)[0]
            rouge_scores.append(r["rouge-l"]["f"])
        except Exception:
            rouge_scores.append(0.0)
        # 术语命中
        gold_terms = set(t for t in TERMS if t in gold)
        if gold_terms:
            hit = sum(1 for t in gold_terms if t in pred)
            term_acc += hit / len(gold_terms)
        # 选项准确率（首字母匹配）
        m_gold = re.match(r"^([A-D])", gold.strip())
        m_pred = re.match(r"^([A-D])", pred.strip())
        if m_gold and m_pred and m_gold.group(1) == m_pred.group(1):
            opt_acc += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] 已评估，用时 {time.time()-t0:.0f}s")
    result = {
        "tag": tag,
        "n": n,
        "BLEU-4": round(sum(bleu_scores) / len(bleu_scores), 4) if bleu_scores else 0,
        "ROUGE-L": round(sum(rouge_scores) / len(rouge_scores), 4) if rouge_scores else 0,
        "术语准确率": round(term_acc / n, 4),
        "选项准确率": round(opt_acc / n, 4),
        "耗时(s)": round(time.time() - t0, 1),
    }
    print(f"  ✅ {tag}: {result}")
    return result

print("\n" + "=" * 70)
print("3. 评估 base（微调前）")
print("=" * 70)
res_base = eval_model(base_model, "base")

print("\n" + "=" * 70)
print("4. 加载 LoRA 并评估（微调后）")
print("=" * 70)
lora_model = PeftModel.from_pretrained(base_model, LORA)
lora_model.eval()
res_lora = eval_model(lora_model, "base+LoRA")

print("\n" + "=" * 70)
print("5. 对比报告")
print("=" * 70)
print(f"{'指标':<15}{'base':<12}{'LoRA':<12}{'提升':<12}")
print("-" * 50)
for k in ["BLEU-4", "ROUGE-L", "术语准确率", "选项准确率"]:
    b, l = res_base[k], res_lora[k]
    delta = l - b
    sign = "+" if delta >= 0 else ""
    print(f"{k:<15}{b:<12}{l:<12}{sign}{delta:.4f}")
print("-" * 50)
print(f"base 耗时: {res_base['耗时(s)']}s | LoRA 耗时: {res_lora['耗时(s)']}s")

# 保存报告
report = {"base": res_base, "lora": res_lora}
os.makedirs("./eval_report", exist_ok=True)
with open("./eval_report/comparison.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("\n✅ 报告已保存：./eval_report/comparison.json")
