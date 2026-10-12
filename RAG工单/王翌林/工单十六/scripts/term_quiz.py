# -*- coding: utf-8 -*-
"""工业术语专项评估：用文本可答的术语题，衡量模型是否正确使用专业术语。
对比 baseline 与 finetuned，输出 docs/screenshots/term_quiz.json。"""
import os, json, torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

MODEL_DIR = "/home/dabaie/models/Qwen2-VL-2B-Instruct"
ADAPTER = "/home/dabaie/code/工单/工单十六/finetune_output/lora_adapter"
OUT = "/home/dabaie/code/工单/工单十六/docs/screenshots"

# 工业术语专项测试集（问题 + 应命中的目标术语）
QUIZ = [
    ("淬火热处理的主要目的是什么？", ["淬火"]),
    ("公差与配合主要用于控制机械零件的什么指标？", ["公差", "配合"]),
    ("链条传动与齿轮传动相比，更适合什么工况？", ["链条", "齿轮"]),
    ("带式输送机通常由什么部件提供驱动？", ["电机", "输送机"]),
    ("气化器的基本作用是什么？", ["气化器"]),
    ("轧制带材时调节张力的主要目的是什么？", ["轧制", "张力", "带材"]),
    ("水洗洗涤塔通常用于处理什么？", ["洗涤塔", "水洗", "气体"]),
    ("联轴器的主要作用是什么？", ["联轴器"]),
    ("喷吹管的作用是什么？", ["喷吹管", "气体"]),
    ("料斗在散料处理中的作用是什么？", ["料斗", "散料"]),
    ("环形通道与径向通道有什么区别？", ["环形", "径向"]),
    ("电极在电加热装置中的作用是什么？", ["电极"]),
]


def build(adapter=None):
    p = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    m = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, device_map="cuda")
    if adapter and os.path.isdir(adapter):
        m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    m.eval()
    return m, p


def gen(m, p, q):
    msgs = [{"role": "user", "content": [{"type": "text", "text": q}]}]
    text = p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = p(text=[text], return_tensors="pt").to(m.device)
    with torch.inference_mode():
        out = m.generate(**inp, max_new_tokens=96, do_sample=False)
    g = out[0][inp["input_ids"].shape[1]:]
    return p.decode(g, skip_special_tokens=True).strip()


def run(tag, adapter):
    m, p = build(adapter)
    rows = []
    for q, terms in QUIZ:
        pred = gen(m, p, q)
        hit = sum(1 for t in terms if t in pred)
        rows.append({"question": q, "terms": terms, "pred": pred,
                     "hit": hit, "total": len(terms),
                     "acc": round(hit / len(terms), 3)})
    del m; torch.cuda.empty_cache()
    acc = sum(r["hit"] for r in rows) / sum(r["total"] for r in rows)
    summary = {"tag": tag, "term_accuracy": round(acc, 4),
               "per_q_avg": round(sum(r["acc"] for r in rows) / len(rows), 4), "num": len(rows)}
    return {"summary": summary, "details": rows}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    base = run("baseline", None)
    fine = run("finetuned", ADAPTER)
    json.dump(base, open(f"{OUT}/termquiz_baseline.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(fine, open(f"{OUT}/termquiz_finetuned.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("baseline  术语准确性:", base["summary"])
    print("finetuned 术语准确性:", fine["summary"])
    delta = fine["summary"]["term_accuracy"] - base["summary"]["term_accuracy"]
    print(f"提升: {delta*100:+.1f} 个百分点")
