# -*- coding: utf-8 -*-
"""
工单十六：微调效果评估脚本
指标：
  1. 通用文本生成：BLEU / ROUGE-L
  2. 工业准确性：a) 专业术语准确性  b) 图纸推理正确性（选择题命中率）
  3. 端到端延迟
对比：基线模型 vs 微调后（LoRA 合并）模型。
"""
import os, sys, json, time, argparse
from loguru import logger
import torch
from PIL import Image

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

MODEL_DIR = "/home/dabaie/models/Qwen2-VL-2B-Instruct"
IMAGE_ROOT = "/home/dabaie/code/工单/工单十六/data/images"
DATA_DIR = "/home/dabaie/code/工单/工单十六/data"
ADAPTER_DIR = "/home/dabaie/code/工单/工单十六/finetune_output/lora_adapter"

# 工业专业术语表（数据驱动：来自 IMDR 专利语料高频专业词，用于术语准确性评估）
INDUSTRY_TERMS = [
    "淬火", "公差", "配合", "链条", "散料", "部件", "装置", "气流", "圆锥形", "圆柱形",
    "外壳", "导管", "料斗", "调节器", "环形", "径向", "切向", "气化器", "电极",
    "洗涤塔", "喷吹管", "水洗", "气体", "浇铸", "速度", "张力", "轧制", "带材",
    "联轴器", "齿轮", "电机", "输送机", "传动", "右侧", "上方", "下方", "内部",
    "顶部", "底部", "混合", "支撑", "控制", "处理", "密封", "管口", "箔衬袋",
]


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_model(adapter_path=None):
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, device_map="cuda")
    if adapter_path and os.path.isdir(adapter_path):
        logger.info(f"加载 LoRA 适配器: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    return model, processor


def generate_answer(model, processor, rec):
    q = rec["question"]
    content = []
    img = None
    if rec.get("image"):
        p = os.path.join(IMAGE_ROOT, rec["image"])
        if os.path.isfile(p):
            im = Image.open(p).convert("RGB")
            max_side = 384
            w, h = im.size
            if max(w, h) > max_side:
                s = max_side / max(w, h)
                im = im.resize((int(w * s), int(h * s)), Image.BILINEAR)
            img = im
            content.append({"type": "image"})
    content.append({"type": "text", "text": q})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img] if img else None, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip()


# ---------- 指标 ----------
def bleu_score(pred, ref):
    import collections, math
    pred_t = pred.replace(" ", "")
    ref_t = ref.replace(" ", "")
    if not pred_t or not ref_t:
        return 0.0
    def ngrams(s, n):
        return collections.Counter(s[i:i+n] for i in range(len(s)-n+1))
    precisions = []
    for n in (1, 2):
        p, r = ngrams(pred_t, n), ngrams(ref_t, n)
        overlap = sum((p & r).values())
        precisions.append(overlap / max(1, sum(p.values())))
    bp = 1.0 if len(pred_t) >= len(ref_t) else math.exp(1 - len(ref_t)/max(1,len(pred_t)))
    return bp * math.exp(sum(math.log(max(x,1e-9)) for x in precisions)/2)


def rouge_l(pred, ref):
    pred_t, ref_t = pred.replace(" ", ""), ref.replace(" ", "")
    if not pred_t or not ref_t:
        return 0.0
    m, n = len(pred_t), len(ref_t)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if pred_t[i-1]==ref_t[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    r = lcs / n
    p = lcs / m
    return 2*r*p/(r+p) if (r+p) > 0 else 0.0


def normalize(s):
    s = s.replace(" ", "").replace("\n", "").lower()
    for a, b in [("内部", "之内"), ("顶端", "顶部"), ("上方", "顶部")]:
        s = s.replace(a, b)
    return s


def exact_match(pred, gold):
    p, g = normalize(pred), normalize(gold)
    return g in p or p in g


def term_coverage(pred):
    """预测答案中命中的工业术语数 / 标准答案术语数（这里用命中数打分）"""
    hit = sum(1 for t in INDUSTRY_TERMS if t in pred)
    return hit


def industry_term_accuracy(pred, gold):
    """工业术语准确性 = 标准答案中的工业术语，有多少被预测正确复述（召回率）。
    衡量模型是否学会使用正确的专业术语，而非泛化描述。"""
    gold_terms = [t for t in INDUSTRY_TERMS if t in gold]
    if not gold_terms:
        return None  # 该题无术语，不参与术语准确性统计
    hit = sum(1 for t in gold_terms if t in pred)
    return hit / len(gold_terms)


def evaluate(model, processor, rows, tag, out_path):
    results = []
    t0 = time.time()
    for rec in rows:
        pred = generate_answer(model, processor, rec)
        gold = rec["answer"]
        ta = industry_term_accuracy(pred, gold)
        item = {
            "question": rec["question"],
            "gold": gold,
            "pred": pred,
            "image": rec.get("image", ""),
            "bleu": round(bleu_score(pred, gold), 4),
            "rouge_l": round(rouge_l(pred, gold), 4),
            "exact": exact_match(pred, gold),
            "term_hits": term_coverage(pred),
            "term_acc": ta,
            "is_image": bool(rec.get("image")),
        }
        results.append(item)
    elapsed = time.time() - t0
    n = len(results)
    term_vals = [r["term_acc"] for r in results if r["term_acc"] is not None]
    summary = {
        "tag": tag,
        "num": n,
        "avg_bleu": round(sum(r["bleu"] for r in results)/n, 4),
        "avg_rouge_l": round(sum(r["rouge_l"] for r in results)/n, 4),
        "exact_acc": round(sum(r["exact"] for r in results)/n, 4),
        "industry_term_acc": round(sum(term_vals)/len(term_vals), 4) if term_vals else 0.0,
        "term_acc_num": len(term_vals),
        "avg_term_hits": round(sum(r["term_hits"] for r in results)/n, 2),
        "latency_per_q_ms": round(elapsed/n*1000, 1),
        # 图纸推理正确性：含图样本的 exact 命中率
        "image_exact_acc": round(sum(r["exact"] for r in results if r["is_image"])/max(1,sum(1 for r in results if r["is_image"])), 4),
    }
    output = {"summary": summary, "details": results}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"[{tag}] {summary}")
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eval", type=int, default=18)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-finetuned", action="store_true")
    args = ap.parse_args()

    eval_rows = load_jsonl(os.path.join(DATA_DIR, "eval.jsonl"))[:args.max_eval]
    logger.info(f"评估集 {len(eval_rows)} 条")

    out_dir = "/home/dabaie/code/工单/工单十六/docs/screenshots"
    if not args.skip_baseline:
        m, p = build_model(adapter_path=None)
        evaluate(m, p, eval_rows, "baseline", os.path.join(out_dir, "eval_baseline.json"))
        del m; torch.cuda.empty_cache()
    if not args.skip_finetuned:
        m, p = build_model(adapter_path=ADAPTER_DIR)
        evaluate(m, p, eval_rows, "finetuned", os.path.join(out_dir, "eval_finetuned.json"))
        del m; torch.cuda.empty_cache()
    print("评估完成")


if __name__ == "__main__":
    main()
