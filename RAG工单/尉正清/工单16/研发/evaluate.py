# 工单16：人工智能NLP-RAG-微调专用视觉语言模型
"""工业场景下的专业评估

工单要求「不仅计算 BLEU/ROUGE，更要设计工业准确性评估」，所以这里有三层指标：

| 指标 | 怎么算 | 对应用途 |
|---|---|---|
| `term_acc` | 标准答案里的**工业术语**是否出现在模型输出里 | a) 专业术语准确性（验收看这个） |
| `drawing_acc` | 图文题（group 2/3）的选项是否答对 | b) 图纸推理正确性 |
| `letter_acc` | 全部题的选项是否答对 | 整体选择题准确率 |
| `bleu` / `rouge_l` | 字符级 4-gram BLEU（带简短惩罚）与 ROUGE-L F1 | 文本重合度，作参考 |

BLEU 和 ROUGE 都自己实现（字符级，中文按字算）—— 不引第三方包，
免得又碰这个项目里深度学习库互相锁版本的老问题。

用法：
    python evaluate.py --tag base                       # 基线（未微调）
    python evaluate.py --tag finetuned --adapter output/qwen2vl-2b-lora-imdr
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).parent


# ---------------------------------------------------------------- 指标

def normalize(s):
    s = re.sub(r"[，。；、,.;:！？!?（）()【】\[\]“”\"'\s]", "", s or "")
    return s.strip().lower()


def char_ngrams(s, n):
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def bleu(pred, ref, max_n=4):
    """字符级 BLEU。中文按字算 n-gram 是通行做法，也不依赖分词器。"""
    p, r = normalize(pred), normalize(ref)
    if not p or not r:
        return 0.0
    logs = []
    for n in range(1, max_n + 1):
        pc, rc = char_ngrams(p, n), char_ngrams(r, n)
        if not pc:
            break
        overlap = sum((pc & rc).values())
        logs.append(overlap / max(1, sum(pc.values())))
    if not logs or min(logs) <= 0:
        return 0.0
    import math
    bp = min(1.0, math.exp(1 - len(r) / max(1, len(p))))
    return bp * math.exp(sum(math.log(x) for x in logs) / len(logs))


def rouge_l(pred, ref):
    """字符级 ROUGE-L F1（最长公共子序列）。"""
    p, r = normalize(pred), normalize(ref)
    if not p or not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(p) + 1)]
    for i in range(1, len(p) + 1):
        for j in range(1, len(r) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if p[i - 1] == r[j - 1] \
                else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[len(p)][len(r)]
    if not lcs:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def term_hit(pred, term):
    """术语是否出现在输出里。术语按空格切开逐段判断，全中才算命中。"""
    if not term:
        return None
    np_, parts = normalize(pred), [normalize(t) for t in term.split() if len(t) >= 2]
    if not parts:
        return None
    # 只要有一段能对上就算命中（术语常写成「链条」或「环形链条」）
    return any(t in np_ for t in parts)


def letter_hit(pred, letter):
    if not letter:
        return None
    return letter.upper() in (pred or "").upper()[:40]


# ---------------------------------------------------------------- 推理

def load_model(base, adapter, cfg):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    kwargs = {"attn_implementation": cfg["model"].get("attn_implementation", "sdpa"),
              "torch_dtype": torch.bfloat16}
    if cfg["model"].get("load_in_4bit"):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = {"": 0}
    model = Qwen2VLForConditionalGeneration.from_pretrained(base, **kwargs)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        base, min_pixels=cfg["image"]["min_pixels"], max_pixels=cfg["image"]["max_pixels"])
    return model, processor


def generate(model, processor, row, max_new_tokens=64):
    import torch
    from PIL import Image

    img = Image.open(row["image"]).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": row["question"]}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0][enc["input_ids"].shape[1]:]
    return processor.tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "finetune_config.json"))
    ap.add_argument("--tag", required=True, help="结果文件名，如 base / finetuned")
    ap.add_argument("--adapter", default="", help="LoRA adapter 目录，基线留空")
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 条（调试用）")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in
            (HERE / cfg["data"]["eval"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"=== 评估 [{args.tag}] ===\n  样本 {len(rows)} 条  "
          f"adapter={args.adapter or '（基线，未微调）'}", flush=True)

    model, processor = load_model(cfg["model"]["base"], args.adapter, cfg)

    out, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        pred = generate(model, processor, r)
        out.append({**{k: r[k] for k in ("question", "answer", "term", "group", "doc")},
                    "pred": pred,
                    "term_hit": term_hit(pred, r.get("term")),
                    "letter_hit": letter_hit(pred, r.get("letter")),
                    "bleu": round(bleu(pred, r["answer"]), 4),
                    "rouge_l": round(rouge_l(pred, r["answer"]), 4)})
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}  已用 {time.time() - t0:.0f}s", flush=True)

    def rate(key):
        vals = [x[key] for x in out if x[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    draw = [x for x in out if x["group"] in (2, 3)]
    summary = {
        "tag": args.tag,
        "adapter": args.adapter,
        "n": len(out),
        "term_acc": round(rate("term_hit"), 4),
        "letter_acc": round(rate("letter_hit"), 4),
        "drawing_acc": round(sum(1 for x in draw if x["letter_hit"]) / max(1, len(draw)), 4),
        "n_drawing": len(draw),
        "bleu": round(sum(x["bleu"] for x in out) / len(out), 4),
        "rouge_l": round(sum(x["rouge_l"] for x in out) / len(out), 4),
        "sec": round(time.time() - t0, 1),
    }

    res = HERE.parent / "测试" / "results"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{args.tag}.json").write_text(
        json.dumps({"summary": summary, "items": out}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    print("\n=== 结果 ===")
    for k in ("term_acc", "letter_acc", "drawing_acc", "bleu", "rouge_l"):
        print(f"  {k:12} {summary[k]:.4f}")
    print(f"  存到 {res / (args.tag + '.json')}")


if __name__ == "__main__":
    main()
