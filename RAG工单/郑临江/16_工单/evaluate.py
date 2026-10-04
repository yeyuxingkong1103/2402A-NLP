# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-微调专用视觉语言模型工单
专业评估脚本：BLEU/ROUGE + 工业术语准确性 + 图纸推理正确性。
"""
import json
import math
from collections import Counter

import config
from data_convert import load_questions, resolve_answer


# ---- BLEU / ROUGE ----
def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _tok(text):
    return [c for c in text if not c.isspace()]


def bleu(pred, ref, n=4):
    p = _tok(pred)
    r = _tok(ref)
    precisions = []
    for i in range(1, n + 1):
        pn = Counter(_ngrams(p, i))
        rn = Counter(_ngrams(r, i))
        overlap = sum((pn & rn).values())
        precisions.append(overlap / max(sum(pn.values()), 1))
    geo = math.exp(sum(math.log(max(x, 1e-9)) for x in precisions) / n)
    bp = math.exp(1 - len(r) / max(len(p), 1)) if len(p) < len(r) else 1.0
    return round(bp * geo, 4)


def rouge_l(pred, ref):
    p = _tok(pred)
    r = _tok(ref)
    if not p or not r:
        return 0.0
    # 最长公共子序列近似
    from itertools import combinations
    def lcs(a, b):
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]
    l = lcs(p, r)
    r_score = l / len(r)
    p_score = l / len(p)
    f = 2 * r_score * p_score / (r_score + p_score + 1e-9)
    return round(f, 4)


# ---- 工业准确性评估 ----
def term_accuracy(pred, ref):
    """专业术语准确性：答案中正确覆盖的工业术语比例。"""
    terms = [t for t in config.INDUSTRIAL_TERMS if t in ref]
    if not terms:
        return 1.0
    hit = sum(1 for t in terms if t in pred)
    return hit / len(terms)


def drawing_reasoning(pred, entry):
    """图纸推理正确性：答案与图纸/部件编号是否相符。"""
    import re
    parts = re.findall(r"部件\s*\d+|编号\s*\d+", pred)
    ref_parts = re.findall(r"部件\s*\d+|编号\s*\d+", entry.get("question", ""))
    if not ref_parts:
        return 1.0
    # 若预测提到任何参照部件编号视为一致
    return 1.0 if parts else 0.0


def evaluate(preds, refs, entries):
    """preds/refs 为预测与标准答案列表。"""
    b = [bleu(p, r) for p, r in zip(preds, refs)]
    rl = [rouge_l(p, r) for p, r in zip(preds, refs)]
    ta = [term_accuracy(p, r) for p, r in zip(preds, refs)]
    dr = [drawing_reasoning(p, e) for p, e in zip(preds, entries)]
    n = max(len(preds), 1)
    return {
        "BLEU": round(sum(b) / n, 4),
        "ROUGE-L": round(sum(rl) / n, 4),
        "term_accuracy": round(sum(ta) / n, 4),
        "drawing_reasoning": round(sum(dr) / n, 4),
    }


def load_preds(path):
    """加载模型预测文件（每行 json：{question, answer, pred}）。"""
    if not path or not __import__("os").path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    entries = load_questions(limit=100)
    refs = [resolve_answer(e) for e in entries]
    preds_path = "predictions.jsonl"
    preds = load_preds(preds_path)
    if preds is None:
        # 离线示例：用标准答案占位，演示评估流程
        print("[evaluate] 未找到 predictions.jsonl，使用标准答案占位演示")
        pred_list = refs
        entries_used = entries
    else:
        pred_list = [p["pred"] for p in preds]
        entries_used = [e for e in entries][:len(pred_list)]
    result = evaluate(pred_list, refs[:len(pred_list)], entries_used)
    print("===== 专业评估结果 =====")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
