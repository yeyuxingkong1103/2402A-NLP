# -*- coding: utf-8 -*-
"""
工单 16 · 工业准确性评估脚本（BLEU/ROUGE + 专业术语 + 图纸推理）

输入 pred/ref（JSONL，每行一条）：
  {"id":1,"question":"...","pred":"模型输出","ref":"标准答案",
   "type":"text|image","terms":["淬火","公差配合"]}
输出（stdout + --out json）：
  - BLEU（字符 4-gram，中文友好；若装了 sacrebleu 可切换）
  - ROUGE-L F1
  - 专业术语准确性：ref 中术语被 pred 覆盖的比例
  - 图纸推理正确性：type=image 的样本中，pred 与 ref 语义/选项一致的比例
  - 总准确率 + 分类型准确率

用法：
  python eval_industrial.py --pred pred.jsonl --ref ref.jsonl [--out result.json]
  python eval_industrial.py --self-test          # 内置样例自检
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from typing import List


# ---------------- 文本相似度 ----------------
def _char_ngrams(text: str, n: int):
    text = "".join(text.split())
    return Counter(text[i:i + n] for i in range(max(0, len(text) - n + 1)))


def char_bleu(pred: str, ref: str, n: int = 4) -> float:
    """字符 n-gram BLEU（带 brevity penalty），中文场景下比词级更稳。"""
    if not pred or not ref:
        return 0.0
    p = _char_ngrams(pred, n)
    r = _char_ngrams(ref, n)
    if not p:
        return 0.0
    overlap = sum((p & r).values())
    prec = overlap / max(1, sum(p.values()))
    bp = min(1.0, math.exp(1 - len(ref) / max(1, len(pred))))
    return round(prec * bp, 4)


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F1（按字符 LCS）。"""
    a, b = "".join(pred.split()), "".join(ref.split())
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[len(a)][len(b)]
    prec, rec = lcs / len(a), lcs / len(b)
    return round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0


def option_hit(pred: str, ref: str) -> bool:
    """选择题判定：取 pred 首个 'A/B/C/D' 字母与 ref 比对；否则做包含匹配。"""
    import re
    p = re.search(r"[ABCDＡ-Ｄ]", pred or "")
    r = re.search(r"[ABCDＡ-Ｄ]", ref or "")
    if p and r:
        return p.group(0).upper() == r.group(0).upper()
    key = "".join(ref.split())
    return bool(key) and key in "".join((pred or "").split())


# ---------------- 评估主流程 ----------------
def evaluate(pairs: List[dict]) -> dict:
    bleus, rouges = [], []
    term_hit = term_tot = 0
    img_ok = img_tot = 0
    txt_ok = txt_tot = 0
    for it in pairs:
        pred, ref = it.get("pred", ""), it.get("ref", "")
        bleus.append(char_bleu(pred, ref))
        rouges.append(rouge_l(pred, ref))
        for t in (it.get("terms") or []):
            term_tot += 1
            if t and t in pred:
                term_hit += 1
        if it.get("type") == "image":
            img_tot += 1
            img_ok += int(option_hit(pred, ref))
        else:
            txt_tot += 1
            txt_ok += int(option_hit(pred, ref))
    n = max(1, len(pairs))
    return {
        "n": len(pairs),
        "bleu4": round(sum(bleus) / n, 4),
        "rouge_l": round(sum(rouges) / n, 4),
        "term_accuracy": round(term_hit / term_tot, 4) if term_tot else None,
        "term_coverage": f"{term_hit}/{term_tot}",
        "image_reasoning_acc": round(img_ok / img_tot, 4) if img_tot else None,
        "text_acc": round(txt_ok / txt_tot, 4) if txt_tot else None,
        "overall_acc": round((img_ok + txt_ok) / max(1, img_tot + txt_tot), 4),
    }


def self_test():
    pairs = [
        {"id": 1, "type": "text", "question": "发明人", "pred": "A. P·吉特勒",
         "ref": "A. P·吉特勒", "terms": ["P·吉特勒"]},
        {"id": 2, "type": "image", "question": "部件4相对部件5", "pred": "A. 部件4位于部件5的左侧",
         "ref": "A. 部件4位于部件5的左侧", "terms": ["部件4", "部件5"]},
        {"id": 3, "type": "image", "question": "x", "pred": "B. 右侧", "ref": "A. 左侧", "terms": ["配气带孔盘"]},
    ]
    print(json.dumps(evaluate(pairs), ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred")
    ap.add_argument("--ref")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    pred = [json.loads(l) for l in open(args.pred, encoding="utf-8") if l.strip()]
    ref = [json.loads(l) for l in open(args.ref, encoding="utf-8") if l.strip()]
    # 以 id 对齐 pred/ref
    rmap = {r.get("id", i): r for i, r in enumerate(ref)}
    pairs = []
    for i, p in enumerate(pred):
        r = rmap.get(p.get("id", i), {})
        pairs.append({**r, **p, "ref": p.get("ref", r.get("ref", ""))})
    res = evaluate(pairs)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.out:
        json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
