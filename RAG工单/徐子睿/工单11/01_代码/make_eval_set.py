# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG 项目-Embeddings 模型微调任务
# 模块：embedding_ft/make_eval_set —— 构造「留出页 + 全新问法」检索测试集
# 说明：工单验收要求「微调后检索效果比微调前要好，且有指标支撑」。只用业务 16 题做测试
#       容易触碰天花板（基座 bge-base-zh 在中文上本就不弱）。因此额外构造一套**留出集**：
#       ① 页面留出：排除 16 题金标准页，避免测试页与训练页重合；
#       ② 问法留出：使用训练阶段**没有出现过**的 8 种口语化/换句式模板，考察泛化；
#       产出 `embedding_ft/data/eval_pairs.jsonl`（query / doc / page / positive）。
# 用法：python embedding_ft/make_eval_set.py [--pages 30]
import argparse
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "app"))
sys.stdout.reconfigure(encoding="utf-8")

CHUNKS = os.path.join(ROOT, "data", "index", "chunks.jsonl")
OUT = os.path.join(HERE, "data", "eval_pairs.jsonl")

# 与训练模板（gen_pairs.py）刻意不同的 8 种问法
TEMPLATES = [
    "招股书里关于「{kw}」是怎么写的？那一页的原文我想看一下",
    "帮我翻一下 {doc}，{kw} 具体是多少来着？",
    "我记不太清了，{kw} 好像是 {val} 左右，对吗？（请给出原文页）",
    "{kw} 在 {doc} 里出现在哪一页？请把那一页内容给我",
    "麻烦查一下 {doc} 的 {kw}，只要原文不要解释",
    "领导问 {kw} 这个数，我该拿哪一页给他看？",
    "有没有哪一页同时提到 {kw} 和具体数字？帮我定位",
    "{kw} 这一项的数据出处在哪里？",
]


def load_pages():
    """(doc,page) -> 文本（仅文本/表格块）"""
    g = {}
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c.get("type") == "image":
                continue
            g.setdefault((c.get("doc"), str(c.get("page"))), []).append(c.get("text") or "")
    return {k: "\n".join(t for t in v if t) for k, v in g.items()}


def gold_pages():
    ex = set()
    for rel, key in (("evaluation/eval_questions.json", "page"),
                     ("evaluation/eval_questions_pdf2.json", "gold_page")):
        d = json.load(open(os.path.join(ROOT, rel), encoding="utf-8"))
        for it in (d.get("questions") or d.get("items")):
            for p in (it.get(key) or []):
                ex.add((it.get("doc") or ("招股说明书2" if "pdf2" in rel else "招股说明书1"), str(p)))
            if it.get("doc") is None:          # 金标准页没有归属文档时，两种文档都排除
                for p in (it.get(key) or []):
                    ex.add(("招股说明书1", str(p)))
                    ex.add(("招股说明书2", str(p)))
    return ex


def pick_kw(text):
    """从页里挑一个「指标词 + 数字」作为关键词，保证问题有据可查。"""
    m = re.search(r"([\u4e00-\u9fa5]{2,10}(?:收入|利润|费用|毛利率|负债率|占比|金额|余额|股数|比例|收益率|资产|负债))"
                  r"[^0-9%]{0,12}([0-9][0-9,\.]*\s*(?:%|万元|亿元|万股|元))", text)
    if m:
        return m.group(1), m.group(2)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed)

    pages = load_pages()
    ex = gold_pages()
    cands = []
    for (doc, page), txt in sorted(pages.items(), key=lambda x: (x[0][0], int(x[0][1]))):
        if (doc, page) in ex or len(txt) < 120:
            continue
        kw, val = pick_kw(txt)
        if not kw:
            continue
        cands.append({"doc": doc, "page": page, "kw": kw, "val": val, "positive": txt[:300]})
    random.shuffle(cands)
    chosen = cands[:max(1, a.pages)]
    rows = []
    for i, c in enumerate(chosen):
        t = TEMPLATES[i % len(TEMPLATES)]
        q = t.format(kw=c["kw"], val=c["val"], doc=c["doc"])
        rows.append({"query": q, "doc": c["doc"], "page": c["page"],
                     "kw": c["kw"], "val": c["val"], "positive": c["positive"], "src": "eval_holdout"})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("留出评测集 =", len(rows), "条（页面留出 + 新问法）")
    for r in rows[:5]:
        print("  -", r["query"][:60], "->", r["doc"], "p" + r["page"])
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
