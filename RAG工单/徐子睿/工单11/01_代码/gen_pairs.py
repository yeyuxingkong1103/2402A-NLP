# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG 项目-Embeddings 模型微调任务
# 关联工单：人工智能NLP-RAG-基于PDF文档的问答系统 | 人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 模块：embedding_ft/gen_pairs —— 领域问答对（训练数据）生成
# 说明：两条路生成「query → 正例段落」：
#       ① 模板路：从招股书分块里识别「指标 + 数值」句，套用业务问法（覆盖广、量足）；
#       ② LLM 路：抽样若干段落，让本地 qwen2:7b 生成「只能由该段回答」的问题（更自然）。
#       产出 `embedding_ft/data/pairs.jsonl`（query / positive / doc / page / src）。
# 用法：python embedding_ft/gen_pairs.py [--llm 120] [--template 900] [--llm-only]
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
OUT = os.path.join(HERE, "data", "pairs.jsonl")

INDICATORS = ["营业收入", "净利润", "归属于母公司股东的净利润", "归属于母公司所有者的净利润",
              "主营业务收入", "毛利率", "资产负债率", "研发费用", "研发投入", "销售费用",
              "管理费用", "财务费用", "基本每股收益", "每股收益", "加权平均净资产收益率",
              "流动比率", "速动比率", "募集资金", "发行股数", "注册资本", "总资产",
              "应收账款", "存货", "经营活动产生的现金流量净额", "前五大客户", "前五大供应商",
              "市场占有率", "军用领域收入", "毛利率", "期间费用率"]
COMPANIES = ["武汉兴图新科电子股份有限公司", "武汉力源信息技术股份有限公司",
             "兴图新科", "力源信息"]

NUM = re.compile(r"\d[\d,]*\.?\d*\s*(?:万元|亿元|元|%|％|股|倍|天)")


def load_chunks():
    out = []
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c.get("type") == "image":
                continue
            t = (c.get("text") or "").strip()
            if 60 <= len(t) <= 900:
                out.append(c)
    return out


def _ngrams(s, n=3):
    s = re.sub(r"\s+", "", s or "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def load_eval_questions():
    """载入评测题（招股书1 中文10题 + 招股书2 6题），用于**去泄漏**。"""
    qs = []
    for rel in ("evaluation/eval_questions.json", "evaluation/eval_questions_pdf2.json"):
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        for it in (d.get("questions") or d.get("items") or []):
            if it.get("question"):
                qs.append(it["question"])
    return qs


def leaks(query, eval_grams, thr=0.5):
    """字符 3-gram Jaccard 超阈 → 视为与评测题重叠（避免训练集泄露评测集）。"""
    g = _ngrams(query)
    if not g:
        return False
    for eg in eval_grams:
        inter = len(g & eg)
        if inter and inter / len(g | eg) >= thr:
            return True
    return False


def template_pairs(chunks, cap):
    """模板路：指标 + 数值句 → 业务问法（多用几种句式，每块可取 2 个不同指标）。"""
    pairs, seen = [], set()
    random.shuffle(chunks)
    ANYNUM = re.compile(r"\d")
    for c in chunks:
        t = c["text"]
        if c.get("type") != "table" and not ANYNUM.search(t):
            continue
        hit_ind = [i for i in INDICATORS if i in t]
        if not hit_ind:
            continue
        hit_co = [x for x in COMPANIES if x in t]
        co = hit_co[0] if hit_co else c.get("doc", "")
        made = 0
        for ind in hit_ind[:2]:
            q = "%s的%s是多少？" % (co, ind)
            if made:
                q = "根据%s，%s的%s为多少？" % (c.get("doc", ""), co, ind)
            key = (q, c.get("page"))
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"query": q, "positive": t, "doc": c.get("doc"),
                          "page": c.get("page"), "src": "template"})
            made += 1
        if len(pairs) >= cap:
            break
    return pairs


def llm_pairs(chunks, n):
    """LLM 路：让 qwen2:7b 依据段落出题（只能由该段回答）。"""
    try:
        import llm
    except Exception as e:  # noqa: BLE001
        print("LLM 路跳过：", e)
        return []
    SYS = ("你是金融文档问答数据标注员。用户给你一段招股说明书原文，"
           "你要提出 1 个**只能依据这段原文回答**的中文问题（10–30 字），"
           "要求包含公司名或关键指标，不要问页码，不要问“这段文字是什么”。只输出问题本身。")
    out = []
    sample = random.sample(chunks, min(n, len(chunks)))
    for i, c in enumerate(sample, 1):
        try:
            q = llm.chat([{"role": "system", "content": SYS},
                          {"role": "user", "content": c["text"][:1200]}],
                         temperature=0.3, num_predict=64).strip()
            q = q.split("\n")[0].strip("。？? ")
            if 8 <= len(q) <= 60:
                out.append({"query": q + "？", "positive": c["text"], "doc": c.get("doc"),
                            "page": c.get("page"), "src": "llm"})
        except Exception as e:  # noqa: BLE001
            print("  [%d] gen fail: %r" % (i, e))
        if i % 20 == 0:
            print("  LLM 路进度 %d/%d" % (i, len(sample)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=int, default=900)
    ap.add_argument("--llm", type=int, default=120)
    ap.add_argument("--llm-only", action="store_true")
    ap.add_argument("--keep-existing", action="store_true", help="保留已有 pairs 并追加")
    a = ap.parse_args()
    random.seed(20261004)
    chunks = load_chunks()
    print("可用段落 =", len(chunks))

    eval_grams = [_ngrams(q) for q in load_eval_questions()]
    print("评测题（用于去泄漏）=", len(eval_grams))

    tpl = llm_p = []
    if not a.llm_only:
        tpl = template_pairs(chunks, a.template)
        n0 = len(tpl)
        tpl = [p for p in tpl if not leaks(p["query"], eval_grams)]
        print("模板路 pairs = %d（去泄漏后 %d）" % (n0, len(tpl)))
    if a.llm:
        llm_p = llm_pairs(chunks, a.llm)
        n1 = len(llm_p)
        llm_p = [p for p in llm_p if not leaks(p["query"], eval_grams)]
        print("LLM 路 pairs = %d（去泄漏后 %d）" % (n1, len(llm_p)))

    allp = []
    if a.keep_existing and os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            line = line.strip()
            if line:
                allp.append(json.loads(line))
        print("已有 pairs =", len(allp))
    seen = {(p["query"], p.get("page")) for p in allp}
    for p in tpl + llm_p:
        k = (p["query"], p.get("page"))
        if k in seen:
            continue
        seen.add(k)
        allp.append(p)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for p in allp:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("saved ->", OUT, "共", len(allp), "对")
    from collections import Counter
    print("来源分布：", dict(Counter(p.get("src", "?") for p in allp)))


if __name__ == "__main__":
    main()
