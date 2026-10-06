# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 关联工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化 | 人工智能NLP-RAG-图像内容解析及检索优化 | 人工智能NLP-RAG-Query理解优化任务 | 人工智能NLP-RAG-混合检索任务
# 模块：compare_before_after —— 优化前后检索精确度对比（消融实验）
# 说明：同一份索引、同一套 10 题，只切换检索侧的两个优化开关，测量“优化前(01 交付版)”
#       与“优化后(02 优化版)”的 Hit@1/@3/@5 与 MRR，并输出逐题对比表。
#       优化前 = caliber="penalty"（口径只降权、无“万元”升权；口径词只看中文）
#                + xling="basic"（英文无条件补公司全名）
#       优化后 = caliber="full" + xling="full"
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
sys.stdout.reconfigure(encoding="utf-8")

import engine        # noqa: E402
import kb as kb_mod  # noqa: E402

BEFORE = {"caliber": "penalty", "xling": "basic"}
AFTER = {"caliber": "full", "xling": "full"}


def expand_gold(item, chunks):
    gold = set(item.get("page") or [])
    for ev in item.get("evidence") or []:
        for c in chunks:
            if ev in c["text"]:
                gold.add(c["page"])
    return gold


def rank_of(hits, gold):
    for i, h in enumerate(hits, 1):
        if h["page"] in gold or (h.get("page_end") in gold if h.get("page_end") else False):
            return i
    return None


def eval_set(rows, chunks, cfg, top_k=5):
    h1 = h3 = h5 = 0
    rr = 0.0
    detail = []
    for item in rows:
        gold = expand_gold(item, chunks)
        qu = engine.query_understanding(item["question"], xling=cfg["xling"])
        hits, conf = engine.retrieve(item["question"], qu, top_k=top_k, caliber=cfg["caliber"])
        rk = rank_of(hits, gold)
        h1 += rk == 1
        h3 += bool(rk and rk <= 3)
        h5 += bool(rk and rk <= 5)
        rr += (1.0 / rk) if rk else 0.0
        detail.append({"id": item["id"], "rank": rk, "conf": round(conf, 3),
                       "top_pages": [h["page"] for h in hits[:5]]})
    n = len(rows)
    return {"n": n, "hit1": h1 / n, "hit3": h3 / n, "hit5": h5 / n, "mrr": rr / n}, detail


def main():
    zh = json.load(open(os.path.join(HERE, "eval_questions.json"), encoding="utf-8"))["questions"]
    zh_by_id = {q["id"]: q for q in zh}
    en_qs = json.load(open(os.path.join(HERE, "eval_questions_en.json"), encoding="utf-8"))["questions"]
    # 英文题沿用中文同 id 的金标准（page + evidence）
    en = [{"id": q["id"], "question": q["question"],
           "page": zh_by_id[q["id"]]["page"], "evidence": zh_by_id[q["id"]]["evidence"]}
          for q in en_qs]

    chunks = kb_mod.get_kb().chunks
    out = {"zh": {}, "en": {}}
    md = ["# 优化前后检索精确度对比（消融实验）", "",
          "> 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化",
          "> 同一索引、同一批题目，仅切换检索侧优化开关；金标准页 = 规范出处页 ∪ 含证据串的页", ""]

    for tag, rows in (("zh", zh), ("en", en)):
        b, bd = eval_set(rows, chunks, BEFORE)
        a, ad = eval_set(rows, chunks, AFTER)
        out[tag] = {"before": b, "after": a, "detail_before": bd, "detail_after": ad}
        name = "中文" if tag == "zh" else "英文（跨语）"
        md += ["", "## %s 10 题" % name, "",
               "| 指标 | 优化前 | 优化后 | 变化 |", "|---|---|---|---|",
               "| Hit@1 | %.1f%% | %.1f%% | %+.1f pt |" % (100 * b["hit1"], 100 * a["hit1"], 100 * (a["hit1"] - b["hit1"])),
               "| Hit@3 | %.1f%% | %.1f%% | %+.1f pt |" % (100 * b["hit3"], 100 * a["hit3"], 100 * (a["hit3"] - b["hit3"])),
               "| Hit@5 | %.1f%% | %.1f%% | %+.1f pt |" % (100 * b["hit5"], 100 * a["hit5"], 100 * (a["hit5"] - b["hit5"])),
               "| MRR | %.4f | %.4f | %+.4f |" % (b["mrr"], a["mrr"], a["mrr"] - b["mrr"]), "",
               "| id | 优化前名次 | 优化后名次 | 优化前 top5 页码 | 优化后 top5 页码 |",
               "|---|---|---|---|---|"]
        for x, y in zip(bd, ad):
            md.append("| %s | %s | %s | %s | %s |" % (
                x["id"], x["rank"] or "miss", y["rank"] or "miss",
                ",".join(map(str, x["top_pages"])), ",".join(map(str, y["top_pages"]))))
        print("[%s] before Hit@1=%.0f%% Hit@3=%.0f%% Hit@5=%.0f%% MRR=%.4f" %
              (tag, 100 * b["hit1"], 100 * b["hit3"], 100 * b["hit5"], b["mrr"]))
        print("[%s] after  Hit@1=%.0f%% Hit@3=%.0f%% Hit@5=%.0f%% MRR=%.4f" %
              (tag, 100 * a["hit1"], 100 * a["hit3"], 100 * a["hit5"], a["mrr"]))

    json.dump(out, open(os.path.join(HERE, "compare_before_after.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    open(os.path.join(HERE, "优化前后对比.md"), "w", encoding="utf-8").write("\n".join(md))
    print("saved -> 优化前后对比.md / compare_before_after.json")


if __name__ == "__main__":
    main()
