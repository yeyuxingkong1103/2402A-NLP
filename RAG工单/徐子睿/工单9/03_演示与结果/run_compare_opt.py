# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
# 关联工单：人工智能NLP-RAG-基于Graph RAG实现金融问答 | 人工智能NLP-RAG-Graph RAG优化任务
# 模块：run_compare_opt —— 工单 09「优化前后对比」：同一批题目分别跑
#       GRAPH_MODE=v1（工单08 初版：叙事页一律置顶）与 GRAPH_MODE=opt（工单09 优化：仅补漏页），
#       对比命中率 / MRR / 关键数值命中率 / 时延，产出 `evaluation/比较_优化前后.md`。
import importlib
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
sys.stdout.reconfigure(encoding="utf-8")

import engine      # noqa: E402
import kb as kb_mod  # noqa: E402

Q_PATH = os.path.join(HERE, "eval_questions_fin.json")
OUT_MD = os.path.join(HERE, "比较_优化前后.md")
OUT_JSON = os.path.join(HERE, "opt_compare.json")

UNIT = r"\d[\d,]*\.?\d*\s*(?:亿元|万元|%)"


def num_hit(text, gold):
    """关键数值命中：金标准里的“数字+单位”是否出现在答案里。"""
    ks = re.findall(UNIT, gold or "")
    if not ks:
        return None
    return all(k.replace(",", "") in (text or "").replace(",", "") for k in ks)


def run_mode(mode, items, km):
    os.environ["GRAPH_MODE"] = mode
    import graph_rag
    importlib.reload(graph_rag)           # 让 PREPEND_MODE 生效
    h1 = h3 = h5 = 0
    rr = 0.0
    ms = []
    detail = []
    for it in items:
        gold_pages = set(it.get("gold_page") or [])
        for c in km.chunks:
            if c["doc"] == it.get("company") and any(k in c["text"] for k in (it.get("gold_kw") or [])):
                gold_pages.add(c["page"])
        t0 = time.time()
        hits, conf, info = graph_rag.retrieve(it["question"], top_k=5)
        dt = int((time.time() - t0) * 1000)
        ms.append(dt)
        rank = next((i for i, x in enumerate(hits, 1) if x["page"] in gold_pages), None)
        h1 += rank == 1
        h3 += bool(rank and rank <= 3)
        h5 += bool(rank and rank <= 5)
        rr += (1.0 / rank) if rank else 0.0
        detail.append({"id": it["id"], "question": it["question"], "pages": [h["page"] for h in hits],
                       "gold_pages": sorted(gold_pages), "rank": rank, "cost_ms": dt})
    n = len(items)
    return {"mode": mode, "n": n, "hit@1": h1 / n, "hit@3": h3 / n, "hit@5": h5 / n, "mrr": rr / n,
            "avg_ms": round(sum(ms) / n, 1), "detail": detail}


def main():
    km = kb_mod.get_kb()
    items = json.load(open(Q_PATH, encoding="utf-8"))["questions"]
    v1 = run_mode("v1", items, km)
    opt = run_mode("opt", items, km)

    def pct(x):
        return "%.1f%%" % (100 * x)

    md = ["# 工单 09 · Graph RAG 优化前后对比", "",
          "> 工单编号：人工智能NLP-RAG-Graph RAG优化任务",
          "> 同题集（10 题）、同索引、同 top_k=5；唯一差异 = 图谱证据页置顶策略。",
          "> - **优化前（v1，工单 08 初版）**：叙事句来源页一律置顶（最多 2 页）",
          "> - **优化后（opt，工单 09）**：仅当「混合检索完全没召回该页」且该页命中 ≥2 个指标时，补充置顶 1 页", ""]
    md += ["## 一、检索指标对比", "",
           "| 指标 | 优化前 v1 | **优化后 opt** | 变化 |",
           "|---|---|---|---|",
           "| Hit@1 | %s | **%s** | %+.1f 个百分点 |" % (pct(v1["hit@1"]), pct(opt["hit@1"]),
                                                 100 * (opt["hit@1"] - v1["hit@1"])),
           "| Hit@3 | %s | **%s** | %+.1f 个百分点 |" % (pct(v1["hit@3"]), pct(opt["hit@3"]),
                                                 100 * (opt["hit@3"] - v1["hit@3"])),
           "| Hit@5 | %s | **%s** | %+.1f 个百分点 |" % (pct(v1["hit@5"]), pct(opt["hit@5"]),
                                                 100 * (opt["hit@5"] - v1["hit@5"])),
           "| MRR | %.4f | **%.4f** | %+.4f |" % (v1["mrr"], opt["mrr"], opt["mrr"] - v1["mrr"]),
           "| 平均检索时延 | %.0f ms | **%.0f ms** | %+.0f ms |" % (v1["avg_ms"], opt["avg_ms"],
                                                            opt["avg_ms"] - v1["avg_ms"]),
           "",
           "> 与工单 07 基线（常规混合检索）对比：Hit@1 60%% / Hit@3 80%% / Hit@5 90%% / MRR 0.725。", ""]
    md += ["## 二、逐题明细（命中排名）", "",
           "| 题号 | 公司 | 问题 | v1 排名 | opt 排名 | opt 命中页（top5） |",
           "|---|---|---|---|---|---|"]
    for a, b in zip(v1["detail"], opt["detail"]):
        md.append("| %s | %s | %s | %s | %s | %s |" %
                  (a["id"], items[int(a["id"]) - 1].get("company", ""), a["question"],
                   a["rank"] or "MISS", b["rank"] or "MISS", b["pages"]))
    md += ["", "## 三、优化点小结", "",
           "1. **不再无条件置顶叙事页**：初版会把「只命中 1 条三元组的噪声页」顶到第 1 位，"
           "把本来排第 1 的正确答案页挤到第 2 位（如 id1、id6、id7）；",
           "2. **只在“混合检索漏召”时补位**：图谱的价值在于补检索的漏，而不是覆盖排序；",
           "3. **要求 ≥2 个指标命中**：过滤单指标误抽（如把“保险业务收入 0.01%”当成 2021 年值）；",
           "4. 效果：Hit@1 %s→%s、MRR %.4f→%.4f，同时保留 id9（混合检索漏召、图谱补位）的命中。" %
           (pct(v1["hit@1"]), pct(opt["hit@1"]), v1["mrr"], opt["mrr"]), ""]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(md))
    json.dump({"v1": v1, "opt": opt}, open(OUT_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("v1 : Hit@1=%s Hit@3=%s Hit@5=%s MRR=%.4f" % (pct(v1["hit@1"]), pct(v1["hit@3"]),
                                                        pct(v1["hit@5"]), v1["mrr"]))
    print("opt: Hit@1=%s Hit@3=%s Hit@5=%s MRR=%.4f" % (pct(opt["hit@1"]), pct(opt["hit@3"]),
                                                        pct(opt["hit@5"]), opt["mrr"]))
    print("saved ->", OUT_MD)


if __name__ == "__main__":
    main()
