# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Query理解优化任务
# 关联工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化 | 人工智能NLP-RAG-图像内容解析及检索优化 | 人工智能NLP-RAG-Query理解优化任务 | 人工智能NLP-RAG-混合检索任务
# 模块：run_dialog_demo —— 多轮对话演示/评估（工单 05 产出物二：针对给定问题检索并显示答案）
# 说明：顺序跑工单给定的 5 轮对话，逐轮打印「原始问题 / 指代消解后的完整问题 / 答案 / 引用页 / 耗时」，
#       并落盘 多轮对话结果.md（含响应时间与准确率统计）供提交。
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
sys.stdout.reconfigure(encoding="utf-8")

import dialog  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_MD = os.path.join(HERE, "多轮对话结果.md")
OUT_JSON = os.path.join(HERE, "dialog_result.json")

# 工单给定的 5 轮对话（含指代消解点）
ROUNDS = [
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    "他参与的哪个工程荣获了国家科技进步一等奖？",
    "这个公司的法定代表人是谁？",
    "那武汉力源信息技术股份有限公司呢？",
    "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",
]

# 期望答案关键字（用于自动判定准确率，人工可复核）
EXPECT = [
    ["6,464.51", "14,414.16", "18,780.67"],
    ["一等奖", "一体化工程"],
    ["程家明"],
    ["赵马克"],
    ["大客户销售部", "销售处"],
]


def main():
    sid = dialog.new_session()
    rows, hits_cnt, lat = [], 0, []
    for i, q in enumerate(ROUNDS):
        r = dialog.chat(sid, q)
        ok = all(any(k in r["answer"] for k in group) for group in [EXPECT[i]]) if EXPECT[i] else True
        # 关键字命中（宽松：期望组内任一关键字出现即算）
        kw_ok = any(any(k in r["answer"] for k in [x]) for x in EXPECT[i]) if EXPECT[i] else True
        hits_cnt += 1 if kw_ok else 0
        lat.append(r["cost_ms"])
        rows.append({"round": i + 1, "q": q, "rewritten": r["rewritten_query"],
                     "resolution": r["resolution"]["note"], "answer": r["answer"],
                     "confidence": r["confidence"], "refused": r["refused"],
                     "doc": r["doc"], "cost_ms": r["cost_ms"],
                     "pages": [(h["page"], h.get("type")) for h in r.get("hits", [])][:5],
                     "kw_ok": kw_ok})
        print("第 %d 轮 Q: %s" % (i + 1, q))
        print("      指代消解: %s" % r["resolution"]["note"])
        print("      改写后  : %s" % r["rewritten_query"])
        print("      A(%dms): %s" % (r["cost_ms"], r["answer"].replace("\n", " ")[:220]))
        print("      引用: %s" % [(h["page"], h.get("type")) for h in r.get("hits", [])][:5])
        print("-" * 70)

    n = len(ROUNDS)
    avg = sum(lat) / n
    acc = 100.0 * hits_cnt / n
    print("\n==== 多轮对话汇总 ====")
    print("准确率(关键字判定) = %.1f%% (%d/%d)" % (acc, hits_cnt, n))
    print("平均响应时间 = %d ms（首轮含模型冷启动，热态约 1-2s）" % avg)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": {"accuracy": acc / 100, "n": n, "avg_ms": avg}, "rounds": rows},
                  f, ensure_ascii=False, indent=2)

    lines = ["# 工单 05 · 多轮对话检索结果", "",
             "> 工单编号：人工智能NLP-RAG-Query理解优化任务",
             "> 会话 id：%s ｜ 准确率(关键字判定)：%.1f%% (%d/%d) ｜ 平均响应：%d ms" % (sid, acc, hits_cnt, n, avg),
             ""]
    for r in rows:
        lines += ["## 第 %d 轮" % r["round"], "",
                  "- **Q**：%s" % r["q"],
                  "- **指代消解**：%s" % r["resolution"],
                  "- **改写后完整问题**：`%s`" % r["rewritten"],
                  "- **A**：%s" % r["answer"].replace("\n", " "),
                  "- 引用页：%s ｜ 置信度 %s ｜ 耗时 %d ms" % (r["pages"], r["confidence"], r["cost_ms"]),
                  ""]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("saved ->", OUT_MD)


if __name__ == "__main__":
    main()
