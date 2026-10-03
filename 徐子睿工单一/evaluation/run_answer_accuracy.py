# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 模块：run_answer_accuracy —— 答案准确率评测（验收口径：准确率 ≥ 90%）
# 说明：对 qa_bilingual_result.json 里 RAG 链路给出的答案，按“金标准事实/证据串是否命中”
#       做规则判分（可与人工核对、可复现），输出逐题判定与总体准确率。
#       判分规则：数值题要求全部数值都在答案中（忽略空格/千分位差异）；
#                 事实/列举题要求各组关键词至少命中一个（组见 CHECKS）。
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")

# 每题：checks = [(组内任一命中即可), ...]；全部组都命中 => 完全正确
CHECKS = {
    "zh": {
        260: [["6,464.51"], ["14,414.16"], ["18,780.67"], ["4,627.14"]],
        95: [["视频指挥系统技术标准", "视频技术规范", "技术标准"]],
        33: [["82.10"], ["97.31"], ["94.84"], ["94.34"]],
        34: [["电子元器件"], ["机箱", "机柜", "金属壳体"]],
        957: [["国防军队视频指挥", "军队视频指挥", "军工", "国防"]],
        793: [["军队"], ["政府机关", "政府"], ["能源"]],
        795: [["国家科技进步一等奖", "科技进步一等奖"], ["情报", "C4ISR", "一体化工程"]],
        543: [["5,520"]],
        531: [["程家明"]],
        207: [["15,000"]],
    },
    "en": {
        260: [["6,464.51"], ["14,414.16"], ["18,780.67"], ["4,627.14"]],
        95: [["视频技术规范", "视频指挥系统技术标准", "standard"]],
        33: [["82.10"], ["97.31"], ["94.84"], ["94.34"]],
        34: [["电子元器件", "component"], ["机箱", "机柜", "cabinet", "enclosure", "metal", "壳体"]],
        957: [["国防军队视频指挥", "军队视频指挥", "military", "defense", "defence"]],
        793: [["军队", "military"], ["政府", "government"], ["能源", "energy"]],
        795: [["国家科技进步一等奖", "科技进步一等奖", "first prize"]],
        543: [["5,520"]],
        531: [["程家明", "cheng jiamin", "cheng"]],
        207: [["15,000"]],
    },
}


def norm(s):
    return s.replace(" ", "").replace("\u3000", "").replace("，", ",")


def judge(text, groups):
    text = norm(text)
    hit_groups, total = 0, len(groups)
    for g in groups:
        if any(norm(k) in text or k.lower() in text.lower() for k in g):
            hit_groups += 1
    return hit_groups, total


def main():
    data = json.load(open(os.path.join(HERE, "qa_bilingual_result.json"), encoding="utf-8"))["records"]
    md = ["# 答案准确率评测（验收口径）", "",
          "> 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化",
          "> 判分：数值题要求全部数值命中；事实/列举题按关键词组判定。完全正确率 = 完全正确题数 / 总题数。", ""]
    summary = {}
    for lang in ("zh", "en"):
        rows = [r for r in data if r["lang"] == lang]
        full = partial = 0
        md += ["", "## %s 10 题" % ("中文" if lang == "zh" else "英文（跨语）"), "",
               "| id | 判定 | 关键组命中 | 引用命中金标准页 | 拒答 |", "|---|---|---|---|---|"]
        for r in rows:
            groups = CHECKS[lang].get(r["id"], [])
            got, tot = judge(r["answer"], groups) if groups else (0, 0)
            ok = tot > 0 and got == tot
            full += ok
            partial += got
            if r["refused"]:
                mark = "拒答"
            elif ok:
                mark = "✅ 完全正确"
            elif got > 0:
                mark = "⚠️ 部分正确"
            else:
                mark = "❌ 错误"
            md.append("| %s | %s | %d/%d | %s | %s |" % (
                r["id"], mark, got, tot, "是" if r["cite_ok"] else "否", "是" if r["refused"] else "否"))
        n = len(rows)
        tot_groups = sum(len(CHECKS[lang].get(r["id"], [])) for r in rows)
        summary[lang] = {"n": n, "full": full, "accuracy": round(full / n, 4),
                         "key_group_hit": round(partial / tot_groups, 4) if tot_groups else None}
        md += ["", "- **完全正确率：%d/%d = %.1f%%**" % (full, n, 100 * full / n),
               "- 关键组命中率（细粒度）：%d/%d = %.1f%%" % (partial, tot_groups, 100 * partial / tot_groups),
               ""]
        print("[%s] 完全正确 %d/%d = %.1f%% | 关键组命中 %.1f%%" %
              (lang, full, n, 100 * full / n, 100 * partial / tot_groups))

    json.dump(summary, open(os.path.join(HERE, "answer_accuracy.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    open(os.path.join(HERE, "答案准确率.md"), "w", encoding="utf-8").write("\n".join(md))
    print("saved -> 答案准确率.md / answer_accuracy.json")


if __name__ == "__main__":
    main()
