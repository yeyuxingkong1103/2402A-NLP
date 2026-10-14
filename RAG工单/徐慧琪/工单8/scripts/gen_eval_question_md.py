# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""由 data/eval/ground_truth.json 生成 eval_question.md（单一数据源）。"""
import io
import json
import os

GT_PATH = os.path.join("data", "eval", "ground_truth.json")
MD_PATH = "eval_question.md"

HEADER = """# eval_question.md

> **工单编号**：人工智能NLP-RAG-基于GraphRAG实现金融问答
>
> **来源说明**：工单原文要求「基于 `eval_question.md` 中的 Question 进行检索」，
> 但附件 `ccf_competition.zip` 内**并未提供该文件**（压缩包只有 9 份年报的 pdf/txt）。
> 本问题集因此构建为两部分：
>
> * **Q01~Q10** —— 完整复用工单07《功能测试及评估》的测试集，含逐题金标准
>   （`evidence_facts` / `key_facts` / `page_hints`），**保证与 07 逐题严格可比**；
> * **Q11~Q14** —— 工单08 扩展的**图谱专属题**（多跳关系型 / 跨文档对比型 /
>   组织结构型 / 全局型），用于检验 Graph RAG 相对普通 RAG 的增量。
>
> 题型覆盖：事实型 / 数值型 / 表格型 / 图像型 / 多跳推理型 / 多跳关系型 /
> 跨文档对比型 / 组织结构型 / 全局型。

## 问题清单

| 编号 | 问题 | 类型 | 来源 | 期望答案要点 |
| --- | --- | --- | --- | --- |
"""


def main() -> None:
    with io.open(GT_PATH, encoding="utf-8") as f:
        gt = json.load(f)
    # HEADER 自带末尾换行，若原样放进 lines，join 后会在表头分隔行与首个数据行
    # 之间多出一个空行，GFM 会在空行处终止表格（14 行全部退化成普通文本）。
    lines = [HEADER.rstrip("\n")]
    for it in sorted(gt, key=lambda x: x["id"]):
        q = it["question"].replace("|", "\\|")
        ref = (it.get("reference") or "").replace("|", "\\|").replace("\n", " ")
        org = "07" if it.get("origin") == "wo07" else "08扩展"
        lines.append(f"| Q{it['id']:02d} | {q} | {it['qtype']} | {org} | {ref} |")
    lines.append("")
    with io.open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("eval_question.md generated, questions:", len(gt))


if __name__ == "__main__":
    main()
