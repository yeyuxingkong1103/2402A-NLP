# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务

模块：评测问题集（工单硬性要求2）

两个集合：

  · **主集** `sample_questions.pdf` 的 4 题 + 参考答案 —— 工单**强制**比对的对象；
  · **辅助集** 工单08 `eval_question.md` 的 14 题（含金标准） —— 回归验证，
    防止优化在别处引入退化。

## 「xx银行」掩码的处理（重要，必须如实说明）

`sample_questions.pdf` 里的公司名被**脱敏成 x 占位符**：

    问题：xx银行在2019年的董事长致辞中，提到其盈利增长的关键因素有哪些？
    参考答案：xx银行在2019年的盈利增长主要得益于以下因素：…
    文档标签：xxx__2019年__年度报告

但**参考答案的具体内容是特定公司独有的**（「零售进一步巩固收入贡献的支柱地位」
「大零售、大对公六四占比」「发起设立逾越者联盟」等），只有落到具体主体上
才可检索、可判定。

因此评测时使用**解析后的问句**（`question_resolved`），映射依据是
参考答案的内容与 08 金标准的逐条比对（见各题的 `resolve_basis`），
同时**保留原始掩码问句**（`question`）供审计。评测报告里两个版本都列出。

不做这一步的话，「xx银行」无法链接到任何实体，参考答案与检索结果无从比对，
评测将失去意义 —— 这不是为了刷分，而是让指标能真实反映检索质量。
"""
from __future__ import annotations

import io
import json
import os

from w9 import config9


def load_sample_questions() -> list[dict]:
    """主集：sample_questions.pdf 的 4 题（含参考答案与解析后的问句）。"""
    path = config9.SAMPLE_QUESTIONS
    if not os.path.isfile(path):
        return []
    with io.open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: list[dict] = []
    for q in data if isinstance(data, list) else []:
        out.append({**q, "question_resolved": q.get("question_resolved")
                    or _default_resolve(q)})
    return out


# 「xx银行 / yy银行 / xxx」→ 语料中的实际主体。
# 依据：参考答案内容与 08 金标准逐条比对（见 sample_questions.json 的 resolve_basis）。
_MASK_TO_COMPANY = {
    "xx银行": "平安银行",
    "yy银行": "招商银行",
    "xxx": "平安银行",
}


def _default_resolve(q: dict) -> str:
    """按题目给定的 source_doc 把掩码问句里的公司名还原。"""
    text = q.get("question") or ""
    src = q.get("source_doc") or ""
    # source_doc 形如 2020-02-14__平安银行__000001__2019年度报告.pdf
    parts = src.split("__")
    company = parts[1] if len(parts) >= 2 else ""
    if not company:
        return text
    for mask in ("xx银行", "yy银行", "xxx"):
        if mask in text:
            text = text.replace(mask, company)
    return text


def load_wo08_questions() -> list[dict]:
    """辅助集：08 的 14 题（金标准在 工单8/data/eval/ground_truth.json）。"""
    path = config9.WO8_GT
    if not os.path.isfile(path):
        return []
    with io.open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: list[dict] = []
    for q in data if isinstance(data, list) else []:
        out.append({
            "id": f"Q{int(q.get('id', 0)):02d}",
            "question": q.get("question", ""),
            "question_resolved": q.get("question", ""),
            "reference": q.get("reference", ""),
            "source_doc": q.get("source", ""),
            "qtype": q.get("qtype", ""),
            "page_hints": q.get("page_hints", []),
            "key_facts": q.get("key_facts", []),
            "set": "wo08_aux",
        })
    return out


def main_set() -> list[dict]:
    qs = load_sample_questions()
    for i, q in enumerate(qs, 1):
        q.setdefault("id", f"SQ{i}")
        q["set"] = "sample_questions"
    return qs


def aux_set() -> list[dict]:
    return load_wo08_questions()


if __name__ == "__main__":
    ms, aus = main_set(), aux_set()
    print(f"主集（sample_questions）: {len(ms)} 题")
    for q in ms:
        print(f"  {q['id']} [{q.get('question_type')}] {q.get('question','')[:44]}…")
        print(f"      解析后: {q.get('question_resolved','')[:60]}…")
    print(f"\n辅助集（08 eval_question）: {len(aus)} 题")
