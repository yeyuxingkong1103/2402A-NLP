"""Faithfulness 打分器自检（批次 11 任务 4）。

用 3 条人工构造的"确定有问题"的回答喂给同一打分器：
  1) 引用了不存在的法条（清单外 + 编造条号）
  2) 法条内容被改写（数字与原文不符）
  3) 结论超出条文范围（原文没有且错误的断言）
期望：每条得分 < 0.6。若打分器给出高分 → 提示词太宽松，需收紧。

用法：python evaluation/selfcheck_faithfulness.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from run_eval import prepare_env  # noqa: E402

prepare_env()

from app.models.llm import build_chat_client_from_settings  # noqa: E402
from faithfulness import score_item  # noqa: E402

# context_excerpts 全部取自库内真实条文原文（batch10 评测报告实测摘录，非编造）
CONTEXT_MATERNITY = [
    {"index": 1, "law_name": "中华人民共和国劳动法", "article_number": "第六十二条",
     "content": "第六十二条 女职工生育享受不少于九十天的产假。"},
]
CONTEXT_WAGE = [
    {"index": 1, "law_name": "中华人民共和国劳动合同法实施条例", "article_number": "第十五条",
     "content": "第十五条 劳动者在试用期的工资不得低于本单位相同岗位最低档工资的80%或者不得低于劳动合同约定工资的80%，并不得低于用人单位所在地的最低工资标准。"},
]

SELF_CHECK_ITEMS = [
    {
        "id": "selfcheck-01-编造法条",
        "type": "selfcheck",
        "question": "女职工产假是多少天？",
        "expect_refusal": False,
        "refused_flag": False,
        "context_excerpts": CONTEXT_MATERNITY,
        "answer": (
            "根据相关规定，女职工生育享受不少于九十八天的产假[1]，"
            "其中《女职工劳动保护特别规定》第七条明确产假为98天，难产的另加30天[2]。"
        ),
    },
    {
        "id": "selfcheck-02-条改写",
        "type": "selfcheck",
        "question": "女职工产假是多少天？",
        "expect_refusal": False,
        "refused_flag": False,
        "context_excerpts": CONTEXT_MATERNITY,
        "answer": "依据[1]，女职工生育享受不少于一百二十天的产假，用人单位不得在产假期间降低工资。",
    },
    {
        "id": "selfcheck-03-超范围",
        "type": "selfcheck",
        "question": "试用期工资有什么底线要求？",
        "expect_refusal": False,
        "refused_flag": False,
        "context_excerpts": CONTEXT_WAGE,
        "answer": (
            "根据[1]，试用期工资不得低于劳动合同约定工资的80%，且不得低于当地最低工资标准。"
            "另外，试用期最长可以约定十二个月，超过部分按转正工资计算。"
        ),
    },
]


def main() -> int:
    llm_client = build_chat_client_from_settings()
    records = []
    for item in SELF_CHECK_ITEMS:
        print(f"打分 {item['id']} …", flush=True)
        records.append(score_item(llm_client, item))

    print()
    all_pass = True
    for record in records:
        score = record.get("score")
        ok = score is not None and score < 0.6
        all_pass = all_pass and ok
        mark = "✅ 识别为坏回答" if ok else "❌ 打分器放行"
        print(f"{mark}  {record['id']}  score={score}")
        print(f"   理由：{record.get('reason')}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "reports"
    out = {
        "selfcheck_at": stamp,
        "all_below_0_6": all_pass,
        "records": records,
    }
    (out_dir / f"faithfulness_selfcheck_{stamp}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "latest_faithfulness_selfcheck.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n自检结论：{'通过（全部 <0.6）' if all_pass else '不通过（打分器太宽松，需收紧提示词）'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
