# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-图像内容解析及检索优化
文件：测试/评测骨架/screenshot_16q_skeleton.py（**T6 骨架**：逐题截图留痕清单与命名规范）

⚠️ 本文件**不驱动浏览器、不生成任何图片**：只输出「该截哪些图、命名是什么、必须能看出什么」的清单，
   供 T6 用 `browser_*` 工具真实截图时逐项打钩（`设计/验收标准.md` §6 + A9）。
   禁止用 PIL 画假图；截图必须来自真实 Web 界面。

用法：
  pwsh -NoProfile -File run_py.ps1 测试/评测骨架/screenshot_16q_skeleton.py --list
  pwsh -NoProfile -File run_py.ps1 测试/评测骨架/screenshot_16q_skeleton.py --list
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "测试" / "测试截图"

#: 16 题截图命名（题号 01..16 对应 golden 的 order；题目内容取自 golden）
#: 命名口径：题号NN = **工单题 id 升序序号**（与用户示例 题号01_发行股数 一致；NN ≠ 提示词顺序）
QUESTION_SHOTS = [
    (1, "题号01_发行股数_回答"),
    (2, "题号02_募投项目_回答"),
    (3, "题号03_控制关系关联方_回答"),
    (4, "题号04_非控制关联方_回答"),
    (5, "题号05_组织结构图_回答"),
    (6, "题号06_IC市场增长率_回答"),
    (7, "题号07_军用收入占比_回答"),
    (8, "题号08_行业上游_回答"),
    (9, "题号09_技术标准_回答"),
    (10, "题号10_补充流动资金_回答"),
    (11, "题号11_军用领域收入_回答"),
    (12, "题号12_法定代表人_回答"),
    (13, "题号13_注册资本_回答"),
    (14, "题号14_行业下游_回答"),
    (15, "题号15_国家科技进步一等奖_回答"),
    (16, "题号16_重要供应商领域_回答"),
]

#: 专项截图（验收标准 §6 清单）
SPECIAL_SHOTS = [
    ("题号05_组织结构图_检索精度", "第 5 题框图解析产物（页 39 结构树）+ 检索命中的 chunk 与相似度"),
    ("题号06_IC市场增长率_检索精度", "第 6 题图表数值（条形图 xref1692 行业-增长率）+ 检索命中"),
    ("拒答红线", "负例 N1~N4 拒答（refused=true，答『不清楚/未披露』）"),
    ("检索精确度", "Top-K 命中块与相似度（含未命中的反例更佳）"),
    ("准确率汇总", "16 题准确率汇总表（**数字来自 T6 实跑**）"),
    ("响应时间", "逐题延迟 + P50/P95/最大（**数字来自 T6 实跑**）"),
    ("图像解析结果", "结构树文本 / 图表数值表 / 候选图去重统计"),
    ("中英文问答", "同一问题中文 + 英文各一次"),
    ("日志可审计性", "结构化日志：函数名/输入摘要/输出摘要/耗时"),
]


def build_plan() -> dict:
    return {
        "out_dir": str(OUT_DIR.relative_to(ROOT)).replace("\\", "/"),
        "naming": "题号NN_题目内容_回答.png（NN=01..16，与 golden.order 一致）；专项截图用固定中文名",
        "hard_rules": [
            "必须用 browser_* 驱动真实 Web 界面截图；禁止 PIL 画假图",
            "截图须能看出：问题、答案、引用页码（文件名: 第N页，N=1-based 物理页）、检索精度",
            "两位数字题号与 golden.order 一一对应，禁止错位",
            "所有与成绩相关的数字只能来自 T6 实跑（待填充），本清单不提供任何数字",
        ],
        "question_shots": [{"order": o, "file": f"{n}.png", "content": "该题问答界面（问题/答案/引用）"}
                           for o, n in QUESTION_SHOTS],
        "special_shots": [{"file": f"{n}.png", "content": c} for n, c in SPECIAL_SHOTS],
        "counts": {"question": len(QUESTION_SHOTS), "special": len(SPECIAL_SHOTS),
                   "total": len(QUESTION_SHOTS) + len(SPECIAL_SHOTS)},
        "status": "待 T6 实跑填充（本计划仅定义清单与命名，未产出任何截图）",
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="16 题截图留痕清单（骨架，不驱动浏览器）")
    p.add_argument("--list", action="store_true", help="打印清单")
    p.add_argument("--out", default=None, help="可选：把清单落盘为 JSON（供 T6 打钩）")
    args = p.parse_args(argv)
    plan = build_plan()
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[OK] 清单已落盘 → {path}")
    if args.list or not args.out:
        print(f"截图目录：{plan['out_dir']}（共 {plan['counts']['total']} 张："
              f"逐题 {plan['counts']['question']} + 专项 {plan['counts']['special']}）")
        for s in plan["question_shots"]:
            print(f"  [{s['order']:>2}] {s['file']}  ← {s['content']}")
        for s in plan["special_shots"]:
            print(f"  [专项] {s['file']}  ← {s['content']}")
        print("⚠️ 骨架：未产出任何截图、未填任何数字（待 T6 实跑填充）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
