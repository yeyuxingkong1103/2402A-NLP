"""
工单5 消融实验：多轮对话「有没有用」、「哪一步在起作用」
工单编号：人工智能NLP-RAG-Query理解优化任务

为什么要单独做一份消融，而不是直接报一个准确率：
    「5 轮都答对了」只能说明**结果对**，说明不了**是不是因为做了多轮消解才对**。
    验收标准写的是「准确性 ≥ 90%」，但对一份工程交付来说，更有说服力的证据是：
    把每一项设计**逐个关掉**，看准确率掉到多少。掉得越狠，说明这项设计越关键。

本脚本跑两组实验，每组都是**单变量**：

--------------------------------------------------------------------------
实验 A：消解方式（4 臂）—— 回答「多轮到底有没有用」
--------------------------------------------------------------------------
    T0 off       完全不看历史。这是**工单前**系统的行为，也是最强对照。
                 第 2~4 轮（含「他」「这个公司」「那X呢」）必然答不上或答串。
    T1 concat    只把最近两轮的原话**拼**到问题上，不做任何消解 ——
                 这是最常见的朴素实现（很多 RAG 系统就这么做）。
    T2 rule      纯规则消解：指代替换 + 省略补全，一次 LLM 都不调。
    T3 rule+llm  规则优先，规则把握不足时（试检索依据分过低）才让大模型补 —— 交付配置。

    T1/T2/T3 之间**只有一个变量**：检索式是「拼历史原文」还是「消解后的自包含问句」，
    以及消解由谁来做。doc_filter / 主体置顶 这两个子优化在三臂里都开着，保持不变。

--------------------------------------------------------------------------
实验 B：子优化项（3 臂）—— 回答「哪一步在起作用」
--------------------------------------------------------------------------
    在 T3 的基础上逐个关掉工单里新加的检索层优化，定位各自的贡献：
      B0 全开
      B1 关 doc_filter（多轮确定主体后硬排除另一份文档）
      B2 关 subject_first（同文档内的主体一致性排序主键 + 放宽绝对闸门）

    这两项**都不是**拍脑袋加的：它们各自对应一次实测定位到的问题，
    关掉它们应当能在同一道题上复现那个问题（见 docs 里的「踩坑记录」）。

用法：
    python scripts/ablation_dialogue.py                # 全跑（4 臂 + 3 臂，约 5~8 分钟）
    python scripts/ablation_dialogue.py --group A      # 只跑消解方式
    python scripts/ablation_dialogue.py --group B      # 只跑子优化项
    python scripts/ablation_dialogue.py --no-warmup    # 不做预热（T1 会含模型加载耗时）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config, rag  # noqa: E402
import report_wot5 as R5  # noqa: E402 —— 复用评测集的判定口径，避免两套标准

WORK_ORDER_NO = config.WORK_ORDER_NO_QU

# 实验 A：消解方式
ARMS_A = [
    ("T0 off", "off", "完全不看历史（工单前系统的行为）"),
    ("T1 concat", "concat", "把最近两轮原话拼上去，不做消解"),
    ("T2 rule", "rule", "纯规则消解，零 LLM 成本"),
    ("T3 rule+llm", "rule+llm", "规则优先 + 规则不确定时 LLM 兜底（交付配置）"),
]


def _warmup() -> float:
    t = time.perf_counter()
    rag.answer("武汉兴图新科电子股份有限公司的注册资本是多少？")
    return time.perf_counter() - t


def _run_arm(label: str, mode: str, desc: str, dialogues: list[dict],
             patch: dict | None = None) -> dict:
    """跑一条臂：**对每一场对话独立开一个会话**跑完，再汇总成总准确率。

    为什么不能把所有场次串成一个会话：不同对话的主题/主体是各自独立的，
    串起来会制造出上下文污染（前面一场的「那X呢」会被后面一场看到），
    测出来的就不是「多轮消解」而是「上下文串味」了。
    """
    per = [R5.run_dialogue(d, mode, True) for d in dialogues]
    n = sum(x["n_turns"] for x in per)
    ok = sum(x["n_correct"] for x in per)
    arm = {
        "label": label, "mode": mode, "desc": desc, "patch": patch or {},
        "per_dialogue": per,
        "n_turns": n, "n_correct": ok,
        "accuracy": round(ok / n, 4) if n else 0.0,
        "subject_acc": round(sum(1 for x in per for t in x["turns"]
                                 if t["judge"]["subject_ok"]) / n, 4) if n else 0.0,
        "fact_acc": round(sum(1 for x in per for t in x["turns"]
                              if t["judge"]["facts_ok"]) / n, 4) if n else 0.0,
        "gold_hit_acc": round(sum(1 for x in per for t in x["turns"]
                                  if t["judge"]["doc_gold_hit"]) / n, 4) if n else 0.0,
        "avg_ms": round(sum(t["total_ms"] for x in per for t in x["turns"]) / n, 2) if n else 0.0,
    }
    return arm


def _show(arm: dict, width: int = 18) -> None:
    print(f"  {arm['label']:<{width}} 准确率 {arm['accuracy']:.1%} "
          f"({arm['n_correct']}/{arm['n_turns']})  主体 {arm['subject_acc']:.1%}  "
          f"事实 {arm['fact_acc']:.1%}  答案页 {arm['gold_hit_acc']:.1%}  "
          f"{arm['avg_ms']:.0f}ms")


def group_a(dialogues: list[dict]) -> list[dict]:
    out = [_run_arm(l, m, d, dialogues) for l, m, d in ARMS_A]
    for r in out:
        _show(r, 14)
    return out


def group_b(dialogues: list[dict]) -> list[dict]:
    """子优化项消融：改 config 后跑，跑完复原。"""
    orig = (config.MULTITURN_DOC_FILTER_ENABLED, config.MULTITURN_SUBJECT_FIRST_ENABLED)
    arms = [
        ("B0 全开", "rule+llm", "交付配置", dict(MULTITURN_DOC_FILTER_ENABLED=True,
                                                 MULTITURN_SUBJECT_FIRST_ENABLED=True)),
        ("B1 关 doc_filter", "rule+llm", "不硬排除另一份文档",
         dict(MULTITURN_DOC_FILTER_ENABLED=False, MULTITURN_SUBJECT_FIRST_ENABLED=True)),
        ("B2 关 subject_first", "rule+llm", "不做同文档内主体消歧",
         dict(MULTITURN_DOC_FILTER_ENABLED=True, MULTITURN_SUBJECT_FIRST_ENABLED=False)),
    ]
    out = []
    try:
        for label, mode, desc, patch in arms:
            for k, v in patch.items():
                setattr(config, k, v)
            out.append(_run_arm(label, mode, desc, dialogues, patch))
    finally:
        config.MULTITURN_DOC_FILTER_ENABLED, config.MULTITURN_SUBJECT_FIRST_ENABLED = orig
    for r in out:
        _show(r)
    return out


def build_md(report: dict) -> str:
    lines = [
        "# 工单5 多轮对话消融实验",
        "",
        f"- **工单编号**：{WORK_ORDER_NO}",
        f"- **生成时间**：{report['ts']}",
        f"- **评测对象**：{[d['id'] for d in report['dialogues']]}（共 {report['n_turns']} 轮）",
        f"- **判定口径**：与 `scripts/report_wot5.py` 完全一致（主体正确 + 关键事实全中 + 未串号）",
        "",
    ]
    if report.get("A"):
        lines += [
            "## 实验 A：消解方式（单变量：检索式怎么来）",
            "",
            "| 臂 | 模式 | 说明 | 准确率 | 主体 | 关键事实 | 答案页 | 平均耗时 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in report["A"]:
            lines.append(
                f"| **{r['label']}** | `{r['mode']}` | {r['desc']} | **{r['accuracy']:.1%}** "
                f"({r['n_correct']}/{r['n_turns']}) | {r['subject_acc']:.1%} | "
                f"{r['fact_acc']:.1%} | {r['gold_hit_acc']:.1%} | {r['avg_ms']:.0f} ms |"
            )
        lines.append("")
    if report.get("B"):
        lines += [
            "## 实验 B：子优化项（单变量：逐个关掉工单新增的检索层优化）",
            "",
            "| 臂 | 说明 | 准确率 | 主体 | 关键事实 | 答案页 | 平均耗时 |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in report["B"]:
            lines.append(
                f"| **{r['label']}** | {r['desc']} | **{r['accuracy']:.1%}** "
                f"({r['n_correct']}/{r['n_turns']}) | {r['subject_acc']:.1%} | "
                f"{r['fact_acc']:.1%} | {r['gold_hit_acc']:.1%} | {r['avg_ms']:.0f} ms |"
            )
        lines.append("")

    # 逐轮明细：只看「哪些轮在哪个臂上挂了」，这是最能说明问题的一张表
    arms = (report.get("A") or []) + (report.get("B") or [])
    for di, d in enumerate(report["dialogues"]):
        lines += [f"### {d['id']}　{d['title']}", "",
                  "| 轮次 | 类型 | 问题 |" + "".join(f" {a['label']} |" for a in arms),
                  "|---|---|---|" + "---|" * len(arms)]
        for ti, spec in enumerate(d["turns"]):
            row = [f"| {ti + 1} | `{spec.get('type')}` | {spec['q'][:38]} |"]
            for a in arms:
                t = a["per_dialogue"][di]["turns"][ti]
                j = t["judge"]
                if j["correct"]:
                    row.append(" ✅ |")
                    continue
                why = "主体错" if not j["subject_ok"] else ""
                if j["must_have_missed"]:
                    why += "漏" + "".join(j["must_have_missed"])[:10]
                if j["leaked"]:
                    why += "串" + "".join(j["leaked"])[:10]
                row.append(f" ❌{why} |")
            lines.append("".join(row))
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="AB", help="A=消解方式 B=子优化项，默认都跑")
    ap.add_argument("--dialogue", default="ALL",
                    help="用哪场对话做消融，ALL=全部场次（默认）。只跑 D1 时敏感度不够——"
                         "工单原题那 5 轮在完全不做多轮的情况下也可能碰巧答对，"
                         "必须用 D2~D4 的指代/省略轮次才能测出真实差距。")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    spec = json.loads(R5.EVAL_FILE.read_text(encoding="utf-8"))
    if args.dialogue.upper() == "ALL":
        dialogues = spec["dialogues"]
    else:
        want = {x.strip().upper() for x in args.dialogue.split(",")}
        dialogues = [d for d in spec["dialogues"] if d["id"].upper() in want]
    if not dialogues:
        print(f"[FAIL] 没有匹配的场次：{args.dialogue}")
        return 2

    print(f"[工单5 消融] 场次：{[d['id'] for d in dialogues]}　"
          f"共 {sum(len(d['turns']) for d in dialogues)} 轮")
    if not args.no_warmup:
        print(f"… 预热中…（{_warmup():.1f}s，不计入任何一臂的耗时）\n")

    report: dict = {
        "work_order_no": WORK_ORDER_NO,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_turns": sum(len(d["turns"]) for d in dialogues),
        "dialogues": [
            {"id": d["id"], "title": d["title"],
             "turns": [{"index": i, "type": t.get("type"), "q": t["q"],
                        "expect_subject": t.get("expect_subject")}
                       for i, t in enumerate(d["turns"], 1)]}
            for d in dialogues
        ],
    }

    if "A" in args.group.upper():
        print("=== 实验 A：消解方式 ===")
        report["A"] = group_a(dialogues)
        print()
    if "B" in args.group.upper():
        print("=== 实验 B：子优化项 ===")
        report["B"] = group_b(dialogues)
        print()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    jp = config.EVAL_DIR / f"工单5_多轮消融_{args.group}_{stamp}.json"
    mp = config.EVAL_DIR / f"工单5_多轮消融_{args.group}_{stamp}.md"
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    mp.write_text(build_md(report), encoding="utf-8")
    print(f"[JSON] {jp}")
    print(f"[MD]   {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
