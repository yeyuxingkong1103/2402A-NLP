# 工单16：人工智能NLP-RAG-微调专用视觉语言模型
"""对比微调前后的评估结果，打印成报告用的表格

读 `测试/results/{base,finetuned}.json`，输出：

  · 各指标的前后对比与变化量（验收看「专业术语准确性」的提升是否 ≥5%）
  · 失败案例分类：答非所问 / 术语没答对 / 图纸关系判错
  · 每条失败样例的题面、标准答案、模型输出

用法：
    python compare.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RES = Path(__file__).parent / "results"
METRICS = [("term_acc", "专业术语准确性"), ("drawing_acc", "图纸推理正确性"),
           ("letter_acc", "选项准确率"), ("bleu", "BLEU"), ("rouge_l", "ROUGE-L")]


def load(tag):
    p = RES / f"{tag}.json"
    if not p.exists():
        raise SystemExit(f"缺少 {p}，先跑 evaluate.py --tag {tag}")
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    base, ft = load("base"), load("finetuned")
    b, f = base["summary"], ft["summary"]

    print("=" * 74)
    print(f"{'指标':<18}{'微调前':>12}{'微调后':>12}{'变化':>12}")
    print("-" * 74)
    for key, name in METRICS:
        delta = f[key] - b[key]
        mark = "✓" if (key == "term_acc" and delta >= 0.05) else ""
        print(f"{name:<18}{b[key]:>12.4f}{f[key]:>12.4f}"
              f"{delta:>+11.4f} {mark}")
    print("=" * 74)
    print(f"评估样本 {b['n']} 条（其中图文关系题 {b['n_drawing']} 条）"
          f"，术语型题按 term 非空统计")

    # ---- 失败案例 ----
    print("\n=== 微调后仍答错的案例（按类型归类）===")
    by_key = {x["question"]: x for x in base["items"]}
    fail = [x for x in ft["items"]
            if x["letter_hit"] is False or (x["term_hit"] is False and x["term"])]

    kinds = Counter()
    for x in fail:
        if x["term"] and x["term_hit"] is False:
            kinds["术语没答对"] += 1
        elif x["letter_hit"] is False:
            kinds["图纸关系判错"] += 1
        else:
            kinds["其它"] += 1
    for k, v in kinds.most_common():
        print(f"  {k}: {v} 条")

    # 前后都错 / 只有微调后错 —— 后者说明微调带来了负作用，最值得看
    both = neither = regressed = 0
    for x in ft["items"]:
        y = by_key.get(x["question"], {})
        ok_b = y.get("letter_hit") or y.get("term_hit")
        ok_f = x["letter_hit"] or x["term_hit"]
        if not ok_b and not ok_f:
            both += 1
        elif ok_b and not ok_f:
            regressed += 1
        elif not ok_b and ok_f:
            neither += 1
    print(f"\n  微调救回来的: {neither} 条   前后都错: {both} 条   微调后反而错的: {regressed} 条")

    print("\n=== 微调后答错的代表样例（最多 8 条）===")
    shown = 0
    for x in fail:
        if shown >= 8:
            break
        y = by_key.get(x["question"], {})
        was = "微调前也错" if not (y.get("letter_hit") or y.get("term_hit")) else "微调前是对的"
        print(f"\n  [{was}] {x['question'][:70]}")
        print(f"    标准答案: {x['answer'][:80]}")
        print(f"    微调前  : {(y.get('pred') or '')[:80]}")
        print(f"    微调后  : {(x['pred'] or '')[:80]}")
        shown += 1

    (RES / "compare.md").write_text(render_md(b, f, kinds, neither, both, regressed, fail),
                                    encoding="utf-8")
    print(f"\n报告已写到 {RES / 'compare.md'}")


def render_md(b, f, kinds, rescued, both, regressed, fail):
    """产出报告里能直接粘的 markdown 片段。"""
    lines = ["## 微调前后对比", "",
             "| 指标 | 微调前 | 微调后 | 变化 |", "|---|---|---|---|"]
    for key, name in METRICS:
        lines.append(f"| {name} | {b[key]:.4f} | {f[key]:.4f} | {f[key] - b[key]:+.4f} |")
    lines += ["", "## 失败案例归类", "", "| 类型 | 条数 |", "|---|---|"]
    for k, v in kinds.most_common():
        lines.append(f"| {k} | {v} |")
    lines += ["", f"- 微调救回来的：{rescued} 条",
              f"- 前后都错：{both} 条",
              f"- 微调后反而错的：{regressed} 条", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
