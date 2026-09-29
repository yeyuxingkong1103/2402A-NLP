# -*- coding: utf-8 -*-
"""对比两次 RAGAS 评测结果，输出优化前后的增益。

    evals/.venv/bin/python evals/compare.py
    evals/.venv/bin/python evals/compare.py --base baseline --opt optimized
"""
import argparse
import json
import os

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

METRICS = ["faithfulness", "answer_relevancy",
           "context_precision", "context_recall"]
LABELS = {
    "faithfulness": "忠实度（回答是否忠于资料）",
    "answer_relevancy": "答案相关性（是否切题）",
    "context_precision": "上下文精确率（检索到的有多少有用）",
    "context_recall": "上下文召回率（该找的是否都找到了）",
}


def load(tag):
    path = os.path.join(EVAL_DIR, "scores_%s.json" % tag)
    if not os.path.exists(path):
        raise FileNotFoundError("找不到评分结果: %s" % path)
    return json.load(open(path, encoding="utf-8"))


def fmt_delta(base, opt):
    d = opt - base
    arrow = "↑" if d > 0.005 else ("↓" if d < -0.005 else "→")
    return "%+.3f %s" % (d, arrow)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="baseline")
    ap.add_argument("--opt", default="optimized")
    args = ap.parse_args()

    base, opt = load(args.base), load(args.opt)

    print("=" * 78)
    print("RAGAS 评测对比：%s  ->  %s" % (args.base, args.opt))
    print("=" * 78)
    print("样本数: %d -> %d" % (base["samples"], opt["samples"]))
    print()

    print("%-30s %10s %10s %12s" % ("指标", args.base, args.opt, "变化"))
    print("-" * 78)
    gains = []
    for m in METRICS:
        b, o = base["overall"][m], opt["overall"][m]
        gains.append(o - b)
        print("%-30s %10.4f %10.4f %12s" % (LABELS[m], b, o, fmt_delta(b, o)))

    avg = sum(gains) / len(gains)
    print("-" * 78)
    print("%-30s %10s %10s %12s" % ("四项平均", "", "", "%+.4f" % avg))
    print()

    # 分角色
    roles = sorted(set(base["per_role"]) | set(opt["per_role"]))
    print("分角色明细")
    print("-" * 78)
    header = "%-20s" % "角色"
    for m in METRICS:
        header += "%14s" % m.replace("_", " ")[:13]
    print(header)
    for r in roles:
        b = base["per_role"].get(r, {})
        o = opt["per_role"].get(r, {})
        line = "%-20s" % r
        for m in METRICS:
            bv, ov = b.get(m, 0), o.get(m, 0)
            line += "%14s" % ("%.3f%s%.3f" % (bv, "→", ov))
        print(line)

    # 结论
    print()
    print("=" * 78)
    improved = [METRICS[i] for i, g in enumerate(gains) if g > 0.005]
    degraded = [METRICS[i] for i, g in enumerate(gains) if g < -0.005]
    print("提升: %s" % (", ".join(improved) or "无"))
    print("下降: %s" % (", ".join(degraded) or "无"))
    print("=" * 78)

    out = os.path.join(EVAL_DIR, "comparison.json")
    json.dump({"base": args.base, "opt": args.opt,
               "overall": {m: {"base": base["overall"][m],
                               "opt": opt["overall"][m],
                               "delta": round(opt["overall"][m] - base["overall"][m], 4)}
                           for m in METRICS},
               "per_role": {r: {m: {"base": base["per_role"].get(r, {}).get(m),
                                    "opt": opt["per_role"].get(r, {}).get(m)}
                                for m in METRICS} for r in roles}},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("已保存: %s" % out)


if __name__ == "__main__":
    main()
