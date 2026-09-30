# -*- coding: utf-8 -*-
"""可选 RAGAS 适配器。

在链路中的位置：
    独立评测脚本。与 eval/evaluate.py 的分工是"评什么"不同：
        evaluate.py        评**检索**（recall@k / MRR），不需要大模型，随时可跑
        本文件             评**生成**（忠实度、答案相关性），需要大模型且要求
                           输入里已经有生成好的答案

用法：
    python eval/ragas_eval.py <数据集.json> [--output eval/ragas_report.json]

数据集要求（JSON 数组）：每行必须有 question / answer / contexts 三个字段；
若每行还提供了 reference 或 reference_answer，则额外计算 context_precision /
context_recall 两个指标。

该脚本不伪造答案：输入必须包含 question、answer、contexts，若提供
reference_answer/reference_contexts 则额外计算对应指标。未安装 ragas 时
返回清晰提示，核心检索评测请运行 evaluate.py。

"不伪造答案"是本项目的评测原则：
    本脚本只负责把已生成的真实答案喂给 RAGAS 计算指标，
    绝不会在缺少 answer 时自己编一个上去 —— 那会得出毫无意义的分数。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    """读取数据集，调 RAGAS 计算指标并写出报告。

    返回：
        0 = 评测完成；2 = 缺少 ragas/datasets 可选依赖。

    用退出码 2 区分"依赖没装"和"评测失败"：
        ragas 是可选依赖（装在 requirements 之外），
        缺它不代表项目有问题，报错信息里也直接给出安装提示。

    指标按条件追加（context_precision/context_recall 需要参考答案）：
        这两个指标衡量"检索到的上下文与标准答案的匹配程度"，
        没有 reference 就无法计算。所以先判断数据集里有没有，
        有才算 —— 硬算的话 RAGAS 会报错或给出无意义的值。

    all(...) 检查每一行都有 reference：
        只要有一行缺失就不计算那两个指标，避免"部分行有、部分行没有"
        导致指标在样本上不一致（那样算出来的分数无法解释）。

    default=str 让 json.dumps 能处理非原生类型：
        RAGAS 的返回值里可能含 numpy 数值或自定义对象，
        不设这个参数会直接抛 TypeError 导致整个报告写不出来。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="JSON 数组，字段 question/answer/contexts，可选 reference_answer")
    parser.add_argument("--output", default="eval/ragas_report.json")
    args = parser.parse_args()
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:
        print(f"RAGAS 评测未运行：缺少可选依赖（{exc}）。请安装 ragas、datasets 后重试。")
        return 2

    rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    required = {"question", "answer", "contexts"}
    # 空数据集时把 required 整体当作缺失，从而走到下面的报错分支 ——
    # 否则会继续往下执行，最后在 Dataset.from_list([]) 或取 rows[0] 处抛出难懂的异常
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise SystemExit(f"数据集缺少字段: {', '.join(sorted(missing))}")
    dataset = Dataset.from_list(rows)
    # faithfulness 与 answer_relevancy 是最基础的两个生成指标，任何数据集都能算
    metrics = [faithfulness, answer_relevancy]
    if all("reference" in row or "reference_answer" in row for row in rows):
        metrics += [context_precision, context_recall]
    result = evaluate(dataset, metrics=metrics)
    # metric_names 单独列出：RAGAS 版本之间返回的键名有变化，
    # 附带一份"本次实际用了哪些指标"能让报告自带上下文、事后可解释
    report = {"metrics": dict(result), "cases": len(rows), "metric_names": [metric.name for metric in metrics]}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
