# -*- coding: utf-8 -*-
"""RAGAS 评测：faithfulness / answer_relevancy / context_precision / context_recall。

用法：
    python ragas_eval.py --dataset tests/eval_samples.jsonl --out tests/ragas_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from logger import log


def load_samples(path: Path) -> list[dict[str, Any]]:
    """加载评测样本（jsonl：每行一个 JSON 对象）。"""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate(samples: list[dict[str, Any]]) -> dict[str, float]:
    """运行 RAGAS 指标。samples 字段：
    question / answer / contexts(list[str]) / ground_truth(list[str] 或 str)
    """
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        log.warning("ragas 未安装，跳过评测: %s", exc)
        return {"error": str(exc)}

    # 字段归一化：contexts 统一为列表，ground_truth 统一为字符串
    for s in samples:
        if isinstance(s.get("contexts"), str):
            s["contexts"] = [s["contexts"]]
        if isinstance(s.get("ground_truth"), list):
            s["ground_truth"] = "\n".join(s["ground_truth"])

    ds = Dataset.from_list(samples)
    try:
        result = ragas_evaluate(
            dataset=ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            raise_exceptions=False,  # 单条失败不中断整体评测
        )
        return {k: float(v) for k, v in result.items()}
    except Exception as exc:
        log.warning("ragas evaluate failed: %s", exc)
        return {"error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS 评测")
    parser.add_argument("--dataset", default="tests/eval_samples.jsonl", help="评测样本文件（jsonl）")
    parser.add_argument("--out", default="tests/ragas_report.json", help="结果输出路径")
    args = parser.parse_args()

    src = Path(args.dataset)
    if not src.exists():
        # 样本文件不存在：生成一条示例，保证脚本能直接跑通
        src.write_text(
            json.dumps(
                {
                    "question": "I miss you 怎么翻译？",
                    "answer": "I miss you 可译为「我很想你」。",
                    "contexts": ["I miss you. <-> 我很想你。"],
                    "ground_truth": "我很想你",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        log.info("已生成示例样本: %s", src)

    samples = load_samples(src)
    log.info("loaded %s samples", len(samples))
    report = evaluate(samples)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("ragas report -> %s", out)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


# =====================================================================
# 知识点说明（RAG：评测 RAGAS）
# ---------------------------------------------------------------------
# 1. RAGAS 四个核心指标：
#    - faithfulness（忠实度）：答案是否忠于检索资料，衡量幻觉程度；
#    - answer_relevancy（答案相关性）：答案是否切题；
#    - context_precision（上下文精确率）：召回资料中相关内容是否排在前；
#    - context_recall（上下文召回率）：标准答案所需信息是否都被召回。
#    前两个评"生成"，后两个评"检索"——可定位 RAG 链路瓶颈在哪一段。
# 2. 评测数据格式：question / answer / contexts（检索到的块）/
#    ground_truth（参考答案）；RAGAS 用"LLM 当裁判"（LLM-as-a-judge）
#    打分，因此需配置 LLM_API_KEY 才能运行。
# 3. 优化闭环：评测分数低 → 定位环节（recall 低 → 改分块/Query 改写/
#    多路召回；precision 低 → 改重排/得分阈值；faithfulness 低 → 改
#    提示词/换模型）→ 再评测，形成 RAG 优化的量化迭代。
# =====================================================================
