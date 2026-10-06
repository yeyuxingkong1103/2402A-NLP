# -*- coding: utf-8 -*-
"""RAG 离线评测入口。

在链路中的位置：
    独立评测脚本，调用 backend/retrieval.py 的检索链路，产出指标报告。
    它是"先立尺子再优化"这句话的落地工具 —— 每一次检索策略的改动，
    都靠它给出的 recall@k / MRR 数字来判断是变好还是变差。

用法：
  python eval/evaluate.py
  python eval/evaluate.py --k 5 --data eval/retrieval_set.json

脚本默认使用现有混合检索链路；若本地模型或 Milvus 未启动，会明确报告
不可用而不是填充虚假分数。安装 ragas 后，可将生成集交给项目的 RAGAS
实验环境计算 faithfulness / answer_relevancy / context_precision。

指标口径（与《项目总览.md》里 V1→V3 的战报一致）：
    recall@k   前 k 条里命中"正确来源+正确页码"的比例。衡量"有没有找回来"
    MRR        命中位置的倒数均值（1/rank）。衡量"找回来的排得靠不靠前"
    refusal_accuracy_proxy  无依据问题的拒答代理指标（说明见下）

输出：eval/report_latest.json，同时打印到屏幕。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 把 backend/ 加进模块搜索路径，这样才能直接 `from retrieval import ...`。
# 之所以这样做而不是 `from backend.retrieval import ...`：
# backend 下的模块内部用的是裸导入（from pipeline import ...），
# 它们之间互相引用时假定彼此同处一个目录。加 search path 是最省事的适配方式。
sys.path.insert(0, str(ROOT / "backend"))


def hit_expected(hit: dict, item: dict) -> bool:
    """判断一条检索结果是否命中了该题的预期依据。

    参数：
        hit: 一条检索结果（含 source / page）
        item: 一道评测题（含 expected_source / expected_pages）
    返回：
        命中返回 True。

    判定条件是"来源匹配 **且** 页码在预期集合内"：
        只判来源太松 —— 一份标准文档有上百页，随便命中一页就送分，
        指标会虚高到没有意义。
        加上页码才能真实反映"有没有找对那一页那一段"。

    用 `source in str(hit.get("source",""))` 做包含判断而不是相等：
        预期值写的是文档标识或文件名片段，实际 source 是完整路径，
        包含匹配让评测集写起来更宽松、也更抗路径变化。

    `bool(source and ...)` 的写法保证 expected_source 为空时直接判不命中：
        避免"没写预期"的题意外变成送分题。
    """
    source = item.get("expected_source", "")
    pages = set(item.get("expected_pages", []))
    return bool(source and source in str(hit.get("source", "")) and int(hit.get("page", -1)) in pages)


def main() -> int:
    """跑完整个评测集并写出报告。

    返回：
        0 = 评测正常完成；2 = 检索链路不可用（Milvus 或模型未就绪）。

    对"不可用"返回独立的退出码 2（而不是 0 或 1）：
        这与"评测完成但分数低"是完全不同的事。
        用 2 区分开，CI 才能分辨"环境没起"和"效果退化"，
        而不是把环境问题误读成指标下降。

    逐题 try/except 而不是让异常中断整个评测：
        一道题失败（如某个查询触发了边界 bug）时，记录该题的 error 并继续 ——
        跑完 20 题拿到部分结论，比在第 3 题崩掉、什么数据都没有强得多。

    各项指标的分母都用 non_refusal（应能回答的题数）：
        无依据的题（should_refuse=True）本来就不该召回任何东西，
        把它们算进 recall 的分母会让分数被无关地拉低。

    除零保护（if non_refusal else None）：
        评测集里全是拒答题时分母为 0。返回 None（JSON 里写 null）
        比返回 0 更诚实 —— 0 表示"一个都没命中"，null 表示"无法计算"，
        这两者对看报告的人来说含义完全不同。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(Path(__file__).with_name("retrieval_set.json")))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    cases = json.loads(Path(args.data).read_text(encoding="utf-8"))
    try:
        from retrieval import retrieve_with_trace
        from pipeline import COLLECTION
        from vector_store import collection_count
        count = collection_count(COLLECTION)
    except Exception as exc:
        # 明确报告不可用及原因，绝不写出一个"看起来正常"的假报告 ——
        # 假报告的危害远大于没有报告：它会让人基于错误数字做决策
        print(json.dumps({"status": "unavailable", "reason": str(exc)}, ensure_ascii=False, indent=2))
        return 2

    hits = 0
    reciprocal_sum = 0.0
    refusal_total = refusal_correct = 0
    details = []
    for item in cases:
        try:
            # candidate_k 至少 12：给精排留出足够的候选池（同 server.py 的考量）
            result = retrieve_with_trace(item["question"], candidate_k=max(args.k, 12), final_k=args.k)
            ranked = result.get("results", [])
            # 找第一条命中预期依据的结果，记下它的排名（从 1 开始；没命中则 None）
            positive = next((idx + 1 for idx, hit in enumerate(ranked) if hit_expected(hit, item)), None)
            expected_refusal = bool(item.get("should_refuse"))
            # 检索评测中的拒答代理：没有命中指定知识来源视为拒答。
            actual_refusal = not bool(positive)
            if expected_refusal:
                refusal_total += 1
                refusal_correct += int(actual_refusal)
            if positive:
                hits += 1
                reciprocal_sum += 1.0 / positive  # MRR 的核心：排第 1 得 1.0，排第 3 得 0.33
            details.append({"question": item["question"], "rank": positive, "actual_refusal_proxy": actual_refusal, "trace": result.get("trace", [])})
        except Exception as exc:
            details.append({"question": item["question"], "error": str(exc)})

    non_refusal = sum(1 for item in cases if not item.get("should_refuse"))
    report = {
        "status": "ok",
        "collection": COLLECTION,
        "collection_points": count,
        "k": args.k,
        "cases": len(cases),
        "recall_at_k": round(hits / non_refusal, 4) if non_refusal else None,
        "mrr": round(reciprocal_sum / non_refusal, 4) if non_refusal else None,
        "refusal_accuracy_proxy": round(refusal_correct / refusal_total, 4) if refusal_total else None,
        # 这句 note 是必要的自我说明：proxy 指标只反映检索层，
        # 不检查答案文本是否忠实，避免读者把 1.0 误读成"答案质量满分"
        "note": "refusal_accuracy_proxy 只评估召回层无命中代理；完整 RAGAS 生成指标需运行模型并提供 reference/answer 数据。",
        "details": details,
    }
    out = Path(__file__).with_name("report_latest.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
