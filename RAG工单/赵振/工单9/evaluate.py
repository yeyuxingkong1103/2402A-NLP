"""工单编号：人工智能NLP-RAG-GraphRAG优化任务：前后评估。"""
import json

from sklearn.metrics import average_precision_score, recall_score
from app import ROOT, ask, load_index, normalize


def supports(text, claim):
    text = normalize(text)
    return all(any(normalize(word) in text for word in group) for group in claim["terms"])


def measure(hits, case):
    # 人工事实标签+词组对齐，保留逐条证据；不宣称是LLM语义判分。
    valid = [h for h in hits if not case.get("companies") or h["company"] in case["companies"]]
    supported = [any(supports(h["text"], c) for h in valid) for c in case["claims"]]
    relevance = [int(h in valid and any(supports(h["text"], c) for c in case["claims"])) for h in hits]
    precision = average_precision_score(relevance, list(range(len(hits), 0, -1))) if any(relevance) else 0
    recall = recall_score([1] * len(supported), supported, zero_division=0)
    return {"context_precision": float(precision), "context_recall": float(recall),
            "claims": [{**c, "supported": v,
                        "evidence": [{"company": h["company"], "page": h["page"]}
                                     for h in valid if supports(h["text"], c)]}
                       for c, v in zip(case["claims"], supported)]}


def main():
    baseline = json.loads((ROOT / "baseline.json").read_text(encoding="utf-8"))
    index = load_index()
    results = []
    for case in json.loads((ROOT / "cases.json").read_text(encoding="utf-8")):
        before = baseline["cases"][case["id"]]["hits"]
        after = ask(case["question"], index)
        results.append({"id": case["id"], "question": case["question"], "reference": case["reference"],
                        "before": {**measure(before, case), "hits": before},
                        "after": {**measure(after["hits"], case), **after}})
    summary = {}
    for side in ("before", "after"):
        summary[side] = {m: round(sum(r[side][m] for r in results) / len(results), 4)
                         for m in ("context_precision", "context_recall")}
    report = {"summary": summary, "results": results,
              "method": "scikit-learn，人工参考事实的词组对齐；不是RAGAS语义评分。",
              "warning": "固定开发题指标，不能证明泛化；原文摘录耗时不是模型生成耗时。"}
    (ROOT / "评估结果.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
