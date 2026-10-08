# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-功能测试及评估
src/evaluation_v7.py —— 工单七 RAG 检索评估框架（新增）

对 10 个测试问题的 RAG 检索结果做可量化评估：
  1. doc_recall@k   —— top-k 检索结果中金标文档的覆盖率（跨文档题按命中文档比例）
  2. mrr            —— 首个金标文档出现排名的倒数（检索排序质量）
  3. context_recall —— 金标关键词在检索上下文中的覆盖率（检索内容相关性）
  4. answer_acc     —— 金标关键词在 RAG 答案中的全命中率（回答正确性代理指标）
  5. issues         —— 自动问题归类（金标文档漏检/跨公司串档/关键词缺口/答案漏点/超时）

框架为纯函数实现，不依赖模型/Milvus，便于单元测试。
"""
import re
from typing import Any, Dict, List, Sequence

WORK_ORDER = "人工智能NLP-RAG-功能测试及评估"

# 工单七：验收性能门槛（沿用工单六，≤3 秒）
LATENCY_LIMIT_MS = 3000.0


def normalize_text(text: str) -> str:
    """工单七：归一化文本——去千分位逗号与所有空白，便于 2,768.09 / 2768.09 等数字比对"""
    if not text:
        return ""
    return re.sub(r"[\s,，]", "", str(text))


def keyword_group_hits(text: str,
                       keyword_groups: Sequence[Sequence[str]]
                       ) -> Dict[str, Any]:
    """工单七：词组命中评估

    keyword_groups: 二维数组，每组同义词任中一个即该组通过
    Returns: {hit, total, rate, hits:[命中的词], missed:[整组未中]}
    """
    norm = normalize_text(text)
    hits, missed_labels = [], []
    for group in keyword_groups:
        group = [g for g in group if g]
        if not group:
            continue
        matched = next((kw for kw in group
                        if normalize_text(kw) in norm), None)
        if matched is not None:
            hits.append(matched)
        else:
            missed_labels.append(group[0])
    total = len(hits) + len(missed_labels)
    return {"hit": len(hits), "total": total,
            "rate": round(len(hits) / total, 3) if total else 0.0,
            "hits": hits, "missed": missed_labels}


def _docs_in_topk(retrieved: Sequence[Dict[str, Any]], k: int) -> List[str]:
    """工单七：top-k 结果中按出现顺序去重的 doc_id 列表"""
    seen, ordered = set(), []
    for h in list(retrieved)[:k]:
        d = h.get("doc_id", "")
        if d and d not in seen:
            seen.add(d)
            ordered.append(d)
    return ordered


def doc_recall_at_k(retrieved: Sequence[Dict[str, Any]],
                    gold_docs: Sequence[str], k: int = 5) -> float:
    """工单七：金标文档在 top-k 的覆盖率（0~1）"""
    if not gold_docs:
        return 0.0
    topk = set(_docs_in_topk(retrieved, k))
    hit = len(set(gold_docs) & topk)
    return round(hit / len(gold_docs), 3)


def reciprocal_rank(retrieved: Sequence[Dict[str, Any]],
                    gold_docs: Sequence[str]) -> float:
    """工单七：MRR——首个金标文档排名倒数（未命中为 0）"""
    gold = set(gold_docs)
    for rank, h in enumerate(retrieved, 1):
        if h.get("doc_id") in gold:
            return round(1.0 / rank, 3)
    return 0.0


def cross_company_noise(retrieved: Sequence[Dict[str, Any]],
                        gold_docs: Sequence[str], k: int = 5) -> List[str]:
    """工单七：top-k 中混入的非金标公司文档（跨文档串档问题）"""
    gold = set(gold_docs)
    return [d for d in _docs_in_topk(retrieved, k) if d not in gold]


def classify_issues(metrics: Dict[str, Any]) -> List[str]:
    """工单七：检索结果问题自动归类（供问题分析章节使用）"""
    issues = []
    if metrics["doc_recall@5"] < 1.0:
        issues.append("金标文档漏检(top5未覆盖全部金标文档)")
    if metrics["mrr"] < 0.5:
        issues.append("金标文档排序靠后(MRR<0.5)")
    if metrics.get("noise_docs"):
        issues.append("跨公司串档(top5混入其他年报)")
    if metrics["context_recall"] < 1.0:
        issues.append("上下文关键词缺口(召回内容未覆盖全部金标关键词)")
    if metrics["answer_acc"] < 1.0:
        issues.append("答案要点遗漏(答案未命中全部金标关键词)")
    if metrics["latency_ms"] > LATENCY_LIMIT_MS:
        issues.append("响应超时(>3s)")
    return issues


def evaluate_case(retrieved: Sequence[Dict[str, Any]],
                  answer: str,
                  gold_docs: Sequence[str],
                  keyword_groups: Sequence[Sequence[str]],
                  latency_ms: float,
                  k: int = 5) -> Dict[str, Any]:
    """工单七：单题检索+答案评估

    Args:
        retrieved: RAG 检索到的文本 chunk（含 doc_id/page/content/score）
        answer: RAG 最终答案
        gold_docs: 金标文档 doc_id 列表
        keyword_groups: 金标关键词组
        latency_ms: 端到端耗时
    """
    context = "\n".join(str(c.get("content", "")) for c in retrieved)
    ctx_kw = keyword_group_hits(context, keyword_groups)
    ans_kw = keyword_group_hits(answer, keyword_groups)
    metrics = {
        "doc_recall@5": doc_recall_at_k(retrieved, gold_docs, k),
        "mrr": reciprocal_rank(retrieved, gold_docs),
        "context_recall": ctx_kw["rate"],
        "context_keyword_hits": ctx_kw["hits"],
        "context_keyword_missed": ctx_kw["missed"],
        "answer_acc": 1.0 if ans_kw["rate"] >= 1.0 else 0.0,
        "answer_keyword_rate": ans_kw["rate"],
        "answer_keyword_hits": ans_kw["hits"],
        "answer_keyword_missed": ans_kw["missed"],
        "noise_docs": cross_company_noise(retrieved, gold_docs, k),
        "latency_ms": round(float(latency_ms), 1),
    }
    metrics["issues"] = classify_issues(metrics)
    return metrics


def summarize(case_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """工单七：10 题汇总指标"""
    n = max(1, len(case_rows))
    avg = lambda key: round(sum(r["metrics"][key] for r in case_rows) / n, 3)
    return {
        "case_count": len(case_rows),
        "doc_recall@5_avg": avg("doc_recall@5"),
        "mrr_avg": avg("mrr"),
        "context_recall_avg": avg("context_recall"),
        "answer_accuracy": round(
            sum(r["metrics"]["answer_acc"] for r in case_rows) / n, 3),
        "latency_avg_ms": round(
            sum(r["metrics"]["latency_ms"] for r in case_rows) / n, 1),
        "latency_ok_rate": round(
            sum(1 for r in case_rows
                if r["metrics"]["latency_ms"] <= LATENCY_LIMIT_MS) / n, 3),
        "issue_distribution": _issue_distribution(case_rows),
    }


def _issue_distribution(case_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """工单七：问题类型分布统计"""
    dist: Dict[str, int] = {}
    for r in case_rows:
        for issue in r["metrics"]["issues"]:
            dist[issue] = dist.get(issue, 0) + 1
    return dict(sorted(dist.items(), key=lambda x: -x[1]))
