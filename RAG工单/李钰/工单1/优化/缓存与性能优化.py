# -*- coding: utf-8 -*-
"""
缓存与性能优化模块
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统

功能:
    1. 使用 functools.lru_cache 缓存常见问题答案, 显著降低重复查询时延
    2. 提供索引预加载与并发安全单例
    3. 提供性能指标统计 (命中率、平均时延)

使用方法 (在 qa_engine 中替换调用):
    from 缓存与性能优化 import cached_answer_question
    result = cached_answer_question(question)
"""
import os
import sys
import time
import json
import threading
import functools
from typing import Dict

# 引入研发目录
_DEV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "研发")
sys.path.insert(0, _DEV_DIR)

import config
import qa_engine

_stats = {
    "total_calls": 0,
    "cache_hits": 0,
    "total_time": 0.0,
}
_lock = threading.Lock()


def _normalize_question(q: str) -> str:
    """问题归一化: 去除首尾空白、统一标点, 提升缓存命中率"""
    if not q:
        return ""
    return q.strip().lower().replace("?", "？").replace(",", "，")


@functools.lru_cache(maxsize=128)
def _cached_answer_impl(question_key: str) -> str:
    """缓存层: 以问题字符串作为 key 缓存完整结果 (JSON)"""
    result = qa_engine.answer_question(question_key)
    return json.dumps(result, ensure_ascii=False)


def cached_answer_question(question: str) -> Dict:
    """
    带缓存的问答接口 (与 qa_engine.answer_question 接口兼容)

    Returns:
        dict (同 qa_engine.answer_question)
    """
    global _stats
    key = _normalize_question(question)
    with _lock:
        _stats["total_calls"] += 1
        if key in _cached_answer_impl.cache_info().cache:
            _stats["cache_hits"] += 1

    start = time.time()
    raw = _cached_answer_impl(key)
    elapsed = time.time() - start
    with _lock:
        _stats["total_time"] += elapsed
    return json.loads(raw)


def get_stats() -> Dict:
    """获取性能统计"""
    info = _cached_answer_impl.cache_info()
    total = _stats["total_calls"]
    return {
        "total_calls": total,
        "cache_hits": _stats["cache_hits"],
        "hit_rate": round(_stats["cache_hits"] / total, 4) if total else 0.0,
        "avg_time": round(_stats["total_time"] / total, 4) if total else 0.0,
        "cache_size": info.currsize,
        "cache_max": info.maxsize,
    }


def warm_up(questions: list):
    """预热缓存: 提前对一组问题进行查询"""
    for q in questions:
        try:
            cached_answer_question(q)
        except Exception as e:
            print(f"[预热失败] {q}: {e}")


# 默认预热问题 (工单验收中的 10 个问题)
WARM_UP_QUESTIONS = [
    "报告期内, 武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少?",
    "武汉兴图新科电子股份有限公司参与制定了哪个技术标准?",
    "报告期内, 武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少?",
    "根据武汉兴图新科电子股份有限公司招股意向书, 电子信息行业的上游涉及哪些企业?",
    "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商?",
    "根据武汉兴图新科电子股份有限公司招股意向书, 电子信息行业的下游主要包括哪些行业?",
    "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖?",
    "武汉兴图新科电子股份有限公司注册资本是多少?",
    "武汉兴图新科电子股份有限公司法定代表人是谁?",
    "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金?",
]


if __name__ == "__main__":
    print("=== 缓存预热 ===")
    warm_up(WARM_UP_QUESTIONS[:3])

    print("\n=== 第二次查询 (应命中缓存) ===")
    q = WARM_UP_QUESTIONS[0]
    start = time.time()
    result = cached_answer_question(q)
    print(f"响应时间: {(time.time() - start):.4f}s")
    print(f"RAG 答案: {result['rag_answer'][:100]}")

    print("\n=== 性能统计 ===")
    print(json.dumps(get_stats(), ensure_ascii=False, indent=2))
