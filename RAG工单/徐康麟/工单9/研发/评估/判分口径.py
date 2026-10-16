# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 说明：本文件复用自工单8（工单9 未修改其业务逻辑），仅补齐工单9 的编号头注释。
"""工单8 判分口径：**直接复用 工单7 的纯函数判分模块**（``工单7/研发/评估/判分.py``）。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

为什么复用而不是重写（关键验收要求）：
    本工单的验收标准之一是「**比对 07 工单的测试结果**」。要谈得上可比，
    两次评测的判分口径必须**逐位一致**：同样的归一化（去空白/千分位/全角转半角）、
    同样的数字边界（``11.65`` 不算命中 ``1.65``）、同样的阈值（关键词 0.6 / 部分 0.3 / 原子 0.6）、
    同样的 recall@k 合并正文口径、同样的页命中区间解析、同样的跨语料泄漏判定。
    任何"自己再写一遍"的做法都可能引入口径漂移，让对比表失去意义。
    因此这里**只做导入与参数固化**，不复制判分逻辑。

降级策略：工单7 目录缺失时，``judge`` 为 ``None``，调用方必须显式报错而不是静默给分。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

WORK_ORDER = "人工智能NLP-RAG-基于Graph RAG 实现金融问答"

#: 工单7 判分模块所在目录（**只读**引用）
BASE_EVAL_DIR = Path(r"E:\gao6gongdan\工单7\研发\评估")

#: 判分阈值（与 工单7 研发/评估/配置.py 逐项一致）
KEYWORD_THRESHOLD = 0.6
PARTIAL_THRESHOLD = 0.3
ATOMIC_THRESHOLD = 0.6
LEAK_TOP_N = 3
RECALL_KS = (5, 10)
SLA_MS = 3000

judge: Any = None
_IMPORT_ERROR: str | None = None
try:  # noqa: SIM105 —— 导入失败必须留痕，故用显式 try/except
    if str(BASE_EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_EVAL_DIR))
    import 判分 as judge  # type: ignore  # noqa: N813 —— 工单7 的模块名就是中文
except Exception as exc:  # noqa: BLE001 —— 工单7 缺失时不得假装能判分
    judge = None
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def available() -> bool:
    """判分模块是否可用（不可用时调用方必须显式失败，不得给分）。"""
    return judge is not None


def import_error() -> str | None:
    return _IMPORT_ERROR


def score_answer(answer: str, keywords: list[str], refused: bool) -> dict[str, Any]:
    """单题答案判分（严格/部分/覆盖率/原子覆盖率），字段名与 工单7 评估 jsonl 对齐。"""
    if judge is None:
        raise RuntimeError(f"判分模块不可用（{_IMPORT_ERROR}）—— 请在 工单7 存在的前提下运行评估")
    hit_count, total, coverage, hit, miss = judge.keyword_coverage(answer, keywords)
    _, _, atomic_cov, atomic_hit, atomic_miss = judge.atomic_coverage(answer, keywords)
    correct = judge.is_correct(coverage, refused, threshold=KEYWORD_THRESHOLD)
    partial = (not correct) and judge.partial_correct(
        coverage, refused, answer=answer, keywords=keywords, threshold=KEYWORD_THRESHOLD,
        partial_threshold=PARTIAL_THRESHOLD, atomic_threshold=ATOMIC_THRESHOLD)
    return {"keyword_hit_count": hit_count, "keyword_total": total,
            "keyword_coverage": round(float(coverage), 4), "keyword_hit": hit, "keyword_miss": miss,
            "atomic_coverage": round(float(atomic_cov), 4), "atomic_hit": atomic_hit,
            "atomic_miss": atomic_miss, "threshold": KEYWORD_THRESHOLD,
            "is_correct": bool(correct), "partial_correct": bool(partial)}


def recall_at_k(hits: list[dict[str, Any]], keywords: list[str], k: int) -> dict[str, Any]:
    """top-k 合并正文的关键词召回（与 工单7 同口径：命中正文而非页码）。"""
    if judge is None:
        raise RuntimeError(f"判分模块不可用（{_IMPORT_ERROR}）")
    texts = [str(h.get("text") or h.get("content") or "") for h in hits[:k]]
    hit_count, total, recall, hit, miss = judge.retrieval_recall_at_k(texts, keywords, k)
    return {f"recall_at_{k}": round(float(recall), 4), f"recall_at_{k}_hit": hit_count,
            f"recall_at_{k}_total": total, f"recall_at_{k}_hit_keywords": hit,
            f"recall_at_{k}_miss_keywords": miss}


def page_hit(hits: list[dict[str, Any]], expected_pages_hint: Any,
             source_filter: Any = None) -> dict[str, Any]:
    """命中页是否落在期望页区间（支持 ``5-51`` 与 ``5-37;7-20`` 多区间）。"""
    if judge is None:
        raise RuntimeError(f"判分模块不可用（{_IMPORT_ERROR}）")
    return judge.page_hit(hits, expected_pages_hint, source_filter=source_filter)


def cross_corpus_leak(hits: list[dict[str, Any]], expected_files: Any,
                      top_n: int = LEAK_TOP_N) -> dict[str, Any]:
    """top-N 命中是否串到非期望语料（公司级串档判定）。"""
    if judge is None:
        raise RuntimeError(f"判分模块不可用（{_IMPORT_ERROR}）")
    return judge.cross_corpus_leak(hits, expected_files, top_n=top_n)


def classify_question(question: str, lang: str, keywords: list[str], source_file: Any = None) -> str:
    """题型分类（定性/数值/英文/跨文档对比，与 工单7 同一分类器）。"""
    if judge is None:
        return "英文题" if str(lang) == "en" else "数值题"
    return str(judge.classify_question(question, lang, keywords, source_file))


def splitters() -> dict[str, Any]:
    """返回 工单7 的语料/公司解析工具（``split_source_files`` / ``company_of`` / ``companies_of``）。"""
    if judge is None:
        return {}
    return {"split_source_files": judge.split_source_files, "company_of": judge.company_of,
            "companies_of": judge.companies_of}
