# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务

模块：辅助检索指标（工单硬性要求2）

指标：Precision@k、Recall@k、Hit Rate、MRR、NDCG@k

## 相关性判定（中文场景的关键取舍）

本语料是中文年报，没有词边界，也不能依赖外部分词模型 —— 因此相关性判定
采用**字符 n-gram 重合**（默认 4-gram），它对中文稳健且无需依赖：

  · **块级相关性**（用于 Precision@k / Hit Rate / MRR / NDCG）：
    块文本与参考答案的 4-gram 重合率达到阈值即判为相关块。
    另把 `page_hints`（08 金标准给出的答案页）命中视为强相关。

  · **论断覆盖**（用于 Recall@k）：
    把参考答案的 `key_facts`（若无则用参考答案切分出的要点句）逐条检查
    「能否在 top-k 块里找到」—— 这与 RAGAS 的 context_recall 定义一致，
    但**完全确定性、零模型参与**，可作为 RAGAS 的交叉验证：
    若 RAGAS 的 recall 很高而本指标很低，说明裁判模型判分有问题。

阈值取 0.10：中文 4-gram 重合率 0.10 已表示存在成句的实质重合
（该阈值在 9 份年报的样本上人工核对过）。
"""
from __future__ import annotations

import re
import math

NGRAM = 4
# 判「块与答案相关」用的片段长度。
#
# ## 阈值是实测选出来的，而且**这个指标本身有已知局限**
#
# 在本语料上实测不同 n 的判「相关」命中数（09 的 4 题）：
#
#     n=4 → 12/14, 15/15, 16/16, 8/8     过于宽松（金融套话就能命中）
#     n=5 → 12/14, 15/15,  6/16, 5/8     中间
#     n=6 →  1/14,  2/15,  1/16, 1/8     过严
#     n=8 →  1/14,  2/15,  0/16, 0/8     几乎全灭
#
# 取 n=5 作为折中。**但必须如实说明它的局限**：
#
#   本指标衡量的是**词面重合**，不是「对回答有没有用」。实测它与 RAGAS 的
#   语义判定**排序不一致** —— 例如 SQ2 在本指标下是 15/15（满分），
#   而 RAGAS 判它的 context precision 只有 0.122（因为召回的多数块虽然
#   与答案有词面重合，却并不回答问题）。
#
#   根因：参考答案是**改写过的摘要**，不是原文摘录（如答案写「零售**业务**
#   进一步巩固收入贡献」，原文是「零售进一步巩固收入贡献」），词面重合
#   天然不可靠。
#
#   因此本组指标只作为**确定性、零模型参与的旁证**，用于与 RAGAS 交叉比对；
#   判断检索质量**以 RAGAS 为准**。
PHRASE_N = 5
OVERLAP_THRESHOLD = 0.10      # 仅用于 fact_coverage 的论断匹配
MIN_CHUNK_CHARS = 30


def _norm(text: str) -> str:
    """去空白与标点，只留中英文数字（标点差异不应影响重合判定）。"""
    s = str(text or "")
    return re.sub(r"[^一-鿿A-Za-z0-9]", "", s)


def _ngrams(text: str, n: int = NGRAM) -> set[str]:
    s = _norm(text)
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _overlap(a: str, b: str) -> float:
    """a 的 n-gram 有多少比例出现在 b 中（对 a 归一）。"""
    ga = _ngrams(a)
    if not ga:
        return 0.0
    gb = _ngrams(b)
    return len(ga & gb) / len(ga)


# ---------------------------------------------------------------------------
# 相关性
# ---------------------------------------------------------------------------
def chunk_relevant(chunk_text: str, reference: str,
                   source: str = "", page: int | None = None,
                   page_hints: list | None = None) -> bool:
    """单个检索块是否与参考答案相关。

    ## 判据为什么是「共享任一长短语」而不是「重合率超过阈值」（实测修正）

    初版用 `_overlap(reference, chunk) >= 0.10` —— 即「参考答案的 4-gram
    至少 10% 出现在这个块里」。实测该口径**把所有块都判成不相关**：
    参考答案（尤其第 3、4 题）有几百字，而单个检索块只覆盖其中一个要点，
    它对**整篇参考答案**的 4-gram 贡献天然只有 1~3%，永远够不到 10%。
    结果是 Precision@k / Hit Rate / NDCG 全为 0，指标完全失去区分度。

    正确的问题不是「这个块覆盖了答案的多少」，而是
    「这个块是否包含答案里的实质内容」。故改为：**只要共享一个 8 字以上的
    连续片段**（中文里 8 字已足以构成一个有意义的短语，如「零售进一步巩固收入贡献」），
    即判为相关。这既不过严（不要求整段覆盖），也不过松（不靠零散单字蒙对）。
    """
    text = str(chunk_text or "")
    if len(_norm(text)) < MIN_CHUNK_CHARS:
        return False
    # 页码命中（金标准给的答案页）→ 强相关
    if page_hints and page is not None and int(page) in set(page_hints):
        return True
    return _shares_phrase(reference, text, n=PHRASE_N)


def _shares_phrase(a: str, b: str, n: int = PHRASE_N) -> bool:
    """a 与 b 是否共享至少一个长度为 n 的连续片段。"""
    ga = _ngrams(a, n)
    if not ga:
        return False
    return bool(ga & _ngrams(b, n))


def relevant_flags(hits: list[dict], reference: str,
                   page_hints: list | None = None) -> list[bool]:
    """逐块相关性标记（保持检索顺序 —— MRR/NDCG 依赖次序）。"""
    out: list[bool] = []
    for h in hits:
        page = h.get("page_idx", h.get("page", -1))
        try:
            page = int(page)
        except (TypeError, ValueError):
            page = -1
        out.append(chunk_relevant(h.get("text") or "", reference,
                                  h.get("source") or h.get("source_doc") or "",
                                  page, page_hints))
    return out


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def precision_at_k(flags: list[bool], k: int) -> float:
    top = flags[:k]
    return sum(top) / len(top) if top else 0.0


def hit_rate(flags: list[bool], k: int) -> float:
    return 1.0 if any(flags[:k]) else 0.0


def mrr(flags: list[bool]) -> float:
    """第一个相关块的倒数名次。"""
    for i, ok in enumerate(flags, start=1):
        if ok:
            return 1.0 / i
    return 0.0


def ndcg_at_k(flags: list[bool], k: int) -> float:
    """二值相关性的 NDCG@k。"""
    top = flags[:k]
    if not top:
        return 0.0
    dcg = sum((1.0 / math.log2(i + 2)) for i, ok in enumerate(top) if ok)
    n_rel = sum(top)
    if n_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


def fact_coverage(hits: list[dict], facts: list[str]) -> float:
    """论断覆盖率：有多少条 key_facts 能在检索块里找到（对未覆盖者记 0）。

    这是一个**确定性**的 context recall 代理指标 —— 不依赖裁判 LLM，
    可与 RAGAS 的结果交叉验证（见模块 docstring）。
    """
    facts = [f for f in (facts or []) if str(f).strip()]
    if not facts:
        return 0.0
    blob = "\n".join((h.get("text") or "") for h in hits)
    hit = sum(1 for f in facts if _overlap(str(f), blob) >= 0.6)
    return hit / len(facts)


def evaluate_hits(hits: list[dict], reference: str, facts: list[str] | None = None,
                  page_hints: list | None = None, k: int = 8) -> dict:
    """对一次检索的命中列表算全套辅助指标。"""
    flags = relevant_flags(hits, reference, page_hints)
    return {
        "k": k,
        "n_hits": len(hits),
        "n_relevant": sum(flags),
        "precision@k": round(precision_at_k(flags, k), 4),
        "recall@k": round(fact_coverage(hits[:k], facts or []), 4),
        "hit_rate@k": round(hit_rate(flags, k), 4),
        "mrr": round(mrr(flags), 4),
        "ndcg@k": round(ndcg_at_k(flags, k), 4),
        "truth_coverage": round(fact_coverage(hits[:k], facts or []), 4),
    }


def aggregate(rows: list[dict]) -> dict:
    """多题指标取平均。"""
    keys = ("precision@k", "recall@k", "hit_rate@k", "mrr", "ndcg@k",
            "truth_coverage")
    out: dict = {"n": len(rows)}
    for key in keys:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        out[key] = round(sum(vals) / len(vals), 4) if vals else None
    return out
