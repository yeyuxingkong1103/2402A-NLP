# 工单编号：人工智能NLP-RAG-混合检索任务
# 工单06 - 混合检索（向量检索 / 全文检索 / 混合检索）
"""
重排器（可插拔）。工单要求「提供至少 3 种重排算法」，这里给了 4 种 + 1 个空操作：

    lexical   工单02 的词法融合（dense + 查询词覆盖率）—— **默认，零成本**
    tfidf     用全文索引的全局 DF 算 TF-IDF 重排
    llm       复用**常驻的 qwen3:8b** 对 top-N 候选打相关性分（见下方成本说明）
    feedback  读工单05 落盘的 `feedback.jsonl`，按人工反馈给页面加权
    none      不重排（fulltext / hybrid 剖面用：关键词路本身就是词法，
              再叠一次词法重排是同义反复，还会掩盖融合的真实效果）

【统一约定】
· 输入是**召回的候选**，输出是**同集合、不同顺序** —— 只改顺序、绝不增删。
  否则"换了重排器"就变成"换了召回集"，评测里无法归因。
· 任何异常（超时 / JSON 解析失败 / 模型不可用）都**回退 lexical** 并在
  `RerankInfo.fallback` 留痕，绝不静默丢弃结果。

【LLM 重排的成本必须说清】它要吃**一次完整的生成调用**，而本项目 TTFT 的
P95 已经是 1915ms（≤3000ms 红线）。所以：
  · 只对前 `rerank_top_n` 个候选打分（默认 5），不重排全部；
  · 用 `rerank_timeout_ms` 硬超时；
  · 它只出现在**演示/离线剖面**（`hybrid_llm`），不进交付默认路径。
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from app.config import settings
from app.core.text_analysis import abstract_query, query_tokens, tokenize
from app.core.vectorstore import SearchHit

log = logging.getLogger("rag.rerank")


@dataclass
class RerankInfo:
    """重排的留痕（进 meta / 报告，演示时要能说清"这一轮谁排的序"）。"""

    reranker: str = "none"
    fallback: bool = False          # 是否因异常回退到 lexical
    feedback_n: int = 0             # 反馈重排器实际用上多少条人工反馈（0 = 冷启动）
    ms: int = 0
    detail: str = ""

    def as_dict(self) -> dict:
        return {"reranker": self.reranker, "fallback": self.fallback,
                "feedback_n": self.feedback_n, "rerank_ms": self.ms,
                "detail": self.detail}


# ======================================================================
# lexical（工单02 的原式，逐字节保留）
# ======================================================================
def _base_value(hit: SearchHit) -> tuple[float, str]:
    """取出用于归一化的「原始分 + 它的尺度」。融合过的命中要用 `base_*`。"""
    if hit.base_score is not None:
        return hit.base_score, (hit.base_kind or hit.score_kind)
    return hit.score, hit.score_kind


def lexical_score(hit: SearchHit, qt: set[str], w_dense: float = 0.5,
                  *, span: float | None = None, lo: float = 0.0) -> float:
    """工单02 的词法融合分：`w_dense * 归一化分 + (1-w_dense) * 词覆盖率`。

    【归一化按**原始分数的尺子**选】
      · `cosine`（COSINE ∈ [-1,1]）→ `(score+1)/2`，**与工单02 逐字节一致**
      · `none`（关键词路独有、向量没召回）→ 记 0，不在稠密项上占便宜
      · 其余（bm25 无上界 / 融合分）→ 按本次候选集 min-max

    【为什么必须看"原始分"而不是融合后的分】实测：同一批候选，把归一化从
    COSINE 换成排名尺度，页召回从 60.4% 掉到 38.5%。原因是 COSINE 分数密集在
    0.75~0.83 的窄带里，融合时约九成权重落在"查询词覆盖率"上；换成铺满 0~1
    的排名分后，覆盖率的信号被盖掉，排序整体变了 —— 而这一切**不报错**。

    【为什么不能对 BM25/RRF 套 (score+1)/2】RRF 的分只有 0.016 量级，
    `(0.016+1)/2 ≈ 0.508`，对候选集里所有块几乎相同，`w_dense * dense` 退化成
    常数项，排序悄悄只剩词覆盖率在起作用。
    """
    if not qt:
        return hit.score
    lex = len(qt & query_tokens(hit.content)) / len(qt)
    val, kind = _base_value(hit)
    if kind == "cosine":
        base = (val + 1.0) / 2.0
    elif kind == "unit":
        base = val                  # 融合时已映射进稠密路区间，直接用
    elif kind == "none":
        base = 0.0
    elif span and span > 1e-12:
        base = (val - lo) / span
    else:
        base = 0.5
    return w_dense * base + (1.0 - w_dense) * lex


def _lexical(hits: Sequence[SearchHit], query: str, *, w_dense: float) -> list[SearchHit]:
    # 【必须与工单02 用同一个词集合】老公式用的是 `abstract_query`（去掉公司全称）
    # 的词集合。用原始问句的话，公司全称那些token永远匹配不上任何片段，
    # 会把覆盖率整体稀释 —— 实测这一步就能把页召回从 60.4% 打到 39.6%。
    qt = query_tokens(abstract_query(query))
    if not qt:
        return list(hits)
    vals = [_base_value(h)[0] for h in hits]
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    span = hi - lo
    return sorted(hits, key=lambda h: -lexical_score(h, qt, w_dense, span=span, lo=lo))


# ======================================================================
# tfidf
# ======================================================================
def _idf(index, term: str) -> float:
    by_field = index.postings.get(term)
    if not by_field:
        return 0.0
    docs: set[int] = set()
    for post in by_field.values():
        docs |= set(post)
    df = len(docs)
    if df == 0:
        return 0.0
    return math.log((index.n_docs - df + 0.5) / (df + 0.5) + 1.0)


def _tfidf(hits: Sequence[SearchHit], query: str, *, index) -> list[SearchHit]:
    """TF-IDF 重排：用**全文索引的全局 DF** 算 IDF，用候选自身的词频算 TF。

    【为什么要用索引里的 DF 而不是候选集里的】候选只有几十条，在其中统计 DF
    会把"常见词"当成"稀有词"—— IDF 就失效了。全局 DF 才有意义。
    """
    terms = {t for t in query_tokens(query)}
    if not terms:
        return list(hits)
    idf = {t: _idf(index, t) for t in terms}
    scored: list[tuple[float, int, SearchHit]] = []
    for h in hits:
        toks = tokenize(h.content)
        if not toks:
            scored.append((0.0, 0, h))
            continue
        tf = Counter(toks)
        total = len(toks)
        s = sum((tf[t] / total) * idf[t] for t in terms if t in tf)
        scored.append((s, 1, h))
    # 分数降序；同分保持原有相对顺序（稳定）
    scored.sort(key=lambda x: (-x[0],))
    return [h for _, _, h in scored]


# ======================================================================
# feedback（工单05 的人工反馈）
# ======================================================================
@dataclass
class _FeedbackStat:
    pos: int = 0
    neg: int = 0

    @property
    def support(self) -> int:
        return self.pos + self.neg


def _feedback_file() -> Path:
    return settings.data_path / "feedback" / "feedback.jsonl"


def feedback_boosts(min_support: int, max_boost: float) -> tuple[dict[str, float], int]:
    """从 `feedback.jsonl` 聚合出「页面 → 加权系数」。

    【键为什么是 (page_label, section_path 前 40 字)】
    `CitationOut` 里**没有 doc_name 字段**（工单05 定的），所以拿不到文档名。
    但两份书的页码格式天然不同（书1 印 `1-1-21`、书2 印裸数字 `21`），
    不会互相撞车；再带上章节路径前缀进一步降低歧义。这个限度写进文档。

    【冷启动必须等价于"不重排"】文件不存在或样本太少时返回空表 ——
    绝不拿一两条反馈去重排整个结果（那是把偶然偏好放大成排序）。
    返回的 int 是"实际参与统计的反馈条数"，进 meta 供演示核对。
    """
    path = _feedback_file()
    if not path.exists():
        return {}, 0
    tally: dict[str, _FeedbackStat] = defaultdict(_FeedbackStat)
    n_records = 0
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                n_records += 1
                rating = rec.get("rating")
                for c in rec.get("citations") or []:
                    if c.get("is_neighbor"):
                        continue
                    key = _feedback_key(c.get("page_label", ""), c.get("section_path", ""))
                    st = tally[key]
                    if rating == "up":
                        st.pos += 1
                    elif rating == "down":
                        st.neg += 1
    except Exception as e:  # noqa: BLE001
        log.warning("读取反馈失败（按无反馈处理）：%s", e)
        return {}, 0

    boosts: dict[str, float] = {}
    for key, st in tally.items():
        if st.support < min_support:
            continue                      # 样本不足 → 不参与
        # 用 pos/(pos+neg) 映射到 [-1,1]，再乘上限 → 单条反馈最多动 max_boost
        ratio = st.pos / st.support
        boosts[key] = (ratio * 2.0 - 1.0) * max_boost
    return boosts, n_records


def _feedback_key(page_label: str, section_path: str) -> str:
    return f"{page_label}||{(section_path or '')[:40]}"


def _feedback(hits: Sequence[SearchHit], *, boosts: dict[str, float]
              ) -> list[SearchHit]:
    if not boosts:
        return list(hits)
    scored = []
    for i, h in enumerate(hits):
        b = boosts.get(_feedback_key(h.page_label, h.section_path), 0.0)
        # 乘性微调，且**保持原顺序作为主键**（同 boost 时不改变相对次序）
        scored.append((h.score * (1.0 + b), -i, h))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    return [h for _, _, h in scored]


# ======================================================================
# llm
# ======================================================================
_LLM_SYS = ("你是检索结果重排器。给定一个问题与若干候选片段，"
            "按与问题的相关性从高到低排序。只输出 JSON，不要解释。")
_LLM_TMPL = """问题：{q}

候选片段：
{items}

只输出 JSON：{{"order": [按相关性从高到低的编号]}}"""


def _parse_order(raw: str, n: int) -> list[int] | None:
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(s[i:j + 1])
    except Exception:  # noqa: BLE001
        return None
    order = obj.get("order") if isinstance(obj, dict) else None
    if not isinstance(order, list):
        return None
    seen: list[int] = []
    for x in order:
        try:
            k = int(x)
        except Exception:  # noqa: BLE001
            continue
        if 1 <= k <= n and k not in seen:
            seen.append(k)
    # 漏掉的编号按原顺序补在后面（模型偶尔少给几个，不能因此丢候选）
    for k in range(1, n + 1):
        if k not in seen:
            seen.append(k)
    return seen


async def _llm(hits: Sequence[SearchHit], query: str, *, profile, client=None
               ) -> tuple[list[SearchHit], bool, str]:
    """LLM 重排。返回 (结果, 是否回退, 说明)。**只重排前 N 个，只改顺序。**"""
    from app.core.ollama_client import get_client

    n = max(1, min(profile.rerank_top_n, len(hits)))
    head, tail = list(hits[:n]), list(hits[n:])
    if len(head) <= 1:
        return list(hits), False, "候选不足 2 条，跳过"

    items = "\n".join(
        f"[{i}] 页码 {h.page_label}｜{(h.section_path or '')[:40]}\n{h.content[:220]}"
        for i, h in enumerate(head, 1))
    msgs = [{"role": "system", "content": _LLM_SYS},
            {"role": "user", "content": _LLM_TMPL.format(q=query, items=items)}]

    # 【复用常驻的 qwen3:8b，绝不给重排单独拉一个模型】8GB 卡上第二个生成模型
    # 会把 qwen3 挤出去，下一次提问要付 7 秒冷启动 —— 省下的那点时间全赔进去。
    cli = client or get_client()
    try:
        raw = await asyncio.wait_for(
            cli.chat(msgs, fmt="json", temperature=0.0, max_tokens=120),
            timeout=max(0.1, profile.rerank_timeout_ms / 1000.0))
    except asyncio.TimeoutError:
        return list(hits), True, f"LLM 重排超时（>{profile.rerank_timeout_ms}ms），已回退"
    except Exception as e:  # noqa: BLE001
        return list(hits), True, f"LLM 重排失败（{type(e).__name__}），已回退"

    order = _parse_order(raw, len(head))
    if order is None:
        return list(hits), True, "LLM 未返回可解析的排序，已回退"
    return [head[k - 1] for k in order] + tail, False, f"LLM 重排前 {len(head)} 条"


# ======================================================================
# 统一入口
# ======================================================================
async def rerank(hits: Sequence[SearchHit], query: str, *, profile,
                 client=None) -> tuple[list[SearchHit], RerankInfo]:
    """按剖面选定的重排器重排候选。**只改顺序，不增删**。"""
    info = RerankInfo(reranker=profile.reranker)
    if not hits or profile.reranker == "none":
        return list(hits), info

    t0 = time.perf_counter()
    try:
        if profile.reranker == "lexical":
            out = _lexical(hits, query, w_dense=profile.w_dense)
        elif profile.reranker == "tfidf":
            from app.core.fulltext import get_index
            out = _tfidf(hits, query, index=get_index())
        elif profile.reranker == "feedback":
            boosts, n = feedback_boosts(profile.feedback_min_support,
                                        profile.feedback_max_boost)
            info.feedback_n = n
            info.detail = (f"反馈 {n} 条，生效页面 {len(boosts)} 个"
                           if boosts else f"反馈 {n} 条（样本不足，等价于不重排）")
            out = _feedback(hits, boosts=boosts)
        elif profile.reranker == "llm":
            out, fallback, detail = await _llm(hits, query, profile=profile, client=client)
            info.fallback = fallback
            info.detail = detail
            if fallback:
                out = _lexical(hits, query, w_dense=profile.w_dense)
        else:  # 兜底：不该发生（profile 的 __post_init__ 已经校验过）
            out = list(hits)
            info.detail = f"未知重排器 {profile.reranker!r}，未重排"
    except Exception as e:  # noqa: BLE001
        log.warning("重排器 %s 异常，回退 lexical：%s", profile.reranker, e)
        info.fallback = True
        info.detail = f"{type(e).__name__}: {e}"
        out = _lexical(hits, query, w_dense=profile.w_dense)

    info.ms = int((time.perf_counter() - t0) * 1000)
    return out, info
