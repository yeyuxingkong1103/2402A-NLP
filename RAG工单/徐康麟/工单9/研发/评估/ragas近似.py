# -*- coding: utf-8 -*-
"""工单9 自研「RAGAS 近似」评估器：context_precision / context_recall / faithfulness / answer_relevancy。

工单编号：人工智能NLP-RAG-Graph RAG 优化任务

================================================================================
⚠️ 口径声明（**必读，验收会看**）
================================================================================
本机 **没有安装 ragas**（`_共享约定.md` §2 缺失包清单 + 实测 `import ragas` 失败），
且不允许裸 `pip install`。因此本模块是**自研的等价实现**，并且：

    **本模块是「关键词 / 页码近似口径」，不是 LLM 裁判（LLM-as-a-judge）。**

与官方 RAGAS 的差异（逐项如实列出，不掩饰）：

| 指标 | 官方 RAGAS 口径 | 本模块近似口径 | 差异影响 |
| --- | --- | --- | --- |
| context_precision | LLM 逐块判「该块对回答参考是否有用」，再取 rank 加权平均 | **规则判**：块命中参考页区间（且属参考语料）**或**含 ≥1 个参考答案关键词 | 不会识别「语义相关但用词不同」的块 ⇒ 偏保守（低估）；不会误判语义无关 ⇒ 无明显高估 |
| context_recall | LLM 把参考答案拆成 claims，逐条判「能否由上下文推出」 | **关键词覆盖**：参考答案的 ``answer_keywords`` 被上下文并集覆盖的比例 | 关键词是人工/规则整理的要点，覆盖 ≠ 语义蕴含 ⇒ 会**高估**「表述不同但意思对」的情形 |
| faithfulness | LLM 把答案拆 claims，判每条能否由上下文支持 | **数值原子 + 参考答案关键词**被上下文支持的比例 | 只看数字与关键词，不看叙述性 claim ⇒ 对定性表述偏宽松 |
| answer_relevancy | LLM 由答案反推问题，算与原问题的向量相似度 | **参考答案关键词覆盖**（复用 工单7 判分口径）+ 长度合理性 | 不依赖嵌入模型，可复现；对「答非所问但含关键词」的情形会高估 |

**结论**：本口径适合做**同一套问题集上的优化前后对比**（同口径、可比、可复现），
绝对值**不应**直接与官方 RAGAS 结果对标。所有汇总产物都带 ``口径`` 字段写明这一点。

================================================================================
公式（全部写在下面，与实现一一对应）
================================================================================
记归一化（复用 工单7 ``判分.normalize``：全角→半角、去千分位、标点→空格、小写）后的：

* ``C = [c_1 … c_K]`` —— **上下文块**（按检索名次排序；``上下文口径`` 见下）
* ``Q = {q_1 … q_n}`` —— 参考答案关键词 ``answer_keywords``
* ``P`` —— 参考页区间（``expected_pages_hint``，支持 ``5-51`` / ``5-37;7-20``）
* ``F`` —— 参考语料文件名（``source_file``，多文件题用 ``;`` 分隔）
* ``contains(t, k)`` —— 带**数字边界**的关键词包含判定（``11.65`` 不命中 ``1.65``）

**(0) 块相关判定**（context_precision 的判据，二者取「或」）::

    relevant(c) = [ file(c) ∈ F  ∧  page(c) ∈ P ]   ∨   [ ∃ q ∈ Q : contains(text(c), q) ]

    多文件题（|F| == |P| > 1）按**下标配对**：第 i 个文件只与第 i 个区间配对
    （与 工单7 ``判分.page_hit`` 的跨文档题语义一致）。

**(1) context_precision（主口径，比例版）**::

    context_precision = |{ c ∈ C : relevant(c) }| / K          （K = |C|，K=0 时记 0）

**(2) context_precision_rank（RAGAS 原版 rank 加权版的规则化）**::

    rel_i   = 1 if relevant(c_i) else 0
    prec@i  = (Σ_{j≤i} rel_j) / i
    context_precision_rank = Σ_i (prec@i × rel_i) / Σ_i rel_i   （Σrel=0 时记 0）
    —— 对「相关块排前面」给更高分，与官方 RAGAS 的排序敏感特性对齐。

**(3) context_recall**::

    context_recall = |{ q ∈ Q : ∃ c ∈ C, contains(text(c), q) }| / |Q|   （Q 空时记 0）

**(4) faithfulness**::

    设 A = 答案里的**数值原子**（金额/比例/年份，正则见 ``_ATOM_RE``）
    faithfulness = |{ a ∈ A : ∃ c ∈ C, contains_numeric(text(c), a) }| / |A|      （A 非空）
                 = |{ q ∈ Q : contains(answer, q) }| / |Q|                        （A 为空，退化口径）

**(5) answer_relevancy**::

    cov          = 参考答案关键词在**答案**里的覆盖率（== 工单7 的 keyword_coverage）
    length_score = 1.0                       若 20 ≤ len(answer) ≤ 800
                 = len(answer)/20            若 len(answer) < 20
                 = 800/len(answer)           若 len(answer) > 800
    answer_relevancy = 0.0                   若 refused（拒答不算相关）
                     = 0.7 × cov + 0.3 × length_score   否则

================================================================================
上下文口径（``上下文口径`` 参数，两套都算、都报）
================================================================================
* ``evidence``（**主口径**）—— **进生成阶段的证据块**（``answer.evidence_chunks``）。
  这才是 RAGAS 语义下的 ``retrieved_contexts``：真正喂给生成模型的那些块。
* ``topk``（附口径）—— 检索返回的 Top-K（``answer.all_chunks[:K]``），
  衡量「召回池」本身的精度/召回，不受生成层选块影响。

用法（可单独 import 做单元测试，不依赖索引/模型/网络）::

    from ragas近似 import evaluate_record
    m = evaluate_record(record, contexts, answer_text, refused=False)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

WORK_ORDER = "人工智能NLP-RAG-Graph RAG 优化任务"

#: 复用 工单7 的纯函数判分模块（**只读引用**，不复制逻辑 ⇒ 保证与 07/08 工单同口径）
BASE_EVAL_DIR = Path(r"E:\gao6gongdan\工单7\研发\评估")
if str(BASE_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_EVAL_DIR))

try:
    import 判分 as _J  # type: ignore  # noqa: N813 —— 工单7 的模块名就是中文
except Exception as exc:  # noqa: BLE001 —— 缺 工单7 时必须显式失败，不给分
    raise SystemExit(f"[ERROR] 需要 工单7 的判分模块（只读）：{type(exc).__name__}: {exc}") from exc


# ---------------------------------------------------------------------------
# 默认阈值（**全部可被配置覆盖**，见 研发/app/graphrag_opt/配置.py 与 部署/配置/优化配置.json）
# ---------------------------------------------------------------------------
DEFAULTS: dict[str, float] = {
    "context_precision_target": 0.80,     # 工单硬指标：上下文精度 ≥ 0.8
    "context_recall_target": 0.90,        # 工单硬指标：上下文召回 ≥ 0.9
    "sla_ms": 3000.0,                     # 工单硬指标：提问→答案 ≤ 3 秒
    "answer_len_min": 20.0,               # answer_relevancy 的长度合理区间下界
    "answer_len_max": 800.0,              # 上界
    "relevancy_keyword_weight": 0.7,      # answer_relevancy = w×cov + (1-w)×length_score
}

#: 数值原子（与生成层护栏同口径）
_ATOM_RE = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*(?:亿元|万元|千元|百万元|元|亿美元|亿港元)"
    r"|\d+(?:\.\d+)?\s*(?:%|个百分点|倍)"
    r"|(?:19|20)\d{2}\s*年(?:度)?"
)
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _cfg(key: str, override: Mapping[str, Any] | None = None) -> float:
    if override and key in override:
        try:
            return float(override[key])
        except (TypeError, ValueError):
            pass
    return float(DEFAULTS[key])


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def norm(text: Any) -> str:
    """归一化（**直接复用 工单7 口径**，不自己再写一遍以免口径漂移）。"""
    return _J.normalize(text)


def contains(hay_norm: str, needle: str) -> bool:
    """带数字边界的关键词包含判定（复用 工单7 口径）。"""
    return _J.contains_keyword(hay_norm, needle)


def _basename(name: Any) -> str:
    return str(name or "").strip().replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _split_source(source_file: Any) -> list[str]:
    return _J.split_source_files(source_file)


def _context_text(item: Mapping[str, Any]) -> str:
    """取上下文块正文（兼容检索 payload 的 ``text`` 与索引原始行的 ``content``）。"""
    return str(item.get("text") or item.get("content") or "")


# ---------------------------------------------------------------------------
# 块相关判定
# ---------------------------------------------------------------------------
def chunk_relevance(item: Mapping[str, Any], *, keywords: Sequence[str],
                    page_ranges: Sequence[tuple[int, int]], files: Sequence[str],
                    keyword_norm: Sequence[str]) -> dict[str, Any]:
    """单块相关判定（公式 (0)）。

    :returns: ``{"relevant": bool, "by_page": bool, "by_keyword": bool, "hit_keywords": [...]}``
    """
    text = norm(_context_text(item))
    hit_keywords = [kw for kw, nk in zip(keywords, keyword_norm) if nk and contains(text, kw)]

    by_page = False
    if page_ranges:
        try:
            page = int(item.get("page") or 0)
        except (TypeError, ValueError):
            page = 0
        name = _basename(item.get("file_name"))
        pairs = (len(files) == len(page_ranges) > 1)          # 多文件题：下标配对
        if files:
            if pairs and name in {_basename(f) for f in files}:
                lo, hi = page_ranges[[_basename(f) for f in files].index(name)]
                by_page = lo <= page <= hi
            elif not pairs:
                in_file = name in {_basename(f) for f in files}
                by_page = in_file and any(lo <= page <= hi for lo, hi in page_ranges)
        else:
            by_page = any(lo <= page <= hi for lo, hi in page_ranges)

    return {"relevant": bool(by_page or hit_keywords), "by_page": bool(by_page),
            "by_keyword": bool(hit_keywords), "hit_keywords": hit_keywords}


def context_metrics(contexts: Sequence[Mapping[str, Any]], *, keywords: Sequence[str],
                    expected_pages_hint: Any = "", source_file: Any = None) -> dict[str, Any]:
    """上下文指标（公式 (1)(2)(3)）。"""
    kws = [str(k) for k in (keywords or []) if str(k).strip()]
    keyword_norm = [norm(k) for k in kws]
    ranges = _J.parse_page_ranges(expected_pages_hint)
    files = _split_source(source_file)

    ctx = list(contexts or [])
    k = len(ctx)
    flags: list[dict[str, Any]] = [chunk_relevance(c, keywords=kws, page_ranges=ranges,
                                                   files=files, keyword_norm=keyword_norm)
                                   for c in ctx]
    rel = [1 if f["relevant"] else 0 for f in flags]
    n_rel = sum(rel)

    precision = round(n_rel / k, 4) if k else 0.0
    # rank 加权版（RAGAS 原版排序敏感特性的规则化）
    if n_rel:
        cum, weighted = 0, 0.0
        for i, r in enumerate(rel, 1):
            cum += r
            if r:
                weighted += (cum / i) * r
        precision_rank = round(weighted / n_rel, 4)
    else:
        precision_rank = 0.0

    merged = norm("\n".join(_context_text(c) for c in ctx))
    hit_kw = [kw for kw in kws if contains(merged, kw)]
    recall = round(len(hit_kw) / len(kws), 4) if kws else 0.0

    return {
        "context_size": k,
        "context_relevant_count": n_rel,
        "context_precision": precision,
        "context_precision_rank": precision_rank,
        "context_recall": recall,
        "context_keyword_hit": hit_kw,
        "context_keyword_miss": [kw for kw in kws if kw not in hit_kw],
        "context_page_rel_count": sum(1 for f in flags if f["by_page"]),
        "context_keyword_rel_count": sum(1 for f in flags if f["by_keyword"]),
        "per_chunk_relevance": [{"chunk_id": c.get("chunk_id"), "page": c.get("page"),
                                 "file_name": c.get("file_name"), **f} for c, f in zip(ctx, flags)],
    }


# ---------------------------------------------------------------------------
# 答案侧指标
# ---------------------------------------------------------------------------
def _numeric_atoms(text: Any) -> list[str]:
    out: list[str] = []
    for m in _ATOM_RE.finditer(str(text or "")):
        atom = re.sub(r"\s+", "", m.group(0))
        if atom and atom not in out:
            out.append(atom)
    return out


def faithfulness(answer: Any, contexts: Sequence[Mapping[str, Any]], *,
                 keywords: Sequence[str] = ()) -> dict[str, Any]:
    """忠实度（公式 (4)）：答案里的数值原子有多少能在上下文里找到。"""
    atoms = _numeric_atoms(answer)
    merged = norm("\n".join(_context_text(c) for c in contexts or []))
    merged_atoms = set(_NUM_RE.findall(merged))
    if atoms:
        hit, miss = [], []
        for atom in atoms:
            nums = _NUM_RE.findall(norm(atom))
            ok = bool(nums) and all(n in merged_atoms for n in nums)
            (hit if ok else miss).append(atom)
        return {"faithfulness": round(len(hit) / len(atoms), 4), "mode": "numeric_atom",
                "supported": hit, "unsupported": miss, "total": len(atoms)}
    kws = [str(k) for k in (keywords or []) if str(k).strip()]
    if not kws:
        return {"faithfulness": 0.0, "mode": "empty", "supported": [], "unsupported": [], "total": 0}
    answer_norm = norm(answer)
    hit = [kw for kw in kws if contains(answer_norm, kw)]
    return {"faithfulness": round(len(hit) / len(kws), 4), "mode": "keyword_fallback",
            "supported": hit, "unsupported": [k for k in kws if k not in hit], "total": len(kws)}


def answer_relevancy(answer: Any, *, keywords: Sequence[str], refused: bool = False,
                     override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """答案相关性（公式 (5)）。"""
    if refused:
        return {"answer_relevancy": 0.0, "keyword_coverage": 0.0, "length_score": 0.0,
                "reason": "拒答 ⇒ 相关性记 0", "answer_chars": len(str(answer or ""))}
    text = str(answer or "")
    n, total, cov, hit, miss = _J.keyword_coverage(text, list(keywords or []))
    lo, hi = _cfg("answer_len_min", override), _cfg("answer_len_max", override)
    chars = len(text)
    if chars >= lo and chars <= hi:
        length_score = 1.0
    elif chars < lo:
        length_score = round(chars / lo, 4) if lo else 0.0
    else:
        length_score = round(hi / chars, 4) if chars else 0.0
    w = _cfg("relevancy_keyword_weight", override)
    value = w * float(cov) + (1.0 - w) * float(length_score)
    return {"answer_relevancy": round(value, 4), "keyword_coverage": round(float(cov), 4),
            "length_score": round(length_score, 4), "keyword_hit": hit, "keyword_miss": miss,
            "keyword_total": total, "answer_chars": chars}


# ---------------------------------------------------------------------------
# 单题总入口
# ---------------------------------------------------------------------------
def evaluate_record(record: Mapping[str, Any], contexts: Sequence[Mapping[str, Any]],
                    answer: Any, *, refused: bool = False,
                    override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """对单题算出全部 4 个指标（``contexts`` 为**生成阶段证据块**，即主口径）。"""
    keywords = list(record.get("answer_keywords") or [])
    page_hint = record.get("expected_pages_hint")
    source_file = record.get("source_file")
    ctx = context_metrics(contexts, keywords=keywords, expected_pages_hint=page_hint,
                          source_file=source_file)
    return {
        "id": record.get("id"),
        "context_precision": ctx["context_precision"],
        "context_precision_rank": ctx["context_precision_rank"],
        "context_recall": ctx["context_recall"],
        "context_size": ctx["context_size"],
        "context_relevant_count": ctx["context_relevant_count"],
        "context_keyword_hit": ctx["context_keyword_hit"],
        "context_keyword_miss": ctx["context_keyword_miss"],
        "context_detail": ctx,
        "faithfulness_detail": faithfulness(answer, contexts, keywords=keywords),
        "answer_relevancy_detail": answer_relevancy(answer, keywords=keywords, refused=refused,
                                                    override=override),
    }


def evaluate_pair(record: Mapping[str, Any], *, evidence_contexts: Sequence[Mapping[str, Any]],
                  topk_contexts: Sequence[Mapping[str, Any]], answer: Any, refused: bool = False,
                  override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """两套上下文口径一起算：``evidence``（主） + ``topk``（附）。"""
    keywords = list(record.get("answer_keywords") or [])
    page_hint = record.get("expected_pages_hint")
    source_file = record.get("source_file")
    ev = context_metrics(evidence_contexts, keywords=keywords, expected_pages_hint=page_hint,
                         source_file=source_file)
    tk = context_metrics(topk_contexts, keywords=keywords, expected_pages_hint=page_hint,
                         source_file=source_file)
    return {
        "id": record.get("id"),
        # ---- 主口径：生成阶段的证据块 ----
        "context_precision": ev["context_precision"],
        "context_precision_rank": ev["context_precision_rank"],
        "context_recall": ev["context_recall"],
        "context_size": ev["context_size"],
        "context_relevant_count": ev["context_relevant_count"],
        "context_keyword_hit": ev["context_keyword_hit"],
        "context_keyword_miss": ev["context_keyword_miss"],
        "context_detail": ev,
        # ---- 附口径：检索 Top-K ----
        "topk_context_precision": tk["context_precision"],
        "topk_context_precision_rank": tk["context_precision_rank"],
        "topk_context_recall": tk["context_recall"],
        "topk_context_size": tk["context_size"],
        "topk_context_relevant_count": tk["context_relevant_count"],
        "topk_context_detail": tk,
        # ---- 答案侧 ----
        "faithfulness_detail": faithfulness(answer, evidence_contexts, keywords=keywords),
        "answer_relevancy_detail": answer_relevancy(answer, keywords=keywords, refused=refused,
                                                    override=override),
    }


def aggregate(rows: Sequence[Mapping[str, Any]], *, override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """宏平均汇总 + 达标判定（阈值全部来自配置）。"""
    ok = [r for r in rows if not r.get("error")]
    n = len(ok)
    if not n:
        return {"题数": 0}

    def mean(key: str) -> float:
        vals = [float(r.get(key) or 0.0) for r in ok if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    cp = mean("context_precision")
    cr = mean("context_recall")
    cpr = mean("context_precision_rank")
    tcp = mean("topk_context_precision")
    tcr = mean("topk_context_recall")
    tcpr = mean("topk_context_precision_rank")
    faith = round(sum(float((r.get("faithfulness_detail") or {}).get("faithfulness") or 0.0)
                      for r in ok) / n, 4)
    ar = round(sum(float((r.get("answer_relevancy_detail") or {}).get("answer_relevancy") or 0.0)
                   for r in ok) / n, 4)
    cp_t = _cfg("context_precision_target", override)
    cr_t = _cfg("context_recall_target", override)
    sla_t = _cfg("sla_ms", override)
    lat = [float(r.get("latency_total_ms") or 0.0) for r in ok]
    lat_sorted = sorted(lat)

    def pct(q: float) -> float:
        if not lat_sorted:
            return 0.0
        idx = min(int(round(q * (len(lat_sorted) - 1))), len(lat_sorted) - 1)
        return round(lat_sorted[idx], 2)

    sla_hit = sum(1 for v in lat if v <= sla_t)
    return {
        "题数": n,
        "context_precision": cp,
        "context_precision_rank": cpr,
        "context_recall": cr,
        "topk_context_precision": tcp,
        "topk_context_precision_rank": tcpr,
        "topk_context_recall": tcr,
        "faithfulness": faith,
        "answer_relevancy": ar,
        "达标": {
            "阈值": {"context_precision": cp_t, "context_recall": cr_t, "sla_ms": sla_t},
            "context_precision达标": bool(cp >= cp_t),
            "context_recall达标": bool(cr >= cr_t),
            "context_precision_rank达标": bool(cpr >= cp_t),
            "响应时间达标率": round(sla_hit / n, 4),
            "SLA达标题数": sla_hit,
        },
        "响应时间ms": {"平均": round(sum(lat) / n, 2), "P50": pct(0.50), "P90": pct(0.90),
                       "P95": pct(0.95), "最大": round(max(lat), 2), "最小": round(min(lat), 2)},
        "口径": ("关键词/页码近似口径（非 LLM 裁判）；上下文主口径 = 生成阶段证据块，"
                 "附口径 = 检索 Top-K。详见 研发/评估/ragas近似.py 模块 docstring。"),
    }


def format_table(rows: Sequence[Mapping[str, Any]], title: str = "") -> str:
    """逐题明细的文本表（落报告用）。"""
    lines = [f"== {title} ==" if title else "=="]
    lines.append(f"{'题号':<6}{'上下文精度':>10}{'rank精度':>10}{'上下文召回':>10}"
                 f"{'TopK精度':>10}{'TopK召回':>10}{'忠实度':>8}{'相关性':>8}{'耗时ms':>10}")
    for r in rows:
        fd = r.get("faithfulness_detail") or {}
        ad = r.get("answer_relevancy_detail") or {}
        lines.append(
            f"{str(r.get('id')):<6}{float(r.get('context_precision') or 0):>10.3f}"
            f"{float(r.get('context_precision_rank') or 0):>10.3f}"
            f"{float(r.get('context_recall') or 0):>10.3f}"
            f"{float(r.get('topk_context_precision') or 0):>10.3f}"
            f"{float(r.get('topk_context_recall') or 0):>10.3f}"
            f"{float(fd.get('faithfulness') or 0):>8.3f}"
            f"{float(ad.get('answer_relevancy') or 0):>8.3f}"
            f"{float(r.get('latency_total_ms') or 0):>10.1f}")
    return "\n".join(lines)


def _selftest() -> int:
    """内置自检（纯函数；pytest 也调用它）。"""
    failures: list[str] = []

    def check(name: str, got: Any, want: Any) -> None:
        if got != want:
            failures.append(f"{name}: got={got!r} want={want!r}")

    rec = {"id": "T1", "answer_keywords": ["1,379.58", "18.2%", "281.95"],
           "expected_pages_hint": "5-51",
           "source_file": "2020-02-14__平安银行股份有限公司__000001__平安银行__2019年__年度报告.pdf"}
    good = [{"chunk_id": "p5-x1", "page": 5, "file_name": rec["source_file"],
             "text": "零售进一步巩固收入贡献的支柱地位，营业收入1,379.58亿元，同比增长18.2%"},
            {"chunk_id": "p19-t0", "page": 19, "file_name": rec["source_file"],
             "text": "净利润281.95亿元"},
            {"chunk_id": "p81-x0", "page": 81, "file_name": "别家.pdf", "text": "无关内容"}]
    m = context_metrics(good, keywords=rec["answer_keywords"],
                        expected_pages_hint=rec["expected_pages_hint"],
                        source_file=rec["source_file"])
    check("precision/3块中2相关", m["context_precision"], 0.6667)
    check("recall/3关键词全覆盖", m["context_recall"], 1.0)
    # rank 加权：相关块都在最前 ⇒ 1.0（这正是 rank 加权与「比例口径」的区别所在）
    check("rank/相关块全在前", m["context_precision_rank"], 1.0)
    check("precision/相关数", m["context_relevant_count"], 2)
    # rank 加权：无关块插在中间 ⇒ 被惩罚（比例口径不变，仍 2/3）
    mixed = [good[0], good[2], good[1]]
    mm = context_metrics(mixed, keywords=rec["answer_keywords"],
                         expected_pages_hint=rec["expected_pages_hint"],
                         source_file=rec["source_file"])
    check("rank/无关块居中被惩罚", mm["context_precision_rank"], 0.8333)
    check("rank/比例口径不受排序影响", mm["context_precision"], 0.6667)

    # 空上下文
    e = context_metrics([], keywords=rec["answer_keywords"], expected_pages_hint="5-51",
                        source_file=rec["source_file"])
    check("空上下文/precision", e["context_precision"], 0.0)
    check("空上下文/recall", e["context_recall"], 0.0)
    # 空关键词
    k0 = context_metrics(good, keywords=[], expected_pages_hint="", source_file=None)
    check("空关键词/recall", k0["context_recall"], 0.0)

    # 数字边界（context_recall 不能把 11.65 当 1.65）
    bad = [{"chunk_id": "x", "page": 5, "file_name": rec["source_file"], "text": "不良贷款率11.65%"}]
    check("数字边界/11.65不命中1.65",
          context_metrics(bad, keywords=["1.65"], expected_pages_hint="", source_file=None)["context_recall"], 0.0)

    # faithfulness
    f1 = faithfulness("营业收入1,379.58亿元，同比增长18.2%，净利润281.95亿元", good)
    check("faithfulness/全部支持", f1["faithfulness"], 1.0)
    f2 = faithfulness("营业收入9,999.99亿元", good)
    check("faithfulness/不支持", f2["faithfulness"], 0.0)

    # answer_relevancy
    a1 = answer_relevancy("营业收入1,379.58亿元，同比增长18.2%，净利润281.95亿元。", keywords=rec["answer_keywords"])
    check("relevancy/全覆盖", a1["keyword_coverage"], 1.0)
    check("relevancy/长度满分", a1["length_score"], 1.0)
    check("relevancy/加权", a1["answer_relevancy"], 1.0)
    a2 = answer_relevancy("不清楚", keywords=rec["answer_keywords"], refused=True)
    check("relevancy/拒答记0", a2["answer_relevancy"], 0.0)

    # 跨文档题：文件与页区间下标配对
    multi = {"id": "T2", "answer_keywords": ["a"], "expected_pages_hint": "5-20;25-40",
             "source_file": "A.pdf;B.pdf"}
    ok_pair = [{"chunk_id": "1", "page": 30, "file_name": "B.pdf", "text": "a"},
               {"chunk_id": "2", "page": 10, "file_name": "A.pdf", "text": "b"}]
    check("跨文档/配对命中", context_metrics(ok_pair, keywords=["a"], expected_pages_hint="5-20;25-40",
                                        source_file="A.pdf;B.pdf")["context_precision"], 1.0)
    bad_pair = [{"chunk_id": "1", "page": 30, "file_name": "A.pdf", "text": "zzz"}]
    check("跨文档/错序不命中", context_metrics(bad_pair, keywords=["a"], expected_pages_hint="5-20;25-40",
                                          source_file="A.pdf;B.pdf")["context_precision"], 0.0)

    # aggregate
    agg = aggregate([{**m, "latency_total_ms": 1000, "faithfulness_detail": f1,
                      "answer_relevancy_detail": a1}])
    check("汇总/题数", agg["题数"], 1)
    check("汇总/precision", agg["context_precision"], 0.6667)
    check("汇总/达标判定", agg["达标"]["context_precision达标"], False)
    check("汇总/SLA", agg["达标"]["响应时间达标率"], 1.0)

    if failures:
        print(f"[RAGAS近似自检] 失败 {len(failures)} 项：")
        for item in failures:
            print("  -", item)
        return 1
    print("[RAGAS近似自检] 全部通过（比例/rank 精度、召回、数字边界、跨文档配对、"
          "忠实度、相关性、空输入、汇总达标）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
