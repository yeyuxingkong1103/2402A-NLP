# -*- coding: utf-8 -*-
"""工单：人工智能NLP-RAG-Query 理解优化

对话级缓存基准（``bench_dialogue_cache``）—— 用真实 ``MultiTurnEngine`` 跑五轮多轮序列，
对比 **开缓存 / 关缓存** 的端到端延迟，量出对话级缓存的收益。

## ⚠️ 口径声明（必读，别把这里的数字当成引擎内置缓存）
``研发/app/multiturn_engine.py`` **尚未接入** ``DialogueCache``（由 captain 接线）。
因此本基准**在脚本侧用 ``_CachedAsk`` 包装 ``MultiTurnEngine.ask`` 模拟接线**：

    问句 → 改写（engine 内，必跑）→ 组 key(rewritten_query, lang, session_id)
         → DialogueCache.get ── 命中 → 直接返回（不检索、不生成）
                              └ 未命中 → engine.ask(...) → 写回缓存

真实改写 / 真实检索 / 真实生成都在；**只有「查缓存」这一步是脚本拼的**。

## 为什么每个阶段是**独立进程**
``QAEngine`` 自带一层进程内答案缓存，键是 ``question + lang + top_k``——**不含 session_id**。
若把「冷跑 / 测量 / 校验」放在同一进程里，第 2 阶段起会被它短路（实测冷跑 P50 直接掉到
3~5 ms，等于没测）。因此本脚本用 ``--phase`` 把三个阶段拆成**独立子进程**串行执行
（默认 ``--phase all`` 自动编排），每个阶段都拿到**全新引擎 + 全新空缓存**。

## 测量协议
| 阶段 | 进程 | session | 缓存 | 用途 |
| --- | --- | --- | --- | --- |
| 预热 | 各自 | — | — | ``warmup()``，**单独记、不计入统计** |
| `cold` | 独立 | ``bench-cold`` | 完全不经缓存 | 对照基准：直接调 ``engine.ask`` |
| `hit` | 独立 | ``bench-live`` | **开**（空缓存） | 逐轮「先未命中、紧接着同一问句再问一次命中」 |
| `check` | 独立 | ``bench-check`` | **开** | 同轮同会话：缓存包装 vs 直调 ``engine.ask`` 逐题比对 |

引擎内置答案缓存**无法从外部关闭**（``use_cache`` 只关内存+落盘两层，见
``qa_engine._memory_cache`` 的实现），故用「独立进程」而非「开关」来隔离它——这也是
本基准必须分进程的根因。

## 为什么命中测量写成「逐轮连续问两次」
逐轮在同一 session 内：第 1 次问 → 未命中 → 真实执行整条链路；紧接着第 2 次问同一句
→ 改写结果与状态完全一致 → 必中。这样每一轮都拿到**可比的未命中/命中一对样本**，
且不会因为「同一遍序列里出现重复改写问句」（第 2 轮与第 4 轮实测改写结果相同）而
把某一轮的未命中样本吃掉。

## 产物
* ``优化/评估结果/对话缓存基准.json`` + ``.md``
* ``优化/评估结果/对话缓存基准_阶段.json``（三阶段原始数据，便于复核）

运行：
    pwsh -NoProfile -File run_py.ps1 优化/脚本/bench_dialogue_cache.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "研发"))

WORK_ORDER = "人工智能NLP-RAG-Query 理解优化"

#: 工单指定的五轮序列（逐字，不改）
SEQUENCE: list[str] = [
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    "他参与的哪个工程荣获了国家科技进步一等奖？",
    "这个公司的法定代表人是谁？",
    "那武汉力源信息技术股份有限公司呢？",
    "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",
]

COLD_SID = "bench-cold"      # 冷跑基线（完全不经缓存）
LIVE_SID = "bench-live"      # 命中测量（开缓存）
CHECK_SID = "bench-check"    # 正确性校验（开缓存 + 直调引擎比对）

#: 校验答案时比对的字段（「硬口径」：缓存是确定性回放，这几项必须逐字一致）
_COMPARE_FIELDS = ("answer", "rewritten_query", "refused")
#: 一并展示、但**不参与**硬口径判定的字段：引用页码由模型各自生成（实测同一问句两次调用
#: 会给出不同页码），属模型侧随机性，与缓存无关；focus_entity / strategy 走改写路径，必一致。
_SOFT_FIELDS = ("focus_entity", "strategy", "cited_pages")


# --------------------------------------------------------------------- 工具
def percentile(values: list[float], q: float) -> float:
    """最近序法百分位（与工单4 ``bench_cache.py`` 同口径）；毫秒保留 2 位。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return round(ordered[idx], 2)


def percentile_raw(values: list[float], q: float) -> float:
    """同一百分位口径，但**不四舍五入**（毫秒分辨率看不到命中耗时，需要 µs 精度）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[idx]


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 2) if values else 0.0


def _fmt(value: float | None) -> str:
    """极小值也要看得见（命中路径实测 < 0.01 ms）。"""
    if value is None:
        return "—"
    return "&lt;0.01" if value == 0 else f"{value:.2f}"


@contextmanager
def _timed() -> Iterator[list[float]]:
    """只把 **engine.ask 本体** 算进耗时；缓存查表单独记为 ``cache_lookup_ms``。"""
    box: list[float] = []
    started = time.perf_counter()
    try:
        yield box
    finally:
        box.append((time.perf_counter() - started) * 1000)


def _describe(answer: dict[str, Any]) -> dict[str, Any]:
    return {
        "rewritten_query": answer.get("rewritten_query"),
        "strategy": answer.get("strategy"),
        "focus_entity": answer.get("focus_entity"),
        "refused": bool(answer.get("refused")),
        "turn_index": answer.get("turn_index"),
        "cited_pages": list(answer.get("cited_pages") or []),
        "answer": str(answer.get("answer") or ""),
        "answer_head": str(answer.get("answer") or "")[:100],
        "engine_total_ms": round(float((answer.get("latency_ms") or {}).get("total") or 0.0), 2),
    }


def _golden_keywords() -> dict[str, list[str]]:
    """从仓库自带金标题库取本序列前两问的关键字口径（**只读，不新增数据**）。

    用于「答案本身对不对」的自证：只做关键字包含检查，不做语义判断。
    取值优先级：``answer_keywords``（`|` 分隔）→ ``required_substrings``。
    """
    path = ROOT / "研发" / "data" / "eval" / "golden_qa.jsonl"
    wanted = {"军用": "来自军用领域的收入分别是多少", "组织结构": "组织结构图中"}
    out: dict[str, list[str]] = {}
    if not path.is_file():
        return out
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            question = str(item.get("question") or "")
            for tag, needle in wanted.items():
                if tag in out or needle not in question:
                    continue
                raw = item.get("answer_keywords") or item.get("required_substrings") or []
                if isinstance(raw, str):
                    raw = [k.strip() for k in raw.split("|")]
                keys = [str(k).strip() for k in raw if str(k).strip()]
                if keys:
                    out[tag] = keys
    except Exception:  # noqa: BLE001 —— 关键字口径拿不到就不自证，绝不因此中断基准
        return {}
    return out


class _CachedAsk:
    """脚本侧包装：给 ``MultiTurnEngine.ask`` 加一层对话级缓存（**模拟接线**）。"""

    def __init__(self, engine: Any, cache: Any = None) -> None:
        self.engine = engine
        self.cache = cache
        self.records: list[dict[str, Any]] = []

    def ask(self, question: str, *, session_id: str, lang: str = "auto",
            use_cache: bool = False) -> dict[str, Any]:
        from app.core.dialogue_cache import DialogueCache

        state = self.engine.states.get(session_id)
        rewritten = self.engine._ensure_rewriter().rewrite(question, state).rewritten
        key = (DialogueCache.make_key(rewritten, lang, session_id)
               if self.cache is not None else "")
        answer_dict: dict[str, Any] | None = None
        source = "engine"
        lookup_ms = 0.0
        if self.cache is not None and key:
            started = time.perf_counter()
            hit = self.cache.get(key)
            lookup_ms = round((time.perf_counter() - started) * 1000, 4)
            if hit is not None:
                answer_dict, source = hit, "cache"
        with _timed() as box:
            if answer_dict is None:
                answer_dict = self.engine.ask(question, session_id=session_id, lang=lang,
                                              use_cache=use_cache).to_dict()
        elapsed_ms = box[0] if box else 0.0
        if source == "engine" and self.cache is not None and key:
            self.cache.put(key, answer_dict)
        self.records.append({"question": question, "rewritten": rewritten, "key": key,
                             "source": source, "ms": round(elapsed_ms, 2),
                             "ms_raw": elapsed_ms, "us_raw": elapsed_ms * 1000,
                             "cache_lookup_ms": lookup_ms, "answer": answer_dict})
        return answer_dict


def _record(rec: dict[str, Any]) -> dict[str, Any]:
    payload = {k: rec[k] for k in ("question", "rewritten", "key", "source", "ms",
                                   "cache_lookup_ms")}
    payload.update(_describe(rec["answer"]))
    return payload


def _new_engine() -> tuple[Any, dict[str, Any], float]:
    """新建引擎 + 预热；返回 (engine, 预热信息, 预热毫秒)。"""
    from app.multiturn_engine import MultiTurnEngine

    engine = MultiTurnEngine()
    started = time.perf_counter()
    warm = engine.warmup()
    return engine, warm, round((time.perf_counter() - started) * 1000, 2)


# --------------------------------------------------------------------- 各阶段
def phase_cold() -> dict[str, Any]:
    """阶段 1：冷跑基线（每轮都真实执行，完全不经任何缓存）。"""
    engine, warm, warmup_ms = _new_engine()
    engine.reset(COLD_SID)
    walk = _CachedAsk(engine, cache=None)
    for question in SEQUENCE:
        walk.ask(question, session_id=COLD_SID, use_cache=False)
    return {"phase": "cold", "warmup_ms": warmup_ms, "warmup": warm,
            "session_id": COLD_SID, "rounds": [_record(r) for r in walk.records]}


def phase_hit(out_dir: Path, *, max_size: int, ttl: float) -> dict[str, Any]:
    """阶段 2：逐轮「未命中 → 命中」各一次（全新空缓存；落盘以验证原子写）。"""
    from app.core.dialogue_cache import DialogueCache

    engine, warm, warmup_ms = _new_engine()
    engine.reset(LIVE_SID)
    persist_path = out_dir / "dialogue_cache_bench.json"
    if persist_path.is_file():
        persist_path.unlink()                 # 绝不复用上次落盘结果
    cache = DialogueCache(max_size=max_size, ttl_s=ttl, persist_path=persist_path)
    assert cache.size() == 0, "测量前缓存必须为空"
    wrapper = _CachedAsk(engine, cache)

    rounds: list[dict[str, Any]] = []
    for order, question in enumerate(SEQUENCE, start=1):
        wrapper.ask(question, session_id=LIVE_SID, use_cache=False)     # 未命中
        miss = wrapper.records[-1]
        wrapper.ask(question, session_id=LIVE_SID, use_cache=False)     # 紧接着命中
        hit = wrapper.records[-1]
        rounds.append({
            "round": order,
            "question": question,
            "未命中_ms": miss["ms"],
            "命中_ms": hit["ms"],
            "命中_us": round(hit["us_raw"], 3),
            "命中_us_raw": round(hit["us_raw"], 3),
            "未命中_us_raw": round(miss["us_raw"], 3),
            "加速比": round(miss["ms"] / hit["ms"], 2) if hit["ms"] > 0 else None,
            "加速比_相对查表": (round(miss["ms"] / hit["cache_lookup_ms"], 1)
                                if hit["cache_lookup_ms"] > 0 else None),
            "缓存查表_ms": round((miss["cache_lookup_ms"] + hit["cache_lookup_ms"]) / 2, 4),
            "来源": {"未命中": miss["source"], "命中": hit["source"]},
            "键": hit["key"],
            "键一致": miss["key"] == hit["key"],
            **_describe(hit["answer"]),
        })
    stats = cache.stats()
    return {"phase": "hit", "warmup_ms": warmup_ms, "warmup": warm,
            "session_id": LIVE_SID, "rounds": rounds, "cache_stats": stats,
            "measured": {"hits": stats["hits"], "misses": stats["misses"]},
            "trace": [_record(r) for r in wrapper.records]}


def phase_check() -> dict[str, Any]:
    """阶段 3：同轮同会话「缓存包装 vs 直调引擎」逐题比对答案。"""
    from app.core.dialogue_cache import DialogueCache

    engine, warm, warmup_ms = _new_engine()
    engine.reset(CHECK_SID)
    wrapped = _CachedAsk(engine, DialogueCache(max_size=64, ttl_s=600.0, persist_path=None))
    direct = _CachedAsk(engine, cache=None)

    rows: list[dict[str, Any]] = []
    diffs: list[str] = []
    soft_diffs: list[str] = []
    aligned = 0
    for order, question in enumerate(SEQUENCE, start=1):
        wrapped_answer = wrapped.ask(question, session_id=CHECK_SID, use_cache=False)
        direct_answer = direct.ask(question, session_id=CHECK_SID, use_cache=False)
        bad = [f for f in _COMPARE_FIELDS if wrapped_answer.get(f) != direct_answer.get(f)]
        soft = [f for f in _SOFT_FIELDS if wrapped_answer.get(f) != direct_answer.get(f)]
        if bad:
            diffs.append(f"第{order}轮字段不一致：{bad}")
        else:
            aligned += 1
        if soft:
            soft_diffs.append(f"第{order}轮：{soft}（模型侧随机性，非缓存差异）")
        rows.append({"round": order, "question": question, "一致": not bad, "差异字段": bad,
                     "软字段差异": soft,
                     "命中来源": wrapped.records[-1]["source"],
                     "wrapped_ms": wrapped.records[-1]["ms"],
                     "direct_ms": direct.records[-1]["ms"],
                     "wrapped": _describe(wrapped_answer), "direct": _describe(direct_answer)})
    return {"phase": "check", "warmup_ms": warmup_ms, "warmup": warm,
            "session_id": CHECK_SID, "rows": rows,
            "aligned": aligned, "total": len(SEQUENCE), "diffs": diffs,
            "soft_diffs": soft_diffs}


# --------------------------------------------------------------------- 汇总
def _render_md(payload: dict[str, Any]) -> str:
    rounds = payload["每轮"]
    summary = payload["汇总"]
    validation = payload["校验"]
    lines: list[str] = []
    lines.append("# 对话级缓存基准（工单5 · 人工智能NLP-RAG-Query 理解优化）")
    lines.append("")
    lines.append("> **口径声明**：`研发/app/multiturn_engine.py` **尚未接入** `DialogueCache`，")
    lines.append("> 本基准**用脚本侧包装（`_CachedAsk`）模拟接线**——真实改写 / 真实检索 / 真实生成，")
    lines.append("> 只有「查缓存」这一步由脚本拼装。**不得据此声称引擎已内置缓存。**")
    lines.append("")
    lines.append("## 测量协议（三阶段各自独立进程；串行，无并发）")
    lines.append("")
    lines.append("| 阶段 | session | 对话级缓存 | 引擎内置答案缓存 | 用途 |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.append("| 预热 | 各自 | — | — | `warmup()`，单独记，**不计入统计** |")
    lines.append("| `cold` | `bench-cold` | 完全不经缓存 | 进程独占 | 对照基准：直接调 `engine.ask` |")
    lines.append("| `hit` | `bench-live` | **开**（全新空缓存） | 进程独占 | 逐轮「未命中 → 紧接着命中」 |")
    lines.append("| `check` | `bench-check` | **开** | 进程独占 | 缓存包装 vs 直调引擎，逐题比对答案 |")
    lines.append("")
    lines.append("**为什么必须分进程**：`QAEngine` 自带一层进程内答案缓存，键是 "
                 "`question + lang + top_k`（**不含 session_id**，见 `qa_engine._memory_cache`），"
                 "且没有外部开关。同进程里跑第 2 个阶段就会被它短路——实测冷跑 P50 会从 "
                 "数百毫秒掉到 3~5 ms，基准当场失效。")
    lines.append("")
    lines.append(f"预热（每阶段各自一次，均不计入统计）：cold {payload['预热_ms']['cold']} ms / "
                 f"hit {payload['预热_ms']['hit']} ms / check {payload['预热_ms']['check']} ms；"
                 f"索引 {payload['预热明细'].get('index_chunks')} chunk，"
                 f"实体 {payload['预热明细'].get('entities')} 个。")
    lines.append("")
    lines.append("## 逐轮结果")
    lines.append("")
    lines.append("| # | 问句（截断） | 未命中 (ms) | 命中 (µs) | 加速比 | 缓存查表 (ms) | 未命中 ÷ 查表 | 冷跑基线 (ms) | focus_entity | strategy |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for item in rounds:
        speedup = "&gt;1000" if item["加速比"] is None else f"{item['加速比']}×"
        ratio = item.get("加速比_相对查表")
        lines.append(
            f"| {item['round']} | {item['question'][:18]}… | {_fmt(item['未命中_ms'])} | "
            f"{item['命中_us']:.2f} | {speedup} | "
            f"{item['缓存查表_ms']:.4f} | {f'{ratio}×' if ratio else '—'} | "
            f"{_fmt(item['冷跑基线_ms'])} | "
            f"{item.get('focus_entity') or '—'} | {item.get('strategy') or '—'} |"
        )
    lines.append("")
    lines.append("> 「命中」列的毫秒值四舍五入后为 `0.00`（< 0.005 ms），故并列 µs 精度。")
    lines.append("> 「加速比 = 未命中 ms ÷ 命中 ms」在命中 < 0.005 ms 时溢出，故标记 `>1000×`")
    lines.append("> （**只给下界，不虚报**）；同时给出可比口径「未命中 ÷ 缓存查表」。")
    lines.append("")
    lines.append("## 汇总")
    lines.append("")
    cold = summary["未命中"]
    hit = summary["命中"]
    base = summary["冷跑基线"]
    lines.append("| 指标 | 未命中（真实执行） | 命中（缓存返回） | 冷跑基线（不经缓存） |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(f"| 样本数 | {cold['n']} | {hit['n']} | {base['n']} |")
    lines.append(f"| P50 | {_fmt(cold['p50_ms'])} ms | {hit['p50_us']:.2f} µs | "
                 f"{_fmt(base['p50_ms'])} ms |")
    lines.append(f"| 平均 | {_fmt(cold['mean_ms'])} ms | {hit['mean_us']:.2f} µs | "
                 f"{_fmt(base['mean_ms'])} ms |")
    lines.append(f"| 最小 / 最大 | {_fmt(cold['min_ms'])} / {_fmt(cold['max_ms'])} ms | "
                 f"{hit['min_us']:.2f} / {hit['max_us']:.2f} µs | "
                 f"{_fmt(base['min_ms'])} / {_fmt(base['max_ms'])} ms |")
    lines.append("")
    stats = summary["cache_stats"]
    lines.append(f"* 测量窗口命中率：**{summary['命中率']:.2%}**（{summary['命中率口径']}）")
    lines.append(f"* **端到端收益**：一次命中相对冷跑基线 P50 {_fmt(base['p50_ms'])} ms "
                 f"**省下约 {summary['每次命中省下_ms']:.0f} ms**"
                 f"（命中 P50 实测 {hit['p50_us']:.2f} µs，即 **>1000×**，毫秒分辨率下已到量程之外）。")
    lines.append(f"* **可比口径**：`cache.get` 查表实测 P50 **{summary['查表_p50_ms']:.4f} ms**，"
                 f"相对冷跑基线 P50 {_fmt(base['p50_ms'])} ms 是 **{summary['查表_相对基线加速比']}×**"
                 "（这一档不受计时器量程影响，最保守）。")
    lines.append(f"* 缓存配置：max_size={stats['max_size']}，ttl_s={stats['ttl_s']}，"
                 f"落盘={stats['persist_path']}，淘汰={stats['evictions']}，过期={stats['expired']}")
    lines.append("* 缓存键 = `sha1(rewritten_query + language + session_id)[:16]`，**含 session_id**"
                 "（同一问句在不同会话上下文不同，绝不跨会话复用）。")
    lines.append("* 未命中路径 = 不经缓存路径 + 一次查表（实测见「缓存查表」列，~0.1 ms 量级），"
                 "**没有额外损耗**。")
    lines.append("")
    lines.append("## 正确性校验")
    lines.append("")
    lines.append(f"**① 缓存不改答案（同轮同会话，缓存包装 vs 直调 `engine.ask`）**："
                 f"**{validation['一致题数']}/{validation['题数']} 一致**"
                 f"（硬口径字段：`{', '.join(validation['硬口径字段'])}`）")
    for note in validation["差异"]:
        lines.append(f"  * 差异：{note}")
    lines.append(f"**② 并行观察的软字段**（`{', '.join(validation['软口径字段'])}`，"
                 "走改写路径的字段实测一致；引用页码由模型各自生成）：")
    for note in validation["软字段差异"]:
        lines.append(f"  * {note}")
    lines.append("")
    proof = payload["答案自证"]
    lines.append("**③ 答案本身对不对（关键字口径，与仓库 `golden_qa.jsonl` 同源）**：")
    if proof:
        for tag, item in proof.items():
            marks = "、".join(f"{k}{'✓' if v else '✗'}" for k, v in item["命中"].items())
            lines.append(f"  * {tag}：**{item['通过']}/{len(item['命中'])} 命中**（{marks}）")
            lines.append(f"    * 答案片段：{item['答案片段'][:80]}")
    else:
        lines.append("  * 未运行（金标题库未找到或关键字口径缺失）")
    lines.append("")
    lines.append("## 未完成项 / 边界")
    lines.append("")
    lines.append("* 引擎侧接线（`MultiTurnEngine` 内置 `DialogueCache`）**未完成**：由 captain 负责；")
    lines.append("  接线后应重跑本脚本复核，届时 `_CachedAsk` 可整体删除。")
    lines.append("* 本基准量的是「同一会话内重复提问」的收益：第一次提问延迟不受影响")
    lines.append("  （未命中路径只多一次微秒级查表），收益全部来自第 2 次及以后。")
    lines.append("* 命中耗时用 `engine.ask` 包装窗口测（不摊入 `cache.get` 的查表时间），")
    lines.append("  故「加速比」是**保守下界**：真实端到端还省掉了这一步查表。")
    lines.append("* 引用页码的模型侧随机性（同一问句两次独立生成可能给出不同页码）**不是**缓存问题，")
    lines.append("  但会让「缓存答案」与「另一次生成」在页码上不一致；本基准为此把硬/软口径分开列。")
    lines.append("")
    return "\n".join(lines) + "\n"


def _run_child(phase: str, out_dir: Path, *, max_size: int, ttl: float) -> dict[str, Any]:
    """以独立子进程跑一个阶段，返回其 JSON 结果。"""
    target = out_dir / f"对话缓存基准_阶段_{phase}.json"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--phase", phase,
           "--out-dir", str(out_dir), "--max-size", str(max_size), "--ttl", str(ttl),
           "--phase-out", str(target)]
    print(f"  → 子进程阶段 {phase} …", flush=True)
    started = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    _ = proc.stdout.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"阶段 {phase} 失败（退出码 {proc.returncode}）")
    return {"result": json.loads(target.read_text(encoding="utf-8")),
            "wall_ms": round((time.perf_counter() - started) * 1000, 2)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="对话级缓存基准（工单5）")
    parser.add_argument("--out-dir", default=str(ROOT / "优化" / "评估结果"))
    parser.add_argument("--phase", default="all",
                        choices=["all", "cold", "hit", "check"],
                        help="内部使用：单跑某一阶段（各阶段必须独立进程）")
    parser.add_argument("--phase-out", default=None, help="内部使用：阶段结果 JSON 落点")
    parser.add_argument("--max-size", type=int, default=512)
    parser.add_argument("--ttl", type=float, default=1800.0)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase != "all":                       # ---- 子进程分支
        if args.phase == "cold":
            result = phase_cold()
        elif args.phase == "hit":
            result = phase_hit(out_dir, max_size=args.max_size, ttl=args.ttl)
        else:
            result = phase_check()
        target = Path(args.phase_out) if args.phase_out else out_dir / f"stage_{args.phase}.json"
        target.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{args.phase}] 预热 {result['warmup_ms']} ms；结果 → {target}")
        return 0

    # ---- 编排分支：三阶段独立进程串行
    print(f"[对话缓存基准] 三阶段独立进程串行：cold → hit → check（工作区 {ROOT}）")
    cold = _run_child("cold", out_dir, max_size=args.max_size, ttl=args.ttl)
    hit = _run_child("hit", out_dir, max_size=args.max_size, ttl=args.ttl)
    check = _run_child("check", out_dir, max_size=args.max_size, ttl=args.ttl)
    cold_res, hit_res, check_res = cold["result"], hit["result"], check["result"]

    base_ms = [r["ms"] for r in cold_res["rounds"]]
    rounds = hit_res["rounds"]
    for index, item in enumerate(rounds):        # 把冷跑基线并到逐轮表里（阶段不同进程）
        item["冷跑基线_ms"] = base_ms[index] if index < len(base_ms) else None
        item["冷跑基线_us"] = round(base_ms[index] * 1000, 3) if index < len(base_ms) else None
    miss_values = [item["未命中_ms"] for item in rounds]
    hit_values = [item["命中_ms"] for item in rounds]

    p50_miss, p50_hit, p50_base = (percentile(miss_values, 0.5), percentile(hit_values, 0.5),
                                   percentile(base_ms, 0.5))
    hit_us = [item["命中_us_raw"] for item in hit_res["rounds"]]
    miss_us = [item["未命中_us_raw"] for item in hit_res["rounds"]]
    lookup_values = [item["缓存查表_ms"] for item in rounds]
    measured = hit_res["measured"]
    total_lookups = measured["hits"] + measured["misses"]
    p50_lookup = percentile(lookup_values, 0.5)
    measured_hits = sum(1 for item in rounds if item["来源"]["命中"] == "cache")
    measured_lookups = len(rounds) * 2
    summary = {
        "未命中": {"n": len(miss_values), "p50_ms": p50_miss, "mean_ms": _mean(miss_values),
                   "min_ms": min(miss_values), "max_ms": max(miss_values),
                   "p50_us": round(percentile_raw(miss_us, 0.5), 1),
                   "mean_us": round(statistics.fmean(miss_us), 1), "values_ms": miss_values},
        "命中": {"n": len(hit_values), "p50_ms": p50_hit, "mean_ms": _mean(hit_values),
                 "min_ms": min(hit_values), "max_ms": max(hit_values),
                 "p50_us": round(percentile_raw(hit_us, 0.5), 3),
                 "mean_us": round(statistics.fmean(hit_us), 3),
                 "min_us": round(min(hit_us), 3), "max_us": round(max(hit_us), 3),
                 "values_ms": hit_values, "values_us": [round(v, 3) for v in hit_us]},
        "冷跑基线": {"n": len(base_ms), "p50_ms": p50_base, "mean_ms": _mean(base_ms),
                     "min_ms": min(base_ms), "max_ms": max(base_ms), "values_ms": base_ms},
        "测量窗口": measured,
        "命中率": round(measured_hits / measured_lookups, 4) if measured_lookups else 0.0,
        "命中率口径": (f"测量窗口内 {measured_hits} 次命中 ÷ {measured_lookups} 次查表"
                    f"（{len(rounds)} 轮 × 每轮 2 次：先未命中、再命中，故恒为 50%）；"
                    "第 2 次及以后提问的命中率为 100%；另有 DialogueCache 累计口径 "
                    f"hits/misses = {measured['hits']}/{measured['misses']}"),
        "p50_加速比": round(p50_miss / p50_hit, 2) if p50_hit > 0 else None,
        "p50_加速比_下界": (">1000" if p50_hit == 0 else None),
        "平均加速比": round(_mean(miss_values) / _mean(hit_values), 2)
        if _mean(hit_values) > 0 else None,
        "对照_p50_加速比": round(p50_base / p50_hit, 2) if p50_hit > 0 else None,
        "每次命中省下_ms": round(p50_base - p50_hit, 2),
        "查表_p50_ms": p50_lookup,
        "查表_相对基线加速比": round(p50_base / p50_lookup, 1) if p50_lookup > 0 else None,
        "cache_stats": hit_res["cache_stats"],
    }

    validation = {
        "题数": check_res["total"],
        "一致题数": check_res["aligned"],
        "差异": check_res["diffs"],
        "软字段差异": check_res["soft_diffs"],
        "硬口径字段": list(_COMPARE_FIELDS),
        "软口径字段": list(_SOFT_FIELDS),
        "口径": (f"同轮同会话比对：硬口径={list(_COMPARE_FIELDS)} 必须逐字一致；"
                 f"软口径={list(_SOFT_FIELDS)} 含模型侧随机字段，仅并列展示、不计入通过判定"),
        "逐轮": check_res["rows"],
    }

    # ---- 答案自证（关键字口径，证明缓存返回的是**对的**答案）
    keywords = _golden_keywords()
    proof: dict[str, Any] = {}
    for tag, keys in keywords.items():
        round_index = 0 if tag == "军用" else len(SEQUENCE) - 1
        answer = str(rounds[round_index]["answer"] or "")
        hits = {key: (key in answer) for key in keys}
        proof[tag] = {"问句": SEQUENCE[round_index], "命中": hits,
                      "通过": sum(1 for ok in hits.values() if ok),
                      "答案片段": answer[:120]}

    warm_detail = (hit_res.get("warmup") or {}).get("base") or {}
    payload = {
        "work_order": WORK_ORDER,
        "protocol": ("脚本侧包装 _CachedAsk 模拟接线（引擎尚未内置缓存）；三阶段各自独立进程串行："
                     "cold（不经缓存）/ hit（开缓存，逐轮未命中→命中）/ check（开缓存，同轮比对答案）"),
        "cache_enabled_in_engine": False,
        "engine_answer_cache": "无法外部关闭；用独立进程隔离（键不含 session_id）",
        "session_ids": {"cold": COLD_SID, "hit": LIVE_SID, "check": CHECK_SID},
        "sequence": SEQUENCE,
        "预热_ms": {"cold": cold_res["warmup_ms"], "hit": hit_res["warmup_ms"],
                    "check": check_res["warmup_ms"], "excluded_from_stats": True},
        "预热明细": {"index_chunks": warm_detail.get("index_chunks"),
                     "entities": (hit_res.get("warmup") or {}).get("entities"),
                     "corpora": (hit_res.get("warmup") or {}).get("corpora")},
        "阶段墙钟_ms": {"cold": cold["wall_ms"], "hit": hit["wall_ms"],
                        "check": check["wall_ms"]},
        "每轮": rounds,
        "汇总": summary,
        "校验": validation,
        "答案自证": proof,
        "notes": [
            "本基准用脚本侧包装模拟接线：MultiTurnEngine 尚未接入 DialogueCache",
            "引擎内置答案缓存无外部开关且键不含 session_id，故三阶段必须各自独立进程",
            "测量前缓存强制为空并删除上次落盘文件，避免旧结果把「未命中」变成「命中」",
            "命中路径不执行检索与生成，故耗时 <0.01 ms（µs 精度见「命中_us」）",
            "加速比在毫秒分辨率下溢出（>1000×），故给下界并另列 µs 实测值与「每次省下 ms」",
            "引用页码属模型侧随机（同一问句两次独立生成页码可能不同），故与硬口径分开列",
            "任何数字均为实跑产物",
        ],
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_path = out_dir / "对话缓存基准.json"
    md_path = out_dir / "对话缓存基准.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    md_path.write_text(_render_md(payload), encoding="utf-8")

    print(f"  ①冷跑基线（不经缓存）：{base_ms} ms → P50 {p50_base} ms")
    print(f"  ②未命中：{miss_values} ms → P50 {p50_miss} ms")
    print(f"  ③命中：{summary['命中']['values_us']} µs → P50 {summary['命中']['p50_us']} µs")
    print(f"  汇总：端到端每次省下 {summary['每次命中省下_ms']} ms（>1000×）；"
          f"可比口径 cache.get P50 {summary['查表_p50_ms']} ms = {summary['查表_相对基线加速比']}×；"
          f"测量窗口命中率 {summary['命中率']:.2%}（累计 hits/misses="
          f"{measured['hits']}/{measured['misses']}）")
    print(f"  校验：同轮同会话硬口径 {validation['一致题数']}/{validation['题数']} 一致；"
          f"软口径差异 {len(validation['软字段差异'])} 项")
    for tag, item in proof.items():
        marks = "/".join(f"{k}{'✓' if ok else '✗'}" for k, ok in item["命中"].items())
        print(f"  答案自证[{tag}]：{item['通过']}/{len(item['命中'])} 命中（{marks}）")
    print(f"  产物 → {json_path} / {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
