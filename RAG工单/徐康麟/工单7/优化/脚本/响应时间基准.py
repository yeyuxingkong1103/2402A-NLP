# -*- coding: utf-8 -*-
"""工单7 性能优化 —— 响应时间基准实测脚本。

工单编号：人工智能NLP-RAG-功能测试及评估

目的（任务书硬指标）
--------------------
任务书要求 **端到端响应时间 ≤ 3 秒**。本脚本用 **真实引擎、真实索引、真实 Ollama** 实测
17 题在三种场景下的 ``latency_ms``（``retrieve`` / ``generate`` / ``first_token`` / ``total``）：

1. ``cold_start``          —— 新进程内**不预热**就发第 1 问（只测 1 题；其余题用预热后数据）；
2. ``warm_no_cache``       —— ``QAEngine.warmup(with_generate=True)`` 后立刻逐题跑，
                              ``use_cache=False``（每题都按「用户第一次问」计，不被缓存污染）；
3. ``answer_cache_hit``    —— 同一问题第 2 次调用（``use_cache=True``，命中答案级内存缓存）。

另外实测两组用于「优化前后对比」的补充场景（同样真实）：

4. ``retrieval_cache_hit`` —— **新建引擎实例**（答案级内存缓存为空）后再问同一批问题，
                              并先清空生成级缓存 ⇒ 只能命中**检索结果缓存**，
                              用于量化「检索缓存」这一优化开关的收益；
5. ``probe_promptk_3`` / ``probe_promptk_2`` ——
                              题号子集先用 ``RAG_RETRIEVAL__PROMPT_TOP_K=3`` 跑一遍，
                              再用 ``RAG_RETRIEVAL__PROMPT_TOP_K=2`` 跑同一子集，
                              量化「减少注入提示词的证据块数」对生成耗时的影响。

另有 ``--mode probe-cache``：单独在一个**干净进程**里做「检索结果缓存 未命中 vs 命中」的
纯检索 A/B（不调用 LLM），用于给出检索层缓存的精确收益。

``--mode probe-cold``：单独在一个**干净进程**里逐个量测冷启动的各个构件
（``load_bundle`` / 分词器 / 实体标注 / 嵌入 / 后端探测 / 生成预热 / 全文倒排索引），
用于解释「不预热为什么必然违约」以及「预热 21 秒花在哪里」。

纪律
----
* **不改 ``研发/`` 下任何代码**；本脚本只读引擎、只写 ``优化/评估结果/``；
* ``use_cache=False`` 会同时关掉检索缓存读取与生成缓存读取（引擎既有语义），
  故场景 2 量到的是**真实检索 + 真实生成**，不含缓存水分；
* Ollama（127.0.0.1:11434）是**CPU 单实例**，与其它任务共享；本脚本**串行**执行、
  ``--batch`` 题一批、批间 ``--batch-sleep`` 秒，并在产物里如实标注竞争波动；
* 预计耗时：``--limit 17`` 约 4~7 分钟（其中预热约 20~40 秒）。

副作用与自清理
--------------
``--mode probe-misc`` 会构造 ``MultiTurnEngine``，其 ``SessionStore`` 默认把会话状态落盘到
``研发/data/runtime/sessions/<session_id>.json``。为了**不污染 ``研发/``**，本脚本在探针结束后
会**只删除自己创建的、前缀匹配 ``--session-prefix`` 的会话文件**（其它任务的会话文件一律不动）。

用法（工作目录 = ``E:\\gao6gongdan\\工单7``）::

    pwsh -NoProfile -File run_py.ps1 优化/脚本/响应时间基准.py --wait-index
    pwsh -NoProfile -File run_py.ps1 优化/脚本/响应时间基准.py --limit 5 --out 优化/评估结果/响应时间基准_烟测.json
    pwsh -NoProfile -File run_py.ps1 优化/脚本/响应时间基准.py --mode probe-cache

输出
----
* ``优化/评估结果/响应时间基准.json``（可由 ``--out`` 改名）
* ``优化/评估结果/响应时间基准.md``（同名不同扩展，含分场景 均值/中位/P50/P90/最大 与达标率）
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 0) 最先执行：禁止写 .pyc（研发/ 是只读区，红线），并记录脚本起点
# ---------------------------------------------------------------------------
import sys

sys.dont_write_bytecode = True

import time  # noqa: E402

_T_SCRIPT_START = time.time()
_T_PERF_START = time.perf_counter()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

WORK_ORDER = "人工智能NLP-RAG-功能测试及评估"

#: 仓库根：本文件位于 <root>/优化/脚本/响应时间基准.py
REPO_ROOT = Path(__file__).resolve().parents[2]

#: 结果目录（本工单专属，绝不写 研发/ 与 测试/）
RESULT_DIR = REPO_ROOT / "优化" / "评估结果"
DEFAULT_OUT = RESULT_DIR / "响应时间基准.json"

QUESTION_SET = REPO_ROOT / "研发" / "评估" / "测试问题集.jsonl"
INDEX_MANIFEST = REPO_ROOT / "研发" / "data" / "index" / "index_manifest.json"

#: 公司评测框架的同一份 SLA 阈值（研发/评估/配置.py: SLA_MS）
SLA_MS = 3000.0

#: 索引就绪判据（与 研发/评估/配置.py 的 INDEX_MIN_CHUNKS 一致）
INDEX_MIN_CHUNKS = 8000

#: 需要在报告里留痕的上游模块（与 研发/评估/运行评估.py 的 module_revision() 同名单）
REVISION_FILES = ("retriever.py", "answer_composer.py", "generator.py", "qa_engine.py",
                  "answerability.py", "query_understanding.py", "reranker.py", "fusion.py",
                  "citation.py", "bm25_index.py", "cache.py", "dialogue_cache.py")

#: 已识别的 RAG_* 环境变量（本脚本会显式设置下面这些；不被 config 识别也无害）
ENV_DEFAULTS = {
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPATH": str(REPO_ROOT / "研发"),
    "RAG_DATA__RAW_DIR": str(REPO_ROOT / "研发" / "data" / "raw"),
    "RAG_DATA__PROCESSED_DIR": str(REPO_ROOT / "研发" / "data" / "processed"),
    "RAG_DATA__INDEX_DIR": str(REPO_ROOT / "研发" / "data" / "index"),
    "RAG_LOG__DIR": str(REPO_ROOT / "部署" / "日志"),
}


def apply_env() -> None:
    """把 :data:`ENV_DEFAULTS` 灌进 ``os.environ``（已存在的不覆盖，便于调用方临时覆盖）。"""
    for key, value in ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)


def _p(message: str = "") -> None:
    """控制台输出（同时被外层 Tee 落成 优化/评估结果/运行日志.txt）。"""
    print(message, flush=True)


def _stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# 1) 统计工具（分位数定义与 研发/评估/运行评估.py::percentile 逐位一致：线性插值）
# ---------------------------------------------------------------------------
def percentile(values: list[float], pct: float) -> float | None:
    """线性插值分位数（``pct`` ∈ [0,100]）；空列表返回 ``None``。"""
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    pos = (len(ordered) - 1) * (float(pct) / 100.0)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return round(ordered[low] + (ordered[high] - ordered[low]) * frac, 2)


def latency_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """对一批记录给出 ``total/retrieve/generate/first_token`` 的完整分布与 SLA 达标率。"""
    ok_records = [r for r in records if not r.get("error")]
    out: dict[str, Any] = {"样本数": len(records), "有效样本数": len(ok_records),
                           "异常样本数": len(records) - len(ok_records)}
    names = {"total": "总耗时", "retrieve": "检索", "generate": "生成", "first_token": "首字"}
    for key, label in names.items():
        values = [float((r.get("latency_ms") or {}).get(key) or 0.0) for r in ok_records]
        if not values:
            out[label] = {"样本数": 0}
            continue
        out[label] = {
            "样本数": len(values),
            "均值": round(statistics.fmean(values), 2),
            "中位": round(statistics.median(values), 2),
            "P50": percentile(values, 50),
            "P90": percentile(values, 90),
            "P95": percentile(values, 95),
            "最大": round(max(values), 2),
            "最小": round(min(values), 2),
        }
    totals = [float((r.get("latency_ms") or {}).get("total") or 0.0) for r in ok_records]
    firsts = [float((r.get("latency_ms") or {}).get("first_token") or 0.0) for r in ok_records]
    if totals:
        ok_n = sum(1 for v in totals if v <= SLA_MS)
        out["SLA"] = {
            "阈值ms": SLA_MS,
            "达标题数": ok_n,
            "样本数": len(totals),
            "达标率": round(ok_n / len(totals), 4),
            "超时题号": [r["id"] for r in ok_records
                         if float((r.get("latency_ms") or {}).get("total") or 0.0) > SLA_MS],
            "达标率_首字": (round(sum(1 for v in firsts if v <= SLA_MS) / len(firsts), 4)
                          if firsts else None),
        }
    return out


# ---------------------------------------------------------------------------
# 2) 环境事实
# ---------------------------------------------------------------------------
def wait_for_index(timeout_s: float, interval_s: float = 20.0) -> dict[str, Any]:
    """``--wait-index``：轮询等索引就绪（默认最多 30 分钟）。

    就绪判据：``研发/data/index/index_manifest.json`` 存在、可解析、
    ``chunk_total >= INDEX_MIN_CHUNKS``（与评测框架同口径）。
    """
    deadline = time.time() + max(0.0, timeout_s)
    attempt = 0
    while True:
        attempt += 1
        info: dict[str, Any] = {"attempts": attempt, "manifest": str(INDEX_MANIFEST)}
        try:
            if INDEX_MANIFEST.is_file():
                data = json.loads(INDEX_MANIFEST.read_text(encoding="utf-8"))
                total = int(data.get("chunk_total") or 0)
                info.update({"chunk_total": total, "run_id": data.get("run_id"),
                             "embed_model": data.get("embed_model")})
                if total >= INDEX_MIN_CHUNKS:
                    _p(f"[索引] 就绪：chunk_total={total} run_id={data.get('run_id')} "
                       f"embed={data.get('embed_model')}")
                    info["ok"] = True
                    return info
                _p(f"[索引] 第 {attempt} 次探测：chunk_total={total} < {INDEX_MIN_CHUNKS}，继续等待…")
            else:
                _p(f"[索引] 第 {attempt} 次探测：清单不存在 {INDEX_MANIFEST}，继续等待…")
        except Exception as exc:  # noqa: BLE001 —— 清单半写状态属正常，继续等
            _p(f"[索引] 第 {attempt} 次探测：清单读取失败 {type(exc).__name__}: {exc}，继续等待…")
        if time.time() >= deadline:
            info["ok"] = False
            info["reason"] = f"等待超时（{timeout_s:.0f}s）"
            _p(f"[索引] !! {info['reason']}")
            return info
        time.sleep(interval_s)


def _sha16(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def module_revision() -> dict[str, str]:
    """被评估/被优化模块的修订指纹（SHA256 前 16 位）——保证结果可归因到单一版本。"""
    core = REPO_ROOT / "研发" / "app" / "core"
    return {name: _sha16(core / name) for name in REVISION_FILES}


def ollama_running_models(base_url: str = "http://127.0.0.1:11434") -> list[dict[str, Any]]:
    """``/api/ps``：当前驻留显存/内存的模型（用于证明「无预热即冷加载」这一事实）。"""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{base_url}/api/ps", timeout=5) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        return [{"name": m.get("name"), "size": m.get("size"),
                 "size_vram": m.get("size_vram"), "expires_at": m.get("expires_at")}
                for m in (data.get("models") or [])]
    except Exception as exc:  # noqa: BLE001 —— 探测失败不阻断基准
        return [{"error": f"{type(exc).__name__}: {exc}"}]


def load_questions(path: Path, limit: int | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if limit:
        items = items[: int(limit)]
    return items


# ---------------------------------------------------------------------------
# 3) 单次问答的量测封装
# ---------------------------------------------------------------------------
def run_one(engine: Any, item: dict[str, Any], *, scenario: str, top_k: int,
            session_prefix: str, pass_tag: str, use_cache: bool,
            capture_text: bool = False) -> dict[str, Any]:
    """跑一题并记录完整分段耗时（异常也落记录，不中断整轮）。"""
    qid = str(item.get("id"))
    question = str(item.get("question"))
    record: dict[str, Any] = {
        "scenario": scenario, "id": qid, "lang": item.get("lang"),
        "question": question, "top_k": top_k, "use_cache": use_cache,
        "asked_at": _stamp(), "latency_ms": {}, "error": None,
    }
    started = time.perf_counter()
    try:
        # 每题一个独立会话：保证 resolved_question == question（不被指代消解影响），
        # 且同一题在各 pass 之间解析结果一致 ⇒ 答案缓存键可稳定命中。
        result = engine.answer(question, lang=str(item.get("lang") or "auto"), top_k=top_k,
                               session_id=f"{session_prefix}-{qid}-{pass_tag}",
                               return_evidence=False, use_cache=use_cache)
    except Exception as exc:  # noqa: BLE001 —— 单题异常不得中断整轮基准
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["error_traceback"] = traceback.format_exc()
        record["wall_ms"] = round((time.perf_counter() - started) * 1000, 2)
        _p(f"    !! {qid} 异常：{record['error']}")
        return record
    record["wall_ms"] = round((time.perf_counter() - started) * 1000, 2)
    record["latency_ms"] = {k: float(v) for k, v in dict(result.latency_ms).items()}
    record.update({
        "refused": bool(result.refused),
        "answer_chars": len(str(result.answer or "")),
        "generation_method": result.generation_method,
        "backend": result.backend,
        "model": result.model,
        "cache": dict(result.cache or {}),
        "cited_pages": list(result.cited_pages or []),
        "refusal_reason": result.refusal_reason,
    })
    if capture_text:
        record["_answer_text"] = str(result.answer or "")
    return record


def run_pass(engine: Any, items: list[dict[str, Any]], *, scenario: str, top_k: int,
             session_prefix: str, pass_tag: str, use_cache: bool, batch: int,
             batch_sleep: float, label: str) -> list[dict[str, Any]]:
    """串行、分批地跑完一批题（批间 sleep，避免与其它任务抢占 CPU 造成尖峰）。"""
    records: list[dict[str, Any]] = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        rec = run_one(engine, item, scenario=scenario, top_k=top_k, session_prefix=session_prefix,
                      pass_tag=pass_tag, use_cache=use_cache)
        records.append(rec)
        lat = rec.get("latency_ms") or {}
        flag = "!!" if rec.get("error") else ("超时" if float(lat.get("total") or 0) > SLA_MS else "ok")
        _p(f"  [{label}] {index}/{total} {rec['id']:<4} "
           f"retrieve={lat.get('retrieve', 0):>8.2f} generate={lat.get('generate', 0):>8.2f} "
           f"first={lat.get('first_token', 0):>8.2f} total={lat.get('total', 0):>8.2f} ms  {flag}")
        if batch > 0 and index % batch == 0 and index < total:
            _p(f"  [{label}] —— 批间休眠 {batch_sleep:.0f}s（与其它任务共享 CPU，避免长时间独占）——")
            time.sleep(batch_sleep)
    return records


def cache_layer_stats() -> dict[str, Any]:
    """读进程级多级缓存的公开统计（检索层/生成层/归一化层）。"""
    try:
        from app.core.cache import get_cache

        return get_cache().stats()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def invalidate_generation_cache() -> dict[str, Any]:
    """清空**生成级**缓存：使「新建引擎实例 + use_cache=True」只可能命中检索缓存。"""
    from app.core.cache import get_cache

    removed = get_cache().generation.invalidate()
    return {"removed": removed}


# ---------------------------------------------------------------------------
# 4) 模式一：probe-cache —— 检索结果缓存 未命中 vs 命中（纯检索 A/B，不调 LLM）
# ---------------------------------------------------------------------------
def mode_probe_cache(args: argparse.Namespace) -> int:
    apply_env()
    if args.wait_index:
        wait_for_index(args.wait_timeout)
    from app.core.config import get_config
    from app.core.qa_engine import QAEngine

    cfg = get_config()
    engine = QAEngine(cfg=cfg)
    # 本探针只量检索层，故只做检索链最小预热（**不**建全文倒排索引、**不**预热生成模型），
    # 省掉约 21 s 的无谓开销；这与「检索缓存」的结论无关。
    _p("[probe-cache] 最小预热：索引 + 分词器 + 嵌入（不预热生成模型与全文倒排索引）…")
    t_warm = time.perf_counter()
    from app.core.embedder import warmup as embed_warmup
    from app.core.retriever import load_bundle
    from app.core.text_utils import warmup_tokenizer

    warm = {"index_chunks": load_bundle(cfg).size, "tokenizer": warmup_tokenizer(),
            "embedding": embed_warmup(cfg),
            "elapsed_ms": None}
    warm["elapsed_ms"] = round((time.perf_counter() - t_warm) * 1000, 2)
    _p(f"[probe-cache] 预热完成：{json.dumps(warm, ensure_ascii=False, default=str)[:300]}")

    items = load_questions(QUESTION_SET, args.limit)
    rows: list[dict[str, Any]] = []
    _p("")
    _p("[probe-cache] 同一问句：第 1 次（检索缓存未命中） vs 第 2 次（检索缓存命中）")
    for index, item in enumerate(items, start=1):
        qid = str(item.get("id"))
        q = str(item.get("question"))
        first = engine.retrieve_only(q, top_k=args.top_k)
        second = engine.retrieve_only(q, top_k=args.top_k)
        miss_ms = float(first["latency_ms"]["total"])
        hit_ms = float(second["latency_ms"]["total"])
        rows.append({"id": qid, "question": q, "miss_ms": round(miss_ms, 2),
                     "hit_ms": round(hit_ms, 2),
                     "saved_ms": round(miss_ms - hit_ms, 2),
                     "speedup": round(miss_ms / hit_ms, 1) if hit_ms > 0 else None,
                     "hit_count": first["hit_count"]})
        _p(f"  {index}/{len(items)} {qid:<4} miss={miss_ms:>8.2f} ms  hit={hit_ms:>8.2f} ms  "
           f"省 {miss_ms - hit_ms:>8.2f} ms")
        if args.batch > 0 and index % args.batch == 0 and index < len(items):
            time.sleep(args.batch_sleep)

    miss_values = [r["miss_ms"] for r in rows]
    hit_values = [r["hit_ms"] for r in rows]
    payload = {
        "工单": WORK_ORDER, "模式": "probe-cache（检索结果缓存 A/B）",
        "生成时间": _stamp(), "top_k": args.top_k, "样本数": len(rows),
        "缓存层统计": cache_layer_stats(),
        "统计": {
            "未命中": latency_stats([{"id": r["id"], "latency_ms": {"total": r["miss_ms"]}}
                                    for r in rows]),
            "命中": latency_stats([{"id": r["id"], "latency_ms": {"total": r["hit_ms"]}}
                                  for r in rows]),
            "未命中_均值": round(statistics.fmean(miss_values), 2) if miss_values else None,
            "命中_均值": round(statistics.fmean(hit_values), 2) if hit_values else None,
            "未命中_P90": percentile(miss_values, 90),
            "命中_P90": percentile(hit_values, 90),
            "平均节省ms": (round(statistics.fmean([r["saved_ms"] for r in rows]), 2)
                        if rows else None),
            "平均加速比": (round(statistics.fmean([r["speedup"] for r in rows]), 1)
                        if rows and all(r["speedup"] for r in rows) else None),
        },
        "逐题": rows,
        "环境": {"解释器": sys.executable, "cwd": str(Path.cwd()),
                 "修订": module_revision()},
    }
    out = Path(args.out) if args.out else (RESULT_DIR / "响应时间基准_检索缓存探针.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _p("")
    _p(f"[probe-cache] 已写出 {out}")
    st = payload["统计"]
    _p(f"[probe-cache] 检索耗时 未命中均值 {st['未命中_均值']} ms → 命中均值 {st['命中_均值']} ms，"
       f"平均省 {st['平均节省ms']} ms/题")
    return 0


# ---------------------------------------------------------------------------
# 5) 模式三：probe-cold —— 冷启动各构件分项实测
# ---------------------------------------------------------------------------
def mode_probe_cold(args: argparse.Namespace) -> int:
    """在**干净进程**里逐个量测冷启动构件；用于解释「不预热必然违约」的成因。"""
    apply_env()
    if args.wait_index:
        wait_for_index(args.wait_timeout)
    out: dict[str, Any] = {
        "工单": WORK_ORDER, "模式": "probe-cold（冷启动构件分项实测）", "生成时间": _stamp(),
        "说明": ("本探针在独立干净进程中按「冷路径的实际调用顺序」逐个量测首次调用开销；"
                 "与 --mode full 的 cold_start 是两次独立测量，CPU 竞争状态略有差异。"),
        "进程内自脚本首行到各阶段": [], "构件": {},
    }
    _p("[probe-cold] 逐项量测冷启动构件（每项都是本进程内的**首次**调用）…")

    from app.core.config import get_config
    from app.core.qa_engine import QAEngine

    cfg = get_config()
    out["构件"]["import_app"] = {"ms": round((time.perf_counter() - _T_PERF_START) * 1000, 2),
                                "note": "脚本首行 → import app.core.config/qa_engine 完成"}

    def _mark(name: str, fn: Any, note: str = "") -> None:
        started = time.perf_counter()
        try:
            value = fn()
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            out["构件"][name] = {"ms": elapsed, "ok": True, "note": note,
                                 "detail": _short(value)}
            _p(f"  {name:<22} {elapsed:>10.2f} ms  {note}")
        except Exception as exc:  # noqa: BLE001 —— 单项失败不阻断其余项
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            out["构件"][name] = {"ms": elapsed, "ok": False, "note": note,
                                 "error": f"{type(exc).__name__}: {exc}"}
            _p(f"  {name:<22} {elapsed:>10.2f} ms  !! {type(exc).__name__}: {exc}")

    from app.core.retriever import load_bundle, warmup_entity_tagger
    from app.core.text_utils import warmup_tokenizer
    from app.core.embedder import warmup as embed_warmup
    from app.core.llm_client import resolve_backend, warmup_generate
    from app.core.retrieval_strategy import warmup_strategy

    _mark("load_bundle", lambda: load_bundle(cfg), "索引加载（chunks/vectors/bm25）")
    _mark("warmup_tokenizer", lambda: warmup_tokenizer(), "jieba 分词器初始化")
    _mark("warmup_entity_tagger", lambda: warmup_entity_tagger(), "实体标注器（词性表）")
    _mark("embed_warmup", lambda: embed_warmup(cfg), "嵌入模型预热（Ollama bge-m3）")
    _mark("resolve_backend", lambda: resolve_backend(cfg), "生成后端探测（≤probe_timeout_s）")
    _mark("warmup_generate", lambda: warmup_generate(cfg), "生成模型预热（keep_alive）")
    _mark("warmup_strategy", lambda: warmup_strategy(cfg=cfg), "多字段倒排索引（全文检索通道）")

    # 以上构件全部就绪后，再建一个引擎提问 —— 这就是「预热后」的对照值
    engine = QAEngine(cfg=cfg)
    started = time.perf_counter()
    res = engine.answer(str(load_questions(QUESTION_SET, 1)[0]["question"]), top_k=args.top_k,
                        session_id=f"{args.session_prefix}-probe-cold-warm", use_cache=False)
    warm_ms = round((time.perf_counter() - started) * 1000, 2)
    out["构件后首问"] = {"ms": warm_ms, "latency_ms": dict(res.latency_ms)}
    _p(f"  构件就绪后首问（对照）  {warm_ms:>10.2f} ms  latency={res.latency_ms}")

    out["构件合计ms"] = round(sum(v.get("ms", 0.0) for v in out["构件"].values()), 2)
    out["进程总墙钟s"] = round(time.time() - _T_SCRIPT_START, 2)
    out["环境"] = {"解释器": sys.executable, "python": sys.version.split()[0],
                  "cwd": str(Path.cwd()), "cpu_count": os.cpu_count(),
                  "修订": module_revision(), "ollama运行前": ollama_running_models()}
    target = Path(args.out) if args.out else (RESULT_DIR / "响应时间基准_冷启动分项.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _p("")
    _p(f"[probe-cold] 构件合计 {out['构件合计ms']} ms；已写出 {target}")
    return 0


def _short(value: Any, limit: int = 220) -> Any:
    """把构件返回值压成可落盘的小摘要。"""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(value)
    return text[:limit]


# ---------------------------------------------------------------------------
# 5b) 模式四：probe-promptk —— prompt_top_k 的性能 + 质量双维度对照
# ---------------------------------------------------------------------------
def _load_judge() -> Any:
    """载入官方判分模块（``研发/评估/判分.py``）——只读复用，口径与评测官完全一致。"""
    import importlib.util

    path = REPO_ROOT / "研发" / "评估" / "判分.py"
    spec = importlib.util.spec_from_file_location("_wo7_judge", path)
    module = importlib.util.module_from_spec(spec)          # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def mode_probe_promptk(args: argparse.Namespace) -> int:
    """同一子集分别在 ``prompt_top_k=3`` 与 ``=2`` 下作答：既量耗时，也用官方判分看质量。

    实验设计（**配对 + 交错 + 对称清缓存**）：
        对同一道题**背靠背**跑两臂（3 先、2 后），且每臂开跑前都清空进程级检索缓存与生成缓存，
        使两臂面对的缓存状态完全相同 —— 消除「后跑的一臂白捡检索缓存」这一混淆因素，
        也把 CPU 竞争漂移限制在「同一题的相邻两次调用」内。
    """
    apply_env()
    if args.wait_index:
        wait_for_index(args.wait_timeout)
    from app.core.cache import get_cache
    from app.core.config import get_config, load_config
    from app.core.qa_engine import QAEngine

    cfg = get_config()
    items = load_questions(QUESTION_SET, args.limit)
    judge = _load_judge()
    _p("=" * 78)
    _p(" prompt_top_k 对照实验（性能 + 质量双维度；配对交错设计）")
    _p("=" * 78)
    _p(f"子集：{'、'.join(str(i.get('id')) for i in items)}（共 {len(items)} 题）")
    _p(f"判分模块：{REPO_ROOT / '研发' / '评估' / '判分.py'}（关键词口径与评测官一致）")
    _p("设计：同题背靠背跑两臂；每臂前清空检索/生成缓存 ⇒ 两臂缓存状态对称")
    _p("")

    engine = QAEngine(cfg=cfg)
    _p("[预热] warmup(with_generate=True)…")
    warm = engine.warmup(with_generate=True)
    _p(f"[预热] {warm.get('elapsed_ms')} ms")

    cfg_pk2 = load_config(env={**os.environ, "RAG_RETRIEVAL__PROMPT_TOP_K": "2"})
    engine_pk2 = QAEngine(cfg=cfg_pk2)
    engine_pk2.warmup(with_generate=False)
    _p(f"[预热] prompt_top_k=2 实例：{cfg_pk2.retrieval.prompt_top_k}（复用已建索引）")
    _p("")

    arms: dict[str, list[dict[str, Any]]] = {"3": [], "2": []}
    pairs: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        row: dict[str, Any] = {"id": str(item.get("id"))}
        for tag, eng in (("3", engine), ("2", engine_pk2)):
            cache = get_cache()
            cache.retrieval.invalidate()        # 对称：两臂都从「检索缓存未命中」开始
            cache.generation.invalidate()       # 对称：两臂都从「生成缓存未命中」开始
            rec = run_one(eng, item, scenario=f"promptk_{tag}", top_k=args.top_k,
                          session_prefix=args.session_prefix, pass_tag=f"pk{tag}i",
                          use_cache=True, capture_text=True)
            answer_text = rec.get("_answer_text") or ""
            keywords = list(item.get("answer_keywords") or [])
            hit_n, total_n, cov, _hit, missing = judge.keyword_coverage(answer_text, keywords)
            rec["keyword_coverage"] = round(float(cov), 4)
            rec["keywords_hit"] = hit_n
            rec["keywords_total"] = total_n
            rec["keywords_missing"] = missing
            rec["answer_excerpt"] = answer_text[:500]
            rec.pop("_answer_text", None)
            arms[tag].append(rec)
            lat = rec.get("latency_ms") or {}
            row[f"total_{tag}"] = lat.get("total")
            row[f"generate_{tag}"] = lat.get("generate")
            row[f"chars_{tag}"] = rec.get("answer_chars")
            row[f"cov_{tag}"] = rec["keyword_coverage"]
        for key in ("total", "generate", "chars", "cov"):
            a, b = row.get(f"{key}_3"), row.get(f"{key}_2")
            row[f"delta_{key}"] = round(b - a, 2) if (a is not None and b is not None) else None
        pairs.append(row)
        lat3 = (arms["3"][-1].get("latency_ms") or {})
        lat2 = (arms["2"][-1].get("latency_ms") or {})
        _p(f"  {index}/{len(items)} {row['id']:<4} "
           f"total 3→2: {lat3.get('total')} → {lat2.get('total')} ms "
           f"({row['delta_total']:+.2f}) ｜ generate: {lat3.get('generate')} → "
           f"{lat2.get('generate')} ms ({row['delta_generate']:+.2f}) ｜ "
           f"覆盖率 {row['cov_3']} → {row['cov_2']}")
        if args.batch > 0 and index % args.batch == 0 and index < len(items):
            time.sleep(args.batch_sleep)

    def _arm_summary(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
        covs = [r["keyword_coverage"] for r in records if not r.get("error")]
        chars = [r.get("answer_chars", 0) for r in records if not r.get("error")]
        return {
            "标签": label, "统计": latency_stats(records), "逐题": records,
            "关键词覆盖率_均值": round(statistics.fmean(covs), 4) if covs else None,
            "答案字符数_均值": round(statistics.fmean(chars), 1) if chars else None,
            "严格判对题数(覆盖率>=0.6且未拒答)": sum(
                1 for r in records if r["keyword_coverage"] >= 0.6 and not r.get("refused")),
        }

    arm3 = _arm_summary(arms["3"], "prompt_top_k=3（引擎默认）")
    arm2 = _arm_summary(arms["2"], "prompt_top_k=2（实验臂）")
    g3 = (arm3["统计"].get("生成") or {}).get("均值")
    g2 = (arm2["统计"].get("生成") or {}).get("均值")
    t3 = (arm3["统计"].get("总耗时") or {}).get("均值")
    t2 = (arm2["统计"].get("总耗时") or {}).get("均值")
    deltas_g = [r["delta_generate"] for r in pairs if r.get("delta_generate") is not None]
    deltas_t = [r["delta_total"] for r in pairs if r.get("delta_total") is not None]
    delta = {
        "总耗时均值ms": {"prompt_top_k=3": t3, "prompt_top_k=2": t2},
        "总耗时P90ms": {"prompt_top_k=3": (arm3["统计"].get("总耗时") or {}).get("P90"),
                     "prompt_top_k=2": (arm2["统计"].get("总耗时") or {}).get("P90")},
        "生成均值ms": {"prompt_top_k=3": g3, "prompt_top_k=2": g2},
        "生成P90ms": {"prompt_top_k=3": (arm3["统计"].get("生成") or {}).get("P90"),
                    "prompt_top_k=2": (arm2["统计"].get("生成") or {}).get("P90")},
        "关键词覆盖率均值": {"prompt_top_k=3": arm3["关键词覆盖率_均值"],
                     "prompt_top_k=2": arm2["关键词覆盖率_均值"]},
        "答案字符数均值": {"prompt_top_k=3": arm3["答案字符数_均值"],
                    "prompt_top_k=2": arm2["答案字符数_均值"]},
        "逐题配对_生成耗时变化": {"题数": len(deltas_g),
                          "均值ms": round(statistics.fmean(deltas_g), 2) if deltas_g else None,
                          "中位ms": round(statistics.median(deltas_g), 2) if deltas_g else None,
                          "变快的题数": sum(1 for d in deltas_g if d < 0),
                          "变慢的题数": sum(1 for d in deltas_g if d > 0)},
        "逐题配对_总耗时变化": {"题数": len(deltas_t),
                         "均值ms": round(statistics.fmean(deltas_t), 2) if deltas_t else None,
                         "中位ms": round(statistics.median(deltas_t), 2) if deltas_t else None,
                         "变快的题数": sum(1 for d in deltas_t if d < 0),
                         "变慢的题数": sum(1 for d in deltas_t if d > 0)},
    }
    if g3 and g2:
        delta["生成耗时降幅%"] = round((g3 - g2) / g3 * 100, 1)
    if t3 and t2:
        delta["总耗时降幅%"] = round((t3 - t2) / t3 * 100, 1)
    if arm3["关键词覆盖率_均值"] is not None and arm2["关键词覆盖率_均值"] is not None:
        delta["关键词覆盖率变化(绝对)"] = round(
            arm2["关键词覆盖率_均值"] - arm3["关键词覆盖率_均值"], 4)

    payload = {
        "工单": WORK_ORDER, "模式": "probe-promptk", "生成时间": _stamp(),
        "子集题号": [str(i.get("id")) for i in items], "top_k": args.top_k,
        "口径": ("两臂除 RAG_RETRIEVAL__PROMPT_TOP_K 外配置完全相同；均 use_cache=True，"
                 "但**每臂开跑前都清空检索缓存与生成缓存**，故两臂的缓存状态对称、"
                 "所测即真实检索 + 真实生成；质量用 研发/评估/判分.py 的 keyword_coverage"
                 "（严格判对阈值 0.6）"),
        "对照": {"3": arm3, "2": arm2}, "逐题配对": pairs, "差值": delta,
        "环境": {"解释器": sys.executable, "python": sys.version.split()[0],
                 "cwd": str(Path.cwd()), "warmup_ms": warm.get("elapsed_ms"),
                 "修订_start": module_revision()},
    }
    payload["环境"]["修订_end"] = module_revision()
    out = Path(args.out) if args.out else (RESULT_DIR / "prompt_top_k对照.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _p("")
    _p("=" * 78)
    _p(" 结论")
    _p("=" * 78)
    _p(json.dumps(delta, ensure_ascii=False, indent=2))
    _p(f"[落盘] {out}")
    return 0


# ---------------------------------------------------------------------------
# 5c) 模式五：probe-misc —— 批量嵌入扫描 + 对话级缓存 A/B
# ---------------------------------------------------------------------------
def mode_probe_misc(args: argparse.Namespace) -> int:
    """两个「等」里的开关：``RAG_EMBEDDING__BATCH_SIZE``（离线/索引侧）与对话级缓存（多轮侧）。"""
    apply_env()
    if args.wait_index:
        wait_for_index(args.wait_timeout)
    from app.core.config import get_config

    cfg = get_config()
    payload: dict[str, Any] = {"工单": WORK_ORDER, "模式": "probe-misc", "生成时间": _stamp(),
                               "环境": {"解释器": sys.executable, "python": sys.version.split()[0],
                                      "cwd": str(Path.cwd()), "修订": module_revision()}}
    _p("=" * 78)
    _p(" 杂项开关实测：① 批量嵌入批大小扫描 ② 对话级缓存 A/B")
    _p("=" * 78)

    # ---- ① 批量嵌入：批大小对嵌入总耗时的影响（离线/索引侧） ----
    _p("")
    _p("[①] 批量嵌入扫描（RAG_EMBEDDING__BATCH_SIZE）")
    _p("    说明：在线单问只嵌 1 条文本，批大小**不影响**在线首字延迟；")
    _p("          该开关只在【建索引/离线批量】时生效，故此处用真实语料块离线量测。")
    from app.core.embedder import embed_texts, warmup as embed_warmup
    from app.core.retriever import load_bundle

    bundle = load_bundle(cfg)
    embed_warmup(cfg)
    texts: list[str] = []
    for cid, row in bundle.chunks.items():
        content = str((row or {}).get("content") or "")
        if content:
            texts.append(content[:600])
        if len(texts) >= 32:
            break
    _p(f"    样本：{len(texts)} 条真实语料块（截断至 600 字符），维度 {cfg.llm.ollama_embed_model}")
    sweep: list[dict[str, Any]] = []
    for batch_size in (1, 2, 4, 8, 16, 32):
        started = time.perf_counter()
        matrix = embed_texts(texts, cfg=cfg, batch_size=batch_size)
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        shape = getattr(matrix, "shape", None)
        sweep.append({"batch_size": batch_size, "总耗时ms": elapsed,
                      "每条ms": round(elapsed / max(1, len(texts)), 3),
                      "往返次数": -(-len(texts) // batch_size), "输出形状": str(shape)})
        _p(f"    batch_size={batch_size:<3} 总耗时={elapsed:>9.2f} ms  "
           f"每条={elapsed / max(1, len(texts)):>7.3f} ms  往返={-(-len(texts) // batch_size)}")
    base_ms = sweep[0]["总耗时ms"]
    for row in sweep:
        row["相对batch1加速比"] = round(base_ms / row["总耗时ms"], 2) if row["总耗时ms"] else None
    payload["批量嵌入扫描"] = {
        "样本数": len(texts), "扫描": sweep,
        "结论": ("批大小只影响离线批量嵌入的往返次数与总耗时，对在线单问延迟无影响"
                 "（在线路径每次只嵌 1 条，任何 batch_size 都退化为 1 次往返）"),
    }

    # ---- ② 对话级缓存：MultiTurnEngine 内置（研发/app/multiturn_engine.py） ----
    _p("")
    _p("[②] 对话级缓存 A/B（MultiTurnEngine 内置 DialogueCache）")
    _p("    协议：同一会话内同一问句问两次 —— 第 1 次（对话缓存未命中）vs 第 2 次（命中）")
    try:
        from app.multiturn_engine import MultiTurnEngine

        mt = MultiTurnEngine(cfg=cfg)
        mt.warmup(with_generate=True)
        rows: list[dict[str, Any]] = []
        for index, item in enumerate(load_questions(QUESTION_SET, args.limit), start=1):
            qid = str(item.get("id"))
            q = str(item.get("question"))
            sid = f"{args.session_prefix}-dlg-{qid}"
            mt.reset(sid)
            first = mt.ask(q, session_id=sid, lang=str(item.get("lang") or "auto"), use_cache=True)
            second = mt.ask(q, session_id=sid, lang=str(item.get("lang") or "auto"), use_cache=True)
            miss_ms = float(first.latency_ms.get("total") or 0.0)
            hit_ms = float(second.latency_ms.get("total") or 0.0)
            layer = str(((second.base or {}).get("cache") or {}).get("layer") or "-")
            rows.append({"id": qid, "miss_ms": round(miss_ms, 2), "hit_ms": round(hit_ms, 2),
                         "second_cache": (second.base or {}).get("cache"),
                         "second_layer": layer,
                         "saved_ms": round(miss_ms - hit_ms, 2),
                         "rewritten_1": first.rewritten_query, "rewritten_2": second.rewritten_query})
            _p(f"    {index}/{len(load_questions(QUESTION_SET, args.limit))} {qid:<4} "
               f"miss={miss_ms:>8.2f} ms  hit={hit_ms:>7.2f} ms  layer={layer}  "
               f"省 {miss_ms - hit_ms:>8.2f} ms")
            time.sleep(args.batch_sleep if args.batch_sleep > 0 else 0)
        hits = [r for r in rows if r["second_layer"] == "dialogue"]
        payload["对话级缓存"] = {
            "逐题": rows, "命中layer=dialogue的题数": len(hits), "样本数": len(rows),
            "未命中均值ms": (round(statistics.fmean([r["miss_ms"] for r in rows]), 2) if rows else None),
            "命中均值ms": (round(statistics.fmean([r["hit_ms"] for r in rows]), 2) if rows else None),
            "平均节省ms": round(statistics.fmean([r["saved_ms"] for r in rows]), 2) if rows else None,
            "引擎缓存统计": mt.stats().get("dialogue_cache"),
        }
    except Exception as exc:  # noqa: BLE001 —— 探针失败不影响主基准
        payload["对话级缓存"] = {"error": f"{type(exc).__name__}: {exc}",
                             "traceback_tail": traceback.format_exc()[-1200:]}
        _p(f"    !! 对话缓存探针失败：{type(exc).__name__}: {exc}")

    out = Path(args.out) if args.out else (RESULT_DIR / "杂项开关实测.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload["会话文件自清理"] = cleanup_own_sessions(args.session_prefix)
    _p(f"[probe-misc] 会话文件自清理：{payload['会话文件自清理']}")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _p("")
    _p(f"[probe-misc] 已写出 {out}")
    return 0


def cleanup_own_sessions(prefix: str) -> dict[str, Any]:
    """删除**本脚本自己创建**的多轮会话落盘文件（只匹配自己的前缀，不碰别人的）。

    ``MultiTurnEngine`` 的 ``SessionStore`` 会把会话状态写到
    ``研发/data/runtime/sessions/<session_id>.json``；探针跑完必须清理，避免污染 ``研发/``。
    """
    directory = REPO_ROOT / "研发" / "data" / "runtime" / "sessions"
    removed: list[str] = []
    if not prefix or not directory.is_dir():
        return {"dir": str(directory), "removed": removed}
    for path in sorted(directory.glob(f"{prefix}*.json")):
        try:
            path.unlink()
            removed.append(path.name)
        except OSError:
            pass
    return {"dir": str(directory), "removed": removed}


# ---------------------------------------------------------------------------
# 6) 模式二：full —— 三种场景（+ 两组补充对照）
# ---------------------------------------------------------------------------
def mode_full(args: argparse.Namespace) -> int:
    apply_env()
    _p("=" * 78)
    _p(f" 工单：{WORK_ORDER}")
    _p(" 响应时间基准（三场景实测）")
    _p("=" * 78)
    _p(f"脚本启动时刻 : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(_T_SCRIPT_START))}")
    _p(f"解释器       : {sys.executable}")
    _p(f"Python       : {sys.version.split()[0]}")
    _p(f"工作目录     : {Path.cwd()}")
    _p(f"SLA 阈值     : {SLA_MS:.0f} ms（任务书「响应时间 ≤ 3 秒」）")
    _p(f"串行/分批    : 每批 {args.batch} 题，批间休眠 {args.batch_sleep:.0f} 秒")
    _p("共享声明     : Ollama(127.0.0.1:11434) 为 CPU 单实例，与其它任务共享，数字含竞争波动")
    _p("")

    if args.wait_index:
        wait_for_index(args.wait_timeout)
    elif not INDEX_MANIFEST.is_file():
        _p(f"[索引] !! 未找到 {INDEX_MANIFEST}；建议加 --wait-index 等待索引就绪")
    else:
        try:
            manifest = json.loads(INDEX_MANIFEST.read_text(encoding="utf-8"))
            _p(f"[索引] run_id={manifest.get('run_id')} chunk_total={manifest.get('chunk_total')} "
               f"embed={manifest.get('embed_model')} backend={manifest.get('vector_backend')}")
        except Exception as exc:  # noqa: BLE001
            _p(f"[索引] !! 清单读取失败：{type(exc).__name__}: {exc}")

    items = load_questions(QUESTION_SET, args.limit)
    _p(f"[问题集] {QUESTION_SET.name} 载入 {len(items)} 题："
       f"{', '.join(str(i.get('id')) for i in items)}")
    _p("")

    rev_start = module_revision()
    ollama_before = ollama_running_models()
    _p(f"[Ollama] 运行前驻留模型：{json.dumps(ollama_before, ensure_ascii=False)[:400]}")
    _p("")

    from app.core.config import get_config, load_config
    from app.core.qa_engine import QAEngine

    cfg = get_config()
    _p("[配置] " + json.dumps({
        "top_k": cfg.retrieval.top_k, "vector_k": cfg.retrieval.vector_k,
        "bm25_k": cfg.retrieval.bm25_k, "prompt_top_k": cfg.retrieval.prompt_top_k,
        "page_context_extra": cfg.retrieval.page_context_extra,
        "cache_size(答案内存缓存)": cfg.retrieval.cache_size,
        "embed_batch_size": cfg.embedding.batch_size,
        "gen_model": cfg.llm.ollama_gen_model, "embed_model": cfg.llm.ollama_embed_model,
        "backend": cfg.llm.backend,
    }, ensure_ascii=False))
    _p("")

    scenarios: dict[str, Any] = {}
    timeline: list[dict[str, Any]] = []

    # ---------------- 场景 1：冷启动（不预热就发第 1 问） ----------------
    _p("-" * 78)
    _p("[场景 1/5] 冷启动：新进程内**不预热**就发第 1 问（use_cache=False）")
    _p("-" * 78)
    t_engine = time.perf_counter()
    engine = QAEngine(cfg=cfg)                       # 构造本身很快（索引是惰性加载）
    engine_construct_ms = round((time.perf_counter() - t_engine) * 1000, 2)
    t_cold = time.perf_counter()
    cold_records = [run_one(engine, items[0], scenario="cold_start", top_k=args.top_k,
                            session_prefix=args.session_prefix, pass_tag="cold", use_cache=False)]
    cold_wall_ms = round((time.perf_counter() - t_cold) * 1000, 2)
    for rec in cold_records:
        lat = rec.get("latency_ms") or {}
        rec["engine_construct_ms"] = engine_construct_ms
        rec["wall_since_script_start_ms"] = round((time.perf_counter() - _T_PERF_START) * 1000, 2)
        # total 里没有被 retrieve/generate 覆盖的部分 = 惰性冷初始化（索引加载 / 分词器 /
        # 问句理解 / 可答性闸门 / 引用构造）。分项证据见 --mode probe-cold。
        rec["unaccounted_ms"] = round(float(lat.get("total") or 0.0)
                                      - float(lat.get("retrieve") or 0.0)
                                      - float(lat.get("generate") or 0.0), 2)
        rec["note"] = ("cold_start = 未预热；total 含索引加载 + 分词器初始化 + 嵌入模型冷加载 + "
                       "生成模型冷加载（若 Ollama 未驻留）；unaccounted_ms 即未被 "
                       "retrieve/generate 两段覆盖的惰性冷初始化")
    c = cold_records[0]
    _p(f"  冷启动首问：total={c['latency_ms'].get('total')} ms "
       f"(retrieve={c['latency_ms'].get('retrieve')} generate={c['latency_ms'].get('generate')} "
       f"first={c['latency_ms'].get('first_token')})；"
       f"引擎构造 {engine_construct_ms} ms；冷问墙钟 {cold_wall_ms} ms；"
       f"脚本首行至今 {c['wall_since_script_start_ms']} ms")
    scenarios["cold_start"] = {
        "说明": "新进程、未预热、第 1 问（只测 1 题）",
        "统计": latency_stats(cold_records), "逐题": cold_records,
    }
    timeline.append({"scenario": "cold_start", "at": _stamp(),
                     "script_uptime_ms": c["wall_since_script_start_ms"]})

    # ---------------- 预热 ----------------
    _p("")
    _p("-" * 78)
    _p("[预热] warmup(with_generate=True)：索引 + 分词器 + 实体标注 + 嵌入 + 全文索引 + 生成模型")
    _p("-" * 78)
    t_warm = time.perf_counter()
    warm_info = engine.warmup(with_generate=True)
    warmup_ms = round((time.perf_counter() - t_warm) * 1000, 2)
    _p(f"  预热完成：{warmup_ms} ms")
    _p(f"  {json.dumps(warm_info, ensure_ascii=False, default=str)[:600]}")
    _p(f"[Ollama] 预热后驻留模型：{json.dumps(ollama_running_models(), ensure_ascii=False)[:400]}")

    # ---------------- 场景 2：预热后逐题（无缓存） ----------------
    _p("")
    _p("-" * 78)
    _p(f"[场景 2/5] 预热后逐题：{len(items)} 题，use_cache=False（真实检索 + 真实生成）")
    _p("-" * 78)
    warm_records = run_pass(engine, items, scenario="warm_no_cache", top_k=args.top_k,
                            session_prefix=args.session_prefix, pass_tag="warm", use_cache=False,
                            batch=args.batch, batch_sleep=args.batch_sleep,
                            label="预热后|无缓存")
    scenarios["warm_no_cache"] = {
        "说明": "warmup(with_generate=True) 后逐题首问（use_cache=False，无缓存水分）",
        "统计": latency_stats(warm_records), "逐题": warm_records,
    }
    timeline.append({"scenario": "warm_no_cache", "at": _stamp()})

    # ---------------- 场景 3：答案缓存命中 ----------------
    _p("")
    _p("-" * 78)
    _p(f"[场景 3/5] 答案缓存命中：同一问题第 2 次调用，use_cache=True（{len(items)} 题）")
    _p("-" * 78)
    hit_records = run_pass(engine, items, scenario="answer_cache_hit", top_k=args.top_k,
                           session_prefix=args.session_prefix, pass_tag="hit", use_cache=True,
                           batch=0, batch_sleep=0.0, label="缓存命中")
    cache_layer_after_hit = cache_layer_stats()
    scenarios["answer_cache_hit"] = {
        "说明": "同一问题同进程第 2 次调用（use_cache=True）→ 命中答案级内存缓存",
        "统计": latency_stats(hit_records), "逐题": hit_records,
        "缓存层统计": cache_layer_after_hit,
    }
    timeline.append({"scenario": "answer_cache_hit", "at": _stamp()})
    hit_n = sum(1 for r in hit_records if (r.get("cache") or {}).get("hit"))
    _p(f"  命中答案缓存的题数：{hit_n}/{len(hit_records)}")
    if hit_n != len(hit_records):
        _p("  !! 存在未命中项，逐题 cache 字段已落 JSON（多为问句被指代消解改写导致键变化）")

    # ---------------- 场景 4：检索结果缓存命中（新引擎实例） ----------------
    _p("")
    _p("-" * 78)
    _p(f"[场景 4/5] 检索结果缓存命中：新建引擎实例（答案缓存为空）+ 清空生成缓存，"
       f"use_cache=True（{len(items)} 题）")
    _p("-" * 78)
    stats_before = cache_layer_stats()
    inv = invalidate_generation_cache()
    _p(f"  已清空生成级缓存：{inv}")
    engine2 = QAEngine(cfg=cfg)
    retr_records = run_pass(engine2, items, scenario="retrieval_cache_hit", top_k=args.top_k,
                            session_prefix=args.session_prefix, pass_tag="retr", use_cache=True,
                            batch=args.batch, batch_sleep=args.batch_sleep,
                            label="检索缓存命中")
    stats_after = cache_layer_stats()
    scenarios["retrieval_cache_hit"] = {
        "说明": ("新引擎实例 + 生成缓存已清空，故只可能命中**检索结果缓存**；"
                 "与场景 2 的 retrieve 分段对照即为检索缓存的收益"),
        "统计": latency_stats(retr_records), "逐题": retr_records,
        "检索层缓存统计_前后": {
            "清空前": (stats_before.get("layers", {}) or {}).get("retrieval"),
            "清空后": (stats_after.get("layers", {}) or {}).get("retrieval"),
        },
    }
    timeline.append({"scenario": "retrieval_cache_hit", "at": _stamp()})

    # ---------------- 场景 5：答案级 SQLite 缓存（跨引擎实例 ≈ 跨进程重启） ----------------
    sqlite_n = int(args.sqlite_probe)
    if sqlite_n > 0 and items:
        subset_s = items[: min(sqlite_n, len(items))]
        _p("")
        _p("-" * 78)
        _p(f"[场景 5/6] 答案级 SQLite 缓存（跨引擎实例 ≈ 服务重启后命中落盘缓存）："
           f"{len(subset_s)} 题")
        _p("-" * 78)
        try:
            from app.storage.sqlite_manager import SqliteManager

            db_path = RESULT_DIR / "响应时间基准_答案缓存.sqlite3"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            if db_path.exists():
                db_path.unlink()          # 从空库开始，保证「填充」与「命中」两段都是实测
            _p(f"  SQLite 缓存库：{db_path}（本次运行专用，不触碰 研发/ 的运行时库）")
            engine_s1 = QAEngine(cfg=cfg, storage=SqliteManager(db_path))
            engine_s1.warmup(with_generate=False)
            fill_records = run_pass(engine_s1, subset_s, scenario="sqlite_fill",
                                    top_k=args.top_k, session_prefix=args.session_prefix,
                                    pass_tag="sqfill", use_cache=True,
                                    batch=args.batch, batch_sleep=args.batch_sleep,
                                    label="SQLite填充")
            # 全新引擎实例：答案级内存缓存为空 ⇒ 只能命中落盘 SQLite 层
            engine_s2 = QAEngine(cfg=cfg, storage=SqliteManager(db_path))
            engine_s2.warmup(with_generate=False)
            s_hit_records = run_pass(engine_s2, subset_s, scenario="sqlite_cache_hit",
                                     top_k=args.top_k, session_prefix=args.session_prefix,
                                     pass_tag="sqhit", use_cache=True,
                                     batch=0, batch_sleep=0.0, label="SQLite命中")
            s_hit_n = sum(1 for r in s_hit_records if (r.get("cache") or {}).get("hit"))
            _p(f"  命中 SQLite 答案缓存的题数：{s_hit_n}/{len(s_hit_records)}"
               f"（layer 应为 sqlite）")
            scenarios["sqlite_fill"] = {
                "说明": ("新引擎实例 + 空 SQLite 库 ⇒ 真实重算并落盘（第 1 次的代价）"),
                "统计": latency_stats(fill_records), "逐题": fill_records}
            scenarios["sqlite_cache_hit"] = {
                "说明": ("**另一个**引擎实例（内存缓存为空）读同一 SQLite 库 ⇒ "
                         "等价于「服务重启后重复提问」"),
                "统计": latency_stats(s_hit_records), "逐题": s_hit_records}
            timeline.append({"scenario": "sqlite_probe", "at": _stamp()})
        except Exception as exc:  # noqa: BLE001 —— 探针失败不阻断主基准
            _p(f"  !! SQLite 探针失败（不影响前三场景）：{type(exc).__name__}: {exc}")
            _p(traceback.format_exc()[-1200:])

    # ---------------- 场景 6：prompt_top_k 对照（可选） ----------------
    probe_n = int(args.promptk_probe)
    promptk_payload: dict[str, Any] | None = None
    if probe_n > 0 and items:
        subset = items[: min(probe_n, len(items))]
        _p("")
        _p("-" * 78)
        _p(f"[场景 6/6] prompt_top_k 对照：同一子集（{len(subset)} 题）先 3 后 2，"
           f"量化「注入提示词的证据块数」对生成耗时的影响")
        _p("-" * 78)
        base_records = run_pass(engine, subset, scenario="probe_promptk_3", top_k=args.top_k,
                                session_prefix=args.session_prefix, pass_tag="pk3", use_cache=False,
                                batch=args.batch, batch_sleep=args.batch_sleep,
                                label="prompt_top_k=3")
        cfg_pk2 = load_config(env={**os.environ, "RAG_RETRIEVAL__PROMPT_TOP_K": "2"})
        _p(f"  已构造 prompt_top_k=2 的配置（retrieval.prompt_top_k={cfg_pk2.retrieval.prompt_top_k}）")
        engine_pk2 = QAEngine(cfg=cfg_pk2)
        t_pk2_warm = time.perf_counter()
        warm_pk2 = engine_pk2.warmup(with_generate=False)
        _p(f"  该实例预热（不重复预热生成模型）用时 "
           f"{round((time.perf_counter() - t_pk2_warm) * 1000, 2)} ms")
        pk2_records = run_pass(engine_pk2, subset, scenario="probe_promptk_2", top_k=args.top_k,
                               session_prefix=args.session_prefix, pass_tag="pk2", use_cache=False,
                               batch=args.batch, batch_sleep=args.batch_sleep,
                               label="prompt_top_k=2")
        scenarios["probe_promptk_3"] = {"说明": "对照臂：RAG_RETRIEVAL__PROMPT_TOP_K=3（引擎默认）",
                                        "统计": latency_stats(base_records), "逐题": base_records}
        scenarios["probe_promptk_2"] = {"说明": "实验臂：RAG_RETRIEVAL__PROMPT_TOP_K=2",
                                        "统计": latency_stats(pk2_records), "逐题": pk2_records}
        promptk_payload = {"子集题号": [str(i.get("id")) for i in subset],
                           "说明": json.dumps(warm_pk2, ensure_ascii=False, default=str)[:400]}
        timeline.append({"scenario": "promptk_probe", "at": _stamp()})

    # ---------------- 汇总落盘 ----------------
    rev_end = module_revision()
    changed = sorted(k for k in rev_start if rev_start.get(k) != rev_end.get(k))
    ended = time.time()
    payload: dict[str, Any] = {
        "工单": WORK_ORDER,
        "生成时间": _stamp(),
        "脚本": str(Path(__file__).resolve()),
        "SLA_ms": SLA_MS,
        "共享CPU声明": ("Ollama(127.0.0.1:11434) 为 CPU 单实例，与「评测官（17 题评估）」"
                    "「测试官（回归）」等任务共享；本基准串行分批执行，批间休眠 "
                    f"{args.batch_sleep:.0f} 秒，但数字仍含竞争波动，未做 CPU 独占隔离。"),
        "配置": {
            "top_k": args.top_k, "use_cache(场景2/5)": False, "session_prefix": args.session_prefix,
            "engine": {"prompt_top_k": cfg.retrieval.prompt_top_k, "vector_k": cfg.retrieval.vector_k,
                       "bm25_k": cfg.retrieval.bm25_k, "cache_size": cfg.retrieval.cache_size,
                       "page_context_extra": cfg.retrieval.page_context_extra},
            "gen_model": cfg.llm.ollama_gen_model, "embed_model": cfg.llm.ollama_embed_model,
            "backend": cfg.llm.backend,
        },
        "环境": {
            "解释器": sys.executable, "python": sys.version.split()[0],
            "cwd": str(Path.cwd()), "cpu_count": os.cpu_count(),
            "脚本启动时刻": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_T_SCRIPT_START)),
            "结束时刻": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ended)),
            "脚本总墙钟s": round(ended - _T_SCRIPT_START, 2),
            "warmup_ms": warmup_ms,
            "warmup明细": warm_info,
            "ollama_运行前": ollama_before, "ollama_预热后": ollama_running_models(),
            "修订_start": rev_start, "修订_end": rev_end,
            "评估期间被修改的模块": changed,
        },
        "场景": scenarios,
        "时间线": timeline,
    }
    if promptk_payload:
        payload["prompt_top_k对照"] = promptk_payload

    out = Path(args.out) if args.out else DEFAULT_OUT
    if not out.is_absolute():
        out = (REPO_ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path = out.with_suffix(".md")
    md_path.write_text(render_markdown(payload), encoding="utf-8")

    if changed:
        _p("")
        _p("!! 警告：以下模块在本次基准运行期间被修改，结果不可严格归因到单一修订：")
        for name in changed:
            _p(f"    {name}: {rev_start[name]} -> {rev_end.get(name)}")

    print_summary(payload)
    _p("")
    _p(f"[落盘] {out}")
    _p(f"[落盘] {md_path}")
    _p(f"[落盘] 脚本总墙钟 {payload['环境']['脚本总墙钟s']} s")
    return 0


def print_summary(payload: dict[str, Any]) -> None:
    _p("")
    _p("=" * 78)
    _p(" 分场景响应时间汇总（ms）")
    _p("=" * 78)
    header = (f"{'场景':<22}{'样本':<6}{'均值':<11}{'中位':<11}{'P90':<11}"
              f"{'最大':<11}{'达标率':<9}{'均值首字':<11}")
    _p(header)
    _p("-" * 78)
    for name, block in payload["场景"].items():
        st = block["统计"]
        total = st.get("总耗时") or {}
        first = st.get("首字") or {}
        sla = st.get("SLA") or {}
        _p(f"{name:<22}{st.get('有效样本数', 0):<6}"
           f"{str(total.get('均值', '-')):<11}{str(total.get('中位', '-')):<11}"
           f"{str(total.get('P90', '-')):<11}{str(total.get('最大', '-')):<11}"
           f"{str(sla.get('达标率', '-')):<9}{str(first.get('均值', '-')):<11}")


# ---------------------------------------------------------------------------
# 6) Markdown 渲染
# ---------------------------------------------------------------------------
def render_markdown(payload: dict[str, Any]) -> str:
    env = payload["环境"]
    cfg = payload["配置"]
    lines: list[str] = []
    add = lines.append
    add("# 响应时间基准（工单7 实测）")
    add("")
    add(f"> 工单编号：**{payload['工单']}**　|　生成时间：{payload['生成时间']}　|　"
        f"SLA 阈值：**{payload['SLA_ms']:.0f} ms**")
    add("")
    add(f"> ⚠️ **共享 CPU 声明**：{payload['共享CPU声明']}")
    add("")
    add("## 一、结论速览")
    add("")
    add("| 场景 | 有效样本 | 均值 | 中位 | P50 | P90 | P95 | 最大 | **≤3000ms 达标率** | 超时题号 |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for name, block in payload["场景"].items():
        st = block["统计"]
        t = st.get("总耗时") or {}
        sla = st.get("SLA") or {}
        add(f"| `{name}` | {st.get('有效样本数', 0)} | {t.get('均值', '-')} | {t.get('中位', '-')} | "
            f"{t.get('P50', '-')} | {t.get('P90', '-')} | {t.get('P95', '-')} | {t.get('最大', '-')} | "
            f"**{sla.get('达标率', '-')}**（{sla.get('达标题数', '-')}/{sla.get('样本数', '-')}） | "
            f"{'、'.join(sla.get('超时题号') or []) or '无'} |")
    add("")
    add("## 二、分段耗时（均值 / P90 / 最大，ms）")
    add("")
    add("| 场景 | 检索 均值 | 检索 P90 | 生成 均值 | 生成 P90 | 首字 均值 | 首字 P90 | 首字 最大 | 首字达标率 |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, block in payload["场景"].items():
        st = block["统计"]
        r = st.get("检索") or {}
        g = st.get("生成") or {}
        f = st.get("首字") or {}
        sla = st.get("SLA") or {}
        add(f"| `{name}` | {r.get('均值', '-')} | {r.get('P90', '-')} | {g.get('均值', '-')} | "
            f"{g.get('P90', '-')} | {f.get('均值', '-')} | {f.get('P90', '-')} | {f.get('最大', '-')} | "
            f"{sla.get('达标率_首字', '-')} |")
    add("")
    add("## 三、逐题明细")
    add("")
    for name, block in payload["场景"].items():
        add(f"### `{name}`")
        add("")
        add(f"- 说明：{block.get('说明')}")
        add("")
        add("| 题号 | retrieve | generate | first_token | total | 墙钟 | 达标 | 拒答 | 生成方式 | 缓存 |")
        add("| --- | ---: | ---: | ---: | ---: | ---: | :-: | :-: | --- | --- |")
        for r in block["逐题"]:
            lat = r.get("latency_ms") or {}
            ok = "✅" if float(lat.get("total") or 0) <= payload["SLA_ms"] else "❌"
            cache = r.get("cache") or {}
            cache_txt = ("命中(" + str(cache.get("layer", "-")) + ")") if cache.get("hit") else "未命中"
            add(f"| {r['id']} | {lat.get('retrieve', '-')} | {lat.get('generate', '-')} | "
                f"{lat.get('first_token', '-')} | {lat.get('total', '-')} | {r.get('wall_ms', '-')} | "
                f"{ok} | {'是' if r.get('refused') else '否'} | {r.get('generation_method', '-')} | "
                f"{cache_txt} |")
        extra = {k: v for k, v in block.items() if k not in {"说明", "统计", "逐题"}}
        if extra:
            add("")
            add(f"<details><summary>附加观测</summary>")
            add("")
            add("```json")
            add(json.dumps(extra, ensure_ascii=False, indent=2)[:4000])
            add("```")
            add("</details>")
        add("")
    add("## 四、复现信息")
    add("")
    add(f"- 复现命令：`pwsh -NoProfile -File run_py.ps1 优化/脚本/响应时间基准.py "
        f"--limit {len(payload['场景']['warm_no_cache']['逐题'])} "
        f"--top-k {cfg['top_k']} --batch 4 --batch-sleep 5 "
        f"--out 优化/评估结果/响应时间基准.json`")
    add(f"- 解释器：`{env['解释器']}`（Python {env['python']}）")
    add(f"- 工作目录：`{env['cwd']}`　CPU 逻辑核数：{env['cpu_count']}")
    add(f"- 脚本启动：{env['脚本启动时刻']}　结束：{env['结束时刻']}　"
        f"总墙钟：**{env['脚本总墙钟s']} s**")
    add(f"- 预热耗时：**{env['warmup_ms']} ms**　明细：`{json.dumps(env['warmup明细'], ensure_ascii=False, default=str)[:400]}`")
    add(f"- 生成模型：`{cfg['gen_model']}`（backend={cfg['backend']}）；"
        f"嵌入模型：`{cfg['embed_model']}`")
    add(f"- 检索 top_k：{cfg['top_k']}；引擎 prompt_top_k：{cfg['engine']['prompt_top_k']}；"
        f"vector_k：{cfg['engine']['vector_k']}；bm25_k：{cfg['engine']['bm25_k']}")
    add(f"- Ollama 运行前驻留：`{json.dumps(env['ollama_运行前'], ensure_ascii=False)[:300]}`")
    add("")
    add("### 被优化代码修订（SHA256 前 16 位）")
    add("")
    add("| 模块 | 运行前 | 运行后 |")
    add("| --- | --- | --- |")
    for name, digest in env["修订_start"].items():
        add(f"| `{name}` | `{digest}` | `{env['修订_end'].get(name)}` |")
    add("")
    if env["评估期间被修改的模块"]:
        add(f"> ⚠️ 以下模块在本次运行期间被修改，结果不可严格归因到单一修订："
            f"{'、'.join(env['评估期间被修改的模块'])}")
        add("")
    if payload.get("prompt_top_k对照"):
        add("## 五、prompt_top_k 对照实验")
        add("")
        add(f"- 子集：{'、'.join(payload['prompt_top_k对照']['子集题号'])}")
        add(f"- 说明：{payload['prompt_top_k对照']['说明']}")
        add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 7) CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"{WORK_ORDER} —— 响应时间基准实测（三场景）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["full", "probe-cache", "probe-cold", "probe-promptk",
                                           "probe-misc"], default="full",
                        help="full=三场景基准（默认）；probe-cache=检索结果缓存纯检索 A/B；"
                             "probe-cold=冷启动各构件分项实测；"
                             "probe-promptk=prompt_top_k 性能+质量双维度对照；"
                             "probe-misc=批量嵌入扫描 + 对话级缓存 A/B")
    parser.add_argument("--wait-index", action="store_true",
                        help="轮询等待索引就绪（默认最多 30 分钟）")
    parser.add_argument("--wait-timeout", type=float, default=1800.0,
                        help="--wait-index 的最长等待秒数（默认 1800 = 30 分钟）")
    parser.add_argument("--limit", type=int, default=None,
                        help="只跑前 N 题（冷启动场景永远只测第 1 题）")
    parser.add_argument("--top-k", type=int, default=10,
                        help="检索 top_k（默认 10，与 研发/评估/配置.py 的 TOP_K 对齐，便于与"
                             "测试/测试数据/评估汇总.json 对照）")
    parser.add_argument("--batch", type=int, default=4, help="每批题数（默认 4；0=不分批）")
    parser.add_argument("--batch-sleep", type=float, default=5.0, help="批间休眠秒数（默认 5）")
    parser.add_argument("--promptk-probe", type=int, default=6,
                        help="prompt_top_k 对照实验的题数（默认 6；0=跳过）")
    parser.add_argument("--sqlite-probe", type=int, default=4,
                        help="跨引擎实例（答案级 SQLite 缓存）探针题数（默认 4；0=跳过）。"
                             "SQLite 文件落在 优化/评估结果/ 下，不触碰 研发/ 的运行时库")
    parser.add_argument("--session-prefix", default="bench7", help="会话前缀（每题再拼题号与 pass）")
    parser.add_argument("--out", default=None, help=f"JSON 输出路径（默认 {DEFAULT_OUT.name}）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "probe-cache":
        return mode_probe_cache(args)
    if args.mode == "probe-cold":
        return mode_probe_cold(args)
    if args.mode == "probe-promptk":
        return mode_probe_promptk(args)
    if args.mode == "probe-misc":
        return mode_probe_misc(args)
    return mode_full(args)


if __name__ == "__main__":
    raise SystemExit(main())
