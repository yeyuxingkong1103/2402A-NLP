# -*- coding: utf-8 -*-
"""工单13 性能埋点与优化开关中心：``研发/app/core/perf.py``。

工单编号：人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化

职责（三个，互不耦合）：

1. **五阶段耗时分解（StageTimer）** —— 把 RAG 链路拆成工单要求的五个阶段：
   ``查询处理与增强`` → ``检索``（向量 / BM25 / 融合+重排 / 证据页补全）→
   ``上下文组装与提示工程`` → ``LLM 生成`` → ``后处理与响应格式化``；
   每阶段内再记子阶段（``embed_query`` / ``vector_search`` / ``bm25_search`` /
   ``fusion_rerank`` / ``page_context`` / ``llm_request`` / ``guard`` …）。
   埋点是**累加式**的：同一子阶段在一次请求里被调用多次则累加，并记次数。
   线程局部（``threading.local``）实现，天然支持并发压测与 ThreadPoolExecutor。

2. **优化开关（OptFlags）** —— 全部优化项可开关、阈值可配置。开关来源优先级：
   显式环境变量 > 配置文件 ``部署/配置/优化开关.json`` > 内置默认。
   **关闭全部开关 = 基线行为**（用于前后对比的 ``优化前`` 口径），这是可验证的等价性前提。

3. **计时口径统一** —— ``now_ms()`` / ``percentile()`` 等工具，供脚本与测试复用。

纪律：本模块**不 import 任何业务模块**（只依赖标准库），避免循环依赖；
所有开关读取都容错（配置坏 ⇒ 回落默认值 + 记 warning，不抛异常）。
"""

from __future__ import annotations

import json
import os
import statistics
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

WORK_ORDER = "人工智能NLP-RAG 项目-RAG 性能瓶颈识别与优化"

# ---------------------------------------------------------------------------
# 1) 五阶段定义（工单原文「查询处理与增强 / 检索 / 上下文组装与提示工程 / LLM 生成 / 后处理」）
# ---------------------------------------------------------------------------
STAGE_QUERY = "1_查询处理与增强"
STAGE_RETRIEVE = "2_检索"
STAGE_CONTEXT = "3_上下文组装与提示工程"
STAGE_GENERATE = "4_LLM生成"
STAGE_POST = "5_后处理与响应格式化"

STAGES: tuple[str, ...] = (STAGE_QUERY, STAGE_RETRIEVE, STAGE_CONTEXT, STAGE_GENERATE, STAGE_POST)
STAGE_LABELS: dict[str, str] = {
    STAGE_QUERY: "查询处理与增强（指代消解 / 语言意图识别 / 查询富化）",
    STAGE_RETRIEVE: "检索（查询嵌入 / 向量检索 / BM25 检索 / RRF 融合+重排 / 证据页补全）",
    STAGE_CONTEXT: "上下文组装与提示工程（证据块选取 / 提示词拼装）",
    STAGE_GENERATE: "LLM 生成（Ollama 调用 / 首字延迟 / 护栏复核）",
    STAGE_POST: "后处理与响应格式化（引用构造 / 结构化输出 / 缓存写入）",
}


class StageTimer:
    """线程局部五阶段计时器（累加式）。

    用法::

        t = StageTimer()
        with t.stage(STAGE_RETRIEVE):
            with t.span("vector_search"):
                ...
        t.snapshot()   # {"stages": {...}, "spans": {...}, "total_ms": ...}
    """

    def __init__(self) -> None:
        self._local = threading.local()

    # -- 内部状态 ---------------------------------------------------------
    def _state(self) -> dict[str, Any]:
        state = getattr(self._local, "state", None)
        if state is None:
            state = {"stages": {}, "spans": {}, "counts": {}, "t0": time.perf_counter()}
            self._local.state = state
        return state

    # -- 生命周期 ---------------------------------------------------------
    def reset(self) -> None:
        """开始一次新的请求计时（清空本线程状态）。"""
        self._local.state = {"stages": {}, "spans": {}, "counts": {}, "t0": time.perf_counter()}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """阶段计时（累加）。"""
        state = self._state()
        started = time.perf_counter()
        try:
            yield
        finally:
            cost = (time.perf_counter() - started) * 1000.0
            state["stages"][name] = state["stages"].get(name, 0.0) + cost

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        """子阶段计时（累加 + 计数）。"""
        state = self._state()
        started = time.perf_counter()
        try:
            yield
        finally:
            cost = (time.perf_counter() - started) * 1000.0
            state["spans"][name] = state["spans"].get(name, 0.0) + cost
            state["counts"][name] = state["counts"].get(name, 0) + 1

    def add(self, name: str, ms: float, *, span: bool = True) -> None:
        """外部已测好的耗时直接累加进来（避免重复包一层计时）。"""
        state = self._state()
        bucket = "spans" if span else "stages"
        state[bucket][name] = state[bucket].get(name, 0.0) + float(ms)
        if span:
            state["counts"][name] = state["counts"].get(name, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        """取当前累计值（不重置）。"""
        state = self._state()
        total = (time.perf_counter() - state["t0"]) * 1000.0
        stages = {k: round(v, 3) for k, v in state["stages"].items()}
        spans = {k: round(v, 3) for k, v in state["spans"].items()}
        for name in STAGES:                       # 五阶段必须齐全（未发生的记 0，不省略）
            stages.setdefault(name, 0.0)
        stages["_其余(未归阶段)"] = round(
            max(0.0, total - sum(v for k, v in stages.items() if k in STAGES)), 3)
        return {"stages": stages, "spans": spans, "counts": dict(state["counts"]),
                "total_ms": round(total, 3)}

    def as_row(self) -> dict[str, float]:
        """扁平化一行（便于写 CSV / jsonl）。"""
        snap = self.snapshot()
        row: dict[str, float] = {f"stage::{k}": v for k, v in snap["stages"].items()}
        row.update({f"span::{k}": v for k, v in snap["spans"].items()})
        row["total_ms"] = snap["total_ms"]
        return row


#: 全局默认计时器（请求级；多线程各自独立）
TIMER = StageTimer()


# ---------------------------------------------------------------------------
# 2) 优化开关
# ---------------------------------------------------------------------------
#: 开关默认值。``False`` / ``0`` 即**基线行为**。键名与 部署/配置/优化开关.json 一一对应。
DEFAULT_FLAGS: dict[str, Any] = {
    # ① 预热与常驻：进程启动时把「索引 + 分词器 + 嵌入模型 + 生成模型」全部热起来
    "warmup_enabled": True,
    # ①b 是否预热「多字段倒排索引」（工单6 全文检索通道）。默认 **False**：
    #     实测该步 19.2 s（8847 块 × 4 字段分词），占冷启动预热的 63%，
    #     而默认检索链路（hybrid_search = FAISS + BM25）**不使用**它 —— 仅 --strategy 时用。
    "warmup_fulltext": False,
    # ② 嵌入后端探测缓存：resolve_embedding_backend 每次调用都 HTTP 探测 /api/tags（实测 3~20 ms/次）
    "embed_backend_cache": True,
    "embed_probe_ttl_s": 300.0,
    # ③ 向量检索与 BM25 检索并行（ThreadPoolExecutor）
    "parallel_channels": True,
    "parallel_workers": 2,
    # ④ 重排候选裁剪（RRF 预算 + 相对分截断）
    "rerank_prune": True,
    "rerank_budget": 40,
    "rerank_min_relative": 0.02,
    # ⑤ 结果缓存（问题级 + 检索级 + 生成级）
    "cache_enabled": True,
    # ⑥ 上下文组装 IO 缓存（页块索引 / 块内容摘要）
    "context_io_cache": True,
    # ⑦ top_k 自适应（依据问句形态收敛检索广度）
    "adaptive_top_k": True,
    "adaptive_top_k_values": {"multi": 5, "single": 3, "default": 5},
    # ⑧ 日志异步化（把同步 write+flush 移出热路径）
    "async_log": True,
    "async_log_queue": 4096,
    # ⑨ 生成预算收敛（max_tokens 上限 + 证据块字符预算）
    "generate_budget": True,
    "generate_max_tokens": 512,
    "context_char_budget": 6000,
    # ⑩ 容错阈值
    "max_query_chars": 2000,
    "ollama_timeout_s": 20.0,
}

_FLAG_ENV: dict[str, str] = {key: f"RAG_PERF__{key.upper()}" for key in DEFAULT_FLAGS}

_CONFIG_CACHE: dict[str, Any] = {"mtime": None, "data": {}}
_CONFIG_LOCK = threading.Lock()
_WARNED: set[str] = set()


def config_path() -> Path:
    """优化开关配置文件路径（``研发/app/core/perf.py`` → 工单根 / 部署/配置/优化开关.json）。"""
    explicit = os.environ.get("RAG_PERF__CONFIG")
    if explicit:
        return Path(explicit)
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "部署" / "配置" / "优化开关.json"


def _load_config_file() -> dict[str, Any]:
    """读配置文件（带 mtime 缓存；坏文件只 warning 不抛）。"""
    path = config_path()
    try:
        mtime = path.stat().st_mtime if path.is_file() else None
    except OSError:
        mtime = None
    with _CONFIG_LOCK:
        if _CONFIG_CACHE["mtime"] == mtime:
            return _CONFIG_CACHE["data"]
        data: dict[str, Any] = {}
        if mtime is not None:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    data = dict(raw.get("flags") or raw)
            except (OSError, ValueError) as exc:      # 显式降级：配置坏 ⇒ 用内置默认
                _warn_once(f"优化开关配置读取失败（{path}）：{type(exc).__name__}: {exc}，改用内置默认")
        _CONFIG_CACHE["mtime"] = mtime
        _CONFIG_CACHE["data"] = data
        return data


def _warn_once(message: str) -> None:
    if message not in _WARNED:
        _WARNED.add(message)
        print(f"[perf 降级] {message}", flush=True)


def _coerce(default: Any, raw: Any) -> Any:
    """按默认值类型把环境变量字符串转成目标类型（失败即回落默认）。"""
    if raw is None:
        return default
    if isinstance(default, bool):
        text = str(raw).strip().lower()
        if text in {"1", "true", "yes", "on", "y"}:
            return True
        if text in {"0", "false", "no", "off", "n"}:
            return False
        return default
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(float(str(raw).strip()))
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(str(raw).strip())
        except ValueError:
            return default
    if isinstance(default, dict):
        try:
            parsed = json.loads(str(raw))
            return parsed if isinstance(parsed, dict) else default
        except ValueError:
            return default
    return raw


def flag(name: str, default: Any = None) -> Any:
    """读单个优化开关：环境变量 > 配置文件 > 内置默认。"""
    base = DEFAULT_FLAGS.get(name) if default is None else default
    env_key = _FLAG_ENV.get(name, f"RAG_PERF__{name.upper()}")
    if env_key in os.environ and str(os.environ[env_key]).strip() != "":
        return _coerce(base, os.environ[env_key])
    file_data = _load_config_file()
    if name in file_data:
        return _coerce(base, file_data[name])
    return base


def flags() -> dict[str, Any]:
    """当前生效的全部开关（写进报告与日志，便于复核）。"""
    return {name: flag(name) for name in DEFAULT_FLAGS}


def apply_profile(profile: str | Mapping[str, Any] | None, *, environ: Any = None) -> dict[str, Any]:
    """按「档位」批量设置开关（写进程环境变量，供子进程/脚本使用）。

    * ``baseline`` / ``off`` —— 全部优化开关关闭（优化前口径）；
    * ``optimized`` / ``on`` —— 全部优化开关打开（优化后口径）；
    * ``Mapping`` —— 逐项设置（用于做「单项开关 A/B 实验」）。

    **唯一例外（必须写进报告，否则前后对比不成立）**：``warmup_enabled`` 在两个档位下
    都保持 True。理由：验收指标「用户输入 query 后 ≤3 s」度量的是**服务已就绪**时的
    单次响应；把「模型冷启动加载」算进每一次查询会同时污染对照组与实验组，
    使对比失去意义。冷启动本身的耗时单独用 ``--no-warmup`` 口径测量并单列（见
    ``测试/测试报告.md`` §冷启动），不混进 SLA 对比。
    """
    environ = os.environ if environ is None else environ
    if profile is None:
        return flags()
    if isinstance(profile, str):
        key = profile.strip().lower()
        if key in {"baseline", "off", "optimize_off"}:
            plan: dict[str, Any] = {name: (False if isinstance(value, bool) else value)
                                    for name, value in DEFAULT_FLAGS.items()}
            plan["warmup_enabled"] = True          # 见 docstring：两组都预热，保证口径可比
        elif key in {"optimized", "on", "opt"}:
            plan = dict(DEFAULT_FLAGS)
        elif key == "config":
            plan = {}
        else:
            raise ValueError(f"未知档位：{profile}（可选 baseline / optimized / config / JSON 对象）")
    else:
        plan = dict(profile)
    for name, value in plan.items():
        env_key = _FLAG_ENV.get(name, f"RAG_PERF__{name.upper()}")
        if isinstance(value, bool):
            environ[env_key] = "1" if value else "0"
        elif isinstance(value, (dict, list)):
            environ[env_key] = json.dumps(value, ensure_ascii=False)
        else:
            environ[env_key] = str(value)
    return flags()


# ---------------------------------------------------------------------------
# 3) 计时工具
# ---------------------------------------------------------------------------
def now_ms() -> float:
    """单调时钟毫秒（性能测量一律用它，不用 wall clock）。"""
    return time.perf_counter() * 1000.0


def percentile(values: list[float], pct: float) -> float:
    """线性插值分位数（``pct`` 取 0~100）；空列表返回 0.0。"""
    data = sorted(float(v) for v in values if v is not None)
    if not data:
        return 0.0
    if len(data) == 1:
        return data[0]
    rank = (len(data) - 1) * (float(pct) / 100.0)
    low = int(rank)
    high = min(low + 1, len(data) - 1)
    frac = rank - low
    return data[low] * (1.0 - frac) + data[high] * frac


def summarize(values: list[float]) -> dict[str, float]:
    """一组耗时的标准统计（均值 / P50 / P90 / P95 / P99 / 最大 / 最小 / 标准差）。"""
    data = [float(v) for v in values if v is not None]
    if not data:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0,
                "max": 0.0, "min": 0.0, "std": 0.0}
    return {
        "count": len(data),
        "mean": round(statistics.fmean(data), 2),
        "p50": round(percentile(data, 50), 2),
        "p90": round(percentile(data, 90), 2),
        "p95": round(percentile(data, 95), 2),
        "p99": round(percentile(data, 99), 2),
        "max": round(max(data), 2),
        "min": round(min(data), 2),
        "std": round(statistics.pstdev(data), 2) if len(data) > 1 else 0.0,
    }


def sla_rate(values: list[float], budget_ms: float = 3000.0) -> float:
    """SLA 达标率（≤ ``budget_ms`` 的比例，0~1）。"""
    data = [float(v) for v in values if v is not None]
    if not data:
        return 0.0
    return round(sum(1 for v in data if v <= float(budget_ms)) / len(data), 4)


def stage_share(stage_means: Mapping[str, float]) -> dict[str, float]:
    """各阶段耗时占比（相对五阶段之和）。"""
    total = sum(float(v) for k, v in stage_means.items() if k in STAGES)
    if total <= 0:
        return {k: 0.0 for k in STAGES}
    return {k: round(float(stage_means.get(k, 0.0)) / total, 4) for k in STAGES}
