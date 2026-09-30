# -*- coding: utf-8 -*-
"""跨平台运行时指标采集器（t6）。

回答用户的问题：「监控 CPU、GPU、显存、内存等占用情况 + 系统层指标，
比如进程内存/CPU、Milvus/Redis 连通性与延迟、Ollama 模型是否驻留」。

两个入口
--------
* :func:`snapshot` —— 一次性采集，返回一个纯 JSON 可序列化的 dict，
  供 ``GET /health`` 与按需诊断使用；默认顺手把数值发布到 t1 的指标注册表。
* :func:`install_periodic_sampler` —— 后台周期采样（守护线程），
  **异常自愈**：任何一次采样失败只记一条 warning，绝不把异常抛给调用方，
  更不会拖垮 FastAPI 主流程。

采集五类
--------
1. 进程级：RSS / CPU 百分比 / 线程数 / 打开句柄(fd)数 / 运行时长 / Python 版本；
2. 主机级：CPU 总利用率与**各核**利用率、内存总量·已用·可用、swap、根分区磁盘；
3. GPU/显存：``nvidia-smi --query-gpu`` 优先，``pynvml`` 可选；
   无卡 / 无命令 / 查询失败 → ``{"available": False, "reason": "..."}``，
   **不抛异常、不静默丢字段**；
4. 依赖健康与延迟：Redis(PING) / Milvus(轻量 gRPC list_collections) /
   Ollama(GET /api/version)，各自 ``up`` + ``latency_ms`` + 失败 ``reason``；
5. Ollama 模型驻留：``GET /api/ps`` → 已加载模型名 / size / size_vram / expires_at。
   空闲时 Ollama 会卸载模型，返回 ``{"models": []}`` —— 这是**正常状态**，
   表达为 ``loaded_count = 0``，不是错误。

依赖策略（务必读）
------------------
* **首选纯标准库**：Linux 读 ``/proc``（meminfo / stat / self/stat / self/statm /
  self/status / uptime / loadavg），Windows 用 ``ctypes`` 调 Win32
  （``GetProcessMemoryInfo`` / ``GetProcessHandleCount`` / ``GetProcessTimes`` /
  ``Toolhelp32`` / ``GlobalMemoryStatusEx`` / ``NtQuerySystemInformation``）；
* ``psutil`` / ``nvidia-ml-py`` 是**可选增强**，延迟导入，缺失时自动回退标准库路径
  并打一条 ``[DEGRADED]`` 日志说明降级；它们不写进 ``requirements.txt`` 硬依赖；
* 两块目标机器的现实差异被显式建模：Windows 本机有 RTX 2060（能采到真值），
  Ubuntu VM 无卡（优雅降级为 unavailable + reason）。

安全红线（本模块只读依赖服务）
------------------------------
* Redis 只发 ``PING``（绝不 FLUSH / 绝不 ``KEYS *``，已有 19 个用户 key 与 0 个
  ``legal_rag:`` key 必须原样保留）；
* Milvus 只调 ``list_collections``（绝不建/删/写 collection）；
* Ollama 只 ``GET /api/version`` 与 ``GET /api/ps``；
* 快照里的 Redis URL 会打码（不把口令写进日志或 /health 响应）。

命名与单位沿用 t1 规范：秒写 ``seconds``、字节写 ``bytes``、
``process_*`` / ``host_*`` / ``gpu_*`` / ``dep_up`` / ``dep_latency_seconds`` /
``ollama_loaded_*``（完整清单见 :data:`METRIC_SPECS` 与 ``docs/MONITORING.md``）。
"""
from __future__ import annotations

import atexit
import ctypes
import datetime
import importlib
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import metrics as M
from .config import RagConfig
from .logging_setup import get_logger

__all__ = [
    # 两个主入口
    "snapshot", "install_periodic_sampler",
    # 采样器生命周期
    "PeriodicSampler", "stop_periodic_sampler", "current_sampler", "sampler_status",
    # 分项采集（t4 / 诊断脚本可直接用）
    "collect_process", "collect_host", "collect_gpu", "collect_dependencies",
    "collect_ollama_models", "probe_redis", "probe_milvus", "probe_ollama",
    # 指标注册与发布
    "METRIC_SPECS", "register_metrics", "publish_metrics", "last_snapshot",
    # /health 摘要
    "health", "summarize",
    # 纯解析函数（跨平台可测）
    "parse_proc_meminfo", "parse_proc_stat", "parse_proc_self_stat",
    "parse_proc_uptime", "parse_gpu_csv", "parse_smi_number", "smi_memory_bytes",
]

_LOGGER = get_logger(__name__)

_IS_WINDOWS = os.name == "nt"
_IS_LINUX = sys.platform.startswith("linux")
_IS_MACOS = sys.platform == "darwin"

#: 依赖探针的默认上限（秒）。真实取值优先用 config，再用本常量兜底/封顶，
#: 避免 config.py（t1 产物，禁止修改）里 30s/60s 的超时把 /health 拖死。
DEFAULT_PROBE_TIMEOUT_SECONDS = 5.0

#: FILETIME(1601-01-01) → Unix epoch 的秒数差
_FILETIME_EPOCH_OFFSET = 11644473600.0

#: reason 字段最长长度（日志与 /health 都不该被一坨 stderr 撑爆）
_MAX_REASON_LEN = 300

#: nvidia-smi 查询字段（顺序即 CSV 列顺序）
#: cuda_version 无法通过 --query-gpu 取（实测报 invalid field），只能从
#: `nvidia-smi` 头部 ``CUDA Version: 12.9`` 正则提取；驱动版本 5 分钟缓存一次。
_GPU_META_TTL_SECONDS = 300.0

_DEGRADATION_LOGGED: set[str] = set()
_DEGRADATION_LOCK = threading.Lock()


# ==========================================================================
# 0. 小工具
# ==========================================================================

def _truncate(text: Any, limit: int = _MAX_REASON_LEN) -> str:
    """压平空白并截断（reason 字段用）。"""
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[: max(0, limit - 3)] + "..."


def _error_text(exc: BaseException) -> str:
    return _truncate(f"{type(exc).__name__}: {exc}")


def _log_degraded_once(key: str, message: str, *args: Any) -> None:
    """同一条降级说明只打一次，避免 15s 一轮的采样把日志刷爆。"""
    with _DEGRADATION_LOCK:
        if key in _DEGRADATION_LOGGED:
            return
        _DEGRADATION_LOGGED.add(key)
    _LOGGER.info("[DEGRADED] " + message, *args)


def _mask_url(url: str) -> str:
    """去掉 URL 里的口令，避免写进日志/响应体（例如 redis://:pwd@host:6379/0）。"""
    return re.sub(r"(?i)(://[^:/@]*):[^@/]*@", r"\1:***@", str(url))


def _positive_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number <= 0 or number != number:
        return default
    return number


def _round(value: Any, digits: int = 2) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return round(number, digits)


def _optional_import(name: str) -> Any:
    """延迟导入可选增强依赖；失败返回 None（并只记一次降级日志）。"""
    try:
        return importlib.import_module(name)
    except Exception as exc:  # ImportError 及其它环境问题
        _log_degraded_once(
            f"optional:{name}",
            "可选依赖 %s 不可用（%s），自动回退纯标准库路径",
            name, type(exc).__name__,
        )
        return None


def _load_config(config: RagConfig | None) -> RagConfig:
    if config is not None:
        return config
    try:
        return RagConfig.from_env()
    except Exception as exc:  # 配置异常也不能让监控挂掉
        _LOGGER.warning("RagConfig.from_env() 失败（%s），改用默认配置", _error_text(exc))
        return RagConfig()


def _probe_timeouts(cfg: RagConfig, override: float | None = None) -> dict[str, float]:
    """三个依赖的探针超时：能取 config 就取，取不到/过大就用常量封顶。"""
    cap = DEFAULT_PROBE_TIMEOUT_SECONDS
    if override is not None:
        cap = _positive_float(override, cap) or cap
    redis_to = _positive_float(getattr(cfg.cache, "connect_timeout", None), cap) or cap
    milvus_to = _positive_float(getattr(cfg.milvus, "connect_timeout", None), cap) or cap
    ollama_to = _positive_float(getattr(cfg.ollama, "embed_timeout", None), cap) or cap
    return {
        "redis": min(redis_to, cap),
        "milvus": min(milvus_to, cap),
        "ollama": min(ollama_to, cap),
    }

def _redis_probe_timeout(cfg: RagConfig, override: float | None = None) -> float:
    return _probe_timeouts(cfg, override)["redis"]


# ==========================================================================
# 1. 纯解析函数（Linux /proc 与 nvidia-smi CSV —— 在两台机器上都能跑单测）
# ==========================================================================

def parse_proc_meminfo(text: str) -> dict[str, int]:
    """``/proc/meminfo`` → ``{字段: 字节}``（只取第一个数值，单位 kB/MB/GB）。"""
    units = {"b": 1, "kb": 1024, "mb": 1024 ** 2, "gb": 1024 ** 3, "tb": 1024 ** 4}
    parsed: dict[str, int] = {}
    for line in str(text).splitlines():
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        parts = rest.split()
        if not parts:
            continue
        try:
            value = float(parts[0])
        except ValueError:
            continue
        unit = parts[1].lower() if len(parts) > 1 else "kb"
        parsed[key.strip()] = int(value * units.get(unit, 1024))
    return parsed


def parse_proc_stat(text: str) -> dict[str, Any]:
    """``/proc/stat`` → ``{"total": (idle, total), "cores": [(idle, total), ...]}``（单位 jiffies）。

    CPU 利用率需要两次采样求差；``total`` 取前 8 个字段（user..steal），
    避免 guest/guest_nice 被重复计入。
    """
    total: tuple[float, float] | None = None
    cores: list[tuple[float, float]] = []
    for line in str(text).splitlines():
        if not line.startswith("cpu"):
            continue
        parts = line.split()
        name = parts[0]
        if name != "cpu" and not name[3:].isdigit():
            continue
        try:
            values = [float(item) for item in parts[1:]]
        except ValueError:
            continue
        if len(values) < 4:
            continue
        idle = values[3] + (values[4] if len(values) > 4 else 0.0)
        total_jiffies = sum(values[:8])
        if name == "cpu":
            total = (idle, total_jiffies)
        else:
            cores.append((idle, total_jiffies))
    return {"total": total, "cores": cores}


def parse_proc_self_stat(text: str) -> dict[str, Any]:
    """``/proc/self/stat`` → 进程 CPU 时间（jiffies）、线程数、启动时刻（jiffies since boot）。"""
    raw = str(text).strip()
    if not raw:
        return {}
    # comm 字段可能含空格与右括号：从最后一个 ') ' 之后开始切
    tail = raw[raw.rfind(")") + 1:].split() if ")" in raw else raw.split()
    # tail[0] 是 state(field 3)，故 field N 对应 tail[N - 3]
    def field(n: int) -> float | None:
        index = n - 3
        if 0 <= index < len(tail):
            try:
                return float(tail[index])
            except ValueError:
                return None
        return None

    utime, stime = field(14), field(15)
    threads, starttime = field(20), field(22)
    result: dict[str, Any] = {}
    if utime is not None and stime is not None:
        result["cpu_jiffies"] = utime + stime
    if threads is not None:
        result["threads"] = int(threads)
    if starttime is not None:
        result["starttime_jiffies"] = starttime
    return result


def parse_proc_uptime(text: str) -> float | None:
    """``/proc/uptime`` → 系统已运行秒数。"""
    parts = str(text).split()
    if not parts:
        return None
    try:
        return float(parts[0])
    except ValueError:
        return None


def parse_loadavg(text: str) -> dict[str, float] | None:
    """``/proc/loadavg`` → ``{"load1":..,"load5":..,"load15":..}``。"""
    parts = str(text).split()
    if len(parts) < 3:
        return None
    try:
        return {"load1": float(parts[0]), "load5": float(parts[1]), "load15": float(parts[2])}
    except ValueError:
        return None


# ==========================================================================
# 【2026-09-29 已停用，待删除】nvidia-smi 文本解析层
# --------------------------------------------------------------------------
# 停用理由：GPU 采集已收敛到 NVML（`nvidia-ml-py` 自 2026-09-29 起是
# `requirements.txt` 的主依赖，见 `_gpu_via_pynvml`）。下面三个函数是
# `nvidia-smi` 子进程路径**唯一**的解析助手，该路径删除后它们再无调用方。
#
# 保留一个版本周期以便回滚（按"先注释、验证、再删"的工作约定）。
# 验证通过后整段删除。
#
# def parse_smi_number(text: Any) -> float | None:
#     """``"6144 MiB"`` / ``"0 %"`` / ``"10.68 W"`` / ``"50"`` / ``"N/A"`` → float 或 None。"""
#     if text is None:
#         return None
#     raw = str(text).strip()
#     if not raw or raw.upper() in {"N/A", "[N/A]", "NA", "NAN", "UNKNOWN", "-", ""}:
#         return None
#     match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", raw)
#     if not match:
#         return None
#     try:
#         return float(match.group())
#     except ValueError:
#         return None
#
#
# def smi_memory_bytes(text: Any) -> int | None:
#     """``"6144 MiB"`` → 6442450944；单位缺失时按 MiB（nvidia-smi 的显存口径）。"""
#     value = parse_smi_number(text)
#     if value is None:
#         return None
#     raw = str(text).lower()
#     if "gib" in raw:
#         scale = 1024 ** 3
#     elif "mib" in raw:
#         scale = 1024 ** 2
#     elif "kib" in raw:
#         scale = 1024
#     elif "gb" in raw:
#         scale = 1000 ** 3
#     elif "mb" in raw:
#         scale = 1000 ** 2
#     else:
#         scale = 1024 ** 2
#     return int(value * scale)
#
#
# def parse_gpu_csv(text: str) -> list[dict[str, Any]]:
#     """``nvidia-smi --query-gpu=... --format=csv`` 输出 → 每个 GPU 一个 dict。
#
#     容忍表头行、空行、``N/A``（部分卡不支持功耗/温度读数）。
#     """
#     devices: list[dict[str, Any]] = []
#     for line in str(text).splitlines():
#         line = line.strip()
#         if not line:
#             continue
#         columns = [item.strip() for item in line.split(",")]
#         if not columns:
#             continue
#         if columns[0].lower().startswith("index") or parse_smi_number(columns[0]) is None:
#             continue  # 表头 / 非数据行
#         try:
#             index = int(float(columns[0]))
#         except ValueError:
#             continue
#
#         def column(position: int) -> str | None:
#             return columns[position] if position < len(columns) else None
#
#         devices.append({
#             "index": index,
#             "name": (column(1) or "").strip() or None,
#             "utilization_percent": _round(parse_smi_number(column(2))),
#             "memory_total_bytes": smi_memory_bytes(column(3)),
#             "memory_used_bytes": smi_memory_bytes(column(4)),
#             "temperature_celsius": _round(parse_smi_number(column(5))),
#             "power_watts": _round(parse_smi_number(column(6))),
#             "driver_version": (column(7) or "").strip() or None,
#         })
#     return devices
# ==========================================================================


# ==========================================================================
# 2. Windows Win32（ctypes）辅助
# ==========================================================================

_WIN32_LOCK = threading.Lock()
_WIN32_API: dict[str, Any] | None = None
_WIN32_ERROR: str | None = None


def _build_win32() -> dict[str, Any]:
    """绑定 Win32 函数签名。

    **必须显式设置 restype/argtypes**：64 位 HANDLE 若按默认 ``c_int`` 返回会被截断，
    ``GetProcessMemoryInfo`` / ``GetProcessHandleCount`` / ``GetProcessTimes``
    会静默失败（实测坑，见 docs/MONITORING.md）。
    """
    import ctypes.wintypes as wt  # 仅 Windows 可导入

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    class FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", wt.DWORD), ("dwHighDateTime", wt.DWORD)]

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    class THREADENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ThreadID", wt.DWORD),
            ("th32OwnerProcessID", wt.DWORD), ("tpBasePri", ctypes.c_long),
            ("tpDeltaPri", ctypes.c_long), ("dwFlags", wt.DWORD),
        ]

    class SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("IdleTime", ctypes.c_longlong), ("KernelTime", ctypes.c_longlong),
            ("UserTime", ctypes.c_longlong), ("DpcTime", ctypes.c_longlong),
            ("InterruptTime", ctypes.c_longlong), ("InterruptCount", wt.ULONG),
        ]

    k32.GetCurrentProcess.argtypes = []
    k32.GetCurrentProcess.restype = wt.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wt.DWORD]
    psapi.GetProcessMemoryInfo.restype = wt.BOOL
    k32.GetProcessHandleCount.argtypes = [wt.HANDLE, ctypes.POINTER(wt.DWORD)]
    k32.GetProcessHandleCount.restype = wt.BOOL
    k32.GetProcessTimes.argtypes = [wt.HANDLE] + [ctypes.POINTER(FILETIME)] * 4
    k32.GetProcessTimes.restype = wt.BOOL
    k32.GetSystemTimes.argtypes = [ctypes.POINTER(FILETIME)] * 3
    k32.GetSystemTimes.restype = wt.BOOL
    k32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
    k32.GlobalMemoryStatusEx.restype = wt.BOOL
    k32.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
    k32.CreateToolhelp32Snapshot.restype = wt.HANDLE
    k32.Thread32First.argtypes = [wt.HANDLE, ctypes.POINTER(THREADENTRY32)]
    k32.Thread32First.restype = wt.BOOL
    k32.Thread32Next.argtypes = [wt.HANDLE, ctypes.POINTER(THREADENTRY32)]
    k32.Thread32Next.restype = wt.BOOL
    k32.CloseHandle.argtypes = [wt.HANDLE]
    k32.CloseHandle.restype = wt.BOOL
    ntdll.NtQuerySystemInformation.argtypes = [
        ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong)]
    ntdll.NtQuerySystemInformation.restype = ctypes.c_long

    return {
        "wt": wt, "k32": k32, "psapi": psapi, "ntdll": ntdll,
        "PROCESS_MEMORY_COUNTERS": PROCESS_MEMORY_COUNTERS,
        "FILETIME": FILETIME, "MEMORYSTATUSEX": MEMORYSTATUSEX,
        "THREADENTRY32": THREADENTRY32,
        "SPPI": SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION,
    }


def _win32() -> dict[str, Any]:
    """惰性初始化 Win32 句柄；失败返回 ``{}``（调用方自然降级）。"""
    global _WIN32_API, _WIN32_ERROR
    if _WIN32_API is not None:
        return _WIN32_API
    if not _IS_WINDOWS:
        _WIN32_API = {}
        return _WIN32_API
    with _WIN32_LOCK:
        if _WIN32_API is not None:
            return _WIN32_API
        try:
            _WIN32_API = _build_win32()
        except Exception as exc:
            _WIN32_ERROR = _error_text(exc)
            _log_degraded_once("win32", "Win32 ctypes 初始化失败（%s），进程/主机指标降级", _WIN32_ERROR)
            _WIN32_API = {}
    return _WIN32_API


def _filetime_seconds(value: Any) -> float:
    """FILETIME → 秒（差值用，不减 epoch）。"""
    ticks = (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)
    return ticks / 1e7


def _filetime_to_unix(value: Any) -> float:
    """FILETIME → Unix 时间戳。"""
    return _filetime_seconds(value) - _FILETIME_EPOCH_OFFSET


def _win_thread_count() -> int | None:
    """CreateToolhelp32Snapshot 数本进程线程数（纯标准库，无 psutil 也能拿到）。"""
    api = _win32()
    if not api:
        return None
    k32, wt = api["k32"], api["wt"]
    TH32CS_SNAPTHREAD = 0x00000004
    snapshot = k32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
    invalid = {None, -1, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF}
    if snapshot in invalid:
        return None
    try:
        entry = api["THREADENTRY32"]()
        entry.dwSize = ctypes.sizeof(api["THREADENTRY32"])
        pid = os.getpid()
        count = 0
        ok = k32.Thread32First(snapshot, ctypes.byref(entry))
        while ok:
            if int(entry.th32OwnerProcessID) == pid:
                count += 1
            ok = k32.Thread32Next(snapshot, ctypes.byref(entry))
        return count
    finally:
        k32.CloseHandle(snapshot)


def _win_per_core_cpu_times() -> list[tuple[float, float]]:
    """NtQuerySystemInformation(SystemProcessorPerformanceInformation) → 各核 (idle, total) 秒。

    ``KernelTime`` 已含 idle，故 total = kernel + user。
    """
    api = _win32()
    if not api:
        return []
    ntdll = api["ntdll"]
    cores = os.cpu_count() or 1
    struct_size = ctypes.sizeof(api["SPPI"])
    size = cores * struct_size
    for _ in range(3):
        buffer = ctypes.create_string_buffer(size)
        status = ntdll.NtQuerySystemInformation(8, buffer, size, None)
        code = int(status) & 0xFFFFFFFF
        if code == 0x00000000:
            array = (api["SPPI"] * cores).from_buffer(buffer)
            return [
                (array[i].IdleTime / 1e7, (array[i].KernelTime + array[i].UserTime) / 1e7)
                for i in range(cores)
            ]
        if code == 0xC0000004:  # STATUS_INFO_LENGTH_MISMATCH：缓冲区不够，加倍重试
            size *= 2
            continue
        return []
    return []


def _win_host_cpu_times() -> tuple[tuple[float, float] | None, list[tuple[float, float]]]:
    """Windows 主机 CPU 累计时间：各核优先，总利用率由各核求和。"""
    cores = _win_per_core_cpu_times()
    if cores:
        return (sum(item[0] for item in cores), sum(item[1] for item in cores)), cores
    api = _win32()
    if not api:
        return None, []
    ft = api["FILETIME"]
    idle, kernel, user = ft(), ft(), ft()
    if not api["k32"].GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        return None, []
    idle_s = _filetime_seconds(idle)
    # GetSystemTimes: kernel 时间**含** idle
    return (idle_s, _filetime_seconds(kernel) + _filetime_seconds(user)), []


def _win_memory() -> dict[str, Any] | None:
    api = _win32()
    if not api:
        return None
    status = api["MEMORYSTATUSEX"]()
    status.dwLength = ctypes.sizeof(api["MEMORYSTATUSEX"])
    if not api["k32"].GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    total = int(status.ullTotalPhys)
    available = int(status.ullAvailPhys)
    swap_total = int(status.ullTotalPageFile)
    swap_available = int(status.ullAvailPageFile)
    # PageFile 口径包含物理内存，扣掉才是「交换文件」部分
    swap_used = max(0, (swap_total - swap_available) - max(0, total - available))
    return {
        "total_bytes": total,
        "available_bytes": available,
        "used_bytes": max(0, total - available),
        "percent": round(float(status.dwMemoryLoad), 2),
        "swap_total_bytes": swap_total,
        "swap_free_bytes": swap_available,
        "swap_used_bytes": swap_used,
    }


# ==========================================================================
# 3. Linux /proc 辅助
# ==========================================================================

def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _clock_ticks() -> float:
    try:
        return float(os.sysconf("SC_CLK_TCK")) or 100.0
    except (AttributeError, ValueError, OSError):
        return 100.0


def _page_size() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) or 4096
    except (AttributeError, ValueError, OSError):
        return 4096


def _linux_process_stats() -> dict[str, Any]:
    """Linux 进程级：RSS / CPU 时间 / 线程数 / fd 数 / 启动时刻。"""
    out: dict[str, Any] = {}
    hz = _clock_ticks()

    stat = parse_proc_self_stat(_read_text("/proc/self/stat"))
    if "cpu_jiffies" in stat:
        out["cpu_seconds"] = stat["cpu_jiffies"] / hz
    if "threads" in stat:
        out["threads"] = stat["threads"]

    statm = _read_text("/proc/self/statm").split()
    if len(statm) >= 2:
        try:
            out["rss_bytes"] = int(statm[1]) * _page_size()
        except ValueError:
            pass
    if "rss_bytes" not in out:
        status = parse_proc_meminfo(_read_text("/proc/self/status"))
        if "VmRSS" in status:
            out["rss_bytes"] = status["VmRSS"]

    try:
        out["open_fds"] = len(os.listdir("/proc/self/fd"))
    except OSError:
        pass

    if "starttime_jiffies" in stat:
        uptime = parse_proc_uptime(_read_text("/proc/uptime"))
        if uptime is not None:
            out["start_time"] = time.time() - uptime + stat["starttime_jiffies"] / hz
    if "threads" not in out:
        try:
            out["threads"] = len(os.listdir("/proc/self/task"))
        except OSError:
            pass
    return out


def _linux_host_cpu() -> tuple[tuple[float, float] | None, list[tuple[float, float]]]:
    parsed = parse_proc_stat(_read_text("/proc/stat"))
    return parsed.get("total"), list(parsed.get("cores") or [])


def _linux_memory() -> dict[str, Any] | None:
    info = parse_proc_meminfo(_read_text("/proc/meminfo"))
    total = info.get("MemTotal")
    if not total:
        return None
    available = info.get("MemAvailable")
    if available is None:
        available = (info.get("MemFree", 0) + info.get("Buffers", 0)
                     + info.get("Cached", 0) + info.get("SReclaimable", 0))
    used = max(0, int(total) - int(available))
    swap_total = int(info.get("SwapTotal", 0))
    swap_free = int(info.get("SwapFree", 0))
    return {
        "total_bytes": int(total),
        "available_bytes": int(available),
        "used_bytes": used,
        "percent": _round(used / int(total) * 100, 2),
        "swap_total_bytes": swap_total,
        "swap_free_bytes": swap_free,
        "swap_used_bytes": max(0, swap_total - swap_free),
    }


# ==========================================================================
# 4. CPU 利用率（需要两次采样求差）
# ==========================================================================

_CPU_LOCK = threading.Lock()
_HOST_CPU_STATE: dict[str, Any] = {"total": None, "cores": []}
_PROC_CPU_STATE: dict[str, Any] = {"cpu_seconds": None, "wall": None}


def _cpu_percent(previous: Sequence[float] | None, current: Sequence[float]) -> float:
    """从 (idle, total) 累计值求利用率百分比。"""
    if not previous or len(previous) < 2 or len(current) < 2:
        return 0.0
    delta_total = float(current[1]) - float(previous[1])
    delta_idle = float(current[0]) - float(previous[0])
    if delta_total <= 0:
        return 0.0
    percent = (delta_total - delta_idle) / delta_total * 100.0
    return round(min(100.0, max(0.0, percent)), 2)


def _host_cpu_percent(
    total: Sequence[float] | None,
    cores: Sequence[Sequence[float]],
) -> tuple[float | None, list[float], str]:
    """主机 CPU 利用率：首轮返回「自开机以来平均」，之后返回两次采样区间值。"""
    if total is None:
        return None, [], "unavailable"
    source = "interval"
    with _CPU_LOCK:
        previous_total = _HOST_CPU_STATE.get("total")
        previous_cores = list(_HOST_CPU_STATE.get("cores") or [])
        _HOST_CPU_STATE["total"] = tuple(total)
        _HOST_CPU_STATE["cores"] = [tuple(item) for item in cores]

    if previous_total is not None:
        overall = _cpu_percent(previous_total, total)
    else:
        source = "since_boot"
        overall = _cpu_percent((0.0, 0.0), total)

    per_core: list[float] = []
    for index, current_core in enumerate(cores):
        if previous_total is not None and index < len(previous_cores):
            per_core.append(_cpu_percent(previous_cores[index], current_core))
        else:
            per_core.append(_cpu_percent((0.0, 0.0), current_core))
    return overall, per_core, source


def _process_cpu_percent(cpu_seconds: float | None, uptime_seconds: float | None) -> tuple[float | None, str]:
    """进程 CPU 百分比（相对单核，可 >100）。

    首次调用没有基线，返回「自进程启动以来的平均值」，之后返回两次采样区间值。
    """
    if cpu_seconds is None:
        return None, "unavailable"
    wall = time.monotonic()
    with _CPU_LOCK:
        previous_cpu = _PROC_CPU_STATE.get("cpu_seconds")
        previous_wall = _PROC_CPU_STATE.get("wall")
        _PROC_CPU_STATE["cpu_seconds"] = float(cpu_seconds)
        _PROC_CPU_STATE["wall"] = wall
    if previous_cpu is not None and previous_wall is not None:
        delta_wall = wall - float(previous_wall)
        if delta_wall > 1e-3:
            percent = (float(cpu_seconds) - float(previous_cpu)) / delta_wall * 100.0
            return round(max(0.0, percent), 2), "interval"
    if uptime_seconds and uptime_seconds > 0:
        return round(max(0.0, float(cpu_seconds) / uptime_seconds * 100.0), 2), "since_start"
    return 0.0, "since_start"


# ==========================================================================
# 5. 进程级采集
# ==========================================================================

def collect_process(*, config: RagConfig | None = None) -> dict[str, Any]:
    """① 进程级：RSS / CPU% / 线程数 / 打开句柄数 / 运行时长 / Python 版本。

    **取数顺序（2026-09-29 调整）**：``psutil`` **优先**，不可用时回落纯标准库路径。

    为什么改成 psutil 优先：标准库路径受操作系统计数器精度限制
    （Windows 的 ``GetProcessTimes`` 是 **100ms 时间片**，实测 ``cpu_seconds``
    与 psutil 差 **83%**），而 psutil 走的是各平台更精确的原生接口。三套实现
    （Linux /proc、Windows win32、psutil）同时维护还会带来**口径不一致**——
    统一到 psutil 后，同一份代码在 Linux 与 Windows 上取数口径一致，对比才有意义。

    ``source`` 字段仍然如实输出（``"psutil"`` / ``"stdlib"``），供 ``/health`` 区分。
    """
    info: dict[str, Any] = {
        "pid": os.getpid(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "source": "psutil",
        "status": "ok",
        "errors": [],
    }

    # ---- 首选：psutil（一次拿全，无平台分支）----
    raw: dict[str, Any] = {}
    try:
        raw = _process_from_psutil()
    except Exception as exc:
        info["errors"].append(f"psutil: {_error_text(exc)}")
        _LOGGER.debug("进程级采集（psutil 路径）失败", exc_info=True)

    # ---- 回落：纯标准库（无 psutil 的环境）----
    if not raw:
        info["source"] = "stdlib"
        try:
            if _IS_LINUX:
                raw = _linux_process_stats()
            elif _IS_WINDOWS:
                raw = _windows_process_stats()
        except Exception as exc:
            info["errors"].append(f"stdlib: {_error_text(exc)}")
            _LOGGER.debug("进程级采集（标准库路径）失败", exc_info=True)
        missing = [key for key in ("rss_bytes", "cpu_seconds", "threads", "open_fds",
                                   "start_time") if raw.get(key) is None]
        if missing:
            _log_degraded_once(
                "process-stdlib-gap",
                "进程级指标部分字段在纯标准库路径下不可用（%s）；"
                "装上 psutil 可拿到全部字段（pip install psutil）",
                ",".join(sorted(missing)),
            )

    uptime = None
    start_time = raw.get("start_time")
    if start_time:
        uptime = max(0.0, time.time() - float(start_time))
    elif _MODULE_START:
        uptime = max(0.0, time.time() - _MODULE_START)

    cpu_percent, cpu_basis = _process_cpu_percent(raw.get("cpu_seconds"), uptime)
    info.update({
        "rss_bytes": raw.get("rss_bytes"),
        "cpu_percent": cpu_percent,
        "cpu_percent_basis": cpu_basis,
        "cpu_seconds": _round(raw.get("cpu_seconds"), 4),
        "threads": raw.get("threads"),
        "open_fds": raw.get("open_fds"),
        "uptime_seconds": _round(uptime, 3),
        "start_time": _round(start_time, 3),
    })
    info["status"] = "ok" if info["rss_bytes"] is not None else "degraded"
    return info


_MODULE_START = time.time()


def _windows_process_stats() -> dict[str, Any]:
    api = _win32()
    if not api:
        return {}
    out: dict[str, Any] = {}
    handle = api["k32"].GetCurrentProcess()
    counters = api["PROCESS_MEMORY_COUNTERS"]()
    counters.cb = ctypes.sizeof(api["PROCESS_MEMORY_COUNTERS"])
    if api["psapi"].GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        out["rss_bytes"] = int(counters.WorkingSetSize)
    counter = api["wt"].DWORD()
    if api["k32"].GetProcessHandleCount(handle, ctypes.byref(counter)):
        out["open_fds"] = int(counter.value)
    created, exited, kernel, user = (api["FILETIME"]() for _ in range(4))
    if api["k32"].GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                 ctypes.byref(kernel), ctypes.byref(user)):
        out["start_time"] = _filetime_to_unix(created)
        out["cpu_seconds"] = _filetime_seconds(kernel) + _filetime_seconds(user)
    threads = _win_thread_count()
    if threads:
        out["threads"] = threads
    return out


def _process_from_psutil() -> dict[str, Any]:
    """**主路径**：用 psutil 一次拿全进程级字段（psutil 缺失时返回空 dict）。

    返回的键与标准库路径**完全一致**（`rss_bytes` / `cpu_seconds` / `threads` /
    `open_fds` / `start_time`），这样两条路径可以互换而不影响上层契约。

    为什么以它为主：一次调用拿到全部字段、无平台分支，且精度高于手写 win32
    （`GetProcessTimes` 只有 100ms 时间片）。`open_fds` 在 Windows 上对应
    `num_handles()`（句柄数），语义与 Linux 的 `num_fds()`（文件描述符）对齐。
    """
    psutil = _optional_import("psutil")
    if psutil is None:
        return {}
    process = psutil.Process(os.getpid())
    out: dict[str, Any] = {}
    try:
        out["rss_bytes"] = int(process.memory_info().rss)
    except Exception:  # noqa: BLE001 - 单个字段失败不影响其余
        _LOGGER.debug("psutil memory_info 失败", exc_info=True)
    try:
        times = process.cpu_times()
        out["cpu_seconds"] = float(times.user) + float(times.system)
    except Exception:  # noqa: BLE001
        _LOGGER.debug("psutil cpu_times 失败", exc_info=True)
    try:
        out["threads"] = int(process.num_threads())
    except Exception:  # noqa: BLE001
        _LOGGER.debug("psutil num_threads 失败", exc_info=True)
    try:
        # Linux/macOS 用 num_fds；Windows 只有 num_handles（句柄）
        out["open_fds"] = int(process.num_fds())
    except (AttributeError, NotImplementedError):
        try:
            out["open_fds"] = int(process.num_handles())
        except Exception:  # noqa: BLE001
            _LOGGER.debug("psutil 句柄数不可用", exc_info=True)
    except Exception:  # noqa: BLE001
        _LOGGER.debug("psutil num_fds 失败", exc_info=True)
    try:
        out["start_time"] = float(process.create_time())
    except Exception:  # noqa: BLE001
        _LOGGER.debug("psutil create_time 失败", exc_info=True)
    return out


# ==========================================================================
# 6. 主机级采集
# ==========================================================================

#: psutil 的 CPU 采样基线是否已预热（首次 ``cpu_percent(interval=None)`` 必然返回 0.0）
_PSUTIL_CPU_PRIMED = False
_PSUTIL_PRIME_LOCK = threading.Lock()


def _prime_psutil_cpu(psutil: Any) -> None:
    """**预热 psutil 的 CPU 采样基线**（只做一次，进程内幂等）。

    为什么必须做：``psutil.cpu_percent(interval=None)`` 是**两次采样求差**的语义，
    **首次调用必然返回 0.0**；而 ``percpu=True`` 的首次调用会顺手初始化它自己的
    基线并返回**全系统**值。实测后果：进程启动后的**第一个**采集点会报出
    脏读数（实测 ``cpu_percent=100.0`` 而各核全是 ``0.0``），并且这个脏值会被
    ``publish_metrics`` 发布成 Gauge、被 ``/health`` 读走。

    修法：在真正取值**之前**先空调一次，把基线建立起来。失败不影响业务
    （``psutil`` 本身不可用时上层已经回落标准库路径）。
    """
    global _PSUTIL_CPU_PRIMED
    if _PSUTIL_CPU_PRIMED:
        return
    with _PSUTIL_PRIME_LOCK:
        if _PSUTIL_CPU_PRIMED:
            return
        try:
            psutil.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None, percpu=True)
        except Exception:  # noqa: BLE001 - 预热失败只是读数可能偏脏，不该影响采集
            _LOGGER.debug("psutil CPU 基线预热失败", exc_info=True)
        _PSUTIL_CPU_PRIMED = True


def collect_host(*, config: RagConfig | None = None) -> dict[str, Any]:
    """② 主机级：CPU 总/各核、内存、swap、根分区磁盘、loadavg。

    **取数顺序（2026-09-29 调整）**：``psutil`` **优先**，不可用时回落纯标准库。

    统一到 psutil 的理由与 :func:`collect_process` 相同：三套实现（Linux /proc、
    Windows win32、psutil）会带来**口径不一致**——实测 ``swap.total_bytes``
    在 Windows 上两套差 **33%**（标准库取 PageFile，含物理内存；psutil 取真实 swap）。
    ``cpu_source`` 字段如实输出 ``"psutil"`` / ``"stdlib"``。
    """
    info: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "cpu_source": "psutil",
        "cpu_percent": None,
        "cpu_per_core": [],
        "cpu_percent_basis": "unavailable",
        "memory": None,
        "swap": None,
        "disk": None,
        "load": None,
        "status": "ok",
        "errors": [],
    }

    # ---- 首选：psutil（内存 / swap / 磁盘 / CPU / loadavg 一次拿全）----
    psutil = _optional_import("psutil")
    if psutil is not None:
        _prime_psutil_cpu(psutil)  # 先建基线，否则首次读数偏脏（见该函数说明）
        try:
            info["cpu_percent"] = float(psutil.cpu_percent(interval=None))
            info["cpu_per_core"] = [round(float(v), 2)
                                    for v in psutil.cpu_percent(interval=None, percpu=True)]
            info["cpu_percent_basis"] = "interval"
            info["cpu_count"] = psutil.cpu_count(logical=True) or info["cpu_count"]
        except Exception as exc:
            info["errors"].append(f"cpu(psutil): {_error_text(exc)}")
        try:
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()
            info["memory"] = {
                "total_bytes": int(vm.total),
                "available_bytes": int(vm.available),
                "used_bytes": int(vm.used),
                "percent": round(float(vm.percent), 2),
            }
            info["swap"] = {
                "total_bytes": int(swap.total),
                "free_bytes": int(swap.free),
                "used_bytes": int(swap.used),
            }
        except Exception as exc:
            info["errors"].append(f"memory(psutil): {_error_text(exc)}")
        try:
            info["load"] = _load_from_psutil(psutil)
        except Exception:  # noqa: BLE001 - loadavg 在 Windows 上本就不可用
            info["load"] = None

    # ---- 回落：纯标准库 ----
    if info["cpu_percent"] is None:
        info["cpu_source"] = "stdlib"
        total_times: tuple[float, float] | None = None
        core_times: list[tuple[float, float]] = []
        try:
            if _IS_LINUX:
                total_times, core_times = _linux_host_cpu()
            elif _IS_WINDOWS:
                total_times, core_times = _win_host_cpu_times()
        except Exception as exc:
            info["errors"].append(f"cpu: {_error_text(exc)}")
            _LOGGER.debug("主机 CPU 采集失败", exc_info=True)
        if total_times is not None:
            overall, per_core, basis = _host_cpu_percent(total_times, core_times)
            info["cpu_percent"] = overall
            info["cpu_per_core"] = per_core
            info["cpu_percent_basis"] = basis
            if not per_core:
                _log_degraded_once(
                    "host-percore",
                    "主机各核 CPU 利用率不可用（当前平台标准库路径不提供各核累计时间），"
                    "已降级为仅总量；装上 psutil 可拿到各核",
                )

    if info["memory"] is None:
        memory = None
        try:
            memory = _linux_memory() if _IS_LINUX else (_win_memory() if _IS_WINDOWS else None)
        except Exception as exc:
            info["errors"].append(f"memory: {_error_text(exc)}")
            _LOGGER.debug("主机内存采集失败", exc_info=True)
        if memory:
            info["memory"] = {key: value for key, value in memory.items()
                              if not key.startswith("swap_")}
            info["swap"] = {key.replace("swap_", ""): value for key, value in memory.items()
                            if key.startswith("swap_")}

    if info["load"] is None:
        if _IS_LINUX:
            info["load"] = parse_loadavg(_read_text("/proc/loadavg"))

    try:
        info["disk"] = _disk_usage()
    except Exception as exc:
        info["errors"].append(f"disk: {_error_text(exc)}")

    if info["cpu_percent"] is None and info["memory"] is None:
        info["status"] = "degraded"
    return info


def _load_from_psutil(psutil: Any) -> dict[str, float] | None:
    """``psutil.getloadavg()`` → 与 ``/proc/loadavg`` 同口径的 dict（不可用返回 None）。"""
    try:
        first, second, third = psutil.getloadavg()
    except (AttributeError, NotImplementedError, OSError):
        return None
    return {"load1": round(first, 2), "load5": round(second, 2), "load15": round(third, 2)}


def _disk_root() -> str:
    """根分区：Windows 用 ``C:\\``（取 SystemDrive，写死 C 会是错的），POSIX 用 ``/``。"""
    if _IS_WINDOWS:
        drive = os.environ.get("SystemDrive") or "C:"
        return drive + os.sep
    return "/"


def _disk_usage() -> dict[str, Any]:
    root = _disk_root()
    usage = shutil.disk_usage(root)
    total = int(usage.total)
    used = int(usage.used)
    return {
        "path": root,
        "total_bytes": total,
        "used_bytes": used,
        "free_bytes": int(usage.free),
        "used_percent": _round(used / total * 100, 2) if total else None,
    }


# ==========================================================================
# 7. GPU / 显存
# ==========================================================================

def _gpu_via_pynvml() -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    """pynvml（nvidia-ml-py）可选路径：无 nvidia-smi 时使用。"""
    nvml = _optional_import("pynvml")
    if nvml is None:
        return [], {}, "pynvml 不可用"
    try:
        nvml.nvmlInit()
    except Exception as exc:
        return [], {}, f"nvmlInit 失败：{_error_text(exc)}"
    devices: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"driver_version": None, "cuda_version": None}
    try:
        try:
            meta["driver_version"] = _truncate(nvml.nvmlSystemGetDriverVersion())
        except Exception:
            meta["driver_version"] = None
        try:
            meta["cuda_version"] = str(nvml.nvmlSystemGetCudaDriverVersion_v2())
        except Exception:
            meta["cuda_version"] = None
        for index in range(int(nvml.nvmlDeviceGetCount())):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            name = _truncate(nvml.nvmlDeviceGetName(handle))
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                utilization = float(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            except Exception:
                utilization = None
            try:
                temperature = float(nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU))
            except Exception:
                temperature = None
            try:
                power = float(nvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
            except Exception:
                power = None
            devices.append({
                "index": index,
                "name": name,
                "utilization_percent": _round(utilization),
                "memory_total_bytes": int(memory.total),
                "memory_used_bytes": int(memory.used),
                "temperature_celsius": _round(temperature),
                "power_watts": _round(power),
                "driver_version": meta.get("driver_version"),
            })
    except Exception as exc:
        return devices, meta, f"pynvml 读取失败：{_error_text(exc)}"
    finally:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass
    return devices, meta, None


def collect_gpu(*, config: RagConfig | None = None, timeout: float = 8.0) -> dict[str, Any]:
    """③ GPU/显存：有卡采真值，无卡/失败 → ``available=False`` + reason（绝不抛异常）。"""
    cfg = _load_config(config)
    result: dict[str, Any] = {
        "available": False,
        "source": None,
        "reason": None,
        "driver_version": None,
        "cuda_version": None,
        "devices": [],
        "error": None,
    }
    if not bool(getattr(cfg.logging, "sys_gpu_enabled", True)):
        result["reason"] = "SYS_GPU_ENABLED=0：GPU 采集已被配置关闭"
        return result

    # ---- 主路径：NVML（nvidia-ml-py，2026-09-29 起为 requirements.txt 主依赖）----
    # 为什么用它取代 nvidia-smi 子进程解析：
    #   ① 无进程开销——`nvidia-smi` 每次采集要 fork/exec 一个二进制（本模块原实现
    #      还带 300 秒的 meta 缓存来缓解，说明这个开销是被感知到的）；
    #   ② 字段更全（温度/功耗/显存/驱动/CUDA 版本一次拿全），不需要解析 CSV 文本；
    #   ③ 少一个"解析层"要维护：原先 `parse_gpu_csv` / `parse_smi_number` /
    #      `smi_memory_bytes` 三个函数共 67 行只为解析 nvidia-smi 的文本输出。
    # 失败时仍给出可读原因（这是既有的降级契约：绝不静默、绝不抛异常）。
    devices, meta, nvml_error = _gpu_via_pynvml()
    if devices:
        result.update({
            "available": True,
            "source": "pynvml",
            "reason": None,
            "devices": devices,
            "driver_version": meta.get("driver_version") or devices[0].get("driver_version"),
            "cuda_version": meta.get("cuda_version"),
        })
        return result

    # ---- 兜底：Linux 上再给一条更具体的原因（无卡机器常见）----
    gpu_hint = _no_gpu_hint()
    parts = [nvml_error or "无可用 GPU 采集通道"]
    if gpu_hint:
        parts.append(gpu_hint)
    result["reason"] = _truncate("；".join(parts))
    result["error"] = result["reason"]
    _log_degraded_once(
        "gpu-unavailable",
        "GPU 指标不可用（优雅降级为 available=false，不抛异常）：%s", result["reason"])
    return result


def _no_gpu_hint() -> str | None:
    """无卡机器上给出更具体的原因（VM 上 ``/dev/nvidia*`` 不存在 / 无 PCI 设备）。"""
    if not _IS_LINUX:
        return None
    try:
        if not any(Path("/dev").glob("nvidia*")):
            return "且 /dev/nvidia* 不存在（该机无 NVIDIA 设备）"
    except OSError:
        return None
    return None


# ==========================================================================
# 8. 依赖健康与延迟（全部只读探针）
# ==========================================================================

def _probe_result(service: str, up: bool, latency_ms: float | None = None,
                  reason: str | None = None, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "service": service,
        "up": bool(up),
        "latency_ms": _round(latency_ms, 3),
        "latency_seconds": _round((latency_ms or 0) / 1000.0, 6) if latency_ms is not None else None,
        "reason": reason,
        "checked_at": round(time.time(), 3),
    }
    payload.update(extra)
    return payload


def probe_redis(cfg: RagConfig, *, timeout: float | None = None) -> dict[str, Any]:
    r"""Redis 连通性：**只发 PING**（绝不 FLUSH、绝不 ``KEYS *``），测往返延迟。"""
    seconds = _redis_probe_timeout(cfg, timeout)
    url = str(getattr(cfg, "redis_url", "") or "")
    endpoint = _mask_url(url)
    redis_lib = _optional_import("redis")
    if redis_lib is not None:
        client = None
        try:
            client = redis_lib.Redis.from_url(
                url, socket_connect_timeout=seconds, socket_timeout=seconds,
                decode_responses=True)
            started = time.perf_counter()
            pong = client.ping()
            latency_ms = (time.perf_counter() - started) * 1000.0
            return _probe_result("redis", bool(pong), latency_ms, None,
                                 endpoint=endpoint, driver="redis-py", pong=bool(pong))
        except Exception as exc:
            return _probe_result("redis", False, None, _error_text(exc),
                                 endpoint=endpoint, driver="redis-py")
        finally:
            _close_quietly(client)
    # 纯标准库兜底：自己发 RESP PING
    try:
        parsed = _parse_redis_url(url)
        started = time.perf_counter()
        with socket.create_connection((parsed["host"], parsed["port"]), timeout=seconds) as sock:
            sock.settimeout(seconds)
            sock.sendall(b"*1\r\n$4\r\nPING\r\n")
            data = sock.recv(64)
        latency_ms = (time.perf_counter() - started) * 1000.0
        up = data.startswith(b"+PONG")
        return _probe_result("redis", up, latency_ms,
                             None if up else f"非预期响应：{data[:32]!r}",
                             endpoint=endpoint, driver="socket")
    except Exception as exc:
        return _probe_result("redis", False, None, _error_text(exc),
                             endpoint=endpoint, driver="socket")


def _parse_redis_url(url: str) -> dict[str, Any]:
    """``redis://[:pwd@]host:port/db`` → host/port/db（不返回口令）。"""
    from urllib.parse import urlsplit
    parts = urlsplit(url if "://" in url else f"redis://{url}")
    if parts.scheme not in {"redis", "rediss"}:
        raise ValueError(f"不支持的 Redis URL scheme：{parts.scheme!r}")
    if (parts.scheme == "rediss"):
        raise ValueError("rediss:// 需要 redis-py 支持，纯标准库路径不支持 TLS")
    return {
        "host": parts.hostname or "127.0.0.1",
        "port": int(parts.port or 6379),
        "db": int((parts.path or "/0").lstrip("/") or 0),
        "has_password": parts.password is not None,
    }


def _close_quietly(client: Any) -> None:
    if client is None:
        return
    for method in ("close", "aclose", "disconnect"):
        close = getattr(client, method, None)
        if callable(close):
            try:
                close()
                return
            except Exception:
                continue
    pool = getattr(client, "connection_pool", None)
    disconnect = getattr(pool, "disconnect", None)
    if callable(disconnect):
        try:
            disconnect()
        except Exception:
            pass


def probe_milvus(cfg: RagConfig, *, timeout: float | None = None) -> dict[str, Any]:
    """Milvus 连通性：一次轻量 gRPC 调用 ``list_collections``（只读）。"""
    seconds = _probe_timeouts(cfg, timeout)["milvus"]
    uri = str(getattr(cfg, "milvus_uri", "") or "")
    pymilvus = _optional_import("pymilvus")
    if pymilvus is None:
        return _probe_result("milvus", False, None, "pymilvus 未安装（requirements.txt 硬依赖）",
                             endpoint=uri)
    client = None
    try:
        kwargs: dict[str, Any] = {"uri": uri, "timeout": seconds}
        token = str(getattr(cfg.milvus, "token", "") or "")
        if token:
            kwargs["token"] = token
        connect_started = time.perf_counter()
        client = pymilvus.MilvusClient(**kwargs)
        connect_ms = (time.perf_counter() - connect_started) * 1000.0
        started = time.perf_counter()
        collections = client.list_collections()
        latency_ms = (time.perf_counter() - started) * 1000.0
        return _probe_result("milvus", True, latency_ms, None, endpoint=uri,
                             connect_ms=_round(connect_ms, 3),
                             collection_count=len(list(collections or [])))
    except Exception as exc:
        return _probe_result("milvus", False, None, _error_text(exc), endpoint=uri)
    finally:
        _close_quietly(client)


def _http_get_json(url: str, *, timeout: float, api_key: str = "") -> tuple[Any, float]:
    """极简 HTTP GET（纯标准库）；返回 (JSON, 耗时秒)。"""
    request = urllib.request.Request(url, method="GET")
    request.add_header("Accept", "application/json")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    elapsed = time.perf_counter() - started
    import json as _json
    return _json.loads(body.decode("utf-8", "replace") or "{}"), elapsed


def probe_ollama(cfg: RagConfig, *, timeout: float | None = None) -> dict[str, Any]:
    """Ollama 连通性：``GET /api/version`` 往返延迟（只读）。"""
    seconds = _probe_timeouts(cfg, timeout)["ollama"]
    base = str(getattr(cfg.ollama, "base_url", "") or "").rstrip("/")
    url = f"{base}/api/version"
    api_key = str(getattr(cfg.ollama, "api_key", "") or "")
    try:
        payload, elapsed = _http_get_json(url, timeout=seconds, api_key=api_key)
        version = payload.get("version") if isinstance(payload, Mapping) else None
        return _probe_result("ollama", True, elapsed * 1000.0, None,
                             endpoint=url, version=version)
    except Exception as exc:
        return _probe_result("ollama", False, None, _error_text(exc), endpoint=url)


def collect_dependencies(cfg: RagConfig | None = None, *,
                         timeout: float | None = None) -> dict[str, Any]:
    """④ Redis / Milvus / Ollama 三件套的 up + 往返延迟 + 失败原因。"""
    config = _load_config(cfg)
    result: dict[str, Any] = {}
    probes: tuple[tuple[str, Callable[..., dict[str, Any]]], ...] = (
        ("redis", probe_redis), ("milvus", probe_milvus), ("ollama", probe_ollama),
    )
    for name, probe in probes:
        try:
            result[name] = probe(config, timeout=timeout)
        except Exception as exc:  # 探针自身异常也要变成 up=False，不能外溢
            _LOGGER.warning("依赖探针 %s 异常（已降级为 down）：%s", name, _error_text(exc))
            result[name] = _probe_result(name, False, None, _error_text(exc))
    return result


# ==========================================================================
# 9. Ollama 模型驻留（/api/ps）
# ==========================================================================

def collect_ollama_models(cfg: RagConfig | None = None, *,
                          timeout: float | None = None) -> dict[str, Any]:
    """⑤ ``GET /api/ps``：已加载模型名 / size / size_vram / expires_at。

    空闲时 Ollama 会卸载模型并返回 ``{"models": []}`` —— 表达为
    ``loaded_count = 0`` 且 ``status = "idle"``，``error`` 保持 None（**不是错误**）。
    """
    config = _load_config(cfg)
    seconds = _probe_timeouts(config, timeout)["ollama"]
    base = str(getattr(config.ollama, "base_url", "") or "").rstrip("/")
    url = f"{base}/api/ps"
    api_key = str(getattr(config.ollama, "api_key", "") or "")
    result: dict[str, Any] = {
        "endpoint": url,
        "status": "unavailable",
        "up": False,
        "loaded_count": 0,
        "loaded_size_bytes": 0,
        "loaded_vram_bytes": 0,
        "models": [],
        "expires_at_min": None,
        "latency_ms": None,
        "reason": None,
        "error": None,
    }
    try:
        payload, elapsed = _http_get_json(url, timeout=seconds, api_key=api_key)
    except urllib.error.HTTPError as exc:
        result["reason"] = f"HTTP {exc.code}"
        result["error"] = result["reason"]
        return result
    except Exception as exc:
        result["reason"] = _error_text(exc)
        result["error"] = result["reason"]
        return result

    raw_models = payload.get("models") if isinstance(payload, Mapping) else None
    models: list[dict[str, Any]] = []
    for item in (raw_models or []):
        if not isinstance(item, Mapping):
            continue
        name = item.get("name") or item.get("model")
        if not name:
            continue
        size = item.get("size")
        vram = item.get("size_vram")
        models.append({
            "name": str(name),
            "size_bytes": int(size) if isinstance(size, (int, float)) else None,
            "size_vram_bytes": int(vram) if isinstance(vram, (int, float)) else None,
            "expires_at": item.get("expires_at"),
            "details": item.get("details") if isinstance(item.get("details"), Mapping) else None,
        })
    total_size = sum(item["size_bytes"] or 0 for item in models)
    total_vram = sum(item["size_vram_bytes"] or 0 for item in models)
    expiries = sorted(item["expires_at"] for item in models if item.get("expires_at"))
    result.update({
        "status": "loaded" if models else "idle",
        "up": True,
        "loaded_count": len(models),
        "loaded_size_bytes": total_size,
        "loaded_vram_bytes": total_vram,
        "models": models,
        "expires_at_min": expiries[0] if expiries else None,
        "latency_ms": _round(elapsed * 1000.0, 3),
        "reason": None,
        "error": None,
        "idle_note": None if models else "Ollama 当前无驻留模型（空闲卸载属正常状态，不是错误）",
    })
    return result


# ==========================================================================
# 10. 指标注册与发布（t1 的 metrics 注册表）
# ==========================================================================

#: 指标名 → (说明, 单位)。命名与单位遵守 t1 规范（秒=seconds、字节=bytes）。
METRIC_SPECS: dict[str, tuple[str, str]] = {
    # ---- 进程级 ----
    "process_rss_bytes": ("进程常驻内存 RSS", "bytes"),
    "process_cpu_percent": ("进程 CPU 使用率（相对单核，可 >100）", "percent"),
    "process_threads": ("进程线程数", "threads"),
    "process_open_fds": ("进程打开的文件描述符/句柄数", "files"),
    "process_uptime_seconds": ("进程已运行时长", "seconds"),
    # ---- 主机级 ----
    "host_cpu_percent": ("整机 CPU 使用率", "percent"),
    "host_cpu_core_percent": ("各核 CPU 使用率（标签 core）", "percent"),
    "host_cpu_count": ("逻辑 CPU 核数", "cores"),
    "host_memory_total_bytes": ("物理内存总量", "bytes"),
    "host_memory_used_bytes": ("已用内存", "bytes"),
    "host_memory_available_bytes": ("可用内存", "bytes"),
    "host_memory_percent": ("内存使用率", "percent"),
    "host_swap_total_bytes": ("swap/页面文件总量", "bytes"),
    "host_swap_used_bytes": ("已用 swap/页面文件", "bytes"),
    "host_disk_used_percent": ("根分区磁盘使用率", "percent"),
    "host_disk_used_bytes": ("根分区已用空间", "bytes"),
    "host_disk_total_bytes": ("根分区总空间", "bytes"),
    "host_disk_free_bytes": ("根分区可用空间", "bytes"),
    "host_load1": ("1 分钟平均负载", "load"),
    # ---- GPU / 显存 ----
    "gpu_available": ("是否存在可用 GPU（1/0）", "bool"),
    "gpu_utilization_percent": ("GPU 利用率（标签 gpu）", "percent"),
    "gpu_memory_used_bytes": ("GPU 已用显存（标签 gpu）", "bytes"),
    "gpu_memory_total_bytes": ("GPU 显存总量（标签 gpu）", "bytes"),
    "gpu_temperature_celsius": ("GPU 温度（标签 gpu）", "celsius"),
    "gpu_power_watts": ("GPU 功耗（标签 gpu）", "watts"),
    "gpu_info": ("GPU 静态信息（标签 gpu/name/driver_version/cuda_version），恒为 1", "bool"),
    # ---- 依赖健康 ----
    "dep_up": ("依赖服务是否可达（标签 service，1/0）", "bool"),
    "dep_latency_seconds": ("依赖服务往返延迟（标签 service）", "seconds"),
    # ---- Ollama 驻留 ----
    "ollama_loaded_models": ("Ollama 当前驻留模型数（空闲为 0，属正常）", "models"),
    "ollama_loaded_vram_bytes": ("Ollama 驻留模型占用的 VRAM 合计", "bytes"),
    "ollama_loaded_size_bytes": ("Ollama 驻留模型体积合计", "bytes"),
    # ---- t1 METRIC_CATALOG 既有口径的兼容别名（同值，新代码请用上面的名字）----
    "sys_cpu_percent": ("兼容别名：= host_cpu_percent", "percent"),
    "sys_memory_percent": ("兼容别名：= host_memory_percent", "percent"),
    "sys_memory_used_bytes": ("兼容别名：= host_memory_used_bytes", "bytes"),
    "sys_disk_used_percent": ("兼容别名：= host_disk_used_percent", "percent"),
    "sys_load1": ("兼容别名：= host_load1", "load"),
    "sys_gpu_util_percent": ("兼容别名：= gpu_utilization_percent（gpu 0）", "percent"),
    "sys_gpu_memory_used_bytes": ("兼容别名：= gpu_memory_used_bytes（gpu 0）", "bytes"),
    "sys_ollama_up": ("兼容别名：= dep_up{service=ollama}", "bool"),
    "sys_redis_up": ("兼容别名：= dep_up{service=redis}", "bool"),
    "sys_milvus_up": ("兼容别名：= dep_up{service=milvus}", "bool"),
}

_REGISTERED_LOCK = threading.Lock()
_REGISTERED = False


def register_metrics() -> list[str]:
    """把 :data:`METRIC_SPECS` 全部注册进 t1 注册表（幂等）。"""
    global _REGISTERED
    names: list[str] = []
    for name, (description, unit) in METRIC_SPECS.items():
        try:
            M.gauge(name, description, unit)
            names.append(name)
        except Exception as exc:  # 注册失败也不能影响采集
            _LOGGER.warning("指标 %s 注册失败：%s", name, _error_text(exc))
    with _REGISTERED_LOCK:
        _REGISTERED = True
    return names


def _set_gauge(flat: dict[str, float], metric: str, value: Any, **labels: Any) -> None:
    """发布一个 Gauge 值；None 视为「本次没采到」，跳过（不写 0，避免假数据）。

    注意：形参名不能叫 ``name``，否则与 ``gpu_info`` 的 ``name`` 标签冲突
    （``TypeError: got multiple values for argument``）。
    """
    if value is None:
        return
    try:
        number = float(value)
    except (TypeError, ValueError):
        return
    if number != number:  # NaN
        return
    try:
        description, unit = METRIC_SPECS.get(metric) or ("", "count")
        M.gauge(metric, description, unit).set(number, **labels)
    except Exception as exc:
        _LOGGER.warning("指标 %s 发布失败：%s", metric, _error_text(exc))
        return
    if labels:
        suffix = "{" + ",".join(f"{key}={item}" for key, item in sorted(labels.items())) + "}"
        flat[f"{metric}{suffix}"] = number
    else:
        flat[metric] = number


def publish_metrics(snap: Mapping[str, Any]) -> dict[str, float]:
    """把一次 snapshot 的数值发布到 t1 注册表，返回 ``{指标[标签]: 数值}`` 扁平表。"""
    register_metrics()
    flat: dict[str, float] = {}
    process = snap.get("process") or {}
    host = snap.get("host") or {}
    gpu = snap.get("gpu") or {}
    deps = snap.get("dependencies") or {}
    ollama = snap.get("ollama") or {}

    # ---- 进程级 ----
    _set_gauge(flat, "process_rss_bytes", process.get("rss_bytes"))
    _set_gauge(flat, "process_cpu_percent", process.get("cpu_percent"))
    _set_gauge(flat, "process_threads", process.get("threads"))
    _set_gauge(flat, "process_open_fds", process.get("open_fds"))
    _set_gauge(flat, "process_uptime_seconds", process.get("uptime_seconds"))

    # ---- 主机级 ----
    _set_gauge(flat, "host_cpu_percent", host.get("cpu_percent"))
    for index, value in enumerate(host.get("cpu_per_core") or []):
        _set_gauge(flat, "host_cpu_core_percent", value, core=str(index))
    _set_gauge(flat, "host_cpu_count", host.get("cpu_count"))
    memory = host.get("memory") or {}
    _set_gauge(flat, "host_memory_total_bytes", memory.get("total_bytes"))
    _set_gauge(flat, "host_memory_used_bytes", memory.get("used_bytes"))
    _set_gauge(flat, "host_memory_available_bytes", memory.get("available_bytes"))
    _set_gauge(flat, "host_memory_percent", memory.get("percent"))
    swap = host.get("swap") or {}
    _set_gauge(flat, "host_swap_total_bytes", swap.get("total_bytes"))
    _set_gauge(flat, "host_swap_used_bytes", swap.get("used_bytes"))
    disk = host.get("disk") or {}
    _set_gauge(flat, "host_disk_used_percent", disk.get("used_percent"))
    _set_gauge(flat, "host_disk_used_bytes", disk.get("used_bytes"))
    _set_gauge(flat, "host_disk_total_bytes", disk.get("total_bytes"))
    _set_gauge(flat, "host_disk_free_bytes", disk.get("free_bytes"))
    load = host.get("load") or {}
    _set_gauge(flat, "host_load1", load.get("load1"))

    # ---- GPU ----
    _set_gauge(flat, "gpu_available", 1 if gpu.get("available") else 0)
    for device in gpu.get("devices") or []:
        label = str(device.get("index"))
        _set_gauge(flat, "gpu_utilization_percent", device.get("utilization_percent"), gpu=label)
        _set_gauge(flat, "gpu_memory_used_bytes", device.get("memory_used_bytes"), gpu=label)
        _set_gauge(flat, "gpu_memory_total_bytes", device.get("memory_total_bytes"), gpu=label)
        _set_gauge(flat, "gpu_temperature_celsius", device.get("temperature_celsius"), gpu=label)
        _set_gauge(flat, "gpu_power_watts", device.get("power_watts"), gpu=label)
        _set_gauge(flat, "gpu_info", 1, gpu=label,
                   name=device.get("name") or "unknown",
                   driver_version=gpu.get("driver_version") or device.get("driver_version") or "unknown",
                   cuda_version=gpu.get("cuda_version") or "unknown")

    # ---- 依赖健康 ----
    for service, payload in (deps or {}).items():
        if not isinstance(payload, Mapping):
            continue
        _set_gauge(flat, "dep_up", 1 if payload.get("up") else 0, service=str(service))
        _set_gauge(flat, "dep_latency_seconds", payload.get("latency_seconds"), service=str(service))

    # ---- Ollama 驻留 ----
    _set_gauge(flat, "ollama_loaded_models", ollama.get("loaded_count"))
    _set_gauge(flat, "ollama_loaded_vram_bytes", ollama.get("loaded_vram_bytes"))
    _set_gauge(flat, "ollama_loaded_size_bytes", ollama.get("loaded_size_bytes"))

    # ---- t1 既有口径的兼容别名 ----
    _set_gauge(flat, "sys_cpu_percent", host.get("cpu_percent"))
    _set_gauge(flat, "sys_memory_percent", (host.get("memory") or {}).get("percent"))
    _set_gauge(flat, "sys_memory_used_bytes", (host.get("memory") or {}).get("used_bytes"))
    _set_gauge(flat, "sys_disk_used_percent", (host.get("disk") or {}).get("used_percent"))
    _set_gauge(flat, "sys_load1", (host.get("load") or {}).get("load1"))
    first_gpu = (gpu.get("devices") or [{}])[0] if gpu.get("devices") else None
    if first_gpu:
        _set_gauge(flat, "sys_gpu_util_percent", first_gpu.get("utilization_percent"))
        _set_gauge(flat, "sys_gpu_memory_used_bytes", first_gpu.get("memory_used_bytes"))
    for service in ("redis", "milvus", "ollama"):
        payload = (deps or {}).get(service)
        if isinstance(payload, Mapping):
            _set_gauge(flat, f"sys_{service}_up", 1 if payload.get("up") else 0)
    return flat


# ==========================================================================
# 11. 一次性快照
# ==========================================================================

_LAST_SNAPSHOT_LOCK = threading.Lock()
_LAST_SNAPSHOT: dict[str, Any] | None = None


def last_snapshot() -> dict[str, Any] | None:
    """最近一次快照（t4 的 /health 可零成本读取；未采样过时为 None）。"""
    with _LAST_SNAPSHOT_LOCK:
        return _LAST_SNAPSHOT


def _section(snap: dict[str, Any], name: str, collector: Callable[[], Any],
             fallback: Mapping[str, Any] | None = None) -> Any:
    """跑一个采集分区：任何异常都记进 errors，绝不让 snapshot 整体失败。"""
    try:
        return collector()
    except Exception as exc:
        detail = _error_text(exc)
        snap.setdefault("errors", []).append({"section": name, "error": detail})
        _LOGGER.warning("[DEGRADED] 采集分区 %s 失败（已跳过）：%s", name, detail, exc_info=True)
        return dict(fallback or {"error": detail, "status": "degraded"})


def snapshot(
    config: RagConfig | None = None,
    *,
    publish: bool = True,
    include_process: bool = True,
    include_host: bool = True,
    include_gpu: bool | None = None,
    include_dependencies: bool = True,
    include_ollama: bool = True,
    timeout: float | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """一次性采集全部运行时指标。

    :param config: 缺省用 ``RagConfig.from_env()``；
    :param publish: 是否把数值发布到 t1 的指标注册表（默认 True，供 /metrics）；
    :param include_gpu: 缺省读 ``config.logging.sys_gpu_enabled``；
    :param timeout: 依赖探针超时上限（秒），缺省按 config 推导并封顶；
    :returns: 纯 JSON 可序列化 dict；**任何环境下都不抛异常**。
    """
    started = time.perf_counter()
    log = logger or _LOGGER
    snap: dict[str, Any] = {
        "ts": round(time.time(), 3),
        "ts_iso": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_ms": None,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "node": platform.node(),
            "python": platform.python_version(),
            "pid": os.getpid(),
        },
        "process": {},
        "host": {},
        "gpu": {},
        "dependencies": {},
        "ollama": {},
        "errors": [],
        "metrics": {},
    }
    cfg: RagConfig | None = None
    try:
        cfg = _load_config(config)
    except Exception as exc:
        snap["errors"].append({"section": "config", "error": _error_text(exc)})

    if include_process:
        snap["process"] = _section(
            snap, "process", lambda: collect_process(config=cfg), {"status": "degraded"})
    if include_host:
        snap["host"] = _section(
            snap, "host", lambda: collect_host(config=cfg), {"status": "degraded"})
    if include_gpu is None:
        include_gpu = bool(getattr(getattr(cfg, "logging", None), "sys_gpu_enabled", True)) if cfg else True
    if include_gpu:
        snap["gpu"] = _section(
            snap, "gpu", lambda: collect_gpu(config=cfg),
            {"available": False, "reason": "采集分区异常", "devices": []})
    if include_dependencies and cfg is not None:
        snap["dependencies"] = _section(
            snap, "dependencies", lambda: collect_dependencies(cfg, timeout=timeout), {})
    if include_ollama and cfg is not None:
        snap["ollama"] = _section(
            snap, "ollama", lambda: collect_ollama_models(cfg, timeout=timeout),
            {"status": "unavailable", "loaded_count": 0, "models": []})

    if publish:
        try:
            snap["metrics"] = publish_metrics(snap)
        except Exception as exc:
            snap["errors"].append({"section": "publish", "error": _error_text(exc)})
            log.warning("指标发布失败（已跳过）：%s", _error_text(exc), exc_info=True)

    snap["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    global _LAST_SNAPSHOT
    with _LAST_SNAPSHOT_LOCK:
        _LAST_SNAPSHOT = snap
    return snap


def summarize(snap: Mapping[str, Any], *, require_gpu: bool = False) -> dict[str, Any]:
    """把 snapshot 压成 ``/health`` 用的紧凑结构（纯函数，不再发起任何探测）。"""
    deps = snap.get("dependencies") or {}
    gpu = snap.get("gpu") or {}
    ollama = snap.get("ollama") or {}
    process = snap.get("process") or {}
    host = snap.get("host") or {}
    failing = [name for name, payload in deps.items()
               if isinstance(payload, Mapping) and not payload.get("up")]
    ok = not failing
    if require_gpu and not gpu.get("available"):
        ok = False
    return {
        "ok": ok,
        "ts": snap.get("ts"),
        "elapsed_ms": snap.get("elapsed_ms"),
        "checks": {
            name: {
                "up": bool(payload.get("up")),
                "latency_ms": payload.get("latency_ms"),
                "reason": payload.get("reason"),
            }
            for name, payload in deps.items() if isinstance(payload, Mapping)
        },
        "gpu": {
            "available": bool(gpu.get("available")),
            "reason": gpu.get("reason"),
            "device_count": len(gpu.get("devices") or []),
            "utilization_percent": ((gpu.get("devices") or [{}])[0] or {}).get("utilization_percent")
            if gpu.get("devices") else None,
            "memory_used_bytes": ((gpu.get("devices") or [{}])[0] or {}).get("memory_used_bytes")
            if gpu.get("devices") else None,
            "memory_total_bytes": ((gpu.get("devices") or [{}])[0] or {}).get("memory_total_bytes")
            if gpu.get("devices") else None,
        },
        "ollama": {
            "up": bool(ollama.get("up")),
            "status": ollama.get("status"),
            "loaded_count": ollama.get("loaded_count"),
            "loaded_vram_bytes": ollama.get("loaded_vram_bytes"),
            "models": [item.get("name") for item in (ollama.get("models") or [])],
            "reason": ollama.get("reason"),
        },
        "process": {
            "rss_bytes": process.get("rss_bytes"),
            "cpu_percent": process.get("cpu_percent"),
            "threads": process.get("threads"),
            "open_fds": process.get("open_fds"),
            "uptime_seconds": process.get("uptime_seconds"),
        },
        "host": {
            "cpu_percent": host.get("cpu_percent"),
            "cpu_count": host.get("cpu_count"),
            "memory_percent": (host.get("memory") or {}).get("percent"),
            "memory_used_bytes": (host.get("memory") or {}).get("used_bytes"),
            "disk_used_percent": (host.get("disk") or {}).get("used_percent"),
        },
        "failures": failing,
        "errors": list(snap.get("errors") or []),
    }


def health(config: RagConfig | None = None, *, timeout: float | None = None,
           require_gpu: bool = False) -> dict[str, Any]:
    """现采一次并返回 ``/health`` 摘要（t4 可直接返回该 dict）。"""
    return summarize(snapshot(config, publish=True, timeout=timeout), require_gpu=require_gpu)


# ==========================================================================
# 12. 后台周期采样（异常自愈）
# ==========================================================================

class PeriodicSampler:
    """守护线程周期采样器。

    设计要点（对应验收「采样器异常自愈、绝不拖垮主流程」）：

    * 每一轮采样整体 try/except，异常只记 warning 并累加失败计数；
    * 连续失败按 ``min(interval * 2**n, 300s)`` 退避，成功后立刻恢复；
    * ``stop()`` 可中断等待，进程退出时 ``atexit`` 自动收尾；
    * ``status()`` 暴露 runs/failures/last_error，方便 /health 与排障。
    """

    def __init__(
        self,
        interval: float = 15.0,
        *,
        config: RagConfig | None = None,
        publish: bool = True,
        logger: logging.Logger | None = None,
        on_sample: Callable[[dict[str, Any]], None] | None = None,
        log_samples: bool = True,
        enabled: bool = True,
    ) -> None:
        self.interval = _positive_float(interval, 15.0) or 15.0
        self.config = _load_config(config)
        self.publish = publish
        self.enabled = bool(enabled)
        self.log = logger or _LOGGER
        self.on_sample = on_sample
        #: 追加的回调（`add_on_sample`）：**必须支持后挂** —— `install_periodic_sampler`
        #: 在实例已运行时直接复用，构造参数里的 `on_sample` 就传不进去了（D4 的
        #: "/health 快照"正需要后挂：采样器常先于 AppState 装好）。
        self._extra_on_sample: list[Callable[[dict[str, Any]], None]] = []
        self.log_samples = log_samples
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.runs = 0
        self.failures = 0
        self.consecutive_failures = 0
        self.skipped = 0
        self.last_error: str | None = None
        self.last_duration_ms: float | None = None
        self.last_run_ts: float | None = None
        self.started_at: float | None = None

    # ---------- 生命周期 ----------
    def start(self) -> "PeriodicSampler":
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self
            if not self.enabled:
                self.log.info("系统指标周期采样未启用（SYS_METRICS_ENABLED=0），跳过启动")
                return self
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="legal-rag-sys-metrics", daemon=True)
            self.started_at = time.time()
            self._thread.start()
            self.log.info(
                "系统指标周期采样已启动：interval=%.1fs publish=%s（守护线程 %s）",
                self.interval, self.publish, self._thread.name)
        return self

    def stop(self, timeout: float = 2.0) -> bool:
        """停止采样；返回线程是否已结束。"""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout)))
        alive = bool(thread is not None and thread.is_alive())
        if not alive:
            self.log.info("系统指标周期采样已停止（runs=%d failures=%d）", self.runs, self.failures)
        return not alive

    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def add_on_sample(self, fn: Callable[[dict[str, Any]], None]) -> None:
        """追加一个每轮采样后的回调（线程安全；实例已在跑也能挂）。

        D4 用它把"刷新 /health 快照"挂到采样器上：采样器一停（``SYS_METRICS_ENABLED=0``），
        快照就不再更新 —— 于是 ``/health`` 必须**回落到现采**并标 ``stale=true``
        （见 `AppState.health_payload`），绝不会把旧数据当新鲜数据发出去。
        """
        with self._lock:
            self._extra_on_sample.append(fn)

    # ---------- 单轮 ----------
    def sample_once(self) -> dict[str, Any] | None:
        """采样一次（同步）；异常不外溢，失败返回 None。"""
        started = time.perf_counter()
        try:
            snap = snapshot(self.config, publish=self.publish)
            self.runs += 1
            self.consecutive_failures = 0
            self.last_duration_ms = round((time.perf_counter() - started) * 1000.0, 3)
            self.last_run_ts = time.time()
            if self.on_sample is not None:
                try:
                    self.on_sample(snap)
                except Exception as exc:
                    self.log.warning("采样回调失败（已忽略）：%s", _error_text(exc))
            for extra in list(self._extra_on_sample):
                # 每个回调**各自兜底**：一个坏回调不许影响采样器，也不许拖累其它回调
                try:
                    extra(snap)
                except Exception as exc:
                    self.log.warning("追加采样回调失败（已忽略）：%s", _error_text(exc))
            if self.log_samples:
                self.log.info("[SYS] %s", self._summary_line(snap))
            return snap
        except Exception as exc:  # 采样失败绝不能影响主流程
            self.failures += 1
            self.consecutive_failures += 1
            self.last_error = _error_text(exc)
            self.last_duration_ms = round((time.perf_counter() - started) * 1000.0, 3)
            self.log.warning("系统指标采样失败（第 %d 次，已自愈继续）：%s",
                             self.consecutive_failures, self.last_error, exc_info=True)
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            delay = self.interval
            if self.consecutive_failures:
                delay = min(self.interval * (2 ** min(self.consecutive_failures, 5)), 300.0)
            if self._stop.wait(delay):
                break

    @staticmethod
    def _summary_line(snap: Mapping[str, Any]) -> str:
        """一行人话摘要（只用 .format()，避免 f-string 跨行语法——Python 3.10 不支持 PEP 701）。"""
        process = snap.get("process") or {}
        host = snap.get("host") or {}
        gpu = snap.get("gpu") or {}
        deps = snap.get("dependencies") or {}
        ollama = snap.get("ollama") or {}
        memory = host.get("memory") or {}
        devices = gpu.get("devices") or []
        dep_text = " ".join(
            "{}={}({}ms)".format(name, "up" if payload.get("up") else "down",
                                 payload.get("latency_ms"))
            for name, payload in deps.items() if isinstance(payload, Mapping)
        )
        parts = [
            "rss={}".format(_human_bytes(process.get("rss_bytes"))),
            "proc_cpu={}%".format(process.get("cpu_percent")),
            "threads={}".format(process.get("threads")),
            "host_cpu={}%".format(host.get("cpu_percent")),
            "mem={}%({}/{})".format(memory.get("percent"),
                                    _human_bytes(memory.get("used_bytes")),
                                    _human_bytes(memory.get("total_bytes"))),
            "disk={}%".format((host.get("disk") or {}).get("used_percent")),
        ]
        if devices:
            device = devices[0]
            parts.append("gpu={} util={}% vram={}/{}".format(
                device.get("name"), device.get("utilization_percent"),
                _human_bytes(device.get("memory_used_bytes")),
                _human_bytes(device.get("memory_total_bytes"))))
        else:
            parts.append("gpu=unavailable({})".format(gpu.get("reason")))
        parts.append("deps[{}]".format(dep_text))
        parts.append("ollama_models={}".format(ollama.get("loaded_count")))
        parts.append("elapsed={}ms".format(snap.get("elapsed_ms")))
        return " ".join(parts)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "running": self.is_running(),
            "interval_seconds": self.interval,
            "publish": self.publish,
            "runs": self.runs,
            "failures": self.failures,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "last_duration_ms": self.last_duration_ms,
            "last_run_ts": self.last_run_ts,
            "started_at": self.started_at,
            "thread": getattr(self._thread, "name", None),
        }


def _human_bytes(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    for unit, scale in (("B", 1), ("KB", 1024), ("MB", 1024 ** 2),
                        ("GB", 1024 ** 3), ("TB", 1024 ** 4)):
        if number < scale * 1024 or unit == "TB":
            return f"{int(number)}B" if scale == 1 else f"{number / scale:.1f}{unit}"
    return f"{int(number)}B"


_SAMPLER_LOCK = threading.RLock()
_SAMPLER: PeriodicSampler | None = None
_ATEXIT_REGISTERED = False


def install_periodic_sampler(
    interval: float | None = None,
    *,
    config: RagConfig | None = None,
    publish: bool = True,
    logger: logging.Logger | None = None,
    on_sample: Callable[[dict[str, Any]], None] | None = None,
    log_samples: bool = True,
) -> PeriodicSampler:
    """安装（并启动）后台周期采样器；重复调用复用已在运行的实例。

    :param interval: 采样间隔秒；缺省取 ``config.logging.sys_metrics_interval``（默认 15s）；
    :param on_sample: 每轮成功后的回调（例如推送到外部监控）；
    :returns: :class:`PeriodicSampler`（``SYS_METRICS_ENABLED=0`` 时返回未启动的实例）。
    """
    global _SAMPLER, _ATEXIT_REGISTERED
    cfg = _load_config(config)
    log = logger or _LOGGER
    enabled = bool(getattr(cfg.logging, "sys_metrics_enabled", True))
    resolved = _positive_float(interval, None)
    if resolved is None:
        resolved = _positive_float(getattr(cfg.logging, "sys_metrics_interval", 15.0), 15.0) or 15.0
        if interval is not None:
            log.warning("采样间隔 %r 非法，退回配置值 %.1fs", interval, resolved)
    with _SAMPLER_LOCK:
        if _SAMPLER is not None and _SAMPLER.is_running():
            log.debug("周期采样器已在运行（interval=%.1fs），复用现有实例", _SAMPLER.interval)
            return _SAMPLER
        sampler = PeriodicSampler(
            resolved, config=cfg, publish=publish, logger=log, on_sample=on_sample,
            log_samples=log_samples, enabled=enabled)
        _SAMPLER = sampler
        sampler.start()
        if enabled and not _ATEXIT_REGISTERED:
            try:
                atexit.register(stop_periodic_sampler)
                _ATEXIT_REGISTERED = True
            except Exception:  # pragma: no cover
                pass
        return sampler


def stop_periodic_sampler(timeout: float = 2.0) -> bool:
    """停止后台采样器（未运行时返回 True）。"""
    with _SAMPLER_LOCK:
        sampler = _SAMPLER
    if sampler is None:
        return True
    return sampler.stop(timeout=timeout)


def current_sampler() -> PeriodicSampler | None:
    """当前采样器实例（未安装时为 None）。"""
    with _SAMPLER_LOCK:
        return _SAMPLER


def sampler_status() -> dict[str, Any]:
    """采样器运行状态；未安装时返回 ``{"installed": False}``。"""
    sampler = current_sampler()
    if sampler is None:
        return {"installed": False, "running": False}
    return {"installed": True, **sampler.status()}


# 模块导入时预注册指标名：让 GET /metrics 在第一次采样前就能看到全部字段口径。
try:  # pragma: no cover - 注册失败不影响采集
    register_metrics()
except Exception:  # pragma: no cover
    _LOGGER.debug("指标预注册失败（延迟到首次快照时再试）", exc_info=True)
