# -*- coding: utf-8 -*-
"""T9 复跑前置检查：**独占 Ollama 条件核实**（不调用模型，零污染）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

为什么必须做（t17 实测结论）：首字验收的**测量前提 = 独占 Ollama**。
    * 独占三臂：max 624.71 / 988.59 / 854.98 ms（无题 >1000 ms）
    * 受控并发（4 路加压）两臂：**10764.76 / 10826.47 ms**
    → 并发条件下测出的首字**不可用于验收**；所以每次正式复跑前必须核实三件事：
        ① `ollama` 进程 4 s 内 CPU 增量 ≈ 0（没有正在推理）；
        ② 本机到 `127.0.0.1:11434` 的已建立连接数 = 0；
        ③ 其它成员的同窗口留痕未在增长（`测试/留痕/`、`部署/日志/` 的最近 mtime 稳定）。
三项全部通过才打印 `EXCLUSIVE_OK=True`；否则给出 `EXCLUSIVE_OK=False` 与原因（调用方据此把首字标注为
「并发口径（不可用于验收）」或稍后重试）。

用法（工作目录 = 工单3）：
    pwsh -NoProfile -File run_py.ps1 优化/脚本/check_exclusive_ollama.py
    pwsh -NoProfile -File run_py.ps1 优化/脚本/check_exclusive_ollama.py --wait-seconds 30
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any, Sequence

# 保留 UTF-8 输出，同时放宽错误策略：Windows 非 UTF-8 控制台/重定向下
# 不会因个别字节无法解码而抛 UnicodeEncodeError（与其他部署脚本同一口径）。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core.logging_conf import get_logger, setup_logging, shutdown_logging  # noqa: E402

WATCH_DIRS = (REPO_ROOT / "测试" / "留痕", REPO_ROOT / "部署" / "日志")


def ollama_cpu_seconds(process_name: str = "ollama") -> float | None:
    """取 ``ollama`` 进程的累计 CPU 秒数（进程不存在返回 ``None``）。"""
    try:
        import subprocess

        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-Process -Name {process_name} -ErrorAction SilentlyContinue | "
             f"Measure-Object -Property CPU -Sum).Sum"],
            capture_output=True, text=True, timeout=20, check=False,
            # 子进程输出按 UTF-8 解码并放宽错误策略：Windows 本地代码页(GBK)
            # 或非 UTF-8 字节不会让读取线程抛 UnicodeDecodeError。
            encoding="utf-8", errors="replace")
        text = (completed.stdout or "").strip()
        return float(text) if text and text not in ("", ".") else None
    except Exception:  # noqa: BLE001 —— 探测失败显式返回 None（调用方按「无法核实」处理）
        return None


def established_connections(host: str = "127.0.0.1", port: int = 11434, logger: Any = None) -> int:
    """本机到 ``host:port`` 的已建立 TCP 连接数（用 psutil 不可用时的 netstat 兜底）。"""
    try:
        import subprocess

        completed = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True,
                                   timeout=20, check=False,
                                   # 同上一处：netstat 输出可能含本地代码页字节
                                   encoding="utf-8", errors="replace")
        needle = f"{host}:{port}"
        count = sum(1 for line in (completed.stdout or "").splitlines()
                    if needle in line and "ESTABLISHED" in line.upper())
        return count
    except Exception as exc:  # noqa: BLE001 —— 显式降级并留痕
        if logger is not None:
            logger.log_event("exclusive.netstat_failed", level="WARNING",
                             error_type=type(exc).__name__, message=str(exc), degrade="连接数按未知处理")
        return -1


def newest_mtime(directory: Path) -> float:
    """目录下最新文件 mtime（目录不存在返回 0.0）。"""
    if not directory.is_dir():
        return 0.0
    latest = 0.0
    for item in directory.rglob("*"):
        if item.is_file():
            try:
                latest = max(latest, item.stat().st_mtime)
            except OSError:
                continue
    return latest


def probe(*, quiet_seconds: float, logger: Any) -> dict[str, Any]:
    """跑一次独占检查：CPU 增量 + 连接数 + 留痕 mtime 稳定性。"""
    with logger.enter("probe", {"quiet_seconds": quiet_seconds}) as span:
        cpu_before = ollama_cpu_seconds()
        mtime_before = {str(path): newest_mtime(path) for path in WATCH_DIRS}
        time.sleep(max(0.0, quiet_seconds))
        cpu_after = ollama_cpu_seconds()
        mtime_after = {str(path): newest_mtime(path) for path in WATCH_DIRS}
        connections = established_connections(logger=logger)
        delta = (None if (cpu_before is None or cpu_after is None) else round(cpu_after - cpu_before, 3))
        touched = [name for name in mtime_after if mtime_after[name] > mtime_before.get(name, 0.0)]
        reasons: list[str] = []
        if delta is None:
            reasons.append("无法读取 ollama 进程 CPU（不视为独占）")
        elif delta > 0.30:
            reasons.append(f"ollama 进程 {quiet_seconds:.0f}s 内 CPU 增量 {delta}s > 0.30s（疑似正在推理）")
        if connections != 0:
            reasons.append(f"到 127.0.0.1:11434 的 ESTABLISHED 连接数 = {connections}（应为 0）")
        if touched:
            reasons.append(f"同窗口留痕仍在增长：{touched}")
        payload = {
            "work_order": WORK_ORDER,
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "quiet_seconds": quiet_seconds,
            "ollama_cpu_before": cpu_before, "ollama_cpu_after": cpu_after, "cpu_delta_s": delta,
            "established_connections_to_11434": connections,
            "watch_dirs": {name: {"before": mtime_before[name], "after": mtime_after[name]}
                           for name in mtime_after},
            "dirs_touched": touched,
            "EXCLUSIVE_OK": not reasons,
            "reasons": reasons,
        }
        logger.log_event("exclusive.check", **{k: v for k, v in payload.items()
                                              if k not in ("work_order", "watch_dirs")})
        span.set_output({"EXCLUSIVE_OK": payload["EXCLUSIVE_OK"], "reasons": reasons})
        return payload


def main(argv: Sequence[str] | None = None) -> int:
    """入口：独占检查（退出码 0 = 可开跑；2 = 非独占，调用方须标注并发口径）。"""
    parser = argparse.ArgumentParser(description="复跑前置检查：Ollama 是否独占")
    parser.add_argument("--quiet-seconds", type=float, default=4.0, help="CPU 采样窗口（默认 4 s，与 t17 口径一致）")
    parser.add_argument("--wait-seconds", type=float, default=0.0, help="等待上限（秒）；>0 时轮询到独占或超时")
    parser.add_argument("--out", default=str(REPO_ROOT / "优化" / "评估结果" / "过程日志" / "_exclusive_check.json"))
    args = parser.parse_args(argv)

    setup_logging(None, force=True)
    log = get_logger("check_exclusive_ollama")
    with log.enter("main", {"quiet_seconds": args.quiet_seconds, "wait_seconds": args.wait_seconds}) as span:
        deadline = time.perf_counter() + max(0.0, args.wait_seconds)
        payload: dict[str, Any] = {}
        while True:
            payload = probe(quiet_seconds=args.quiet_seconds, logger=log)
            if payload["EXCLUSIVE_OK"] or time.perf_counter() >= deadline:
                break
            print(f"… 非独占（{payload['reasons']}），等待中")
            time.sleep(5.0)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"EXCLUSIVE_OK = {payload['EXCLUSIVE_OK']}")
        print(f"cpu_delta_s = {payload['cpu_delta_s']}　ESTABLISHED(11434) = "
              f"{payload['established_connections_to_11434']}　dirs_touched = {payload['dirs_touched'] or '无'}")
        for reason in payload["reasons"]:
            print(f"  ⚠️ {reason}")
        print(f"留痕：{out_path.relative_to(REPO_ROOT)}")
        span.set_output({"EXCLUSIVE_OK": payload["EXCLUSIVE_OK"], "out": str(out_path)})
    shutdown_logging()
    return 0 if payload.get("EXCLUSIVE_OK") else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 —— 顶层兜底
        try:
            get_logger("check_exclusive_ollama").log_event("exclusive.failed", level="ERROR",
                                                           error_type=type(exc).__name__, message=str(exc),
                                                           stack=__import__("traceback").format_exc())
        finally:
            import traceback

            traceback.print_exc()
        raise SystemExit(1)
