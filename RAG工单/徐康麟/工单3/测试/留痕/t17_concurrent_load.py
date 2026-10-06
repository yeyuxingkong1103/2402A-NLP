# -*- coding: utf-8 -*-
"""t17 并发加压器：模拟「其他成员同时占用本地 Ollama」，用于复现首字超标（只读诊断工具）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

用途：t17 要求给出「独占 vs 并发」两组首字数字。本脚本在评测运行期间持续向
``127.0.0.1:11434`` 发起并行生成请求，占用同一个 Ollama 进程，从而在**受控条件**下复现
run2 观测到的 3274.04 ms（该次观测窗口 17:50:04 与 tester 的 17:50:21 留痕重叠）。

只依赖标准库；不写任何产品数据；输出一行结构化 JSON 汇总（请求数 / 延迟统计）。

用法（工作目录 = 工单3）：
    pwsh -NoProfile -File run_py.ps1 测试/留痕/t17_concurrent_load.py --workers 4 --seconds 90
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"
DEFAULT_URL = "http://127.0.0.1:11434/api/generate"
PROMPT = "请用中文详细说明视频指挥控制系统的技术架构与部署方式，尽可能长。"


def _one_request(model: str, *, num_predict: int, timeout: float) -> float:
    """发一次生成请求，返回耗时（ms）；失败抛异常由调用方记录。"""
    payload = json.dumps({"model": model, "prompt": PROMPT, "stream": False,
                          "options": {"num_predict": int(num_predict), "temperature": 0.0}}).encode("utf-8")
    request = urllib.request.Request(DEFAULT_URL, data=payload,
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()
    return round((time.perf_counter() - started) * 1000, 2)


def worker(model: str, deadline: float, *, num_predict: int, timeout: float,
           latencies: list[float], errors: list[str], lock: threading.Lock) -> None:
    """一个加压线程：在 deadline 前不断发请求。"""
    while time.perf_counter() < deadline:
        try:
            elapsed = _one_request(model, num_predict=num_predict, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 —— 加压器失败必须留痕（进 errors），不静默
            with lock:
                errors.append(f"{type(exc).__name__}: {exc}")
            time.sleep(0.2)
            continue
        with lock:
            latencies.append(elapsed)


def main(argv: list[str] | None = None) -> int:
    """入口：起 N 个线程持续加压 ``--seconds`` 秒，打印汇总 JSON。"""
    parser = argparse.ArgumentParser(description="t17 并发加压器")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=60.0)
    # 注意：不要用 ``--out`` 作选项名 —— PowerShell 会把它前缀匹配成 `-OutVariable`/`-OutBuffer`
    # 而报「参数名称 'out' 存在歧义」（实测 2026-10-04 18:57，导致加压器未启动、该轮评测退化为独占口径）。
    parser.add_argument("--save", default="", help="汇总 JSON 落盘路径（UTF-8；便于留痕）")
    args = parser.parse_args(argv)

    deadline = time.perf_counter() + float(args.seconds)
    latencies: list[float] = []
    errors: list[str] = []
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(args.model, deadline), kwargs={
        "num_predict": args.num_predict, "timeout": args.timeout,
        "latencies": latencies, "errors": errors, "lock": lock}, daemon=True)
        for _ in range(max(1, int(args.workers)))]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    summary = {
        "event": "t17.concurrent_load", "work_order": WORK_ORDER,
        "model": args.model, "workers": int(args.workers), "seconds": args.seconds,
        "num_predict": int(args.num_predict),
        "requests_ok": len(latencies), "requests_err": len(errors),
        "mean_ms": round(statistics.fmean(latencies), 2) if latencies else None,
        "max_ms": max(latencies) if latencies else None,
        "wall_s": round(time.perf_counter() - started, 2),
        "sample_errors": errors[:3],
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.save:
        from pathlib import Path

        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"汇总已落盘：{out_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 —— 顶层兜底：非零退出，绝不静默
        import traceback

        print(json.dumps({"event": "t17.concurrent_load_failed", "level": "ERROR",
                          "error_type": type(exc).__name__, "message": str(exc),
                          "stack": traceback.format_exc()}, ensure_ascii=False), flush=True)
        raise SystemExit(1)

