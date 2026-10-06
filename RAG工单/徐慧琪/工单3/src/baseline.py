# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
模块：「无表格优化」基线桥接（只读调用工单1/02 系统）

用途：对比实验/界面对比页需要**无表格优化系统**的真实结果。
工单1（`..\\工单1`）的代码与索引原样保留，本模块通过子进程
（scripts/baseline_ask.py，以工单1为根）调用其 rag.ask，绝不修改工单1任何文件。

为什么用子进程：两侧包名同为 `src`，同进程导入会冲突。

该基线的定位（工单3 对比口径）：
  · 只索引了《招股说明书1》，**没有《招股说明书2》** → id 1~4 天然无法作答；
  · 表格是"整表一个块"的朴素处理，没有表头/行级结构化 → 表格题召回差。
  因此它在 14 题上的表现正是"无表格优化"的量化基线。

  - 结果结构：工单1 的 RagResult.to_dict()（answer/contexts/citations/timings）；
  - 容错：超时/失败返回 {"error": ...}，对比表格中如实标注，不伪造数据。
"""
from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import json
import os
import subprocess
import sys
import time

from src import config

SCRIPT = os.path.join(config.ROOT, "scripts", "baseline_ask.py")


def is_available() -> bool:
    """优化前系统资源是否齐备（工单1 目录 + 索引 + 脚本）。"""
    return (os.path.isdir(config.BASELINE_ROOT)
            and os.path.isdir(config.BASELINE_QDRANT_DIR)
            and os.path.isfile(SCRIPT))


def ask_baseline(question: str, timeout: int = 300) -> dict:
    """调用工单1系统回答单题。返回 dict（含 answer/timings，或 error）。"""
    if not is_available():
        return {"error": "工单1 基线不可用（目录/索引缺失）", "question": question}
    t0 = time.time()
    env = dict(os.environ)
    env["BASELINE_ROOT"] = config.BASELINE_ROOT
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run([sys.executable, SCRIPT, question],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout, env=env,
                              cwd=config.ROOT)
    except subprocess.TimeoutExpired:
        return {"error": f"baseline 超时(>{timeout}s)", "question": question,
                "elapsed_s": round(time.time() - t0, 2)}
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc), "question": question}

    out = proc.stdout or ""
    marker = "<<<RESULT>>>"
    if marker not in out:
        return {"error": "baseline 无结果输出",
                "stderr_tail": (proc.stderr or "")[-500:], "question": question}
    payload = out.split(marker, 1)[1].strip()
    # 取最后一行合法 JSON
    data: dict | None = None
    for line in reversed(payload.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if data is None:
        return {"error": "baseline 输出无法解析", "question": question}
    data["wall_s"] = round(time.time() - t0, 2)
    return data


def ask_baseline_batch(questions: list[str], timeout: int = 2400) -> list[dict]:
    """批量调用工单1系统（一次子进程处理全部题目，模型只加载一次）。

    返回与 questions 等长的结果列表；无法解析的第 i 项为 {"error": ...}。
    内存提示：必须在**调用方尚未加载大模型**时使用（见 evaluate_compare.py
    的 Pass 1/2 设计），否则 Windows 页面文件不足会让子进程的 cublas 加载失败。
    """
    empty = [{"error": "空输入"}] * len(questions)
    if not questions:
        return []
    if not is_available():
        return [{"error": "工单1 基线不可用（目录/索引缺失）"}] * len(questions)

    t0 = time.time()
    env = dict(os.environ)
    env["BASELINE_ROOT"] = config.BASELINE_ROOT
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run([sys.executable, SCRIPT, "--stdin"],
                              input="\n".join(questions), capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, env=env, cwd=config.ROOT)
    except subprocess.TimeoutExpired:
        return [{"error": f"baseline 批量超时(>{timeout}s)"}] * len(questions)
    except Exception as exc:  # noqa: BLE001
        return [{"error": repr(exc)}] * len(questions)

    out = proc.stdout or ""
    marker = "<<<RESULT>>>"
    if marker not in out:
        return [{"error": "baseline 无结果输出",
                 "stderr_tail": (proc.stderr or "")[-300:]}] * len(questions)

    results: list[dict] = []
    for block in out.split(marker)[1:]:
        data = None
        for line in reversed(block.strip().splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        results.append(data if data is not None else {"error": "输出无法解析"})
    # 数量对齐（缺的用错误项补齐）
    while len(results) < len(questions):
        results.append({"error": "结果数量不足"})
    for r in results:
        r["wall_s_batch"] = round(time.time() - t0, 2)
    return results[: len(questions)]


def info() -> dict:
    return {
        "root": config.BASELINE_ROOT,
        "collection": config.BASELINE_COLLECTION,
        "available": is_available(),
    }
