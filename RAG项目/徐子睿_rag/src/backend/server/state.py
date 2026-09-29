# -*- coding: utf-8 -*-
"""server/state.py —— 构建状态的进程内单例。

在链路中的位置：
    上传接口启动后台构建线程写它，构建进度接口与删除接口读它，
    build_pipeline 的 progress 回调也经由 update_step 写它。

为什么必须串行（同一时间只允许一个构建任务）：
    构建要"先删旧向量再写新向量"，两个任务并行会互相删掉对方刚写入的数据；
    而且向量化和 LLM 都吃资源，并行只会让两个任务都变慢。
"""
from __future__ import annotations

import time
from typing import Any

# 同一时间只允许一个 PDF 构建任务。
# 为什么必须串行：构建要删旧向量再写新向量，两个任务并行会互相删掉对方刚写入的数据；
# 而且向量化和 LLM 都吃资源，并行只会让两个任务都变慢。
build_state: dict[str, Any] = {"running": False, "steps": [], "error": None, "doc": None}

def update_step(step: str, status: str, detail: str) -> None:
    """更新构建进度中的某一步状态（作为 build_pipeline 的 progress 回调）。

    参数：
        step: 步骤名（如 "解析PDF" / "向量化"）
        status: "running" / "done" / "error"
        detail: 人类可读的补充说明（如 "128 块"）

    同名步骤已存在就原地替换、否则追加：
        这样前端看到的始终是"每个步骤一行、状态实时刷新"，
        而不是同一个步骤在列表里堆出多行。
    """
    item = {"step": step, "status": status, "detail": detail, "t": time.strftime("%H:%M:%S")}
    for index, old in enumerate(build_state["steps"]):
        if old.get("step") == step:
            build_state["steps"][index] = item
            return
    build_state["steps"].append(item)
