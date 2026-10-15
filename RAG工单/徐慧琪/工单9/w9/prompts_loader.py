# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务

模块：工单09 的 Prompt 加载器

## 为什么要自己写一个

08 的 `src/kg_prompt.load_prompt()` 把 Prompt 目录硬解析到
`工单8/prompts/`（`config.PROMPTS_DIR` 由 `bootstrap.project_root()` 基于
**文件位置**推导）。09 的优化版 Prompt（v2）放在 `工单9/w9/prompts/`，
用 08 的加载器找不到 —— 而且我们也不应该往 08 的目录里放文件
（那会破坏「08 物理只读」的承诺）。

因此本模块提供 09 侧的加载器，**查找顺序**：

    1. `工单9/w9/prompts/{name}.yaml`   —— 09 的优化版（优先）
    2. `工单8/prompts/{name}.yaml`      —— 08 的版本（未优化的沿用）

这样 09 只覆盖**确实做了优化**的 Prompt（kg_extract_v2、graph_qa_v2），
其余（如 08 的 graph_qa 基线 Prompt）继续从 08 读取，避免无谓的复制发散。
"""
from __future__ import annotations

import os
import threading

from w9 import config9

_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.RLock()

_SEARCH_DIRS = (config9.PROMPTS9, os.path.join(config9.ROOT8, "prompts"))


def prompt_path(name: str) -> str | None:
    """按查找顺序返回 Prompt 文件路径；都没有返回 None。"""
    for d in _SEARCH_DIRS:
        p = os.path.join(d, f"{name}.yaml")
        if os.path.isfile(p):
            return p
    return None


def load_prompt(name: str) -> dict:
    """读取 Prompt YAML（带 mtime 缓存，文件改动后自动重载）。"""
    path = prompt_path(name)
    if path is None:
        raise FileNotFoundError(
            f"Prompt 不存在：{name}.yaml\n查找路径：\n  " +
            "\n  ".join(_SEARCH_DIRS))

    mtime = os.path.getmtime(path)
    with _lock:
        cached = _cache.get(name)
        if cached and cached[0] == mtime and cached[2] == path:
            return cached[1]

    import yaml
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Prompt YAML 结构异常：{path}")

    with _lock:
        _cache[name] = (mtime, data, path)
    return data


def source_of(name: str) -> str:
    """该 Prompt 实际来自哪个工单（供报告标注「优化前 vs 优化后」）。"""
    path = prompt_path(name)
    if not path:
        return "缺失"
    return "工单09（优化后）" if path.startswith(config9.PROMPTS9) \
        else "工单08（优化前）"
