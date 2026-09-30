# -*- coding: utf-8 -*-
"""当轮运行变量采集（`docs/目录与命名约定.md` 第 10 条：评测报告必须记录当轮变量）。

为什么要有这个模块：评测数字只有在"同一套变量"下才可比。LLM 模型、提示词文本、召回窗口、
拒答阈值任何一项变了，回答侧指标（无引用数 / 引用条数 / 拒答）都可能整体漂移 —— 批次 29 就是
因为提示词里少了一条"必须标编号"的硬要求，20 题的无引用回答从 4 条涨到 16 条；而当时的报告里
没有任何字段能说明"这一轮和上一轮不是同一套变量"，只能靠人肉翻文件时间戳去猜。

所以报告头自动记录当轮变量，并在 JSON 里以 `run_config` 字段提供，机器可读、便于跨轮 diff。

本模块只做"读"：不写任何状态、不改任何配置。读不到的项如实写 None（并在报告里显示 unknown），
不做任何猜测或缺省填充——报告里出现一个假值，比出现一个空值更误导人。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

# 需要记录 md5 的提示词文件（相对项目根）。SYSTEM_PROMPT 常量就在 prompt_builder.py 里；
# 以后提示词若再拆文件，往这里加路径即可，报告字段与渲染都不用改（渲染按本元组遍历）。
PROMPT_FILES: tuple[str, ...] = ("backend/app/chat/prompt_builder.py",)


def file_md5(path: Path) -> str | None:
    """文件内容的 md5；文件不存在返回 None（不抛异常——出报告不该被缺文件打断）。"""
    if not path.exists():
        return None
    return hashlib.md5(path.read_bytes()).hexdigest()


def collect_run_config(project_root: Path, eval_set_path: Path | None = None) -> dict[str, Any]:
    """采集当轮可复现变量。

    Args:
        project_root: 项目根目录（用于定位提示词文件）
        eval_set_path: 本轮评测集路径；给了就一并记录它的 md5
            （题目集换了内容、文件名却没变，同样是"悄悄换了变量"）

    Returns:
        扁平字典（便于跨轮逐字段 diff）+ 一个 `prompt_files` 子字典（相对路径 → md5）：
        llm_model / llm_temperature / llm_max_tokens / llm_timeout_seconds /
        recall_vector_limit / recall_keyword_limit / rerank_candidate_limit /
        refusal_min_vector_score / eval_set_md5 / prompt_files

        配置读取失败时附 `error` 字段，其余项保持 None。
    """
    config: dict[str, Any] = {
        "llm_model": None,
        "llm_temperature": None,
        "llm_max_tokens": None,
        "llm_timeout_seconds": None,
        "recall_vector_limit": None,
        "recall_keyword_limit": None,
        "rerank_candidate_limit": None,
        "refusal_min_vector_score": None,
        "eval_set_md5": file_md5(eval_set_path) if eval_set_path else None,
        "prompt_files": {rel: file_md5(project_root / rel) for rel in PROMPT_FILES},
    }
    try:
        # 调用方（run_eval.prepare_env）已把 backend 放进 sys.path 并载入 .env，
        # 这里的 settings 即"本轮真正生效的那份配置"，不是 .env.example 之类的样本值。
        from app.core.config import settings
    except Exception as error:  # noqa: BLE001 - 配置不可读也要能出报告，如实标注原因
        config["error"] = f"{type(error).__name__}: {error}"
        return config
    config.update(
        {
            "llm_model": settings.llm_model,
            "llm_temperature": settings.llm_temperature,
            "llm_max_tokens": settings.llm_max_tokens,
            "llm_timeout_seconds": settings.llm_timeout_seconds,
            "recall_vector_limit": settings.recall_vector_limit,
            "recall_keyword_limit": settings.recall_keyword_limit,
            "rerank_candidate_limit": settings.rerank_candidate_limit,
            "refusal_min_vector_score": settings.refusal_min_vector_score,
        }
    )
    return config


def format_run_config_lines(config: dict[str, Any]) -> list[str]:
    """把当轮变量排成 Markdown 列表行（报告头用）。

    渲染与采集同源：字段名只在本模块出现一次，避免"报告写了 A、JSON 里叫 B"。
    """
    if not config:
        return ["- 当轮变量：**未记录**（旧版报告，先于治理规则第 10 条）"]
    unknown = "unknown"

    def show(value: Any) -> str:
        return unknown if value is None else str(value)

    prompt_bits = [
        f"`{name}` md5 `{md5 or unknown}`" for name, md5 in (config.get("prompt_files") or {}).items()
    ]
    lines = [
        "- 当轮变量（`docs/目录与命名约定.md` 第 10 条；JSON 里同源字段 `run_config`）：",
        f"  - LLM：`{show(config.get('llm_model'))}`"
        f"（temperature {show(config.get('llm_temperature'))}"
        f" / max_tokens {show(config.get('llm_max_tokens'))}"
        f" / timeout {show(config.get('llm_timeout_seconds'))}s）",
        f"  - 提示词：{'; '.join(prompt_bits) or unknown}",
        f"  - 召回窗口：向量 {show(config.get('recall_vector_limit'))}"
        f" + 关键词 {show(config.get('recall_keyword_limit'))}"
        f" → 融合前 {show(config.get('rerank_candidate_limit'))} → 重排",
        f"  - 拒答阈值：REFUSAL_MIN_VECTOR_SCORE = {show(config.get('refusal_min_vector_score'))}",
        f"  - 评测集 md5：`{show(config.get('eval_set_md5'))}`",
    ]
    if config.get("error"):
        lines.append(f"  - ⚠️ 配置读取失败：{config['error']}")
    return lines
