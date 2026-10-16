# -*- coding: utf-8 -*-
"""工单4 提示词加载器（``研发/app/prompts/*.txt``，UTF-8）。

工单编号：人工智能NLP-RAG 项目-金融问答系统部署

契约：
    * 提示词**只从文件读取**（便于评审与逐版留痕），代码里不内联长提示词；
    * 文件缺失不得静默：``load_prompt`` 抛 ``RagError``（提示词是答案质量的组成部分，
      缺文件必须让人看见，而不是回落到空字符串）；
    * 缓存按 ``(name, mtime)`` 失效，支持运行期改提示词后立即生效。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import RagError

WORK_ORDER = "人工智能NLP-RAG 项目-金融问答系统部署"

PROMPTS_DIR: Path = Path(__file__).resolve().parents[1] / "prompts"
_CACHE: dict[str, tuple[float, str]] = {}


def _lazy_logger(logger: Any, module: str = "prompts_loader") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


def prompt_path(name: str) -> Path:
    """提示词文件路径（``name`` 不含扩展名）。"""
    return PROMPTS_DIR / f"{name}.txt"


def load_prompt(name: str, *, logger: Any = None) -> str:
    """读取提示词正文（带 mtime 缓存）；缺失抛 ``RagError``。"""
    log = _lazy_logger(logger)
    path = prompt_path(name)
    with log.enter("load_prompt", {"name": name, "path": str(path)}) as span:
        if not path.is_file():
            raise RagError(f"提示词文件缺失：{path}", code="RAG-5001", stage="prompt",
                           detail={"name": name, "path": str(path)})
        mtime = path.stat().st_mtime
        cached = _CACHE.get(name)
        if cached is not None and cached[0] == mtime:
            span.set_output({"chars": len(cached[1]), "cached": True})
            return cached[1]
        text = path.read_text(encoding="utf-8")
        _CACHE[name] = (mtime, text)
        log.log_event("prompt.loaded", name=name, chars=len(text), path=str(path))
        span.set_output({"chars": len(text), "cached": False})
        return text


def render(template: str, mapping: dict[str, Any]) -> str:
    """安全渲染：只替换 ``{key}``，**不执行** str.format（正文里含 ``{}`` 也不会炸）。"""
    out = template
    for key, value in mapping.items():
        out = out.replace("{" + key + "}", str(value))
    return out
