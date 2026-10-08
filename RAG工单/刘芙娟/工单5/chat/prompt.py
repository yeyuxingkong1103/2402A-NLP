"""多轮对话的角色设定装载。

    load_system_prompt() -> str     启动期调用一次

---

## 为什么角色设定是**数据**而不是代码

与 `backend/generate/__init__.py` 对 `data/prompts/` 的处置同一取向：
提示词会被**非开发人员**审阅 —— 医疗内容的措辞需要临床人员把关。
写成 Python 字符串的后果是可预期的：改一个词要走代码评审、要懂 diff、
还要担心缩进；放 `.md` 里则任何编辑器都能改，改动在 diff 里是一段可读的中文。

## 为什么缺失时**抛错**而不是回退到内置默认值

一份悄悄生效的默认提示词意味着：线上模型的行为与仓库里的文件不一致，
**且没有任何迹象**。使用者改了 `data/prompts/med_qa_system.md` 却看不到效果，
排查方向会被引向"是不是没保存""是不是改错文件了"。
抛错的代价是服务起不来（可见、可修），回退的代价是长期的行为漂移（不可见）。

这条与 `backend/generate/prompt.py` 的 `load_prompt` 完全一致。

## ⚠️ 角色设定 MUST NOT 含逐字话术

紧急前置话术与免责声明由 `backend/api/__init__.py` 唯一定义、
由 `pipeline.assemble_answer_text()` 唯一拼接。把它们写进角色设定会导致
回答里出现两遍，且「全仓只应命中一处」这条约束静默失效 ——
`backend/generate/prompt.py:_assert_no_verbatim_boilerplate` 检查的是
**答案提示词**，不覆盖本文件，所以这里靠的是编写时遵守约定。

（首次写这份文件时就踩过一次：初稿里写了"如果紧急，先提示就医"，
虽然没有逐字复制话术，但那正是让模型自己复述免责声明/紧急提示的开端。
现在的版本只说"不做诊疗"，把措辞留给装配层。）
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from . import ChatError

__all__ = ["SYSTEM_PROMPT_PATH", "load_system_prompt"]

# backend/chat/prompt.py → parents[0]=chat, [1]=backend, [2]=仓库根
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT_PATH = PROJECT_ROOT / "data" / "prompts" / "med_qa_system.md"


@lru_cache(maxsize=1)
def load_system_prompt(path: str | None = None) -> str:
    """读取角色设定全文。**结果按路径缓存。**

    ⚠️ 缓存它的理由不是性能（文件约 2 KB，每次读盘可忽略），
    是**一致性**：一次对话中途文件被改，读到半新半旧的内容比读到旧内容更糟。
    启动期读一次、进程生命周期内不变 —— 与 `backend/api/config.py` 同取向。
    """

    target = Path(path) if path else SYSTEM_PROMPT_PATH

    if not target.is_file():
        raise ChatError(
            "角色设定文件不存在：%s\n"
            "  它是多轮对话的固定人设，缺失时 MUST NOT 回退到内置默认值 ——\n"
            "  那样线上模型的行为会与仓库里的文件不一致，且没有任何迹象。" % target
        )

    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise ChatError("角色设定文件无法读取：%s —— %s" % (target, type(exc).__name__)) from exc

    if not text.strip():
        raise ChatError("角色设定文件为空：%s" % target)

    return text
