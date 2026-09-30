# -*- coding: utf-8 -*-
"""链路分发器：按 `config.CHAIN_BACKEND` 选择手写链路或 LangChain 链路。

设计意图
--------
路由层只 import 本模块，无需感知具体实现 —— 这样切换链路不改调用方代码。
答辩现场只要改一个环境变量（`RAGLORA_CHAIN=langchain`）重启，就能对比两条链路。

两条链路的关系是**并存**，不是替换：

    langchain —— `lc_chain`，LCEL 编排版，**2026-09-16 起为默认链路**
    manual    —— `rag_chain`，手写的已验证基线（100% 来源命中率/引用标注率）

二者刻意复用同一批底层能力（`persona.render_system` 渲染 prompt、
`rag_chain.postprocess` 做后处理、同一套 `retrieval` + `rerank`），
差异只在编排层 —— 实测两条链路的检索与引用指标一致（见 docs/06-双链路对比.md）。

`manual` 随时可用 `RAGLORA_CHAIN=manual` 切回，作为出问题时的退路。

未知取值一律回退 `DEFAULT_BACKEND`：配置写错不该让服务起不来。
"""
from collections.abc import Iterator

from ..core import config
from ..core.logging import get_logger
from ..models import Character
from . import lc_chain, rag_chain

log = get_logger("chain")

# 链路名 -> 模块。模块需实现同名的 prepare / ask / ask_stream。
_BACKENDS = {
    "manual": rag_chain,
    "langchain": lc_chain,
}

DEFAULT_BACKEND = "langchain"


def active_backend() -> str:
    """当前生效的链路名。未知值回退 `manual` 并记一条告警。"""
    name = config.CHAIN_BACKEND
    if name not in _BACKENDS:
        log.warning("未知链路 %r，已回退 %s", name, DEFAULT_BACKEND)
        return DEFAULT_BACKEND
    return name


def _impl():
    """当前生效的链路模块。"""
    return _BACKENDS[active_backend()]


def prepare(character: Character, question: str, memory: list[dict]) -> dict:
    return _impl().prepare(character, question, memory)


def ask(character: Character, question: str, memory: list[dict]) -> dict:
    return _impl().ask(character, question, memory)


def ask_stream(character: Character, question: str,
               memory: list[dict]) -> Iterator[tuple[str, dict]]:
    return _impl().ask_stream(character, question, memory)
