# -*- coding: utf-8 -*-
"""链路分发器测试：默认走 langchain，可切回 manual，未知值回退默认链路。

本模块在守护什么
================

`app/services/chain.py` 是路由层唯一的链路入口。它存在的意义是让
「LangChain 链路（lc_chain）」与「手写链路（rag_chain）」**可切换**，
而不必改动调用方代码。

三条必须守住的性质：

  1. `RAGLORA_CHAIN` 未设置时是 `langchain`（2026-09-16 起用户指定的默认）。
  2. 设置成 `manual` 时能切回手写链路 ——
     它是已验证基线（100% 来源命中率/引用标注率），是出问题时的退路，
     **这条退路必须始终可用**。
  3. 设置成**未知值**时回退默认链路而**不是抛异常** ——
     配置写错不该让服务起不来。

注意：这里用 `importlib.reload(config)` 让环境变量被重新读取，
因为 `CHAIN_BACKEND` 是在模块导入时求值的。
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _reload_with(monkeypatch, value: str | None):
    """把 RAGLORA_CHAIN 设为 value（None 表示删除），重载 config 与 chain，返回 chain 模块。"""
    if value is None:
        monkeypatch.delenv("RAGLORA_CHAIN", raising=False)
    else:
        monkeypatch.setenv("RAGLORA_CHAIN", value)

    from app.core import config
    importlib.reload(config)

    from app.services import chain
    importlib.reload(chain)
    return chain


def test_default_is_langchain(monkeypatch):
    """未设置环境变量时走 LangChain 链路（2026-09-16 起的默认）。"""
    ch = _reload_with(monkeypatch, None)
    assert ch.active_backend() == "langchain"


def test_can_fall_back_to_manual(monkeypatch):
    """手写链路必须始终可切回 —— 它是出问题时的退路。"""
    ch = _reload_with(monkeypatch, "manual")
    assert ch.active_backend() == "manual"


def test_unknown_value_falls_back_to_default(monkeypatch):
    """未知取值必须回退默认链路，而不是让服务起不来。"""
    ch = _reload_with(monkeypatch, "no-such-chain")
    assert ch.active_backend() == "langchain"


def test_dispatch_targets_real_modules(monkeypatch):
    """分发确实指向对应的模块对象（而非复制了一份实现）。"""
    from app.services import lc_chain, rag_chain

    ch = _reload_with(monkeypatch, "manual")
    assert ch._impl() is rag_chain

    ch = _reload_with(monkeypatch, "langchain")
    assert ch._impl() is lc_chain


def test_cleanup_restores_default(monkeypatch):
    """收尾：恢复默认，避免污染后续测试。"""
    ch = _reload_with(monkeypatch, None)
    assert ch.active_backend() == "langchain"
