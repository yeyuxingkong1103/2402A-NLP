# -*- coding: utf-8 -*-
"""短期记忆窗口的"单一来源"绑定（批次 41）。

为什么需要这个测试：窗口 20 条在生产里有**两个装配点** ——
`app/api/chat_persistence.get_short_term_memory`（写回侧）与
`app/retrieval/assembly.build_default_retrieval_service`（检索侧），
评测侧还有 `evaluation/multi_turn_grading.SESSION_WINDOW`。
三处必须相等，否则"写回 20 条、改写只读 N 条"这类偏差不会有任何报错 ——
这正是"注释说一致、实际无绑定"的静默漂移。

为什么不用 import 绑定（原本的设想）：评测模块由 `python -m evaluation.run_eval`
导入时 backend 尚未进 sys.path（由 `prepare_env()` 在**运行时**插入），
顶层 `from app.* import` 会在 import 阶段直接 ModuleNotFoundError；而给纯函数模块
加 sys.path 副作用，会把 FastAPI/SQLAlchemy 拖进 `aggregate` / `render_report`
的依赖链。故改为本测试强制。

本测试检查的是**真实装配点的取值**，不是注释里的说法：任一处被改都会变红。
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.api.chat_persistence import SHORT_TERM_MAX_MESSAGES, get_short_term_memory
from app.memory.short_term import ShortTermMemoryStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_GRADING_PATH = PROJECT_ROOT / "evaluation" / "multi_turn_grading.py"
ASSEMBLY_PATH = PROJECT_ROOT / "backend" / "app" / "retrieval" / "assembly.py"


def _load_eval_grading() -> Any:
    """按文件路径加载评测模块（它只依赖 re/typing，可脱离 evaluation 包独立加载）。"""
    spec = importlib.util.spec_from_file_location("_b41_eval_multi_turn_grading", EVAL_GRADING_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _max_messages_values(path: Path, function_name: str) -> list[ast.expr]:
    """取指定函数体内所有 `max_messages=` 关键字的实参节点。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    values += [kw.value for kw in inner.keywords if kw.arg == "max_messages"]
    return values


def test_eval_window_equals_production_constant() -> None:
    """评测侧 SESSION_WINDOW 必须等于生产常量（消除"注释说一致"的漂移）。"""
    module = _load_eval_grading()
    assert module.SESSION_WINDOW == SHORT_TERM_MAX_MESSAGES


def test_writeback_store_actually_uses_the_constant() -> None:
    """写回侧真实装配出的 store，窗口必须等于常量（不是注释，是对象属性）。"""
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    store = get_short_term_memory(request)
    assert isinstance(store, ShortTermMemoryStore)
    assert store.max_messages == SHORT_TERM_MAX_MESSAGES
    # 懒建入口会缓存到 app.state，供同一进程后续请求复用
    assert request.app.state.short_term_memory is store


def test_retrieval_assembly_window_matches_the_constant() -> None:
    """检索侧装配点的 max_messages 必须等于常量（写字面量或引用常量都接受）。"""
    values = _max_messages_values(ASSEMBLY_PATH, "build_default_retrieval_service")
    assert values, "assembly.build_default_retrieval_service 里找不到 max_messages"
    for value in values:
        if isinstance(value, ast.Constant):
            assert value.value == SHORT_TERM_MAX_MESSAGES, (
                f"检索侧窗口 {value.value} 与常量 {SHORT_TERM_MAX_MESSAGES} 不一致"
            )
        elif isinstance(value, (ast.Name, ast.Attribute)):
            name = value.id if isinstance(value, ast.Name) else value.attr
            assert name == "SHORT_TERM_MAX_MESSAGES", (
                f"检索侧窗口引用了 {name}，请确认为同一个常量"
            )
        else:
            raise AssertionError(f"检索侧 max_messages 写法无法静态核对：{ast.dump(value)}")
