# -*- coding: utf-8 -*-
"""ragas 兼容垫片（RAGLoRA 仓库内补丁，非上游代码）。

背景：需绕过两个彼此独立的上游 / 环境问题，才能让 `import ragas` 跑通
=====================================================================

问题 ①（ragas 上游硬导入已删除的模块）
---------------------------------------
`ragas` 在 `ragas/llms/base.py` 的**模块顶层**写死了一行硬导入：

    from langchain_community.chat_models.vertexai import ChatVertexAI

然而 `langchain-community>=0.4` 已经**删除**了 `chat_models/vertexai.py`
（VertexAI 的对话模型实现被移出，只保留了 `langchain_community.llms.VertexAI`
这一 LLM 版本）。经逐版本核验，ragas 从 0.2.15 到 0.4.3 的**所有版本**都存在
这个硬导入，因此**降级 ragas 无法解决**。

在本项目里 ragas 只是评测框架，裁判 LLM 用的是本地 Ollama（`ChatOllama`），
**完全不使用 VertexAI 的任何真实功能**。ragas 对该类的用法仅有两处：

    1) `MULTIPLE_COMPLETION_SUPPORTED = [..., ChatVertexAI, VertexAI]`
    2) `is_multiple_completion_supported()` 里对它做 `isinstance(llm, ...)`

也就是说，它只需要一个**能参与 isinstance 判断、能取到类型属性的类对象**。

本垫片向 `sys.modules` 预注册一个假的
`langchain_community.chat_models.vertexai` 模块，导出一个 `ChatVertexAI`
占位类，使 ragas 那行硬导入命中缓存、不再去磁盘找已删除的文件。

问题 ②（torch / pyarrow 原生库加载顺序冲突，会段错误）
------------------------------------------------------
本环境的 `torch 2.5.1+cu121` 与 `pyarrow 24.0.0` 存在原生 DLL 加载顺序冲突：

    先 `import torch`（加载 torch._C）→ 再 `import pyarrow.dataset`  → **段错误 (0xC0000005)**
    先 `import pyarrow.dataset`    → 再 `import torch` / `datasets`  → 正常

而 ragas 的导入链恰好会踩中它：`ragas` → `datasets` → `pandas` →
`pyarrow.dataset`；只要在它之前有任何东西先把 `torch` 拉进来，`import ragas`
就会**直接进程崩溃（segmentation fault），而不是抛 Python 异常**。

特别注意：**初版垫片曾让占位类继承 `langchain_core` 的 `BaseChatModel`**
（想让类型更“合法”）。但 langchain 1.x 的 `BaseChatModel` 导入链会拉起
`torch`，于是**垫片自己成了崩溃的触发源**。因此本版占位类**刻意做成普通
Python 类**，绝不导入 langchain_core，并在注入前**尽力预载
`pyarrow.dataset`**，把原生库顺序钉死在 torch 之前。

影响范围（重要）
----------------
* 仅影响**本进程内**：向 `sys.modules` 注入一个伪模块，并（尽力）预载
  `pyarrow.dataset`。不改任何全局状态、不写磁盘。
* 不修改 site-packages 里的任何文件（硬约束）。
* 不改动 `langchain-community` / `langchain` / `torch` / `pyarrow` 的版本
  （会破坏 langchain 1.2.18 与 transformers 的依赖链，是硬约束）。
* 不覆盖任何**真实存在**的模块：若将来上游修复、该模块真实存在，函数会直接
  返回，不做任何事。
* **不掩盖致命的加载顺序错误**：若调用时 `torch` 已先于 `pyarrow.dataset`
  导入（原生库顺序已无可挽回），`ensure_ragas_compat()` 会**抛
  `RuntimeError`**，而不是返回成功。这是刻意的——继续 `import ragas` 必然
  段错误，只有抛异常才能被调用方看见。详见下节「用法」。

何时应当删除此文件
------------------
满足**任一**条件即可删除本垫片并移除调用点：
  * ragas 上游把该硬导入改成惰性导入 / try-except，或不再引用 `ChatVertexAI`；
  * `langchain-community` 重新提供 `chat_models.vertexai` 模块；
  * 本项目不再使用 ragas（例如自研评测指标）。

在此之前，任何 `import ragas` 的入口都应先调用一次 `ensure_ragas_compat()`。

用法
----
    from app.core.ragas_compat import ensure_ragas_compat
    ensure_ragas_compat()          # 必须在 import ragas 之前，越早越好
    import ragas

**调用时机是硬要求**：`ensure_ragas_compat()` 不仅是「越早越好」，而是**必须
早于任何会拉起 `torch` 的导入**。本项目会拉起 torch 的模块有
`app.services.embed` 与 `app.services.rerank`（以及任何间接导入它们的模块）。
正确做法是把这两行放在入口脚本的最前面：

    # backend/eval/run_ragas.py 的正确开头
    from app.core.ragas_compat import ensure_ragas_compat
    ensure_ragas_compat()          # ← 必须最先，早于任何 app.services 导入
    import ragas                   # 此后才安全

本函数**幂等**，重复调用安全（首次 `True`，其后 `False`）；本模块也可被反复
import，无副作用。

异常（重要）
------------
若调用时已经错过时机——**`torch` 已在 `sys.modules` 且 `pyarrow.dataset`
尚未加载**——本函数**抛 `RuntimeError`**，消息中包含原因、后果与修复方式。

* **为什么抛而不是返回 `False`/`True`**：此状态下随后 `import ragas` 会
  **段错误**（退出码 139，非 Python 异常，try/except 捕不到）。返回 `True`
  会给出致命的假成功信号，返回 `False` 会被误读成「安全的幂等命中」，
  只有硬报错能被看见。这是未来实现者唯一可靠的诊断线索。
* **不会误伤正常路径**：仅当上述两条**同时**成立时才抛。若
  `pyarrow.dataset` 已加载（无论 torch 在不在），原生库顺序已钉死，函数
  **正常返回**（`True`/`False` 语义同前）。`pyarrow` 缺失或损坏等其他
  失败场景也不抛，交由 `import ragas` 以普通异常报错。
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Any

logger = logging.getLogger(__name__)

# 垫片要伪造的模块全名，以及 ragas 期望从中导入的符号名。
_TARGET_MODULE = "langchain_community.chat_models.vertexai"
_TARGET_ATTR = "ChatVertexAI"

# 记录本进程是否已经由本垫片注入过，用于日志与幂等判断。
_INJECTED_FLAG = "_ragas_compat_injected"


def _build_chat_vertexai_stub() -> type:
    """构造一个「占位用」的 ChatVertexAI 类。

    设计目标是「被引用不会报错」，具体要满足 ragas 的用法：
      * 作为 `list` 的成员 —— 普通类对象即可；
      * 参与 `isinstance(llm, ChatVertexAI)` —— 必须是**真正的类**，
        不能是 `object()` 实例或 `types.SimpleNamespace`；
      * 被取类属性（`__name__` / `__mro__` / `__doc__`）—— 普通类天然满足。

    **刻意不继承任何 langchain / pydantic 类型**（见模块 docstring 的问题 ②）：
    langchain 1.x 的 `BaseChatModel` 导入链会拉起 `torch`，从而在外层
    `import ragas` 时触发 torch→pyarrow 的段错误。普通类的类型信息已足够
    支撑 `isinstance`，且零副作用。

    该类**永远不会被实例化**（本项目不使用 VertexAI）。若被误实例化，
    `__init__` 会抛出明确的 NotImplementedError，而不是难以理解的抽象类错误。
    """

    class ChatVertexAI:
        """`ChatVertexAI` 的占位类（RAGLoRA 兼容垫片，非真实实现）。

        仅为满足 ragas 的类型引用 / isinstance 检查而存在，**不实现**任何
        VertexAI 真实功能，也不继承任何 langchain 类型。
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError(
                "ChatVertexAI 是 RAGLoRA 的兼容垫片（绕过 ragas 上游 bug），"
                "只有类型占位作用，不提供真实功能，不可实例化。"
            )

    # 让日志/调试里显示的名字与真实类一致（类名已天然一致，这里只补全限定名）。
    ChatVertexAI.__qualname__ = "ChatVertexAI"
    ChatVertexAI.__module__ = _TARGET_MODULE
    return ChatVertexAI


def _preload_native_import_order() -> bool:
    """尽力预载 `pyarrow.dataset`，把原生库加载顺序钉在 `torch` 之前。

    返回 True 表示本次完成了预载 / 之前已预载；False 表示跳过。

    **必须在 torch 尚未导入时调用**：若 `torch` 已在 `sys.modules` 中，
    此刻再 `import pyarrow.dataset` 会**段错误**（见模块 docstring 问题 ②），
    所以此处显式检查并跳过，只记一条告警，绝不冒险。
    """
    if "pyarrow.dataset" in sys.modules:
        return True

    if "torch" in sys.modules:
        # 已经错过时机：现在导入 pyarrow.dataset 会崩进程。不尝试、只告警。
        logger.warning(
            "ragas_compat: 检测到 torch 已先于 pyarrow.dataset 导入，"
            "无法再修复原生库加载顺序；随后 import ragas 可能触发段错误。"
            "请确保在导入 torch（如 app.services.embed）之前先调用 "
            "ensure_ragas_compat()。"
        )
        return False

    try:
        import pyarrow.dataset  # noqa: F401

        logger.debug("ragas_compat: 已预载 pyarrow.dataset（钉住原生库加载顺序）")
        return True
    except Exception:  # pragma: no cover - pyarrow 缺失/损坏时静默降级
        logger.warning(
            "ragas_compat: 预载 pyarrow.dataset 失败；若环境存在 torch/pyarrow "
            "加载顺序冲突，import ragas 仍可能段错误。",
            exc_info=True,
        )
        return False


def _native_order_is_doomed() -> bool:
    """判断原生库加载顺序是否已经无可挽回（随后 `import ragas` 必段错误）。

    判定条件（**必须两条同时成立**）：

      * `torch` 已在 `sys.modules`（torch._C 已加载）；
      * `pyarrow.dataset` **尚未**加载。

    此时若继续 `import ragas`（其导入链 `ragas → datasets → pandas →
    pyarrow.dataset`）会触发 torch→pyarrow 的原生 DLL 顺序冲突，
    进程**段错误（退出码 139）**，不是 Python 异常，try/except 兜不住。

    反之，只要 `pyarrow.dataset` 已加载，原生库顺序就已钉死在 torch 之前，
    **无论 torch 在不在**都应当正常放行。
    """
    return "torch" in sys.modules and "pyarrow.dataset" not in sys.modules


def _module_exists() -> bool:
    """判断目标模块是否真实存在（而非由本垫片注入）。

    先看 `sys.modules`：若已存在且**不是**本垫片注入的，说明是真货或被别的
    垫片处理过，我们不应覆盖。
    再用 `importlib.util.find_spec` 探测磁盘上是否真的还有该模块。
    """
    existing = sys.modules.get(_TARGET_MODULE)
    if existing is not None:
        # 已由本垫片注入过 -> 不视为「真实存在」（幂等路径另行处理）。
        if getattr(existing, _INJECTED_FLAG, False):
            return False
        return True

    try:
        import importlib.util

        return importlib.util.find_spec(_TARGET_MODULE) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        # find_spec 在父包不存在等情况会抛异常，一律按「不存在」处理。
        return False


def ensure_ragas_compat() -> bool:
    """在 `import ragas` 之前调用，注入绕过上游 bug 的兼容垫片。

    返回 `True` 表示本次调用**实际注入**了垫片；`False` 表示无需注入
    （真实模块已存在，或此前已注入过——即幂等命中）。

    幂等：重复调用安全，不会重复注入、不会报错。

    异常：若检测到原生库加载顺序已经无可挽回（`torch` 已先于
    `pyarrow.dataset` 导入，见 `_native_order_is_doomed()`），**抛
    `RuntimeError`**。此时继续 `import ragas` 必然段错误，因此**不能**返回
    `True`（那会给调用方一个致命的假成功信号），也不能返回 `False`
    （会被误读成「安全的幂等命中」）——只有硬报错才能被看见。
    """
    # 0) 先钉住原生库加载顺序（幂等；torch 已导入时会安全跳过并告警）。
    #    放在最前面，确保在注入过程中任何可能的 import 之前完成。
    _preload_native_import_order()

    # 0.1) 预载之后若仍处于「必崩」状态，立即硬报错，绝不返回 True。
    #      刻意用独立判定而非依赖 _preload_native_import_order() 的返回值：
    #      后者在「pyarrow 缺失/损坏」等场景也会返回 False，那些场景不该抛
    #      （它们不是段错误，ragas 导入会以普通异常失败）。这里只认真正的
    #      段错误前提条件。
    if _native_order_is_doomed():
        raise RuntimeError(
            "ragas 兼容垫片无法安全初始化：检测到 torch 已先于 pyarrow.dataset "
            "导入，原生库（DLL）加载顺序冲突已无可挽回。\n"
            "后果：随后 import ragas 会直接段错误（segmentation fault，退出码 139），"
            "不是 Python 异常，try/except 无法捕获。\n"
            "修复：必须在**任何会拉起 torch 的模块之前**调用本函数。本项目会拉起 "
            "torch 的模块包括 app.services.embed、app.services.rerank —— 请把它们 "
            "的导入移到 ensure_ragas_compat() 之后；典型做法是把\n"
            "    from app.core.ragas_compat import ensure_ragas_compat\n"
            "    ensure_ragas_compat()\n"
            "放在入口脚本（如 backend/eval/run_ragas.py）的最前面。\n"
            f"当前状态：'torch' in sys.modules="
            f"{'torch' in sys.modules}，"
            f"'pyarrow.dataset' in sys.modules="
            f"{'pyarrow.dataset' in sys.modules}。"
        )

    # 1) 幂等命中：本进程已经注入过，直接返回。
    if getattr(sys.modules.get(_TARGET_MODULE), _INJECTED_FLAG, False):
        return False

    # 2) 真实模块存在则不干预（上游修复后此垫片自动失效）。
    if _module_exists():
        logger.debug("ragas_compat: %s 真实存在，跳过垫片注入", _TARGET_MODULE)
        return False

    # 3) 构造并注册伪模块。
    stub_cls = _build_chat_vertexai_stub()

    module = types.ModuleType(_TARGET_MODULE)
    module.__doc__ = (
        "RAGLoRA 兼容垫片：伪造已被 langchain-community 移除的 "
        "vertexai chat model 模块，仅供 ragas 做类型/isinstance 引用。"
    )
    module.__dict__[_TARGET_ATTR] = stub_cls
    # 便于排查：标记来源，并给出 `__all__`。
    module.__dict__[_INJECTED_FLAG] = True
    module.__dict__["__all__"] = [_TARGET_ATTR]
    # `__ragas_compat__` 只是**给人看的调试面包屑**（在 REPL / 调试器里一眼看出
    # 这个模块来自垫片），**不参与任何逻辑判断** —— 控制流一律走
    # `_INJECTED_FLAG`（`_ragas_compat_injected`）。两者同时存在是有意为之，
    # 不要互相替代。
    module.__dict__["__ragas_compat__"] = True

    sys.modules[_TARGET_MODULE] = module

    # 4) 同时挂到父包属性上，兼容 `langchain_community.chat_models.vertexai.X`
    #    这类点号访问路径（父包可能尚未导入，惰性处理）。
    try:
        import langchain_community.chat_models as _chat_models  # noqa: F401

        setattr(_chat_models, "vertexai", module)
    except Exception:  # pragma: no cover - 父包不可用时降级，但仍需可见
        # 用 warning 而非 debug：`langchain_community.chat_models.vertexai.X`
        # 这类点号访问路径是真实的兼容性保证，挂载失败会在默认日志级别下
        # 不可见，必须让调用方察觉到。
        logger.warning(
            "ragas_compat: 无法将垫片挂到父包 langchain_community.chat_models；"
            "经由点号路径访问 %s.%s 可能失败。",
            _TARGET_MODULE,
            _TARGET_ATTR,
            exc_info=True,
        )

    logger.info(
        "ragas_compat: 已注入兼容垫片 %s.%s（绕过 ragas 上游硬导入 bug）",
        _TARGET_MODULE,
        _TARGET_ATTR,
    )
    return True
