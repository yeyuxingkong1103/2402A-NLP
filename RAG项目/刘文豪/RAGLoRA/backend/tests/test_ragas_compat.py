# -*- coding: utf-8 -*-
"""`app.core.ragas_compat` 垫片的守护测试。

这组测试在守护什么
==================

`backend/app/core/ragas_compat.py` 是绕过 `import ragas` 两处环境/上游问题的
兼容垫片。它的正确性此前只靠一次性临时脚本验证，没有任何自动化回归保护。
最要命的是它同时踩在两条隐性约定上：

  1. **垫片自身绝不能引入 `torch`。**
     本环境 `torch 2.5.1+cu121` 与 `pyarrow 24.0.0` 存在原生 DLL 加载顺序
     冲突：若 torch 先于 `pyarrow.dataset` 加载，随后 `import ragas` 会
     **段错误（退出码 139）**，不是 Python 异常 —— try/except 兜不住。
     初版垫片曾让占位类继承 langchain 的 `BaseChatModel`，而该导入链会拉起
     torch，垫片自己就成了崩溃源。本组测试把「垫片不引入 torch」钉死为
     回归不变量。

  2. **错误路径必须硬报错，而不是静默返回「成功」。**
     若已经错过时机（torch 已导入、pyarrow.dataset 未加载），
     `ensure_ragas_compat()` 必须抛 `RuntimeError`。否则调用方拿到 True 后
     紧接着 `import ragas` 就会段错误，进程直接死且无任何可诊断信息。

设计约定：所有会污染 `sys.modules` 的操作都在**子进程**里跑
（`subprocess.run([sys.executable, "-c", ...])`），保证测试进程自身不被注入
或拉起 torch；也正因如此，本模块**不得**导入任何 `app.services.*` 模块。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# backend/ 目录（本文件位于 backend/tests/ 下），让子进程能 import app.core.*
_BACKEND_DIR = str(Path(__file__).resolve().parents[1])

# 段错误的退出码：POSIX shell 里是 139（= 128 + SIGSEGV 11），
# Windows 上是 0xC0000005；Python subprocess 在 Windows 也可能报负数
# （-1073741819 == 0xC0000005 as signed int）。逐一排除。
_SIGSEGV_EXIT_CODES = {139, -11, -1073741819, 3221225477}


def _run_py(code: str, timeout: float = 180.0) -> subprocess.CompletedProcess:
    """在子进程里跑一段 Python，返回 CompletedProcess（文本模式）。"""
    env = dict(os.environ)
    # 保证子进程能解析 app.core.*（并让 backend 优先于任何已安装的同名包）。
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([_BACKEND_DIR, existing]) if existing else _BACKEND_DIR

    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=_BACKEND_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _assert_not_segfaulted(proc: subprocess.CompletedProcess) -> None:
    """断言子进程不是段错误退出。"""
    assert proc.returncode not in _SIGSEGV_EXIT_CODES, (
        f"子进程段错误！returncode={proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def test_shim_does_not_import_torch() -> None:
    """【最关键】垫片自身不引入 torch，从而不会触发 torch/pyarrow 段错误。

    在全新子进程里 import 垫片并调用 `ensure_ragas_compat()`，
    断言 `"torch" not in sys.modules`。
    """
    code = (
        "import sys\n"
        "# 前提校验：干净进程里 torch 本来就不该在。\n"
        "assert 'torch' not in sys.modules, 'subprocess 起始状态就带了 torch'\n"
        "import app.core.ragas_compat as rc\n"
        "# 仅 import 垫片本身就不应拉起 torch。\n"
        "assert 'torch' not in sys.modules, 'import 垫片时引入了 torch'\n"
        "rc.ensure_ragas_compat()\n"
        "# 注入过程同样不得拉起 torch。\n"
        "assert 'torch' not in sys.modules, 'ensure_ragas_compat() 引入了 torch'\n"
        "print('OK_NO_TORCH')\n"
    )
    proc = _run_py(code)
    _assert_not_segfaulted(proc)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    assert "OK_NO_TORCH" in proc.stdout, proc.stdout


def test_ensure_ragas_compat_is_idempotent() -> None:
    """幂等：连续调用两次不报错，首次 True、第二次 False。"""
    code = (
        "from app.core.ragas_compat import ensure_ragas_compat\n"
        "first = ensure_ragas_compat()\n"
        "second = ensure_ragas_compat()\n"
        "assert first is True, f'首次应为 True，实为 {first!r}'\n"
        "assert second is False, f'二次应为 False，实为 {second!r}'\n"
        "print('OK_IDEMPOTENT', first, second)\n"
    )
    proc = _run_py(code)
    _assert_not_segfaulted(proc)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    assert "OK_IDEMPOTENT True False" in proc.stdout, proc.stdout


def test_does_not_override_real_module() -> None:
    """真实模块存在时不注入、返回 False，且原有模块对象不被覆盖。

    用「往 `sys.modules` 预置一个无 `_ragas_compat_injected` 标记的同名模块」
    来模拟上游修复后真实存在的情形。
    """
    code = (
        "import sys, types\n"
        "TARGET = 'langchain_community.chat_models.vertexai'\n"
        "sentinel = types.ModuleType(TARGET)\n"
        "sentinel.ChatVertexAI = object  # 任意非垫片内容，用于验证未被覆盖\n"
        "sys.modules[TARGET] = sentinel\n"
        "from app.core.ragas_compat import ensure_ragas_compat\n"
        "result = ensure_ragas_compat()\n"
        "assert result is False, f'真实模块存在时应返回 False，实为 {result!r}'\n"
        "assert sys.modules[TARGET] is sentinel, '垫片覆盖了真实模块！'\n"
        "assert sys.modules[TARGET].ChatVertexAI is object, '真实模块属性被改写！'\n"
        "print('OK_NOT_OVERRIDDEN')\n"
    )
    proc = _run_py(code)
    _assert_not_segfaulted(proc)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    assert "OK_NOT_OVERRIDDEN" in proc.stdout, proc.stdout


def test_torch_already_imported_raises_runtime_error() -> None:
    """torch 先导入时抛 RuntimeError，且进程不得段错误（退出码非 139）。

    这条同时验证「不再是静默的 True」——若修复回退成返回 True/False，
    本测试会因未捕到 RuntimeError 而以非 1 退出码失败。
    """
    code = (
        "import sys\n"
        "import torch  # 先把 torch 拉起来，制造「必崩」前置状态\n"
        "from app.core.ragas_compat import ensure_ragas_compat\n"
        "try:\n"
        "    result = ensure_ragas_compat()\n"
        "except RuntimeError as exc:\n"
        "    msg = str(exc)\n"
        "    assert 'app.services.embed' in msg, '错误消息未提及 app.services.embed'\n"
        "    assert 'app.services.rerank' in msg, '错误消息未提及 app.services.rerank'\n"
        "    print('GOT_RUNTIME_ERROR')\n"
        "    sys.exit(1)\n"
        "print(f'NO_ERROR result={result!r}')  # 修复回退时会走到这里\n"
        "sys.exit(0)\n"
    )
    proc = _run_py(code)
    # 关键：进程绝不能是段错误。
    assert proc.returncode not in _SIGSEGV_EXIT_CODES, (
        f"子进程段错误（退出码 {proc.returncode}）——错误路径未在 import ragas 前"
        f"拦下。\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert proc.returncode == 1, (
        f"期望退出码 1（已抛 RuntimeError），实为 {proc.returncode}。\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "GOT_RUNTIME_ERROR" in proc.stdout, proc.stdout
    assert "NO_ERROR" not in proc.stdout, "未抛异常，修复失效"
