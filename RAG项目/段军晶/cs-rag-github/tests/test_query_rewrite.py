# -*- coding: utf-8 -*-
"""查询改写测试

核心约定：rewrite() 永不抛异常，失败一律返回空列表（降级到只用原问题）。

测试全部使用假 LLM（monkeypatch 替换 qr.get_chat_llm），不真实调用 API，
因此可纳入默认的快速测试集（不带 slow 标记）。
"""

import pytest

from backend.rag_pipeline.query_rewrite import QueryRewriter, _parse_queries

import backend.rag_pipeline.query_rewrite as qr


def test_解析_多行输出拆成多条():
    assert _parse_queries("查询一\n查询二\n查询三", max_n=3) == ["查询一", "查询二", "查询三"]


def test_解析_去除序号与项目符号():
    assert _parse_queries("1. 查询甲\n2. 查询乙", max_n=2) == ["查询甲", "查询乙"]


def test_解析_按_max_n_截断():
    assert _parse_queries("a\nb\nc\nd", max_n=2) == ["a", "b"]


def test_解析_去重且去空行():
    assert _parse_queries("a\n\nA\na\nb", max_n=5) == ["a", "b"]


def test_解析_空输入返回空列表():
    assert _parse_queries("", max_n=3) == []
    assert _parse_queries("   \n  \n", max_n=3) == []


def test_解析_不误剥标准编号的小数点():
    """回归：'3.5 版本 质量特性' 不得被削成 '5 版本 质量特性'。

    旧实现的正则 `\\d+[\\.\\)、]` 会把 "3." 当成行首序号剥掉。
    标准文本里「3.5 术语」「4.1 软件质量模型」这类编号极常见。
    """
    assert _parse_queries("3.5 版本 质量特性", max_n=3) == ["3.5 版本 质量特性"]
    assert _parse_queries("4.1 软件质量模型", max_n=3) == ["4.1 软件质量模型"]
    # 真正的序号（后跟空白）仍要剥离
    assert _parse_queries("3. 版本 质量特性", max_n=3) == ["版本 质量特性"]


def test_解析_行尾的孤立序号被剥离后视为空行():
    """序号后跟行尾时同样剥离，剥离后为空则该行被丢弃"""
    assert _parse_queries("1.\n2)", max_n=3) == []
    assert _parse_queries("1.\n有内容", max_n=3) == ["有内容"]


def test_解析异常时返回空列表而非抛错(monkeypatch):
    """契约自证：解析步骤也在保护范围内，不依赖外部类型约定"""

    def _boom(*a, **k):
        raise ValueError("模拟解析异常")

    monkeypatch.setattr(qr, "_parse_queries", _boom)

    class _Ok:
        def chat(self, *a, **k):
            return "任意内容"

    monkeypatch.setattr(qr, "get_chat_llm", lambda: _Ok())
    assert QueryRewriter().rewrite("任意问题") == []


def test_LLM_异常时返回空列表而非抛错(monkeypatch):
    class _Boom:
        def chat(self, *a, **k):
            raise RuntimeError("模拟 LLM 故障")

    monkeypatch.setattr(qr, "get_chat_llm", lambda: _Boom())
    assert QueryRewriter().rewrite("任意问题") == []


def test_LLM_返回空时返回空列表(monkeypatch):
    class _Empty:
        def chat(self, *a, **k):
            return ""

    monkeypatch.setattr(qr, "get_chat_llm", lambda: _Empty())
    assert QueryRewriter().rewrite("任意问题") == []


def test_超时返回空列表(monkeypatch):
    import time as _time

    class _Slow:
        def chat(self, *a, **k):
            _time.sleep(3)
            return "慢查询"

    monkeypatch.setattr(qr, "get_chat_llm", lambda: _Slow())
    r = QueryRewriter(timeout=1)
    assert r.rewrite("任意问题") == []


def test_超时不挂起_耗时应远小于任务本身耗时(monkeypatch):
    """锁定超时实现：调用方必须在超时点拿回控制权，不得阻塞等任务跑完。

    - `with ThreadPoolExecutor(...)` 退出时执行 shutdown(wait=True)，阻塞等任务
      跑完，超时形同虚设（Task 5 修复轮实测 3.00s）。
    - 显式 shutdown(wait=False) 能让调用方 1.00s 返回，但工作线程仍是非 daemon，
      进程退出时会被 atexit 的 _python_exit join（见下一条 daemon 测试）。

    本测试断言：超时场景下 rewrite() 的实际耗时明显小于假 LLM 自身的
    sleep 时长（3s），即「超时真的生效、不挂起」。
    """
    import time as _time

    class _Slow:
        def chat(self, *a, **k):
            _time.sleep(3)
            return "慢查询"

    monkeypatch.setattr(qr, "get_chat_llm", lambda: _Slow())
    r = QueryRewriter(timeout=1)

    started = _time.time()
    assert r.rewrite("任意问题") == []
    elapsed = _time.time() - started

    # 任务自身耗时 3s；正确的实现约 1s 返回。阈值取 2s：
    # 既容忍 CI 抖动，又能把「阻塞等 3s」的写法判为失败。
    assert elapsed < 2.0, (
        f"超时未生效：elapsed={elapsed:.2f}s，"
        f"接近/超过被调用任务的 3s（疑似用了 with ThreadPoolExecutor）"
    )


def test_超时后残留的工作线程必须是_daemon(monkeypatch):
    """区分实现方式的测试：残留线程必须是 daemon，否则会拖住进程退出。

    ThreadPoolExecutor 的工作线程是非 daemon 线程，concurrent.futures.thread
    在 atexit 注册的 _python_exit 会 join 所有遗留线程 —— 于是
    shutdown(wait=False) 只让调用方在超时点拿回控制权，**并不减少进程墙钟
    时间**（实测：rewrite 1.00s 返回，进程退出再 +3.00s）。一次性脚本
    （如 Task 9 的对照实验）会被拖住，最坏 100s+。

    本测试锁定：超时后仍在跑的那个工作线程是 daemon，进程可立即退出。
    """
    import threading
    import time as _time

    class _Slow:
        def chat(self, *a, **k):
            _time.sleep(3)
            return "慢查询"

    monkeypatch.setattr(qr, "get_chat_llm", lambda: _Slow())
    r = QueryRewriter(timeout=1)

    before = set(threading.enumerate())
    assert r.rewrite("任意问题") == []          # timeout=1 先返回
    new_threads = [t for t in threading.enumerate() if t not in before]

    # 假 LLM 还在 sleep(3)，此刻线程必然仍在运行，否则本测试失去意义
    assert new_threads, "预期能观察到超时后仍在运行的工作线程"
    assert all(t.daemon for t in new_threads), (
        f"残留线程必须为 daemon（否则会被 atexit 的 _python_exit join，拖住进程退出）："
        f"{[(t.name, t.daemon) for t in new_threads]}"
    )


def test_线程启动失败时返回空列表而非抛错(monkeypatch):
    """覆盖度回归：线程/FD 耗尽时 Thread.start() 抛 RuntimeError，不得逃出 rewrite()。

    旧写法（ThreadPoolExecutor）的线程是在 try 内的 pool.submit() 里启动的，
    该异常会被 except Exception 接住并降级；改造为 daemon 线程后，若把
    start() 放在 try 之外，这层覆盖就丢了，「永不抛异常」契约会被击穿
    （长驻服务里超时泄漏的 daemon 线程会累积，使其并非纯理论）。
    """
    class _FakeThread:
        """start() 必定抛 RuntimeError 的线程替身（模拟线程/FD 耗尽）"""

        def __init__(self, *a, **k):
            pass

        def start(self) -> None:
            raise RuntimeError("can't start new thread")

    class _FakeThreading:
        """替换模块内的 threading 引用；不触碰真实 threading.Thread"""

        Thread = _FakeThread

    class _Ok:
        def chat(self, *a, **k):
            return "任意内容"

    monkeypatch.setattr(qr, "get_chat_llm", lambda: _Ok())
    monkeypatch.setattr(qr, "threading", _FakeThreading)

    assert QueryRewriter().rewrite("任意问题") == []


def test_配置项存在时以_settings_为准(monkeypatch):
    """锁定「以 settings 为准」：两个字段随 settings 变化，不写死默认值。

    query_rewrite_max_queries / query_rewrite_timeout 已在 backend/config.py
    （十四、查询改写）中定义。本测试用一个取值不同于默认值的 settings 替身，
    确保模块读的是 settings 而非硬编码默认值。
    """

    class _SettingsStub:
        query_rewrite_max_queries = 2
        query_rewrite_timeout = 7
        llm_max_tokens = 4096

    monkeypatch.setattr(qr, "settings", _SettingsStub())
    r = QueryRewriter()
    assert r.max_queries == 2
    assert r.timeout == 7


def test_显式入参优先于配置(monkeypatch):
    """显式传入的 max_queries / timeout 优先，便于测试与实验调整"""
    r = QueryRewriter(max_queries=4, timeout=3)
    assert r.max_queries == 4
    assert r.timeout == 3


def test_正常改写返回解析结果(monkeypatch):
    class _Ok:
        def chat(self, *a, **k):
            return "功能充分性 功能性 质量特性"

    monkeypatch.setattr(qr, "get_chat_llm", lambda: _Ok())
    out = QueryRewriter().rewrite("功能充分性属于哪个质量特性的子特性？")
    assert out == ["功能充分性 功能性 质量特性"]
