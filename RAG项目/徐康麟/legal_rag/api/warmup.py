"""启动预热的后台策略：等 ``lifespan`` 把 engine 建出来，再预热。

**为什么要单独一个模块**（真机踩坑，v3 第 1 次跑吃了这个亏）：
``AppState.engine`` 是 ``AppState.build()`` 里才造的，而 ``build()`` 在 FastAPI 的
``lifespan`` 里执行 —— 比 ``create_app()`` 返回**晚**。启动预热如果紧跟着
``create_app()`` 起线程，拿到的是 ``None``：

    WARNING | __main__ | 预热线程异常（不影响服务）：AttributeError: 'NoneType' object has no attribute 'warm_up'

结果是**预热静默失效**（只留一条 WARNING）、BM25 仍由第一题懒触发，
前 ~10 题落在「资料还在预热」的降级窗口里 —— v3 的评测数据就是这么被污染的。

因此这里的逻辑是「**轮询等 engine 出现**（默认最多 60s）→ 再预热」，并且：
* 预热失败/超时**只记日志**，绝不影响服务启动；
* 必须用**真实角色**预热（BM25 索引按 scope 分签名，不带 role_id 建出来的索引运行时用不上）。
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from ..roles import DEFAULT_ROLE_ID, get_role

logger = logging.getLogger(__name__)

#: 等 engine 出现的最长时间（lifespan 正常是毫秒级；给足余量）
WAIT_SECONDS = 60.0
#: 轮询间隔
POLL_SECONDS = 0.5


def wait_for_engine(app: Any, *, timeout: float = WAIT_SECONDS,
                    poll: float = POLL_SECONDS) -> Any | None:
    """轮询等 ``app.state.app_state.engine`` 就绪；超时返回 ``None``（并记 warning）。"""
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        engine = getattr(getattr(getattr(app, "state", None), "app_state", None), "engine", None)
        if engine is not None:
            return engine
        if time.monotonic() >= deadline:
            logger.warning("预热放弃：等了 %.0fs 仍拿不到 engine（lifespan 没跑起来？）"
                           "—— 不影响服务，只是首查仍会冷加载、BM25 仍要懒建",
                           timeout)
            return None
        time.sleep(max(0.01, poll))


def start_warmup(app: Any, *, role_id: str = DEFAULT_ROLE_ID,
                 timeout: float = WAIT_SECONDS,
                 warm: Callable[[Any, Any], dict] | None = None) -> threading.Thread:
    """后台线程里完成「等 engine → 预热」。返回该线程（便于测试 join）。"""
    def _run() -> None:
        try:
            engine = wait_for_engine(app, timeout=timeout)
            if engine is None:
                return
            role = get_role(role_id)
            runner = warm or (lambda eng, r: eng.warm_up(r))
            result = runner(engine, role)
            logger.info("预热线程完成：%s", result)
        except Exception as exc:  # noqa: BLE001 - 预热绝不能把服务带崩
            logger.warning("预热线程异常（不影响服务，首查仍会冷加载）：%s: %s",
                           type(exc).__name__, exc)

    thread = threading.Thread(target=_run, name="warmup", daemon=True)
    thread.start()
    return thread
