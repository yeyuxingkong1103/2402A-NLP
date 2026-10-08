"""纯函数自检的执行器。**不连 Milvus、不加载权重、不读 `.env`。**

断言本身与其依赖的构造语料在 `selftest_cases.py`；本模块只负责**跑**它们：
逐条执行、收集失败、渲染成终端可读的若干行。

---

## 为什么 `selfcheck` 什么前置都不要

本项目未安装 pytest（research R11 —— 引入测试框架是独立决策，不该夹带在检索
特性里），纯函数的断言因此必须有一个可执行的落点，这就是它。

而它必须**能在 Milvus 没起来、权重没下载的情况下跑** —— 否则"先验证公式对不对"
这件事就要排在"把整套环境搭起来"之后，而公式错了的话，后面搭起来的东西
给出的排名会是错的、且看起来很正常。

## 为什么失败不中断

逐条跑完全部 10 项再报告，而不是第一处失败就退出。若断言 A 的实现坏了，
一次跑完能看到"它带坏了哪几条"，这比修一条跑一次快得多 —— 而这几条本来就
共享同一批底层函数（`fuse.merge` / `lexical.*`）。
"""

from __future__ import annotations

from .selftest_cases import CHECKS

__all__ = ["run_selfcheck"]


def run_selfcheck() -> tuple[int, list[str]]:
    """跑全部断言。返回 `(失败项数, 输出行)`。"""

    failures = 0
    lines: list[str] = []

    for name, check in CHECKS:
        try:
            check()
        except AssertionError as exc:
            failures += 1
            lines.append("  ✗ %s —— %s" % (name, exc))
        except Exception as exc:  # noqa: BLE001 —— 自检本身出错也要报告
            failures += 1
            lines.append(
                "  ✗ %s —— 检查抛异常：%s: %s" % (name, type(exc).__name__, exc)
            )
        else:
            lines.append("  ✓ %s" % name)

    lines.append("")
    lines.append(
        "自检%s（%d 项，%d 项失败）"
        % ("通过" if not failures else "未通过", len(CHECKS), failures)
    )
    return failures, lines
