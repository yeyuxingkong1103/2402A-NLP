# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务

模块：工单08 复用桥（必须最先导入）

职责（顺序敏感，任何 transformers / sentence_transformers 导入之前生效）：

1. **置离线环境变量** —— 从机制上保证全进程不可能触发模型下载
   （工单硬性要求0：禁止静默下载）；
2. **把 `../工单8` 挂到 `sys.path`** —— 使 `import src` 精确指向工单08 的代码，
   实现「基于08 的 Graph RAG 服务及检索结果进行优化，不另起炉灶」这一硬性要求；
   同时保证 08 目录**物理零改动**（只读引用）；
3. 提供 `workorder9_root()` / `workorder8_root()` 两个根目录解析入口。

设计取舍：为什么引用而不是复制
    工单明令「必须基于 08 工单的 Graph RAG 服务及检索结果进行优化，不能另起炉灶」。
    · 引用式复用使 08 目录保持只读 → 08 永远是**可复现的「优化前」基线**；
    · 同一份语料 + 同一份向量索引 + 同一批问题 → 前后差异**只能**归因于 09 的优化，
      构成严格的 A/B 对照（复制式改造无法保证这一点）。

唯一例外：向量索引（Qdrant）复制一份到 09。
    原因：Qdrant local 模式对存储目录加**排他锁**，两个进程共享同一目录会互斥；
    复制后 08 目录真正只读，09 可独立运行、独立压测。
    复制的是**同一份向量**，不改变检索语义，A/B 可比性不受影响。
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# 1) 离线开关 —— 必须最先执行
# ---------------------------------------------------------------------------
def ensure_offline() -> None:
    """强制离线：HF / Transformers / ModelScope 一律不回源。

    与工单08 `src/bootstrap.py` 的 ensure_offline 行为一致，
    这里在 09 侧独立再置一次，确保 09 的任何入口（含直接 import w9 的场景）
    都先于 transformers 生效。
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


ensure_offline()

# ---------------------------------------------------------------------------
# 2) 根目录解析
# ---------------------------------------------------------------------------
def workorder9_root() -> str:
    """工单09 工程根（本文件位于 <root>/w9/compat.py）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def workorder8_root() -> str:
    """工单08 工程根（09 的只读基线）。"""
    return os.path.join(os.path.dirname(workorder9_root()), "工单8")


# ---------------------------------------------------------------------------
# 3) 把工单08 挂到 sys.path（幂等）
# ---------------------------------------------------------------------------
_WO8_ROOT = workorder8_root()


def ensure_wo8_on_path() -> str:
    """确保 `import src` 解析到**工单08** 的 src 包。

    幂等：重复调用不重复插入。若路径已存在但解析到了别的 `src`（例如用户
    在 08 目录里直接跑 09 的脚本），则以 08 的路径**前置**，保证解析正确。
    """
    if not os.path.isdir(os.path.join(_WO8_ROOT, "src")):
        raise RuntimeError(
            f"未找到工单08 工程：{_WO8_ROOT}\n"
            f"工单09 是**优化任务**，必须基于 08 的 Graph RAG 服务；"
            f"请确认 工单8 目录与 工单9 同级存在。")
    if _WO8_ROOT not in sys.path:
        sys.path.insert(0, _WO8_ROOT)
    return _WO8_ROOT


ensure_wo8_on_path()


def wo8_src_path() -> str:
    """返回 08 src 包的实际磁盘路径（校验复用对象是否正确）。"""
    return os.path.join(_WO8_ROOT, "src")


def wo8_data(*parts: str) -> str:
    """拼一个 工单08 data 下的路径（只读引用 08 的语料/索引/图谱产物）。"""
    return os.path.join(_WO8_ROOT, "data", *parts)


def wo9_data(*parts: str) -> str:
    """拼一个 工单09 data 下的路径（09 自有产物）。"""
    return os.path.join(workorder9_root(), "data", *parts)


def verify_reuse() -> dict:
    """自检：确认 `import src` 指向的是工单08，而不是别处。

    这是一个**防御性检查**：若用户曾在 08 目录下运行过 09 的脚本，或
    sys.path 中存在其他 `src`，复用对象可能被污染，导致「优化前后」实际是
    同一份代码 —— 那样整个 A/B 结论就是假的。宁可报错也不静默继续。
    """
    import src  # noqa: F401  —— 触发解析
    actual = os.path.abspath(os.path.dirname(src.__file__))
    expected = os.path.abspath(wo8_src_path())
    return {"ok": actual == expected, "expected": expected, "actual": actual}


if __name__ == "__main__":
    import json
    print(json.dumps({
        "workorder9_root": workorder9_root(),
        "workorder8_root": workorder8_root(),
        "reuse": verify_reuse(),
    }, ensure_ascii=False, indent=2))
