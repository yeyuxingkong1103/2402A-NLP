# -*- coding: utf-8 -*-
"""设备（CPU / GPU）解析工具。

统一决定「向量化 / 重排 / RAGAS 评估」等需要加载本地模型的组件，应把模型放到
CPU 还是 CUDA（GPU）上运行：

    - ``device="auto"``（默认）：自动探测 ``torch.cuda.is_available()``，
      有 CUDA 用 ``cuda``，否则回退 ``cpu``；
    - ``device="cuda"`` / ``"cuda:0"``：显式指定 GPU，CUDA 不可用时回退 ``cpu`` 并告警；
    - ``device="cpu"``：强制 CPU。

为什么要单独封装：``sentence-transformers`` / ``ragas`` 的 ``device`` 参数并不理解
``"auto"`` 这个取值，所以必须在真正加载模型前先把它解析成 ``"cuda"`` 或 ``"cpu"``。

用法：

    from rag2.device import resolve_device

    dev = resolve_device("auto")   # -> "cuda"（有 GPU）或 "cpu"（无 GPU）
"""

from __future__ import annotations

import logging

logger = logging.getLogger("rag2.device")


def cuda_available() -> bool:
    """探测 torch CUDA 是否可用（懒加载 torch，未装 torch 或探测失败均视为不可用）。"""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - 无 torch / 驱动异常等，一律按「无 GPU」处理
        return False


def resolve_device(device: str | None = None) -> str:
    """把请求的设备名解析为模型可直接加载的设备字符串。

    参数：
        device: 期望设备（"auto" / "cuda" / "cuda:0" / "cpu" / None）。
                "auto" 或 None：自动选择（有 CUDA 用 cuda，否则 cpu）。

    返回：
        "cuda" / "cuda:0" / "cpu" 之一。
    """
    requested = (device or "auto").strip().lower()

    if requested in ("", "auto"):
        resolved = "cuda" if cuda_available() else "cpu"
        if resolved == "cuda":
            logger.info("device=auto → 检测到 CUDA，模型将在 GPU 上运行")
        else:
            logger.info("device=auto → 未检测到 CUDA，模型回退 CPU")
        return resolved

    if requested.startswith("cuda"):
        if cuda_available():
            return requested
        logger.warning("配置 device=%r 但 CUDA 不可用，回退 cpu", device)
        return "cpu"

    if requested.startswith("cpu"):
        return "cpu"

    logger.warning("未知设备 %r，按 auto 处理", device)
    return "cuda" if cuda_available() else "cpu"
