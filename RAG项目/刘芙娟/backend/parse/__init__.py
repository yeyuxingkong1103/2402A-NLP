# -*- coding: utf-8 -*-
"""S2 解析的共享定义：退出码、仓库路径、默认权重目录、ScriptError。

只放被多方共享、且不属于任何一方的常量与异常，不放逻辑——因此本模块不 import
包内其它模块，无循环导入风险。
"""

from __future__ import annotations

import os
from pathlib import Path

EXIT_OK, EXIT_ARG, EXIT_VALIDATION, EXIT_DEPENDENCY = 0, 1, 2, 3

BACKENDS = ("pipeline", "vlm-engine")
DEFAULT_BACKEND = "pipeline"

# 三层 parent：本文件 → parse/ → backend/ → 仓库根。
# 不能写成两层——那会指向 backend/，产物会被静默写到 backend\data\parsed\，
# 不报错，只是找不着。
REPO_ROOT = Path(
    os.environ.get("MED_RAG_REPO_ROOT") or Path(__file__).resolve().parent.parent.parent)
SOURCE_DIR = REPO_ROOT / "data" / "source_data"
PARSED_DIR = REPO_ROOT / "data" / "parsed"
DEFAULT_CONFIG_PATH = REPO_ROOT / ".mineru" / "mineru.json"

# 两套权重均已在本地。此处仅为默认值，可用命令行选项覆盖。
DEFAULT_PIPELINE_MODELS = Path(
    r"C:\Users\Lenovo\.cache\modelscope\models"
    r"\OpenDataLab--PDF-Extract-Kit-1.0\snapshots\master")
DEFAULT_VLM_MODELS = Path(
    r"C:\Users\Lenovo\.cache\modelscope\models"
    r"\OpenDataLab--MinerU2.5-Pro-2605-1.2B\snapshots\master")


class ScriptError(Exception):
    """携带退出码的显式失败。禁止用 except 吞掉它。"""

    def __init__(self, exit_code: int, message: str) -> None:
        super().__init__(message)
        self.exit_code = exit_code
