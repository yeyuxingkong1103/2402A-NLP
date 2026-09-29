"""tests/conftest.py —— pytest 的全局配置与路径准备。

在链路中的位置：
    pytest 在收集测试时自动导入本文件（无需在测试里 import），
    它的作用是让所有测试模块都能 import 到项目里的模块。

为什么需要它：
    pytest 只把"测试文件所在目录"加入 sys.path，
    而本项目的测试要导入两处东西：
        src.*           新架构（需要项目根在 path 上）
        pipeline/retrieval  老架构（需要 backend/ 在 path 上）
    在 conftest 里一次把路径准备好，各个测试文件就不必各写一遍。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # tests/ 的上一级即项目根
sys.path.insert(0, str(ROOT))
