# -*- coding: utf-8 -*-
"""pytest 公共配置

职责：
    1. 把项目根目录加入 sys.path，使测试能 import backend / scripts
    2. 说明 slow 标记的含义

关于 slow 标记：
    标记那些「需要加载大模型（约 2.3GB）或依赖外部服务（Milvus）」的慢速测试。
    pytest.ini 中已配置 addopts = -m "not slow"，因此默认只跑快速单测。
    需要跑慢速测试时显式指定：

        .venv/Scripts/python.exe -m pytest tests/ -v -m slow

    该标记的**说明文字写在 Python 文件里而非 pytest.ini**：Windows 下
    pytest 以 GBK 读取 ini 文件，中文会导致解码失败。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
