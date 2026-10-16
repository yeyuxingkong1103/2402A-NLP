# -*- coding: utf-8 -*-
"""工单9 pytest 配置：把 研发/ 与 研发/评估/ 加入 sys.path，并关闭字节码写入。

工单编号：人工智能NLP-RAG-Graph RAG 优化任务
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "研发", ROOT / "研发" / "评估", ROOT / "测试", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
