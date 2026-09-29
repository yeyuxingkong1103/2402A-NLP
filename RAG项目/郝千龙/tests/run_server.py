# -*- coding: utf-8 -*-
"""压测用服务启动脚本：在导入项目模块前设置环境变量，确保纯检索模式。"""
import os
import sys
from pathlib import Path

# 项目根目录（run_server.py 位于 tests/ 下，往上回溯一层）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 必须在 import api 之前设置，覆盖 .env 中的 LLM_API_KEY
os.environ["LLM_API_KEY"] = ""
os.environ["APP_ENV"] = "testing"

# 把工作目录切到项目根，保证 .env 里的相对路径（如 sqlite:///data/app.db）
# 解析到项目根下的 data/app.db，而不是当前运行目录
os.chdir(_PROJECT_ROOT)

# 让 Python 能找到项目根下的 api 模块
sys.path.insert(0, str(_PROJECT_ROOT))

import uvicorn

uvicorn.run("api:app", host="127.0.0.1", port=8000)
