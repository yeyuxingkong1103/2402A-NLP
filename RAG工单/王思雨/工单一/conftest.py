# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 本文件职责：pytest 根配置 —— 把项目根目录加入 sys.path，使 tests/ 下可以直接 import app 包。
# 用法：在项目根目录执行 python -m pytest tests -v

import sys                        # 修改模块搜索路径
from pathlib import Path          # 路径处理

ROOT = Path(__file__).resolve().parent                       # 项目根目录（本文件所在目录）
if str(ROOT) not in sys.path:                                # 避免重复插入
    sys.path.insert(0, str(ROOT))                            # 让 tests/ 能 import app
