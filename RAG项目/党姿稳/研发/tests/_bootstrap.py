"""
_bootstrap.py — 测试公共准备

把项目根目录加入 sys.path，强制测试走本地模式，并把运行期数据指向临时目录，
避免测试污染开发环境的 storage/、logs/，以及**真实的 Milvus 与 Redis**。

注意：config.STORAGE_DIR 必须在导入业务模块之前改掉——session_memory 的会话文件
路径是导入期绑定的常量。各测试模块先 import 本模块再 import 业务模块即可。

同理，LOCAL_MODE / EMBEDDING_BACKEND 也在这里压掉：生产模式跑测试会把数据写进
老师要看的 Milvus，而且每次都要加载 2.3GB 的 bge-m3。
"""

from __future__ import annotations

import atexit
import shutil
import sys
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config  # noqa: E402

TEMP_DIR = Path(tempfile.mkdtemp(prefix="rag-test-"))
config.STORAGE_DIR = TEMP_DIR
config.LOG_DIR = TEMP_DIR

# 测试绝不碰真 Milvus / Redis，也不加载 bge-m3。
# 这几个开关都是"调用时读取"的，所以模块级改值有效。
config.LOCAL_MODE = True
config.EMBEDDING_BACKEND = "hash"

atexit.register(lambda: shutil.rmtree(TEMP_DIR, ignore_errors=True))

import vector_store  # noqa: E402

vector_store.reset_store()
