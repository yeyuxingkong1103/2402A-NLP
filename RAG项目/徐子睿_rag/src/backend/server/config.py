# -*- coding: utf-8 -*-
"""server/config.py —— 服务路径、模型地址、拒答文案与日志配置。

在链路中的位置：
    server 包内各模块共用：registry 用 DATA_DIR / DOCS_JSON 定位登记表，
    routes_* 用 logger 打日志，answer 用 LLM_URL / LLM_MODEL / REFUSAL 调模型与拒答，
    __init__ 用 STATIC_DIR 挂载前端目录。

注意 BASE_DIR 的层级：
    本文件位于 backend/server/ 下，比原来的 backend/server.py 深了一层，
    所以上溯到项目根要用 parents[2]。这是"模块改包"时最容易出错的一处。

PDF_DIR.mkdir 放在本文件（而不是等到用的时候）：
    上传接口会把 PDF 写进 PDF_DIR，目录不存在时写入会失败。
    在导入配置时就建好，比在每个使用点都判断一次更可靠。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

try:
    from ..pipeline import PDF_DIR
except ImportError:
    from pipeline import PDF_DIR

BASE_DIR = Path(__file__).resolve().parents[2]  # 项目根目录（本文件位于 backend/server/ 下，故上跳三级）

STATIC_DIR = BASE_DIR / "static"          # 前端页面目录，最后挂载到 "/"
DATA_DIR = BASE_DIR / "data"              # 数据根目录，知识库体积统计就是扫这里
DOCS_JSON = DATA_DIR / "documents.json"   # 文档登记表：记录每份 PDF 的页数/块数/构建时间
LLM_URL = os.getenv("LLM_URL", "http://localhost:11434/api/chat")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2:7b")
REFUSAL = "知识库中未找到相关依据，无法回答。"  # 固定的拒答文案。用常量而不是各处硬编码，
                                              # 是为了让下面的引用清空判断能精确匹配到它

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)
PDF_DIR.mkdir(parents=True, exist_ok=True)  # 首次启动时目录可能还不存在，先建出来
