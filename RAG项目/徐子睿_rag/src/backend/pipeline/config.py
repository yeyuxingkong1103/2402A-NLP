# -*- coding: utf-8 -*-
"""pipeline/config.py —— 构建管线的路径与调参常量。

在链路中的位置：
    pipeline 包内各模块共用这些常量；本文件不导入任何同包模块，因此是依赖的最内层
    （避免 config 与其它模块形成循环导入）。

注意 BASE_DIR 的层级：
    本文件位于 backend/pipeline/ 下，比原来的 backend/pipeline.py 深了一层，
    所以上溯到项目根要三级（parents[2]）而不是两级（parent.parent）。
    这是"模块改包"时最容易出错、且报错信息最不直观的一处。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # 项目根目录（本文件位于 backend/pipeline/ 下，故上跳三级）
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")  # Ollama 服务地址，换机器/远端部署时用环境变量覆盖
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")                  # 向量化模型：bge-m3 输出 1024 维，中英双语，与 Milvus 索引维度对齐

PDF_DIR = BASE_DIR / "data" / "pdfs"                              # 上传的原始 PDF 落盘位置
MAX_CHUNK = 400  # 分块目标长度（字符）。太大：一个问题会被无关内容淹没，检索精度下降；太小：上下文断裂，答案不成句
OVERLAP = 40     # 超长文本硬切时的重叠字符数，避免一句话正好被切断、语义被劈成两半丢失

# 章节标题，形如 "1 范围" / "3.2 设备要求"。要求编号后紧跟中文或字母，且整行不超过 40 字，
# 这样才不会把正文里出现的数字误判成标题。
HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+[一-鿿A-Za-z].{0,40}$")
# 目录点线，如 "1 范围 ······ 1"。连续 4 个以上点线符号才认定是目录，避免正文省略号被误杀
TOC_RE = re.compile(r"[·\.。…]{4,}|(?:\s[·\.。…]){4,}")
# 页眉页脚：罗马数字页码（Ⅰ Ⅱ Ⅲ / IVXLCDM）或 "GB/T 44653-2024" 这类标准号反复出现的行
HEADER_RE = re.compile(r"^(?:[Ⅰ-ⅩIVXLCDM]+|GB/T\s*\d+\s*[—-]\s*\d{4})$")
