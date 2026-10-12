# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-金融问答系统部署
配置：数据路径（本地 / 容器内均可用，通过环境变量 DATA_DIR 覆盖）。
"""
import os

# 容器内通过 Docker 卷挂载到 /data；本地默认使用附件目录
DATA_DIR = os.environ.get("DATA_DIR", r"D:\软件\QQ\data\RAG 工单\附件")
PDF1 = os.path.join(DATA_DIR, "招股说明书1.pdf")
PDF2 = os.path.join(DATA_DIR, "招股说明书2.pdf")

# 服务配置
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 3
