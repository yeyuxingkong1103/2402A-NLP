# -*- coding: utf-8 -*-
# 【全局配置模块 · config.py】集中管理金融问答容器服务的路径、解析、分块与检索参数
# 工单编号：人工智能NLP-RAG-金融问答系统部署
# 说明：本模块为容器化部署专用的自包含配置，全部参数支持环境变量覆盖，
#       不依赖任何其他工单目录，便于在 Docker 内通过环境变量调整行为。

"""全局配置：路径自动适配“容器 / 本机”两种运行形态。

- 容器内：数据目录固定为挂载点 ``/data``，索引目录固定为 ``/app/index``；
- 本机直跑（python service.py）：自动回退到工程目录 ``02_研发/data`` 与
  ``02_研发/index``，与 docker-compose 的 bind mount 布局一一对应。
"""
import os
from pathlib import Path

# app 目录（本文件所在目录），其上级即 02_研发 工程目录
_APP_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _APP_DIR.parent


def _default_data_dir() -> str:
    """解析默认语料目录：环境变量 > 容器挂载点 /data > 本机工程 data 目录。

    :return: 数据目录绝对路径字符串
    """
    env_dir = os.environ.get("FINQA_DATA_DIR")
    if env_dir:
        return env_dir
    if Path("/data").exists():  # Dockerfile 中显式创建的挂载点
        return "/data"
    return str(_PROJECT_DIR / "data")


def _default_index_dir() -> str:
    """解析默认索引持久化目录：环境变量 > 容器卷 /app/index > 本机工程 index 目录。

    :return: 索引目录绝对路径字符串
    """
    env_dir = os.environ.get("FINQA_INDEX_DIR")
    if env_dir:
        return env_dir
    if Path("/app").exists():  # 容器 WORKDIR
        return "/app/index"
    return str(_PROJECT_DIR / "index")


class Config:
    """系统配置数据类（解析、分块、检索、服务参数集中管理）。"""

    # ---------- 路径配置 ----------
    data_dir: str = _default_data_dir()
    index_dir: str = _default_index_dir()

    # ---------- PDF 解析 ----------
    # 页眉页脚正则：招股书页眉“招股意向书 1-1-52”等噪声标记
    header_footer_patterns = (
        r"招股意向书\s*\d{1,2}-\d{1,2}-\d{1,3}",
        r"招股说明书\s*\d{1,2}-\d{1,2}-\d{1,3}",
        r"招股意向书\s*$",
        r"^\s*\d{1,3}\s*$",
        r"^\s*\d{1,2}-\d{1,2}-\d{1,3}\s*$",
    )
    noise_words = ("北京八维信息集团",)
    # 单页字符数低于阈值时，判定为表格/图片页并回退 pdfplumber 提表
    table_fallback_chars = int(os.environ.get("FINQA_TABLE_FALLBACK_CHARS", "60"))
    # 标题模式：第X节、一、（一）、1. 等
    heading_pattern = (
        r"^(第[一二三四五六七八九十百]+[章节]|[一二三四五六七八九十]+、"
        r"|（[一二三四五六七八九十]+）|\d+[\.、])"
    )

    # ---------- 分块 ----------
    chunk_size = int(os.environ.get("FINQA_CHUNK_SIZE", "450"))
    chunk_overlap = int(os.environ.get("FINQA_CHUNK_OVERLAP", "80"))

    # ---------- 检索 ----------
    dense_top_k = 20          # TF-IDF 向量召回数
    bm25_top_k = 20           # BM25 关键词召回数
    rrf_k = 60                # RRF 融合常数
    final_top_k = int(os.environ.get("FINQA_TOP_K", "3"))  # 最终返回证据数
    candidate_n = 30          # 进入离线精排的融合候选数
    max_features = 20000      # TF-IDF 字符 ngram 特征上限（控制内存）

    # ---------- HTTP 服务 ----------
    host = os.environ.get("FINQA_HOST", "0.0.0.0")
    port = int(os.environ.get("FINQA_PORT", "8000"))

    # ---------- 索引版本 ----------
    # 解析/分块/索引算法升级时递增，使旧卷内索引自动失效重建
    index_version = 1


# 全局单例配置
CONFIG = Config()
