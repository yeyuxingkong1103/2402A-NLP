# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-金融问答系统部署
# 模块职责：集中管理部署配置，所有可变项均支持环境变量覆盖，便于容器化编排

import os                                                       # 导入 os 用于读取环境变量
import sys                                                      # 导入 sys 用于修正标准输出编码

# 容器内标准输出为管道，默认编码可能为 ASCII/GBK，统一改为 UTF-8 避免中文日志报错
for _name in ("stdout", "stderr"):                              # 依次处理标准输出与标准错误
    _stream = getattr(sys, _name, None)                         # 取出对应流对象
    if _stream is not None and hasattr(_stream, "reconfigure"):  # 支持重配置时（Python 3.7+）
        try:                                                    # 尝试重配置
            _stream.reconfigure(encoding="utf-8", errors="replace")  # 改为 UTF-8 并容忍无法编码字符
        except Exception:                                       # 环境不支持时忽略
            pass                                                # 不影响主流程


def _env_str(key, default):                                     # 定义字符串型环境变量读取函数
    """读取环境变量，未设置或为空时返回默认值。"""
    val = os.environ.get(key)                                   # 读取环境变量
    return val if val not in (None, "") else default            # 空值与未设置统一回退默认值


def _env_int(key, default):                                     # 定义整型环境变量读取函数
    """读取整型环境变量，解析失败时回退默认值。"""
    try:                                                        # 尝试转换
        return int(os.environ.get(key, default))                # 返回转换结果
    except (TypeError, ValueError):                             # 非法值容错
        return default                                          # 回退默认值


def _env_bool(key, default):                                    # 定义布尔型环境变量读取函数
    """读取布尔型环境变量，接受 true/false、1/0、yes/no 等写法。"""
    val = str(os.environ.get(key, default)).strip().lower()      # 统一为小写字符串便于比较
    return val in ("1", "true", "yes", "y", "on")               # 命中真值集合则返回 True


# ---------------------------------------------------------------------------
# 一、路径配置（容器内默认值，可被 docker-compose 的环境变量覆盖）
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))            # 应用代码所在目录
DATA_DIR = _env_str("RAG_DATA_DIR", "/app/data")                # 数据目录：索引缓存与问答历史（挂载卷）
LOG_DIR = _env_str("RAG_LOG_DIR", "/app/logs")                  # 日志目录（挂载卷）
CORPUS_DIR = _env_str("RAG_CORPUS_DIR", "/app/corpus")          # 语料目录（镜像内置，可挂载覆盖）
MODEL_PATH = _env_str("EMBED_MODEL_PATH", "/models/m3e-base")   # 嵌入模型目录（挂载卷，只读）

INDEX_FILE = os.path.join(DATA_DIR, "index.json")               # 分块与 BM25 索引缓存
VECTOR_FILE = os.path.join(DATA_DIR, "vectors.npz")             # 向量矩阵缓存
HISTORY_FILE = os.path.join(DATA_DIR, "qa_history.jsonl")       # 问答历史（验证卷持久化）

# ---------------------------------------------------------------------------
# 二、检索与生成参数
# ---------------------------------------------------------------------------
EMBED_ENABLED = _env_bool("EMBED_ENABLED", "true")              # 是否启用向量检索（模型已挂载时开启）
EMBED_DIM = _env_int("EMBED_DIM", 768)                          # 向量维度（m3e-base 为 768）
CHUNK_SIZE = _env_int("CHUNK_SIZE", 400)                        # 分块目标字符数
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 50)                   # 相邻块重叠字符数
BM25_TOP_K = _env_int("BM25_TOP_K", 8)                          # BM25 召回条数
VECTOR_TOP_K = _env_int("VECTOR_TOP_K", 8)                      # 向量召回条数
FINAL_TOP_K = _env_int("FINAL_TOP_K", 5)                        # 最终送入生成的上下文条数
RRF_K = _env_int("RRF_K", 60)                                   # RRF 融合平滑常数
CONTEXT_MAX_CHARS = _env_int("CONTEXT_MAX_CHARS", 3500)         # 上下文最大字符数

# ---------------------------------------------------------------------------
# 三、大模型配置
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY = _env_str("DEEPSEEK_API_KEY", "sk-REDACTED")  # 密钥（建议用环境变量注入）
DEEPSEEK_BASE_URL = _env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")  # 接口地址
DEEPSEEK_MODEL = _env_str("DEEPSEEK_MODEL", "deepseek-chat")    # 模型名称
LLM_TIMEOUT = _env_int("LLM_TIMEOUT", 60)                       # 请求超时（秒）
LLM_MAX_RETRY = _env_int("LLM_MAX_RETRY", 3)                    # 最大重试次数

# ---------------------------------------------------------------------------
# 四、服务配置
# ---------------------------------------------------------------------------
# 服务端口：工单要求"在指定端口提供服务"，默认 8000，由 compose 映射到宿主机
SERVICE_PORT = _env_int("SERVICE_PORT", 8000)                   # 服务监听端口
SERVICE_HOST = _env_str("SERVICE_HOST", "0.0.0.0")              # 监听地址（容器内需 0.0.0.0）
APP_VERSION = "1.0.0"                                           # 服务版本号


def ensure_dirs():                                              # 定义目录初始化函数
    """确保运行时需要的目录存在（挂载卷首次启动时目录可能为空）。"""
    for d in (DATA_DIR, LOG_DIR):                               # 遍历需要保证存在的目录
        os.makedirs(d, exist_ok=True)                           # 不存在则创建
