# -*- coding: utf-8 -*-  # 声明源文件编码为 UTF-8，避免中文注释乱码
"""【全局配置 · config.py】读取 .env 环境变量，集中管理模型、检索、数据库、Milvus、Redis 等所有可调参数。"""  # 模块文档字符串：中文名 + 文件名 + 一句话作用，便于快速识别
import os  # 导入 os 模块，用于读取环境变量和路径操作
from pathlib import Path  # 导入 Path 类，提供跨平台的路径对象操作

from dotenv import load_dotenv  # 导入 load_dotenv，用于从 .env 文件加载环境变量到 os.environ

# 加载项目根目录下的 .env 配置文件（存在才生效，不存在则用默认值）
load_dotenv(Path(__file__).resolve().parent / ".env")  # 加载项目根目录的 .env 文件，把键值对写入 os.environ

# 项目根目录及各数据目录（不存在则自动创建）
PROJECT_ROOT = Path(__file__).resolve().parent  # 定位项目根目录（本文件所在绝对路径）
CACHE_DIR = PROJECT_ROOT / ".cache"  # BM25 索引等缓存目录，存放检索中间产物
LOG_DIR = PROJECT_ROOT / "logs"  # 运行日志目录，记录程序运行时信息
DATA_DIR = PROJECT_ROOT / "data"  # 数据目录，存放 SQLite 库等持久化数据
UPLOAD_DIR = DATA_DIR / "uploads"  # 用户上传的知识库文件存放目录
for _p in (CACHE_DIR, LOG_DIR, DATA_DIR, UPLOAD_DIR):  # 遍历所有数据目录
    _p.mkdir(exist_ok=True)  # 创建目录，exist_ok=True 表示已存在不报错

# 运行环境：development（DEBUG 日志 + 热重载）/ testing / production
APP_ENV = os.getenv("APP_ENV", "development")  # 运行环境标识：development/testing/production


def tsv_path() -> Path:  # 定义函数：返回主数据集 TSV（中英翻译句对）文件路径
    """主数据集路径：中英翻译句对 TSV（可用环境变量 TSV_PATH 覆盖）。

    查找顺序：环境变量 TSV_PATH → 项目内 data/uploads/老师/中英翻译.tsv。
    用项目内回退保证跨平台（Windows/WSL）与文件迁移后仍能启动。
    """
    env_path = Path(os.getenv("TSV_PATH", r"c:\Users\h1981\Desktop\中英翻译.tsv"))  # 读取环境变量指定的 TSV 路径，默认指向桌面文件
    if env_path.exists():  # 若环境变量指定的路径确实存在
        return env_path  # 直接使用该路径
    fallback = PROJECT_ROOT / "data" / "uploads" / "老师" / "中英翻译.tsv"  # 回退路径：项目内默认位置
    return fallback if fallback.exists() else env_path  # 回退存在则用回退，否则返回环境变量路径（可能不存在）


TSV_PATH = tsv_path()  # 调用上面的函数，确定最终使用的 TSV 数据集路径

# ===== 大模型（兼容 OpenAI 协议：DeepSeek/豆包/千问/本地 vLLM 等）=====
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""  # 大模型 API 密钥，优先 LLM_API_KEY，其次 OPENAI_API_KEY，为空则走纯检索模式
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")  # 大模型接口地址，默认指向 DeepSeek 官方 API
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")  # 使用的模型名称，默认 deepseek-chat

# ===== 检索 / RAG 参数 =====
TOP_K = int(os.getenv("TOP_K", "6"))  # 召回条数：BM25 初步检索返回的文档数量
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "4"))  # 重排后保留条数：经重排模型筛选后的最终数量
SHORT_MEMORY_TURNS = int(os.getenv("SHORT_MEMORY_TURNS", "8"))  # 短期记忆保留轮数：对话历史保留的最大轮数
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.0"))  # BM25 得分过滤阈值：低于此分的召回结果被丢弃

# ===== 数据库（默认 SQLite；生产可换 MySQL，见 .env.example）=====
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'data' / 'app.db'}")  # 数据库连接串，默认本地 SQLite，生产可换 MySQL

# ===== Redis 短期记忆（REDIS_ENABLED=1 启用；未启用退回进程内字典）=====
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")  # Redis 连接地址，默认本地 6379 的 0 号库
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "0") == "1"  # 是否启用 Redis 短期记忆（0=关闭退回进程内字典，1=开启）

# ===== Milvus 向量库（MILVUS_ENABLED=1 启用；未启用仅 BM25 检索）=====
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")  # Milvus 服务地址，默认本机
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")  # Milvus 服务端口，默认 19530
MILVUS_ENABLED = os.getenv("MILVUS_ENABLED", "0") == "1"  # 是否启用 Milvus 向量检索（0=仅 BM25，1=开启向量检索）
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "roleplay_kb")  # Milvus 集合名，存储知识库向量

# ===== 向量化模型 BGE-m3 / 重排模型 BGE-rerank（默认关闭，按需启用）=====
EMBEDDING_ENABLED = os.getenv("EMBEDDING_ENABLED", "0") == "1"  # 是否启用向量化（0=关闭，1=开启 BGE-m3 向量编码）
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")  # 向量化模型，默认 BGE-m3（支持中英多语言）
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "0") == "1"  # 是否启用重排（0=关闭，1=开启 BGE-rerank 精排）
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")  # 重排模型，默认 BGE-reranker-base

# ===== 认证与服务 =====
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")  # token 签名密钥，生产环境必须替换为强随机值
API_HOST = os.getenv("API_HOST", "0.0.0.0")  # API 监听地址，0.0.0.0 表示对所有网卡开放
API_PORT = int(os.getenv("API_PORT", "8000"))  # API 监听端口，默认 8000

# 多角色系统：默认角色 code（可被前端/CLI 选择覆盖）
DEFAULT_ROLE_CODE = os.getenv("DEFAULT_ROLE_CODE", "teacher")  # 默认角色，teacher 为英语教师
