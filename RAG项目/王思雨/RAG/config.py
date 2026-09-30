# -*- coding: utf-8 -*-
"""配置模块：只负责从 .env 读取全局配置，第 0 步不包含任何业务逻辑。"""

import os                      # 导入 os 模块，用于读取系统环境变量与拼接路径
import re                      # 导入 re，用于识别带盘符的绝对路径
from pathlib import Path       # 导入 Path 类，用于定位文件路径
from dotenv import load_dotenv  # 导入 dotenv 加载函数，用于把 .env 读入环境变量

BASE_DIR = Path(__file__).resolve().parent  # 项目根目录（当前文件所在目录）
ENV_FILE = BASE_DIR / ".env"                # .env 配置文件的完整路径
load_dotenv(ENV_FILE)                       # 加载 .env 文件中的键值对到环境变量

# ===== 大模型配置 =====
LLM_API_KEY = os.getenv("LLM_API_KEY", "")                          # 大模型 API Key，只从 .env 读取
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")  # 大模型服务地址
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")             # 大模型名称
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))        # 生成温度，越低越稳定
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))           # 单次生成最大 token 数
LLM_HISTORY_LIMIT = int(os.getenv("LLM_HISTORY_LIMIT", "10"))       # 多轮对话最多带入的历史条数

# ===== MySQL 配置 =====
MYSQL_HOST = os.getenv("MYSQL_HOST", "")        # MySQL 主机地址
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))  # MySQL 端口，转为整数
MYSQL_USER = os.getenv("MYSQL_USER", "")        # MySQL 用户名
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")  # MySQL 密码
MYSQL_DB = os.getenv("MYSQL_DB", "")            # MySQL 数据库名

# ===== Redis 配置 =====
REDIS_HOST = os.getenv("REDIS_HOST", "")        # Redis 主机地址
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))  # Redis 端口，转为整数
REDIS_DB = int(os.getenv("REDIS_DB", "0"))      # Redis 库编号，转为整数

# ===== Milvus 向量库配置 =====
MILVUS_HOST = os.getenv("MILVUS_HOST", "")      # Milvus 主机地址
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))  # Milvus 端口，转为整数
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "power_repair_docs")  # 向量集合名称
MILVUS_TIMEOUT = int(os.getenv("MILVUS_TIMEOUT", "5"))  # Milvus 连接超时秒数

# ===== 本地模型路径配置 =====
BGE_M3_PATH = os.getenv("BGE_M3_PATH", "")            # BGE-M3 向量模型本地路径
# MinerU 模型来源：huggingface / modelscope / local。国内直连 HuggingFace 容易超时，
# 默认走 ModelScope 镜像。这个变量是 MinerU 自己读的（mineru 3.4.5 支持），
# 这里读一遍只为能在启动日志里打出来核对，不参与别处逻辑。
MINERU_MODEL_SOURCE = os.getenv("MINERU_MODEL_SOURCE", "modelscope")   # MinerU 模型下载源
# MinerU 模型目录：**仅作记录**。mineru 3.4.5 不读这个变量，实际路径来自
# 用户目录下的 ~/mineru.json 的 models-dir 字段（本机指向 C:/mineru_models）。
MINERU_MODEL_DIR = os.getenv("MINERU_MODEL_DIR", "")        # MinerU 模型目录（记录用）
# 是否把 MinerU 当作主链路解析器：默认 False，主链路仍走 PyMuPDF + pdfplumber。
# 原因见设计文档 11.5：MinerU 吃显存，5 个 PDF 的解析时间会从十几秒涨到几分钟。
USE_MINERU_AS_MAIN = os.getenv("USE_MINERU_AS_MAIN", "false").strip().lower() == "true"   # 主链路开关
RERANKER_PATH = os.getenv("RERANKER_PATH", "")        # BGE-Reranker 重排模型本地路径

# ===== 数据目录配置 =====
PDF_DIR = os.getenv("PDF_DIR", "")                    # 待入库 PDF 资料目录
# 待解析的 PDF 清单，逗号分隔。留空表示"扫描 PDF_DIR 下的全部 PDF"。
# 增删条目后需要跑一次重建（bash scripts/reindex.sh）才会生效。
PDF_FILES = os.getenv("PDF_FILES", "")                 # PDF 清单，逗号分隔；空＝扫全目录


def get_pdf_files():
    """解析 PDF_FILES 配置。

    :return: 空值或未设置时返回 None，表示"扫描 PDF_DIR 全目录"；
             非空时返回去掉首尾空白、丢掉空项之后的条目列表。
    """
    raw = (PDF_FILES or "").strip()                    # 去掉整体首尾空白
    if not raw:                                        # 空值或未配置
        return None                                    # None 是"扫全目录"的约定
    return [item.strip() for item in raw.split(",") if item.strip()]   # 切分、去空白、丢空项


def resolve_pdf_path(name: str) -> str:
    r"""把清单里的一个条目解析成路径。

    规则：带盘符（如 D:\ 或 D:/）或以斜杠开头的，视为绝对路径原样返回；
    其余一律按 PDF_DIR 拼接。这样"只写文件名"和"写完整路径"两种写法都支持。
    """
    text = (name or "").strip()                        # 去掉首尾空白
    if not text:                                       # 空条目不处理
        return text                                    # 原样返回空串
    if re.match(r"^[A-Za-z]:[\/]", text) or text.startswith(("/", "\\")):   # 带盘符或以斜杠开头
        return text                                    # 绝对路径，原样返回
    return os.path.join(PDF_DIR, text)                 # 否则按 PDF_DIR 拼接

# ===== OCR 模型目录配置 =====
# PaddleOCR 的推理模型目录，必须放在纯英文路径下，中文路径会导致模型文件打不开
OCR_MODEL_DIR = os.getenv("OCR_MODEL_DIR", "C:/paddleocr_models")   # OCR 模型根目录

# ===== HTTP 服务配置 =====
API_HOST = os.getenv("API_HOST", "0.0.0.0")                    # 服务监听地址
API_PORT = int(os.getenv("API_PORT", "8000"))                  # 服务监听端口


def _parse_origins(raw: str) -> list:          # 内部函数：解析 CORS 白名单
    """解析跨域白名单：支持 JSON 数组写法，也支持逗号分隔写法。"""
    text = (raw or "").strip()                 # 去掉首尾空白
    if text.startswith("["):                   # 形如 ["http://a", "http://b"]
        try:                                   # 尝试按 JSON 解析
            import json                        # 延迟导入，仅在需要时使用
            return [str(item) for item in json.loads(text)]   # 返回解析后的列表
        except Exception:                      # 解析失败
            text = text.strip("[]")            # 退化成去掉方括号再按逗号切
    return [part.strip().strip('"').strip("'") for part in text.split(",") if part.strip()]   # 逗号分隔


API_CORS_ORIGINS = _parse_origins(os.getenv("API_CORS_ORIGINS", '["http://localhost:5173"]'))   # 跨域白名单

# ===== RAG 优化配置 =====
CACHE_EXPIRE = int(os.getenv("CACHE_EXPIRE", "3600"))          # 答案缓存过期秒数，默认 1 小时
DEDUP_THRESHOLD = float(os.getenv("DEDUP_THRESHOLD", "0.95"))  # 语义去重的余弦相似度阈值
PARENT_SIZE = int(os.getenv("PARENT_SIZE", "3"))               # 每个父块包含的子块数
ENRICH_MIN_LEN = int(os.getenv("ENRICH_MIN_LEN", "80"))        # 低质过滤：块最小长度
ENRICH_MAX_LEN = int(os.getenv("ENRICH_MAX_LEN", "2000"))      # 低质过滤：块最大长度

# ===== 记忆与安全配置 =====
MEMORY_COLLECTION = os.getenv("MEMORY_COLLECTION", "power_repair_memory")  # 长期记忆集合名
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "3"))        # 长期记忆召回条数
PASSWORD_SALT = os.getenv("PASSWORD_SALT", "change_me_in_prod")   # 密码哈希盐值，生产环境必须改

# ===== 检索与重排配置 =====
RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.3"))  # 精排分数过滤阈值
RRF_K = int(os.getenv("RRF_K", "60"))                     # RRF 融合公式中的常数 k
HYBRID_TOP_K = int(os.getenv("HYBRID_TOP_K", "20"))       # Milvus 混合检索返回条数
MYSQL_RECALL_TOP_K = int(os.getenv("MYSQL_RECALL_TOP_K", "10"))  # MySQL 路召回条数
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))        # 精排后最终返回条数
MYSQL_RECALL_WEIGHT = float(os.getenv("MYSQL_RECALL_WEIGHT", "0.5"))   # MySQL 召回路的权重，降低避免历史提问压过知识


def get_config() -> dict:
    """以字典形式返回当前配置，方便启动时统一打印检查（不含任何业务逻辑）。"""
    return {                                   # 返回配置字典
        "mysql": f"{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}",      # MySQL 连接描述
        "redis": f"{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",      # Redis 连接描述
        "milvus": f"{MILVUS_HOST}:{MILVUS_PORT}",              # Milvus 连接描述
        "collection": MILVUS_COLLECTION,                       # 向量集合名称
        "bge_m3_path": BGE_M3_PATH,                            # 向量模型路径
        "reranker_path": RERANKER_PATH,                        # 重排模型路径
        "mineru_model_source": MINERU_MODEL_SOURCE,            # MinerU 模型下载源
        "pdf_files": get_pdf_files(),                 # 配置的 PDF 清单（None 表示全目录）
        "mineru_model_dir": MINERU_MODEL_DIR,                  # MinerU 模型目录（记录用）
        "use_mineru_as_main": USE_MINERU_AS_MAIN,              # 是否用 MinerU 当主链路
    }                                          # 配置字典结束
