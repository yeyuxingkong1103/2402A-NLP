# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 本模块职责：集中读取 .env 与环境变量，统一定义路径与检索参数，供其它模块导入。
# 设计约束：路径一律使用正斜杠 —— .env 中 D:\模型\bge-m3 的 "\b" 会被解析成退格符。

import os                        # 读取环境变量
from pathlib import Path         # 跨平台路径处理，避免手工拼字符串

BASE_DIR = Path(__file__).resolve().parent.parent  # 项目根目录（app 的上一级）


def _load_env() -> None:  # 极简 .env 加载器：不引入 python-dotenv 依赖，减少体积
    """把项目根目录 .env 的键值对写入 os.environ（已存在的环境变量不覆盖）。"""
    env_file = BASE_DIR / ".env"                       # .env 路径
    if not env_file.exists():                          # 文件不存在就直接返回，不报错
        return                                         # 保证缺 .env 时程序仍能启动
    for raw in env_file.read_text(encoding="utf-8").splitlines():  # 逐行读取
        line = raw.strip()                             # 去掉首尾空白
        if not line or line.startswith("#") or "=" not in line:    # 跳过空行/注释/非法行
            continue                                   # 处理下一行
        key, _, value = line.partition("=")            # 以第一个 = 切分键值
        key, value = key.strip(), value.split("#")[0].strip()      # 去掉行内注释与空格
        if key and key not in os.environ:              # 已存在的真实环境变量优先级更高
            os.environ[key] = value                    # 写入环境变量


_load_env()  # 模块导入时立即加载，保证后续 os.getenv 能读到


def _env(key: str, default: str) -> str:  # 读取字符串型配置的简写
    return os.getenv(key, default)        # 找不到就用默认值


def _env_int(key: str, default: int) -> int:  # 读取整型配置
    try:                                      # 防止 .env 里填了非数字导致启动崩溃
        return int(os.getenv(key, str(default)))  # 正常转换
    except ValueError:                        # 转换失败
        return default                        # 静默退回默认值，保证高可用


def _env_float(key: str, default: float) -> float:  # 读取浮点型配置
    try:                                            # 同样做容错
        return float(os.getenv(key, str(default)))  # 正常转换
    except ValueError:                              # 转换失败
        return default                              # 退回默认值


# ---------------- 大模型（DeepSeek） ----------------
# 工单要求 API Key 从环境变量 DEEPSEEK_API_KEY 读取；.env 中的 LLM_API_KEY 作为补充来源
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY") or _env("LLM_API_KEY", "")   # 两级回退
LLM_BASE_URL = _env("LLM_BASE_URL", "https://api.deepseek.com/v1")       # 接口地址
LLM_MODEL = _env("LLM_MODEL", "deepseek-v4-flash")                       # 模型名
LLM_TIMEOUT = _env_float("LLM_TIMEOUT", 60.0)                            # 单次请求超时（秒）
EVAL_JUDGE_MAX_TOKENS = _env_int("EVAL_JUDGE_MAX_TOKENS", 2048)          # RAGAS 判官输出上限（推理模型需留足）
# 推理档位：deepseek-v4-flash 默认会先产出大量 reasoning_tokens，实测把 600 token 预算全部吃光、
# content 返回空串，且首字延迟 5.2s；设为 none 后首字 1.6s 且答案非空。生成任务本身是"从资料摘取"，
# 不需要长推理，故默认关闭。
LLM_REASONING_EFFORT = _env("LLM_REASONING_EFFORT", "none")

# ---------------- 本地模型 ----------------
EMBED_MODEL_PATH = _env("EMBEDDING_MODEL_PATH", "D:/模型/bge-m3")               # 向量模型
RERANK_MODEL_PATH = _env("RERANKER_MODEL_PATH", "D:/模型/bge-reranker-v2-m3")   # 重排模型
EMBED_DEVICE = _env("EMBED_DEVICE", "cuda")       # 推理设备，显存不足可改 cpu
EMBED_DIM = _env_int("EMBED_DIM", 1024)           # bge-m3 稠密向量维度，固定 1024
EMBED_MAX_TOKENS = _env_int("EMBED_MAX_TOKENS", 1024)   # 编码截断长度：8192 会因 padding 拖慢数十倍

# ---------------- 向量库与缓存 ----------------
MILVUS_URI = _env("MILVUS_URI", "http://localhost:19530")   # Docker 版地址
MILVUS_LITE_PATH = _env("MILVUS_LITE_PATH", str(BASE_DIR / "data/milvus/lite.db"))  # 降级库文件
MILVUS_COLLECTION = _env("MILVUS_COLLECTION", "pdf_chunks")  # 集合名
REDIS_URL = _env("REDIS_URL", "redis://localhost:6379/0")    # 缓存地址
REDIS_PROTOCOL = _env_int("REDIS_PROTOCOL", 2)               # 本机 Redis 5 不支持 RESP3，必须为 2

# ---------------- 检索参数（可在优化阶段通过 .env 热调，无需改代码） ----------------
CHUNK_MIN_CHARS = _env_int("CHUNK_MIN_CHARS", 350)            # 语义分块最小字符数
CHUNK_MAX_CHARS = _env_int("CHUNK_MAX_CHARS", 600)            # 语义分块最大字符数
SEMANTIC_BREAK_PERCENTILE = _env_int("SEMANTIC_BREAK_PERCENTILE", 50)  # 断句相似度分位阈值
RETRIEVE_TOP_K = _env_int("RETRIEVE_TOP_K", 80)               # 每路召回的候选数
FUSION_TOP_N = _env_int("FUSION_TOP_N", 60)                   # RRF 融合后进入重排的数量
RERANK_TOP_K = _env_int("RERANK_TOP_K", 8)                # 去重后进入上下文
RERANK_POOL = _env_int("RERANK_POOL", 14)              # 重排候选池大小（去重前）
RERANK_MIN_SCORE = _env_float("RERANK_MIN_SCORE", 0.15)       # 重排分数阈值，低于则丢弃
CONTEXT_MAX_CHARS = _env_int("CONTEXT_MAX_CHARS", 5000)       # 拼给 LLM 的上下文上限

# ---------------- 解析器 ----------------
MINERU_MODEL_SOURCE = _env("MINERU_MODEL_SOURCE", "modelscope")              # 模型下载源
MINERU_LANGFIX_DIR = _env("MINERU_LANGFIX_DIR", "C:/mineru_models/langfix")  # 中文路径补丁目录
MINERU_TIMEOUT = _env_int("MINERU_TIMEOUT", 3600)                            # 解析超时（秒）

# ---------------- 目录 ----------------
DATA_DIR = BASE_DIR / "data"                          # 数据总目录
UPLOAD_DIR = DATA_DIR / "uploads"                     # 上传的 PDF
PARSED_DIR = DATA_DIR / "parsed"                      # 解析产物
INDEX_DIR = DATA_DIR / "index"                        # BM25 语料等索引文件
EVAL_DIR = BASE_DIR / "eval"                          # 评估题目与结果
FRONTEND_DIR = BASE_DIR / "frontend"                  # 前端静态文件

for _d in (DATA_DIR, UPLOAD_DIR, PARSED_DIR, INDEX_DIR, EVAL_DIR):  # 启动时补齐目录
    _d.mkdir(parents=True, exist_ok=True)             # 不存在则创建，存在则忽略


def check_env() -> dict:  # 供 /api/health 与启动日志使用的配置自检
    """返回关键配置项的就绪情况，不抛异常。"""
    return {                                          # 逐项检查并返回字典
        "llm_key": bool(LLM_API_KEY),                 # Key 是否配置
        "embed_model": Path(EMBED_MODEL_PATH).exists(),   # 向量模型目录是否存在
        "rerank_model": Path(RERANK_MODEL_PATH).exists(),  # 重排模型目录是否存在
        "mineru_langfix": Path(MINERU_LANGFIX_DIR).exists(),  # 中文路径补丁是否存在
    }
