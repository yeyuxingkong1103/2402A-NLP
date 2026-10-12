# 工单编号：人工智能NLP-RAG-混合检索任务
# 文件说明：全局配置。路径/阈值/开关全部集中在本文件，其余模块只读不硬编码；真实环境变量优先于 .env。
import os                                                                       # 读操作系统环境变量
from pathlib import Path                                                        # 跨平台路径拼接
from dotenv import load_dotenv                                                  # 解析项目根目录的 .env

ROOT = Path(__file__).resolve().parent.parent                                   # 项目根目录
load_dotenv(ROOT / ".env", override=False)                                      # override=False：已存在的真实环境变量优先
_s = lambda k, d="": os.environ.get(k, d)                                       # 取字符串配置
_i = lambda k, d: int(os.environ.get(k, d) or d)                                # 取整型，空串回落默认值
_f = lambda k, d: float(os.environ.get(k, d) or d)                              # 取浮点，空串回落默认值
_b = lambda k, d=False: str(os.environ.get(k, d)).lower() in ("1", "true", "yes", "on")  # 取布尔

# ===== 一、大模型 DeepSeek =====
DEEPSEEK_API_KEY = _s("DEEPSEEK_API_KEY")                                       # 密钥只从环境变量取，不落代码
LLM_BASE_URL = _s("LLM_BASE_URL", "https://api.deepseek.com")                   # DeepSeek 基址（OpenAI 兼容）
LLM_MODEL = _s("LLM_MODEL", "deepseek-chat")                                    # 对话模型名
LLM_TIMEOUT = _i("LLM_TIMEOUT", 25)                                             # 单次生成超时秒数，防止拖垮 3 秒指标
LLM_MAX_TOKENS = _i("LLM_MAX_TOKENS", 800)                                      # 生成上限，引用类答案足够
LLM_TEMPERATURE = _f("LLM_TEMPERATURE", 0.0)                                    # 温度 0：招股书问答要可复现
LLM_REASONING_EFFORT = _s("LLM_REASONING_EFFORT", "")                           # 推理型模型必须传 none，否则思考吃光 max_tokens 返回空
LLM_RETRIES = _i("LLM_RETRIES", 2)                                              # 失败重试次数
# 单次问答的总预算（秒）：检索+生成合起来超过它，就放弃生成、直接返回检索原文。
# 这是把"响应 ≤3 秒"从"通常满足"变成"有上界"的兜底；代价是极少数抖动请求拿到的是原文而非成文答案。
LLM_DEADLINE_S = _f("LLM_DEADLINE_S", 3.0)                                      # 设为 0 或负数即关闭预算限制

# ===== 二、本地模型（路径必须用正斜杠：反斜杠的 \b 会被 .env 当转义符吃掉）=====
EMBED_MODEL_PATH = _s("EMBED_MODEL_PATH", "D:/模型/bge-m3")                     # bge-m3 向量模型路径
RERANK_MODEL_PATH = _s("RERANK_MODEL_PATH", "D:/模型/bge-reranker-v2-m3")       # 重排模型路径
DEVICE = _s("DEVICE", "cuda")                                                   # 推理设备
EMBED_DIM = _i("EMBED_DIM", 1024)                                               # bge-m3 向量维度，须与 Milvus 一致
EMBED_BATCH = _i("EMBED_BATCH", 16)                                             # 编码批大小，8G 显存下的稳妥值
RERANK_MAX_CHARS = _i("RERANK_MAX_CHARS", 512)                                  # 重排打分输入截断：不截断会按最长序列 padding 拖慢数倍
EMBED_MODELS = {"bge-m3": EMBED_MODEL_PATH}                                     # 多嵌入模型注册表：新增 bge/m3e 只需在此加本地路径

# ===== 三、Milvus 与 Redis =====
MILVUS_URI = _s("MILVUS_URI", "http://localhost:19530")                         # Milvus 地址
MILVUS_COLLECTION = _s("MILVUS_COLLECTION", "gd6_chunks")                       # 本工单专属集合名
MILVUS_TIMEOUT = _i("MILVUS_TIMEOUT", 10)                                       # 连接与检索超时，挂了要快速失败
MILVUS_BATCH = _i("MILVUS_BATCH", 256)                                          # 写库批大小
REDIS_URL = _s("REDIS_URL", "redis://localhost:6379/0")                         # Redis 地址
REDIS_PROTOCOL = _i("REDIS_PROTOCOL", 2)                                        # 本机 Redis 5.0.14 不支持 RESP3，redis-py 8 必须显式 2
CACHE_TTL = _i("CACHE_TTL", 86400)                                              # 问答/检索缓存有效期（秒）
HISTORY_TURNS = _i("HISTORY_TURNS", 6)                                          # 多轮对话保留轮数
HISTORY_TTL = _i("HISTORY_TTL", 7200)                                           # 会话历史有效期（秒）
SESSION_MAX_CHARS = _i("SESSION_MAX_CHARS", 512)                                # 单条历史截断长度，防膨胀

# ===== 四、数据与 PDF 解析 =====
RAW_DIR = ROOT / "data" / "raw"                                                 # 原始 PDF 目录
PARSED_DIR = ROOT / "data" / "parsed"                                           # MinerU 解析产物目录
INDEX_FILE = ROOT / "data" / "index.json"                                       # 分块结果落盘
BM25_FILE = ROOT / "data" / "bm25.pkl"                                          # BM25 索引落盘
MINERU_ENABLED = _b("MINERU_ENABLED", True)                                     # 是否优先用 MinerU（缺失时走 pymupdf 兜底）
MINERU_LANGFIX_DIR = _s("MINERU_LANGFIX_DIR", "C:/mineru_models/langfix")       # 中文用户名下 fasttext 打不开默认模型，缺了必失败
MINERU_TIMEOUT = _i("MINERU_TIMEOUT", 7200)                                     # MinerU 子进程超时（秒）

# ===== 五、语义分块 =====
CHUNK_SIZE = _i("CHUNK_SIZE", 512)                                              # 单块目标字符数
CHUNK_OVERLAP = _i("CHUNK_OVERLAP", 80)                                         # 相邻块重叠字符数
SEM_THRESHOLD = _f("SEM_THRESHOLD", 0.62)                                       # 相邻句与块中心余弦低于此值即切块
SEM_MIN_CHUNK = _i("SEM_MIN_CHUNK", 120)                                        # 过小的块向后合并
SEM_MAX_CHUNK = _i("SEM_MAX_CHUNK", 900)                                        # 硬上限，防止超长块撑爆上下文

# ===== 六、检索 / 融合 / 重排 =====
VEC_TOP_K = _i("VEC_TOP_K", 30)                                                 # 向量通道召回条数
BM25_TOP_K = _i("BM25_TOP_K", 30)                                               # BM25 通道召回条数
FUSION_TOP_K = _i("FUSION_TOP_K", 30)                                           # 融合后候选条数
RERANK_TOP_N = _i("RERANK_TOP_N", 8)                                            # 重排后交给大模型的上下文条数
VEC_WEIGHT = _f("VEC_WEIGHT", 0.7)                                              # 向量通道权重
BM25_WEIGHT = _f("BM25_WEIGHT", 0.3)                                            # BM25 通道权重
RRF_K = _i("RRF_K", 60)                                                         # RRF 融合常数
FIELD_WEIGHTS = {"title": _f("W_TITLE", 2.0),                                   # BM25 多字段权重：标题
                 "body": _f("W_BODY", 1.0),                                     # BM25 多字段权重：正文
                 "abstract": _f("W_ABSTRACT", 0.6)}                             # BM25 多字段权重：摘要
FUZZY_CUTOFF = _i("FUZZY_CUTOFF", 65)                                           # 模糊匹配阈值（partial_ratio 0-100）；人名错 1 字只有 66.7，故不能设 80
ADAPTIVE_ALPHA = _f("ADAPTIVE_ALPHA", 0.5)                                      # 自适应重排的反馈影响强度
CASCADE_MID = _i("CASCADE_MID", 24)                                             # 级联重排中间截断条数：设小会误杀后段金标（实测 12 时被杀），故只削尾
DEFAULT_RERANKERS = _s("DEFAULT_RERANKERS", "bge").split(",")                     # 默认重排器：单级 bge。级联预筛实测伤召回，故不设为默认

# ===== 七、服务与评估 =====
API_HOST = _s("API_HOST", "127.0.0.1")                                          # 监听地址
API_PORT = _i("API_PORT", 8010)                                                 # 监听端口：8000 已被工单五服务占用
LOG_LEVEL = _s("LOG_LEVEL", "INFO")                                             # 日志级别
EVAL_FILE = ROOT / "tests" / "questions.json"                                   # 中英文评估题集
RESULT_DIR = ROOT / "results"                                                   # 评估结果落盘目录
EVAL_MAX_WORKERS = _i("EVAL_MAX_WORKERS", 2)                                    # 默认 16 并发会打满 DeepSeek 限流把指标打成 nan
TARGET_ACCURACY = _f("TARGET_ACCURACY", 0.90)                                   # 目标准确率
TARGET_RECALL = _f("TARGET_RECALL", 0.95)                                       # 目标召回率
TARGET_LATENCY = _f("TARGET_LATENCY", 3.0)                                      # 目标响应时间（秒）
