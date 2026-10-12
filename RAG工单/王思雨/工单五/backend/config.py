# 工单编号：人工智能NLP-RAG-Query理解优化
# 文件说明：全局集中配置。所有可调参数、路径、开关都收敛在本文件，其余模块只读取、不硬编码；
#           .env 中已存在的真实环境变量优先，便于部署时用环境变量覆盖而不改代码。
import os                                                                       # 标准库：读取操作系统环境变量
from pathlib import Path                                                        # 标准库：跨平台路径拼接，避免手写反斜杠
from dotenv import load_dotenv                                                  # 第三方：解析项目根目录下的 .env 文件

ROOT = Path(__file__).resolve().parent.parent                                   # 项目根目录（backend 的上一级目录）
load_dotenv(ROOT / ".env", override=False)                                      # 载入 .env；override=False 表示真实环境变量优先于文件

_s = lambda k, d="": os.environ.get(k, d)                                       # 读字符串配置：未设置时返回默认值
_i = lambda k, d: int(os.environ.get(k, d) or d)                                # 读整型配置：空串会回落默认值，避免 ValueError
_f = lambda k, d: float(os.environ.get(k, d) or d)                              # 读浮点配置：空串会回落默认值
_b = lambda k, d=False: str(os.environ.get(k, d)).lower() in ("1", "true", "yes", "on")  # 读布尔配置：兼容 1/true/yes/on

# ==================== 一、大模型：DeepSeek（OpenAI 兼容接口）====================
DEEPSEEK_API_KEY = _s("DEEPSEEK_API_KEY")                                       # 密钥只从环境变量取，绝不写进代码或仓库
LLM_BASE_URL = _s("LLM_BASE_URL", "https://api.deepseek.com")                   # DeepSeek 接口基址
LLM_MODEL = _s("LLM_MODEL", "deepseek-chat")                                    # 对话模型名；deepseek-chat 即 V3 对话档
LLM_TIMEOUT = _i("LLM_TIMEOUT", 25)                                             # 单次生成超时（秒），防止大模型卡死拖垮接口
LLM_MAX_TOKENS = _i("LLM_MAX_TOKENS", 900)                                      # 生成上限；答引用类问题 900 token 足够，且省时
LLM_TEMPERATURE = _f("LLM_TEMPERATURE", 0.0)                                    # 温度设 0：招股书问答要可复现，不能有随机发挥
LLM_REASONING_EFFORT = _s("LLM_REASONING_EFFORT", "")                           # 坑：推理型模型必须显式传 none，否则思考过程吃光 max_tokens 返回空内容
LLM_RETRIES = _i("LLM_RETRIES", 2)                                              # 网络抖动重试次数（首次 + 2 次重试）
LLM_RETRY_SLEEP = _f("LLM_RETRY_SLEEP", 1.5)                                    # 每次重试前的退避秒数

# ==================== 二、本地模型：路径一律用正斜杠 ====================
# 坑：在 .env 里写 D:\模型\bge-m3 时，"\b" 会被解析成退格符，加载时报路径不存在。
EMBED_MODEL_PATH = _s("EMBED_MODEL_PATH", "D:/模型/bge-m3")                     # bge-m3 向量模型本地路径（中文+英文都支持）
RERANK_MODEL_PATH = _s("RERANK_MODEL_PATH", "D:/模型/bge-reranker-v2-m3")       # bge-reranker-v2-m3 重排模型本地路径
DEVICE = _s("DEVICE", "cuda")                                                   # 推理设备；无 GPU 的机器改 cpu 即可
EMBED_DIM = _i("EMBED_DIM", 1024)                                               # bge-m3 向量维度，与 Milvus 集合定义必须一致
EMBED_BATCH = _i("EMBED_BATCH", 16)                                             # 向量化批大小：8G 显存下 16 是实测稳妥值

# ==================== 三、Milvus 向量库 ====================
MILVUS_URI = _s("MILVUS_URI", "http://localhost:19530")                         # Milvus 地址
MILVUS_COLLECTION = _s("MILVUS_COLLECTION", "gd5_chunks")                       # 本工单专属集合名，避免与工单二/三/四的数据串味
MILVUS_TIMEOUT = _i("MILVUS_TIMEOUT", 10)                                       # 连接与检索超时（秒）；Milvus 挂了要快速失败而不是吊死接口
MILVUS_INSERT_BATCH = _i("MILVUS_INSERT_BATCH", 256)                            # 写库批大小，控制单次显存与内存峰值

# ==================== 四、Redis 缓存与会话 ====================
REDIS_URL = _s("REDIS_URL", "redis://localhost:6379/0")                         # Redis 地址
REDIS_PROTOCOL = _i("REDIS_PROTOCOL", 2)                                        # 坑：本机 Redis 5.0.14 不支持 RESP3(HELLO)，redis-py 8 必须显式指定 protocol=2
CACHE_TTL = _i("CACHE_TTL", 86400)                                              # 问答缓存有效期（秒），默认一天
HISTORY_TURNS = _i("HISTORY_TURNS", 6)                                          # 多轮对话保留的历史轮数（一问一答算一轮）
HISTORY_TTL = _i("HISTORY_TTL", 7200)                                           # 会话历史有效期（秒），默认两小时

# ==================== 五、知识库：工单指定的两篇招股说明书 ====================
RAW_DIR = ROOT / _s("RAW_DIR", "data/raw")                                      # 原始 PDF 目录（工单验收使用这里预置的两份）
PARSED_DIR = ROOT / _s("PARSED_DIR", "data/parsed")                             # MinerU 解析产物目录
INDEX_FILE = ROOT / "data" / "index.json"                                       # 分块结果落盘，重启免于重建
BM25_FILE = ROOT / "data" / "bm25.pkl"                                          # BM25 索引落盘，重启免于重建
def raw_pdfs() -> list:                                                         # 返回知识库内的全部 PDF
    """扫描 data/raw 目录，新上传的 PDF 自动纳入知识库，无需改代码。"""
    return sorted(RAW_DIR.glob("*.pdf"))                                        # 按文件名排序，保证多次运行顺序稳定

# ==================== 六、PDF 解析（MinerU 主 + pymupdf/pdfplumber/PaddleOCR 兜底）====================
MINERU_ENABLED = _b("MINERU_ENABLED", True)                                     # 是否优先尝试 MinerU；解析慢，已有缓存时直接读缓存
MINERU_PYTHON = _s("MINERU_PYTHON")                                             # 装了 MinerU 的解释器路径；留空=用当前解释器
MINERU_LANGFIX_DIR = _s("MINERU_LANGFIX_DIR", "C:/mineru_models/langfix")       # 中文路径补丁目录：用户名含中文时 fasttext 打不开模型，缺了必失败
MINERU_TIMEOUT = _i("MINERU_TIMEOUT", 7200)                                     # 单篇解析超时（秒）；548 页那篇约需数分钟
PADDLEOCR_MODEL_DIR = _s("PADDLEOCR_MODEL_DIR", "C:/paddleocr_models")         # 纯英文的 PaddleOCR 模型根目录（中文用户名下 C++ 推理层打不开默认路径）
OCR_MIN_CHARS = _i("OCR_MIN_CHARS", 30)                                         # 整页可抽取文字少于该值即判定为扫描件，转走 OCR 分支
OCR_SCALE = _f("OCR_SCALE", 3.0)                                                # 扫描页渲染放大倍数；1x 会把图表里的负号读丢，3x 才稳定

# ==================== 七、语义分块（工单建议 chunk_size=512, overlap=80）====================
CHUNK_SIZE = _i("CHUNK_SIZE", 512)                                              # 单块目标字符数（工单明确建议值）
CHUNK_OVERLAP = _i("CHUNK_OVERLAP", 80)                                         # 相邻块重叠字符数（工单明确建议值）
SEM_THRESHOLD = _f("SEM_THRESHOLD", 0.62)                                       # 相邻句子与当前块中心的余弦相似度低于此值即切块
SEM_MIN_CHUNK = _i("SEM_MIN_CHUNK", 120)                                        # 小于该长度的块向后合并，避免碎片挤占检索前排
SEM_MAX_CHUNK = _i("SEM_MAX_CHUNK", 900)                                        # 硬上限：即便语义都很像也必须切开，防止超长块撑爆上下文

# ==================== 八、混合检索与重排 ====================
VEC_TOP_K = _i("VEC_TOP_K", 30)                                                 # 向量通道召回条数
BM25_TOP_K = _i("BM25_TOP_K", 30)                                               # BM25 关键词通道召回条数
FUSION_TOP_K = _i("FUSION_TOP_K", 30)                                           # 融合后送入重排的候选条数
RERANK_TOP_N = _i("RERANK_TOP_N", 8)                                            # 重排后交给大模型的上下文条数
RERANK_MAX_CHARS = _i("RERANK_MAX_CHARS", 512)                                  # 重排打分的单条截断字数：不截断会把成对打分开销放大数倍
VEC_WEIGHT = _f("VEC_WEIGHT", 0.7)                                              # 融合时向量得分的权重
BM25_WEIGHT = _f("BM25_WEIGHT", 0.3)                                            # 融合时 BM25 得分的权重
RRF_K = _i("RRF_K", 60)                                                         # RRF 融合常数，用于把两路不可比的得分折算成排名分

# ==================== 九、多轮对话 Query 理解 ====================
REWRITE_ENABLED = _b("REWRITE_ENABLED", True)                                   # 是否启用 Query 改写（指代消解）；关掉即退化为单轮检索
REWRITE_TIMEOUT = _i("REWRITE_TIMEOUT", 12)                                     # 改写调用的超时（秒）；改写失败必须快速回退到原始问题
REWRITE_MAX_HISTORY = _i("REWRITE_MAX_HISTORY", 4)                              # 送给改写模型的历史轮数上限，越少越快
SESSION_MAX_CHARS = _i("SESSION_MAX_CHARS", 512)                                # 单条会话消息入库前的截断长度，防止历史无限膨胀
ENTITY_TRACKING = _b("ENTITY_TRACKING", True)                                   # 是否启用实体追踪（切换公司后，"他/这个公司"应指向新公司）

# ==================== 十、验收兼容开关（默认关闭）====================
# 背景：工单五 Q5 的"期望答案"是"渠道销售部"，但招股说明书2 内部组织结构图实际画的是
#      "大客户销售部"下辖 6 个销售处。本系统默认按 PDF 原件作答（事实模式）。
#      若验收方强制要求工单口径，把本开关置 true，只会"追加一行对照说明"，不会篡改主答案。
Q5_ACCEPTANCE_COMPAT = _b("Q5_ACCEPTANCE_COMPAT", False)                        # 验收兼容模式开关，默认关闭
Q5_ACCEPTANCE_NOTE = _s("Q5_ACCEPTANCE_NOTE",                                   # 兼容模式下追加的说明文本
    "工单参考口径：渠道销售部，下辖珠海销售处、深圳销售处、北京销售处、武汉销售处、广州销售处（5个）。"
    "注意：该口径与招股说明书2 原件不一致，仅用于工单对照，不作为事实依据。")

# ==================== 十一、FastAPI 服务 ====================
API_HOST = _s("API_HOST", "127.0.0.1")                                          # 监听地址
API_PORT = _i("API_PORT", 8000)                                                 # 监听端口（避开已被占用的 8001/8080 等）
LOG_LEVEL = _s("LOG_LEVEL", "INFO")                                             # 日志级别
WARMUP_ON_START = _b("WARMUP_ON_START", True)                                   # 启动时预热模型与索引，避免首个请求承担加载耗时

# ==================== 十二、评估（ragas）====================
EVAL_FILE = ROOT / "eval" / "questions.json"                                    # 评估题集（工单 5 题 + 变体 + 英文题）
EVAL_RESULT_DIR = ROOT / "results"                                              # 评估结果落盘目录
EVAL_MAX_WORKERS = _i("EVAL_MAX_WORKERS", 2)                                    # 坑：ragas 默认 16 并发会打满 DeepSeek 限流，把指标打成 nan
EVAL_TARGET_ACCURACY = _f("EVAL_TARGET_ACCURACY", 0.90)                         # 目标准确率（工单要求 ≥90%）
EVAL_TARGET_LATENCY = _f("EVAL_TARGET_LATENCY", 3.0)                            # 目标响应时间（工单要求 ≤3 秒）
