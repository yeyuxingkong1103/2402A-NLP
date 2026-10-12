# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 文件说明：集中配置。从 .env 与环境变量读取参数——密钥不落代码，检索与分块参数改配置即可调优。
import os                                                                        # 读取环境变量
from pathlib import Path                                                         # 跨平台路径处理
from dotenv import load_dotenv                                                   # 解析 .env 文件

ROOT = Path(__file__).resolve().parent.parent                                    # 项目根目录（app 的上一级）
load_dotenv(ROOT / ".env", override=False)                                       # 载入 .env；已存在的真实环境变量优先，便于 CI 覆盖
_s = lambda k, d="": os.environ.get(k, d)                                        # 字符串配置读取，缺省返回默认值
_i = lambda k, d: int(os.environ.get(k, d) or d)                                 # 整型配置读取，空串回落默认值
_f = lambda k, d: float(os.environ.get(k, d) or d)                               # 浮点配置读取，空串回落默认值

# ==================== 大模型：DeepSeek（OpenAI 兼容接口）====================
DEEPSEEK_API_KEY = _s("DEEPSEEK_API_KEY")                                        # 密钥只从环境变量取，绝不写进代码
LLM_BASE_URL = _s("LLM_BASE_URL", "https://api.deepseek.com")                    # 接口基址
LLM_MODEL = _s("LLM_MODEL", "deepseek-chat")                                     # 对话模型名
LLM_REASONING_EFFORT = _s("LLM_REASONING_EFFORT", "")                            # 推理型模型必须显式传 none，否则思考过程吃光 max_tokens 后返回空内容

# ==================== 本地模型：路径一律正斜杠 ====================
# 坑：写成 D:\模型\bge-m3 时 "\b" 会被 .env 解析成退格符，加载报路径不存在
EMBED_MODEL_PATH = _s("EMBED_MODEL_PATH", "D:/模型/bge-m3")                       # bge-m3 向量模型本地路径
RERANK_MODEL_PATH = _s("RERANK_MODEL_PATH", "D:/模型/bge-reranker-v2-m3")         # bge-reranker-v2-m3 重排模型本地路径
DEVICE = _s("DEVICE", "cuda")                                                    # 推理设备；无 GPU 时改 cpu
EMBED_DIM = _i("EMBED_DIM", 1024)                                                # bge-m3 向量维度

# ==================== 向量库与缓存 ====================
MILVUS_URI = _s("MILVUS_URI", "http://localhost:19530")                          # Milvus 地址
MILVUS_COLLECTION = _s("MILVUS_COLLECTION", "gd3_chunks")                        # 块向量集合（两篇文档共用，靠 doc 字段区分）
REDIS_URL = _s("REDIS_URL", "redis://localhost:6379/0")                          # Redis 地址
CACHE_TTL = _i("CACHE_TTL", 86400)                                               # 问答缓存有效期（秒）

# ==================== 知识库：工单指定的两篇招股说明书 ====================
RAW_DIR = ROOT / _s("RAW_DIR", "data/raw")                                       # 原始 PDF 目录，/upload 也落盘于此
PARSED_DIR = ROOT / _s("PARSED_DIR", "data/parsed")                              # MinerU 解析产物目录
def docs() -> list:                                                              # 当前知识库内的全部 PDF
    return sorted(RAW_DIR.glob("*.pdf"))                                         # 目录扫描，新上传的文档自动纳入，无需改代码

# ==================== MinerU：主解析器 ====================
# 本机 MinerU 3.4.5 与主程序依赖（pymilvus/FlagEmbedding/torch-cuda/ragas）都装在同一个解释器里，
# 故留空即用当前解释器。若你的环境把 MinerU 单独装到另一个解释器，把 MINERU_PYTHON 指向它即可。
MINERU_PYTHON = _s("MINERU_PYTHON")                                               # 装了 MinerU 的解释器路径，留空=当前解释器
MINERU_LANGFIX_DIR = _s("MINERU_LANGFIX_DIR", "C:/mineru_models/langfix")         # 中文路径补丁目录：用户名含中文时 fasttext 打不开模型，缺了必失败
MINERU_TIMEOUT = _i("MINERU_TIMEOUT", 7200)                                      # 单篇解析超时（秒），548 页约需数分钟

# PaddleOCR 兜底：它的模型缓存路径在 2.7.3 里写死为 ~/.paddleocr/，没有环境变量可改。
# 用户名含中文时，Paddle 的 C++ 推理层打不开这个路径（报 Cannot open file ... pdmodel，
# 但文件其实存在），所以把模型另存到纯英文路径并显式传给 PaddleOCR。
# 首次使用请执行：cp -r ~/.paddleocr/whl/{det,rec,cls} C:/paddleocr_models/
PADDLEOCR_MODEL_DIR = _s("PADDLEOCR_MODEL_DIR", "C:/paddleocr_models")           # 纯英文的 PaddleOCR 模型根目录

# ==================== 分块与检索参数（调参入口，改这里不用改代码）====================
SEM_CHUNK_MAX = _i("SEM_CHUNK_MAX", 700)                                         # 语义分块单块最大字符数
SEM_THRESHOLD = _f("SEM_THRESHOLD", 0.62)                                        # 相邻单元与当前块中心相似度低于此值即切块
SECTION_SPAN = _i("SECTION_SPAN", 3)                                             # 章节名作语境的有效页跨度，超出即失效，避免"张冠李戴"的语境污染
MERGE_MIN = _i("MERGE_MIN", 150)                                                 # 短于该长度的块向后合并，避免碎片挤占检索前排
VEC_TOP_K = _i("VEC_TOP_K", 20)                                                  # 向量通道召回条数
BM25_TOP_K = _i("BM25_TOP_K", 20)                                                # BM25 通道召回条数
FUSION_TOP_K = _i("FUSION_TOP_K", 25)                                            # 融合后进入重排的候选数
RERANK_TOP_N = _i("RERANK_TOP_N", 6)                                             # 重排后交给大模型的上下文条数
RERANK_MAX_CHARS = _i("RERANK_MAX_CHARS", 512)                                   # 重排打分时的单条截断字数：整表块可达数千字，不截断会把成对打分的填充开销放大十倍
VEC_WEIGHT = _f("VEC_WEIGHT", 0.7)                                               # 融合时向量得分权重
BM25_WEIGHT = _f("BM25_WEIGHT", 0.3)                                             # 融合时 BM25 得分权重

# ==================== 索引与题目落盘位置 ====================
INDEX_FILE = ROOT / "data" / "index.json"                                        # 分块结果，重启免于重建
BM25_FILE = ROOT / "data" / "bm25.pkl"                                           # BM25 索引持久化
QUESTIONS = ROOT / "tests" / "test_questions.json"                                # 工单 14 题（含标准答案与出处）
