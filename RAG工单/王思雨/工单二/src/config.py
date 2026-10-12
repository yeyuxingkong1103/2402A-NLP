# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""配置模块：从 .env 与环境变量集中读取参数，密钥不落代码，检索参数不改代码即可调优。"""
import os                                                                                     # 读取环境变量
from pathlib import Path                                                                      # 跨平台路径
from dotenv import load_dotenv                                                                # 解析 .env 文件

ROOT = Path(__file__).resolve().parent.parent                                                  # 项目根目录（src 的上一级）
load_dotenv(ROOT / ".env", override=False)                                                     # 载入 .env，已存在的真实环境变量优先
_s = lambda k, d="": os.environ.get(k, d)                                                      # 字符串型配置读取
_i = lambda k, d: int(os.environ.get(k, d) or d)                                               # 整型配置读取，空串回落默认值
_f = lambda k, d: float(os.environ.get(k, d) or d)                                             # 浮点型配置读取

DEEPSEEK_API_KEY = _s("DEEPSEEK_API_KEY")                                                      # 密钥只从环境变量取，绝不硬编码
LLM_BASE_URL = _s("LLM_BASE_URL", "https://api.deepseek.com")                                  # DeepSeek OpenAI 兼容接口基址
LLM_MODEL = _s("LLM_MODEL", "deepseek-chat")                                                   # 默认对话模型
EMBED_MODEL_PATH = _s("EMBED_MODEL_PATH", "D:/模型/bge-m3")                                     # bge-m3 路径，正斜杠避免反斜杠被当转义
RERANK_MODEL_PATH = _s("RERANK_MODEL_PATH", "D:/模型/bge-reranker-v2-m3")                       # bge-reranker-v2-m3 路径
DEVICE = _s("DEVICE", "cuda")                                                                  # 推理设备，无 GPU 时可改 cpu
EMBED_DIM = _i("EMBED_DIM", 1024)                                                              # bge-m3 向量维度
MILVUS_URI = _s("MILVUS_URI", "http://localhost:19530")                                         # Milvus 地址
MILVUS_COLLECTION = _s("MILVUS_COLLECTION", "gd2_opt_chunks")                                   # 优化后方案的向量集合
MILVUS_COLLECTION_BASE = _s("MILVUS_COLLECTION_BASE", "gd2_base_chunks")                        # 优化前基线的向量集合
REDIS_URL = _s("REDIS_URL", "redis://localhost:6379/0")                                         # Redis 地址，用于结果缓存
PDF_PATH = ROOT / _s("PDF_PATH", "data/招股说明书1.pdf")                                         # 知识库 PDF
MINERU_LANGFIX_DIR = _s("MINERU_LANGFIX_DIR", "C:/mineru_models/langfix")                       # 中文路径补丁目录，缺了 MinerU 必失败
BASE_CHUNK_SIZE = _i("BASE_CHUNK_SIZE", 500)                                                    # 优化前基线：固定分块字符数
SEM_CHUNK_MAX = _i("SEM_CHUNK_MAX", 700)                                                        # 语义分块单块最大字符数
SEM_THRESHOLD = _f("SEM_THRESHOLD", 0.62)                                                       # 相邻单元相似度低于该值即切块
VEC_TOP_K = _i("VEC_TOP_K", 20)                                                                 # 向量检索召回条数
BM25_TOP_K = _i("BM25_TOP_K", 20)                                                               # BM25 关键词召回条数
FUSION_TOP_K = _i("FUSION_TOP_K", 25)                                                           # 融合后进入重排的候选数
RERANK_TOP_N = _i("RERANK_TOP_N", 6)                                                            # 重排后交给大模型的上下文条数
VEC_WEIGHT = _f("VEC_WEIGHT", 0.7)                                                              # 融合时向量得分权重
BM25_WEIGHT = _f("BM25_WEIGHT", 0.3)                                                            # 融合时关键词得分权重
QUESTIONS = ROOT / "data" / "eval" / "questions.json"                                           # 工单问题清单（含标准答案与出处页码）
