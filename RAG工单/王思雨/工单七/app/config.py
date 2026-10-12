# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估 —— 全局配置层（路径/阈值/开关集中于此，其余模块只读）
# 硬性约定：路径一律写正斜杠，写成 D:\模型\bge-m3 时 "\b" 会被 .env 解析成退格符导致路径不存在
import os                                                                     # 读取环境变量
from pathlib import Path                                                      # 路径对象，跨平台且无转义歧义
from dotenv import load_dotenv                                                # 载入密钥文件
ROOT = Path(__file__).resolve().parent.parent                                 # 项目根目录（工单七/）
DATA, RESULTS = ROOT / "data", ROOT / "results"                               # 数据目录（分块表/BM25/解析产物）与结果目录
QUESTIONS_FILE = ROOT / "tests" / "questions.json"                             # 10 题题集（题面 + 标准答案 ground_truth）
load_dotenv(ROOT / ".env", override=False)                                    # 真实环境变量优先，缺失项才由 .env 补齐
def _e(k, d=""): return os.getenv(k, d).strip()                            # 统一取参口：去首尾空白，避免 .env 行尾空格污染路径
EMBED_MODEL_PATH = _e("EMBED_MODEL_PATH", "D:/模型/bge-m3")                    # 向量模型：bge-m3
RERANK_MODEL_PATH = _e("RERANK_MODEL_PATH", "D:/模型/bge-reranker-v2-m3")      # 重排模型：bge-reranker-v2-m3
DEVICE, EMBED_DIM = _e("DEVICE", "cuda"), int(_e("EMBED_DIM", "1024"))         # 推理设备 / bge-m3 稠密向量维度
EMBED_BATCH, RERANK_MAX_CHARS = int(_e("EMBED_BATCH", "16")), int(_e("RERANK_MAX_CHARS", "512"))  # 编码批量 16（8G 显存放得下）/ 重排输入截断 512 字符（不截断会按最长序列 padding 拖慢一倍）
LLM_API_KEY = _e("DEEPSEEK_API_KEY")                                           # 密钥只从环境变量/.env 读，不进代码
LLM_BASE_URL = _e("LLM_BASE_URL", "https://api.deepseek.com")                  # OpenAI 兼容端点（注意不带 /v1）
LLM_MODEL = _e("LLM_MODEL", "deepseek-chat")                                   # 生成用模型
LLM_TIMEOUT, LLM_MAX_TOKEN = float(_e("LLM_TIMEOUT", "30")), int(_e("LLM_MAX_TOKEN", "1024"))  # 单次超时秒数 / 生成上限
LLM_REASONING_EFFORT = _e("LLM_REASONING_EFFORT", "")                          # 推理型模型必须传 none，否则思考吃光 max_tokens 返回空
MILVUS_URI = _e("MILVUS_URI", "http://localhost:19530")                        # 本机容器已在跑，勿再起 compose
MILVUS_COLLECTION = _e("MILVUS_COLLECTION", "gd7_chunks")                      # 本工单独立集合，不污染工单六的 gd6_chunks
REDIS_URL = _e("REDIS_URL", "redis://localhost:6379/0")                        # Redis 缓存地址
REDIS_PROTOCOL = int(_e("REDIS_PROTOCOL", "2"))                                # 本机 Redis 5 不支持 RESP3，必须 protocol=2，否则连接直接失败
CACHE_TTL = int(_e("CACHE_TTL", "86400"))                                      # 缓存有效期（秒）
CHUNK_SIZE, CHUNK_OVERLAP = int(_e("CHUNK_SIZE", "512")), int(_e("CHUNK_OVERLAP", "80"))  # 语义分块目标块长与相邻重叠
SEM_THRESHOLD, SEM_MAX_CHUNK = float(_e("SEM_THRESHOLD", "0.62")), int(_e("SEM_MAX_CHUNK", "900"))  # 语义漂移阈值 / 硬上限
VEC_TOP_K, BM25_TOP_K = int(_e("VEC_TOP_K", "30")), int(_e("BM25_TOP_K", "30"))  # 两通道各自召回条数
FUSION_TOP_K = int(_e("FUSION_TOP_K", "30"))                                   # 融合后送重排的候选数
RERANK_TOP_N = int(_e("RERANK_TOP_N", "8"))                                    # 重排后喂给大模型的片段数
VEC_WEIGHT, BM25_WEIGHT = float(_e("VEC_WEIGHT", "0.7")), float(_e("BM25_WEIGHT", "0.3"))  # 融合权重（和为 1）
CORPUS_DIR = Path(_e("CORPUS_DIR", "D:/Users/雨子/Desktop/专高六/liu-专高六/RAG 工单/附件/ccf_competition"))  # 语料根目录
API_PORT, EVAL_MAX_WORKERS = int(_e("API_PORT", "8010")), int(_e("EVAL_MAX_WORKERS", "2"))  # 端口 8010（8000 被占用）/ RAGAS 并发 2（默认 16 会打满限流把样本打成 nan）
# 盘上文件名里的中文已被 U+FFFD 破坏且不可逆，只有股票代码幸存且稳定，故用代码回认公司名（唯一可靠标识）
CODE2NAME = {"000001": "平安银行", "601318": "中国平安", "600036": "招商银行", "601658": "邮储银行", "600030": "中信证券",
             "601628": "中国人寿", "601601": "中国太保", "600999": "招商证券", "601211": "国泰君安"}
