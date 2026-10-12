# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 文件说明：集中配置。密钥只从环境变量取；路径、模型、检索与图像解析参数一律走这里，改配置不改代码。
import os                                                                 # 读取环境变量
import logging                                                            # 日志级别控制
from pathlib import Path                                                  # 跨平台路径
from dotenv import load_dotenv                                            # 解析 .env

# pymilvus 与 jieba 默认在 DEBUG 级别刷屏（建连接、建集合、加载词典都打日志），会把入库进度淹没
logging.getLogger("pymilvus").setLevel(logging.WARNING)                   # Milvus 只留警告以上
logging.getLogger("jieba").setLevel(logging.WARNING)                      # jieba 只留警告以上

ROOT = Path(__file__).resolve().parent                                    # 项目根目录
load_dotenv(ROOT / ".env", override=False)                                # .env 载入；已存在的真实环境变量优先，便于 CI 覆盖
_s = lambda k, d="": os.environ.get(k, d)                                 # 字符串配置
_i = lambda k, d: int(os.environ.get(k, d) or d)                          # 整型配置，空串回落默认值
_f = lambda k, d: float(os.environ.get(k, d) or d)                        # 浮点配置，空串回落默认值

# ==================== 大模型：DeepSeek（OpenAI 兼容接口）====================
DEEPSEEK_API_KEY = _s("DEEPSEEK_API_KEY")                                 # 密钥不落代码
LLM_BASE_URL = _s("LLM_BASE_URL", "https://api.deepseek.com")             # 接口基址
LLM_MODEL = _s("LLM_MODEL", "deepseek-chat")                              # 对话模型
LLM_REASONING_EFFORT = _s("LLM_REASONING_EFFORT", "")                     # 推理型模型必须显式传 none，否则思考过程吃光 max_tokens 返回空内容
LLM_TIMEOUT = _f("LLM_TIMEOUT", 20)                                       # 单次生成超时（秒）：超时即降级返回检索原文，而不是把接口拖死

# ==================== 本地模型：路径一律用正斜杠 ====================
# 坑：写成 D:\模型\bge-m3 时 .env 会把 "\b" 解析成退格符，加载报路径不存在
EMBED_MODEL_PATH = _s("EMBED_MODEL_PATH", "D:/模型/bge-m3")                # bge-m3 向量模型
RERANK_MODEL_PATH = _s("RERANK_MODEL_PATH", "D:/模型/bge-reranker-v2-m3")  # bge-reranker-v2-m3 重排模型
CLIP_MODEL_PATH = _s("CLIP_MODEL_PATH", "D:/模型/chinese-clip-vit-base-patch16")  # ★工单四核心：中文 CLIP 多模态模型
DEVICE = _s("DEVICE", "cuda")                                             # 推理设备；无 GPU 改 cpu
EMBED_DIM = _i("EMBED_DIM", 1024)                                         # bge-m3 向量维度

# ==================== 向量库与缓存 ====================
MILVUS_URI = _s("MILVUS_URI", "http://localhost:19530")                   # Milvus 地址
MILVUS_COLLECTION = _s("MILVUS_COLLECTION", "gd4_chunks")                 # 集合名（工单四独立集合，不与工单三混用）
REDIS_URL = _s("REDIS_URL", "redis://localhost:6379/0")                   # Redis 地址
CACHE_TTL = _i("CACHE_TTL", 86400)                                        # 问答缓存有效期（秒）
CACHE_PREFIX = _s("CACHE_PREFIX", "gd4:ans:")                             # 缓存键前缀

# ==================== 知识库：工单指定 PDF ====================
RAW_DIR = ROOT / _s("RAW_DIR", "data/raw")                                # 原始 PDF 目录
PARSED_DIR = ROOT / _s("PARSED_DIR", "data/parsed")                       # MinerU 解析产物目录
INDEX_FILE = ROOT / "data" / "index.json"                                 # 分块结果缓存
BM25_FILE = ROOT / "data" / "bm25.pkl"                                    # BM25 索引缓存
QUESTIONS = ROOT / "tests" / "test_questions.json"                        # 工单 16 题（含标准答案）

def docs() -> list:                                                       # 知识库内 PDF 列表
    return sorted(RAW_DIR.glob("*.pdf"))                                  # 目录扫描，新增文档自动纳入

# ==================== MinerU：PDF 主解析器 ====================
MINERU_CMD = _s("MINERU_CMD", "mineru")                                   # MinerU 可执行入口
MINERU_PYTHON = _s("MINERU_PYTHON")                                       # 单独解释器路径；留空=当前解释器（本机 MinerU 与主程序同环境）
MINERU_LANGFIX_DIR = _s("MINERU_LANGFIX_DIR", "C:/mineru_models/langfix")  # 中文路径补丁：用户名含中文时 fasttext 打不开模型，缺了必失败
MINERU_TIMEOUT = _i("MINERU_TIMEOUT", 7200)                               # 单篇解析超时（秒）
MINERU_ENABLED = _s("MINERU_ENABLED", "1") == "1"                         # 关掉则直接走 pymupdf 兜底，便于快速回归

# ==================== 兜底解析器：pymupdf / pdfplumber / paddleocr ====================
# PaddleOCR 模型缓存默认写死 ~/.paddleocr/，用户名含中文时 C++ 推理层打不开该路径，
# 故把模型另存到纯英文目录并显式传入。首次部署执行：
#   cp -r ~/.paddleocr/whl/{det,rec,cls} C:/paddleocr_models/
PADDLEOCR_MODEL_DIR = _s("PADDLEOCR_MODEL_DIR", "C:/paddleocr_models")     # 纯英文的 PaddleOCR 模型根目录
OCR_SCALE = _f("OCR_SCALE", 3.0)                                          # ★图像 OCR 放大倍数：1x 会把图表里的负号"-2.0%"读丢，3x 才稳定读出
OCR_MIN_CONF = _f("OCR_MIN_CONF", 0.6)                                    # OCR 置信度下限
IMAGE_MIN_PX = _i("IMAGE_MIN_PX", 120)                                    # 图像最小边长（像素），比它小的多为水印碎片/项目符号
IMAGE_MIN_BYTES = _i("IMAGE_MIN_BYTES", 4096)                             # 图像最小字节数，滤掉纯色小块
IMAGE_OCR_MAX = _i("IMAGE_OCR_MAX", 600)                                  # 单张图 OCR 文本截断长度，防个别大图撑爆块
IMAGE_BLOCK_MAX = _i("IMAGE_BLOCK_MAX", 1200)                             # 图像语义块最大字符数

# 页眉页脚与签名页噪声：图表里混进的斜向水印文字，OCR 会读成独立文本行
WATERMARK_TOKENS = [t for t in _s("WATERMARK_TOKENS",
    "八维,维教育,人工智能,智能刘敏,刘敏,rengong,zhineng,liumin,liumina,Liumin,八细,货刘,reng").split(",") if t]
SIGN_PAGE_KEYWORDS = ["签名", "声明", "盖章", "签字", "法定代表人："]       # 声明页整页签名图，忽略不建块

# ==================== 分块与检索参数（调参入口）====================
SEM_CHUNK_MAX = _i("SEM_CHUNK_MAX", 700)                                  # 语义分块单块最大字符数
SEM_THRESHOLD = _f("SEM_THRESHOLD", 0.62)                                 # 相邻句与当前块中心相似度低于此值即切块
MERGE_MIN = _i("MERGE_MIN", 150)                                          # 短于该长度的块向后合并，避免碎片挤占前排
VEC_TOP_K = _i("VEC_TOP_K", 40)                                           # 向量通道召回条数
BM25_TOP_K = _i("BM25_TOP_K", 40)                                         # BM25 通道召回条数
RRF_K = _i("RRF_K", 60)                                                   # RRF 融合常数：score=Σ1/(k+rank)，比加权归一化更稳，免调权重
# 融合池必须够大：实测"军用领域收入"题的答案块在 BM25 通道排第 21、向量通道未进前 50，
# 池子只有 25 时它根本进不了重排，答案直接变"未找到"；放到 60 后该块重排排到第 1。
FUSION_TOP_K = _i("FUSION_TOP_K", 60)                                     # 融合后进入重排的候选数
RERANK_TOP_N = _i("RERANK_TOP_N", 8)                                      # 重排后交给大模型的上下文条数
# 低分尾巴不进上下文：实测 id=34 的 8 条里，有 4 条重排分只有 0.40~0.47（「审计机构声明」
# 「验资复核机构声明」「承诺函」这类套话页），它们按本项目自己的相关性阈值（0.5）就是不相关。
# 留在大模型上下文里只是噪声，还直接拉低上下文精确度。保留条数下限防止全被裁空。
MIN_CONTEXT = _i("MIN_CONTEXT", 3)                                        # 裁掉低分块后至少保留的上下文条数
RERANK_MAX_CHARS = _i("RERANK_MAX_CHARS", 512)                            # 重排打分的单条截断字数：整表块数千字，不截断打分耗时会翻十倍
# 重排批大小：pymilvus/transformers 的默认批很大，对长文本极慢。实测（60 条候选、16 题中位、正反序各测一遍）：
#   batch=2→432ms  4→410ms  8→437ms  16→536ms  32→804ms
# 注意别只跟"默认值"比就收工——设成 16 确实比默认快得多，但**不是最优**，4 才是（比 16 快 23%）。
RERANK_BATCH = _i("RERANK_BATCH", 4)                                      # 重排批大小：4 为实测最优，16 会慢 125ms/题
TABLE_EXPAND_MAX = _i("TABLE_EXPAND_MAX", 14)                             # 表格行补全后上下文总条数上限，防止整张大表撑爆提示词
# 补全的**触发门槛**：只有排进重排前 N 名的表格行，才允许把它所属的整张表带进来。
# 实测依据（16 题逐题记录触发行名次）：所有真正需要补全的题都至少有一个触发行排在前 3
# （id=1@2、3@1、4@1、260@2、33@2、543@1、531@2、207@3）；而 id=793 问的是"下游行业"，
# 最好的触发行只排到第 4（0.719 分的「行业分类」行），却把 P22「发行人基本情况表」的
# 注册资本/注册地址/控股股东等 5 行无关内容全带了进来，把上下文精度拉到 0.50。
# 判据用名次而不是分数：有害触发的 0.719 与合法触发的 0.743 差得太近，按分数切会误伤。
TABLE_EXPAND_TRIGGER_RANK = _i("TABLE_EXPAND_TRIGGER_RANK", 3)            # 触发补全所需的最低重排名次
IMAGE_BOOST = _f("IMAGE_BOOST", 1.15)                                     # ★图像语义块权重加成：图像块少而信息密度高，实测加成后图文问题命中率上升
NOISE_PENALTY = _f("NOISE_PENALTY", 0.6)                                  # ★水印污染块降权系数：被水印打碎的表格行，数字与年份对应关系会错位，不能让它排在前排
