# -*- coding: utf-8 -*-
"""全局配置：路径、本地模型、数据库、缓存、检索参数。

所有路径与地址均可被同名环境变量覆盖，便于后续换机部署。
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- 路径
BACKEND_DIR = Path(__file__).resolve().parents[2]          # backend/
BASE_DIR = BACKEND_DIR.parent                              # RAGLoRA/
QDRANT_DIR = BASE_DIR / "qdrant_storage"
DATASETS_DIR = BASE_DIR / "datasets"
LOG_DIR = BACKEND_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 本地模型（绝不联网下载）
EMBED_MODEL_PATH = os.environ.get("EMBED_MODEL_PATH", r"D:\桌面\模型\嵌入模型\bge-m3")
RERANK_MODEL_PATH = os.environ.get("RERANK_MODEL_PATH", r"D:\桌面\模型\精排模型\bge-reranker-v2-m3")
EMBED_DIM = 1024
EMBED_MAX_LEN = 512

# ---------------------------------------------------------------- MySQL
MYSQL_HOST = os.environ.get("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "123456")
MYSQL_DB = os.environ.get("MYSQL_DB", "raglora")
DB_URL = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
)
SQL_ECHO = os.environ.get("SQL_ECHO", "0") == "1"

# ---------------------------------------------------------------- Redis（短期记忆）
REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
MEMORY_TURNS = 6            # 保留最近 N 轮（1 轮 = 问 + 答）
MEMORY_TTL = 3600           # 秒

# Redis 其余数据类型（2026-09-27）：String 限流 / Hash 角色缓存 / Set 分类索引 / zSet 热度榜
RATE_LIMIT_PER_MIN = int(os.environ.get("RAGLORA_RATE_LIMIT", "30"))  # 每用户每分钟问答上限，0=关闭
CHAR_CACHE_TTL = int(os.environ.get("RAGLORA_CHAR_CACHE_TTL", "300"))  # 角色详情缓存秒数

# ---------------------------------------------------------------- Ollama（生成）
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "ollama")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b")
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 800

# ---------------------------------------------------------------- 鉴权
JWT_SECRET = os.environ.get("JWT_SECRET", "raglora-dev-secret-change-me-in-production")
JWT_ALG = "HS256"
JWT_EXPIRE_HOURS = int(os.environ.get("JWT_EXPIRE_HOURS", "72"))
PBKDF2_ROUNDS = 200_000

# ---------------------------------------------------------------- 检索
RECALL_TOP_K = 20           # 混合检索召回条数（进精排的候选池）
RERANK_TOP_K = 5            # 精排后送给大模型的条数
RERANK_ENABLED = True       # 精排总开关（模型不可用时自动降级）
RERANK_MAX_SEQ = 512        # cross-encoder 输入截断
SPARSE_MIN_WEIGHT = 0.01    # 稀疏词权重过滤阈值
CHUNK_MAX_CHARS = 800       # 单条 chunk 写入 prompt 前的截断长度
CHUNK_SIZE = 512            # PDF 段落分块目标字符数
CHUNK_OVERLAP = 50

# ---------------------------------------------------------------- 查询改写
REWRITE_ENABLED = True      # 多轮对话时做指代消解改写

# ---------------------------------------------------------------- 多路召回（2026-09-17）
# 三路：向量库(语义) + MySQL(文档级) + Neo4j(法条关系)
# 关闭时退回单路向量检索（兼容旧行为，也便于对比两模式的实际差异）
MULTI_RECALL_ENABLED = os.environ.get("RAGLORA_MULTI_RECALL", "1") == "1"

# RRF 融合常数。沿用 Qdrant 侧默认值，保持两条路径口径一致。
RRF_K = int(os.environ.get("RAGLORA_RRF_K", "60"))

# 各路权重。图谱路做的是**精确**匹配（指名法条），比语义路更可信，故略高；
# 文档限定路是「缩小范围后的语义检索」，与主向量路同质，权重相同以免重复计票。
# ⚠️ 这些权重是**先验设定，未经过调参**。实际增益需用
#    `scripts/eval_multirecall.py` 实测（见 docs/08-多路召回说明.md）。
MULTI_RECALL_WEIGHTS = {
    "vector": float(os.environ.get("RAGLORA_W_VECTOR", "1.0")),
    "graph": float(os.environ.get("RAGLORA_W_GRAPH", "1.2")),
}

# 当 query 指名了某份库内文档时，对该文档内的命中**加分**。
# 注意是「加分」而非「另开一路计票」—— 后者会与向量路重复计票，
# 把图谱路的精确定中挤下去（实测踩过，见 retrieval.multi_recall 注释）。
DOC_BOOST = float(os.environ.get("RAGLORA_DOC_BOOST", "1.5"))

# ---------------------------------------------------------------- 启动预热
# 预热会把 bge-m3(fp32, ~2.3GB) 与精排(fp16, ~1.1GB) 常驻内存，首个请求因此快 7~10 秒。
# 实测本机 16GB 内存下，后端 2.8GB + Ollama 4.7GB 已接近上限，曾两次触发
# Windows 资源耗尽保护（事件 2004）导致进程被终结。内存吃紧时可置 0 关掉预热，
# 改为首次使用时按需加载。
WARMUP_ENABLED = os.environ.get("RAGLORA_WARMUP", "1") == "1"

# ---------------------------------------------------------------- 知识库集合
COLLECTION_MEDICAL = "kb_medical"
COLLECTION_LEGAL = "kb_legal"

# ---------------------------------------------------------------- 链路与组件开关（2026-09-15 新增）
# 手动链路(manual) 与 LangChain 链路(langchain) 并存。
# 2026-09-16 起默认走 langchain（用户指定）；
# 手写链路保留为可切换备选：RAGLORA_CHAIN=manual 即可回退到已验证基线。
CHAIN_BACKEND = os.environ.get("RAGLORA_CHAIN", "langchain")

# 向量库选择：milvus（Docker，默认） / qdrant（嵌入式） / both（双库对比）
# 2026-09-16 起默认 milvus（用户指定必须使用）。
# ⚠️ 用 milvus 需先起 Docker 与容器：bash tools/demo/up.sh milvus
#    起不来时可用 RAGLORA_VECTOR_STORE=qdrant 临时回退（无需 Docker）。
VECTOR_STORE = os.environ.get("RAGLORA_VECTOR_STORE", "milvus")

# OCR 分流：页均字符数低于该值判为扫描件（详见设计文档 §3.5）
OCR_TEXT_THRESHOLD = int(os.environ.get("RAGLORA_OCR_TEXT_THRESHOLD", "50"))

# ---------------------------------------------------------------- 外部服务（Docker）
MILVUS_URI = os.environ.get("MILVUS_URI", "http://127.0.0.1:19530")
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "raglora123")

# ---------------------------------------------------------------- 嵌入模型对比
# D:\桌面\模型\嵌入模型\ 下有 bge-m3 / m3e-base / bert-base-chinese
EMBED_MODELS_DIR = os.environ.get("EMBED_MODELS_DIR", r"D:\桌面\模型\嵌入模型")
