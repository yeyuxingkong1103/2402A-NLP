# -*- coding: utf-8 -*-
"""
全局配置：所有组件的连接地址、模型路径、参数集中管理
从环境变量读取敏感信息（API Key、数据库密码），不写死在代码里
"""

import os  # 读取环境变量

# ==================== 一、Milvus 向量数据库 ====================
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")  # Milvus 服务地址

# 多集合架构：每个角色一个独立知识库
MILVUS_COLLECTIONS = {
    "lawyer": os.getenv("MILVUS_COLLECTION_LEGAL", "rag_legal"),          # 法律顾问
    "psychologist": os.getenv("MILVUS_COLLECTION_PSYCH", "rag_psychology"),  # 心理专家
    "virtual_friend": os.getenv("MILVUS_COLLECTION_COMPANION", "rag_companion"),  # 虚拟朋友
}

# 默认集合（兼容旧代码和入库脚本）
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_legal")

MILVUS_MEMORY_COLLECTION = os.getenv("MILVUS_MEMORY_COLLECTION", "rag_long_term_memory")  # 长期记忆集合名
MILVUS_INDEX_TYPE = "AUTOINDEX"  # 索引类型：交给 Milvus 自动选择（standalone 下通常 HNSW）
MILVUS_METRIC_TYPE = "COSINE"  # 度量方式：余弦相似度（bge-m3 输出已归一化）

# 角色 key 到集合名的映射函数
def get_collection_for_role(role_key: str) -> str:
    """根据角色 key 返回对应的 Milvus 集合名"""
    return MILVUS_COLLECTIONS.get(role_key, MILVUS_COLLECTION)

# ==================== 二、Redis 短期记忆 ====================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")  # Redis 地址
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))  # Redis 端口
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")  # Redis 密码（无密码留空）
REDIS_DB = int(os.getenv("REDIS_DB", "0"))  # Redis 库编号
MEMORY_WINDOW = 20  # 短期记忆窗口：保留最近 20 条对话（10 轮问答）

# ==================== 三、MySQL 业务数据库 ====================
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")  # MySQL 地址
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))  # MySQL 端口
MYSQL_USER = os.getenv("MYSQL_USER", "root")  # MySQL 用户名
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "123456")  # MySQL 密码
MYSQL_DB = os.getenv("MYSQL_DB", "rag")  # 数据库名

# ==================== 四、Ollama 向量模型 ====================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # Ollama 服务地址
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3:567m")  # 向量模型标签
EMBED_DIM = 1024  # bge-m3 输出维度（换模型必须同步改，且重建集合）

# ==================== 五、重排模型（本地 modelscope 路径） ====================
RERANK_MODEL = os.getenv(
    "RERANK_MODEL",
    r"D:\model\modelscope\models\BAAI--bge-reranker-v2-m3\snapshots\master",  # modelscope 本地路径
)
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cpu")  # 有 GPU 可改 "cuda"

# ==================== 六、大模型（DeepSeek 为默认，支持切换其他） ====================
# 在线 API 方式（默认 DeepSeek）
LLM_API_KEY = os.getenv("deepseek_api_key1", "")  # DeepSeek API Key（从环境变量读）
LLM_BASE_URL = os.getenv("deepseek_base_url", "https://api.deepseek.com/v1")  # 接口地址
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-flash")  # 模型名
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))  # 温度：角色扮演可适当调高到 0.5-0.7
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))  # 最大输出 token

# 本地部署方式（vLLM / SGLang / xInference）—— 与上面二选一，设 LOCAL_LLM_BASE_URL 即切换
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "")  # 如 http://localhost:8000/v1（vLLM 启动地址）
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen2.5-7B-Instruct")  # 本地模型名

# ==================== 七、文本切分参数 ====================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # 单块目标字数
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))  # 块间重叠字数
TOP_K_RECALL = 10  # 每路召回条数（向量 10 + BM25 10）
TOP_K_RERANK = 5  # 重排后默认保留条数
SCORE_THRESHOLD = 0.3  # 余弦相似度过滤阈值：低于此分的文档不喂给 LLM

# ==================== 八、FastAPI 服务 ====================
API_HOST = os.getenv("API_HOST", "0.0.0.0")  # 监听地址（0.0.0.0 = 对外开放）
API_PORT = int(os.getenv("API_PORT", "8000"))  # 端口

# ==================== 九、日志 ====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # 日志级别：DEBUG / INFO / WARNING / ERROR
LOG_FILE = os.getenv("LOG_FILE", "rag_roleplay.log")  # 日志文件路径
