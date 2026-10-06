# -*- coding: utf-8 -*-
"""
全局配置模块：所有组件的连接地址、模型路径、检索参数集中在这里管理。

在系统中的位置：
    本模块是最底层的「配置中心」，被 logger / db_redis / db_mysql / db_milvus /
    retriever / main 等几乎所有业务模块 import；它自己不 import 任何业务模块，
    也不产生副作用（不建连接、不加载模型），因此不会形成循环依赖。

职责与设计取舍：
    1. 敏感信息（API Key、数据库密码）一律走 os.getenv 读取，代码里只留能跑通
       的本地默认值，既不写死密钥又方便教学演示时开箱即用；
    2. 每个角色对应一个独立的 Milvus 集合，用字典做「角色 key → 集合名」映射，
       新增角色只改这里，检索层代码不必改动；
    3. 与向量维度有关的 EMBED_DIM 单独列出并加注释提醒：换模型必须同步改这里
       并重建集合，否则写入会因维度不匹配直接报错。
"""

import os  # 读取环境变量（os.getenv 的第二个参数就是「取不到时的默认值」）

# ==================== 一、Milvus 向量数据库 ====================
# 向量库地址：Milvus 3.0.0 跑在本机（或 WSL 里做端口转发），默认 19530 端口；
# 必须带 http:// 协议头，只写 host:port 会被 pymilvus 当成非法 URI。
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")  # Milvus 服务地址

# 多集合架构：每个角色一个独立知识库，检索时按角色路由，互不干扰
# key   = 角色 key（请求参数里的角色标识，main.py 用它做路由）
# value = Milvus 集合名（集合不存在时由 LangChain 的封装自动创建）
MILVUS_COLLECTIONS = {
    "lawyer": os.getenv("MILVUS_COLLECTION_LEGAL", "rag_legal"),          # 法律顾问知识库
    "psychologist": os.getenv("MILVUS_COLLECTION_PSYCH", "rag_psychology"),  # 心理专家知识库
    "virtual_friend": os.getenv("MILVUS_COLLECTION_COMPANION", "rag_companion"),
}

# 默认集合（兼容旧代码和入库脚本）：入库脚本不显式传集合名时落到这个集合
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_legal")

# 长期记忆集合名：所有角色共用同一个集合，靠 user_id 字段过滤，避免为每个用户建集合
MILVUS_MEMORY_COLLECTION = os.getenv("MILVUS_MEMORY_COLLECTION", "rag_long_term_memory")  # 长期记忆集合名
MILVUS_INDEX_TYPE = "AUTOINDEX"  # 索引类型：交给 Milvus 自动选择（standalone 下通常 HNSW，教学场景不必手工调参）
MILVUS_METRIC_TYPE = "COSINE"  # 度量方式：余弦相似度（bge-m3 输出已归一化，此时 COSINE 与内积等价，取值落在 [-1,1] 便于设阈值）

# 角色 key 到集合名的映射函数
def get_collection_for_role(role_key: str) -> str:
    """
    根据角色 key 返回对应的 Milvus 集合名。

    参数：
        role_key：角色标识字符串，由请求参数传入，预期取值 "lawyer" /
                  "psychologist" / "virtual_friend"（即上面字典的键）。
    返回：
        str，命中的集合名；若 role_key 不认识（前端传了拼写错误的值，或角色表
        新增了角色但忘了在这里加映射），降级返回 MILVUS_COLLECTION，
        即走到法律知识库，保证检索链路不会因为配置缺失而直接 500。
    """
    return MILVUS_COLLECTIONS.get(role_key, MILVUS_COLLECTION)

# ==================== 二、Redis 短期记忆 ====================
# Redis 装在 WSL2 Ubuntu-22.04 里，通过 localhost 端口转发访问；requirepass 设为 123456
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")  # Redis 地址
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))  # Redis 端口（环境变量取出来是字符串，必须 int 转换）
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "123456")  # Redis 密码（WSL 里 requirepass 设的就是这个）
REDIS_DB = int(os.getenv("REDIS_DB", "0"))  # Redis 库编号（0~15，不同项目用不同编号可以隔离数据）
MEMORY_WINDOW = 20  # 短期记忆窗口：保留最近 20 条对话（10 轮问答），太大浪费 token，太小会丢上下文

# ==================== 三、MySQL 业务数据库 ====================
# 同样在 WSL 里，存放用户 / 角色 / 对话日志等需要长期持久化的结构化数据
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")  # MySQL 地址
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))  # MySQL 端口
MYSQL_USER = os.getenv("MYSQL_USER", "root")  # MySQL 用户名
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "123456")  # MySQL 密码
MYSQL_DB = os.getenv("MYSQL_DB", "rag")  # 数据库名（须提前手动建库，代码只负责建表）

# ==================== 四、向量模型（本地 bge-m3 权重，不再走 Ollama） ====================
# 已从 Ollama 的 HTTP 调用改为 langchain_huggingface 的 HuggingFaceEmbeddings 直载本地权重：
# 好处是离线可用、没有网络往返和 Ollama 服务依赖；代价是进程首次调用要花几秒加载权重。
EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH", r"D:\Projects\Models\bge-m3")  # 本地模型目录（用 r"" 原始字符串，避免 \P 之类被当成转义符）
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")  # 运行设备：cpu 最稳；有 GPU 可改 "cuda"（需装对应 CUDA 版 torch）
EMBED_DIM = 1024  # bge-m3 输出维度（换模型必须同步改，且重建集合：Milvus 的向量维度在建集合时就固化了）

# ==================== 五、重排模型（本地权重目录） ====================
RERANK_MODEL = os.getenv(
    "RERANK_MODEL",
    r"D:\Projects\Models\BAAI--bge-reranker-large\bge-reranker-large",  # HF 缓存外层多套一层目录（真正权重在嵌套目录里，路径少写一层会加载失败）
)
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cpu")  # 运行设备：有 GPU 可改 "cuda"（重排是逐条推理，GPU 提速明显）

# ==================== 六、大模型（DeepSeek 为默认，支持切换其他） ====================
# 在线 API 方式（默认 DeepSeek）：用 langchain_openai.ChatOpenAI，它兼容 OpenAI 协议，换 base_url 即可换厂商
LLM_API_KEY = os.getenv("deepseek_api_key1")  # DeepSeek API Key（从环境变量读；取不到是 None，调用时才会报鉴权错）
LLM_BASE_URL = os.getenv("deepseek_base_url")  # 接口地址
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-flash")  # 模型名
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))  # 温度：0 最确定、1 最发散；角色扮演可适当调高到 0.5-0.7
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))  # 单次回答最大输出 token（限制成本与响应延迟）

# 本地部署方式（vLLM / SGLang / xInference）—— 与上面二选一，设 LOCAL_LLM_BASE_URL 即切换
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "")  # 如 http://localhost:8000/v1（vLLM 启动地址）；空串表示不启用本地模型
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen2.5-7B-Instruct")  # 本地模型名（要与 vLLM 的 --served-model-name 一致）

# ==================== 七、文本切分与检索参数 ====================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # 单块目标字数：太大召回不精准，太小语义不完整
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))  # 块间重叠字数：防止答案正好被切断在块边界上
TOP_K_RECALL = 10  # 每路召回条数（向量 10 + BM25 10），两路各自召回后再交给 RRF 融合
TOP_K_RERANK = 5  # 重排后默认保留条数，即最终喂给 LLM 的文档数
SCORE_THRESHOLD = 0.3  # 余弦相似度过滤阈值：低于此分的文档不喂给 LLM（宁可少给，也别用无关内容污染回答）

# ==================== 八、FastAPI 服务 ====================
API_HOST = os.getenv("API_HOST", "0.0.0.0")  # 监听地址（0.0.0.0 = 对外开放，局域网可访问；127.0.0.1 只允许本机）
API_PORT = int(os.getenv("API_PORT", "8000"))  # 端口

# ==================== 九、日志 ====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # 日志级别：DEBUG / INFO / WARNING / ERROR（DEBUG 可看到检索命中的原始文本）
LOG_FILE = os.getenv("LOG_FILE", "persona_rag.log")  # 日志文件路径（相对路径时落在启动进程的工作目录）