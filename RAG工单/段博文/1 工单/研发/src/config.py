# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""
全局配置模块：所有组件的连接地址、模型路径、检索参数集中在这里管理。

在系统中的位置：
    本模块是最底层的「配置中心」，被 logger / db_milvus / retriever /
    llm_client / main / ingest_pdf 等几乎所有业务模块 import；它自己不
    import 任何业务模块，也不产生副作用（不建连接、不加载模型），因此不会
    形成循环依赖。

职责与设计取舍：
    1. 敏感信息（API Key）一律走 os.getenv 读取，代码里不写死密钥，既安全
       又方便教学演示时开箱即用；
    2. 单集合架构：所有 PDF 文档入库到同一个 Milvus 集合，简化路由；
    3. 与向量维度有关的 EMBED_DIM 单独列出并加注释提醒：换模型必须同步改
       这里并重建集合，否则写入会因维度不匹配直接报错。
"""

import os  # 读取环境变量（os.getenv 的第二个参数就是「取不到时的默认值」）

# ==================== 一、Milvus 向量数据库 ====================
# 向量库地址：Milvus 3.0.0 跑在本机（或 WSL 里做端口转发），默认 19530 端口；
# 必须带 http:// 协议头，只写 host:port 会被 pymilvus 当成非法 URI。
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")  # Milvus 服务地址

# 单集合架构：所有 PDF 文档统一入库到这一个集合，检索时不做角色路由
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_pdf_qa")  # 集合名（不存在时由 LangChain 自动创建）

# 索引与度量
MILVUS_INDEX_TYPE = "AUTOINDEX"  # 索引类型：交给 Milvus 自动选择（standalone 下通常 HNSW，教学场景不必手工调参）
MILVUS_METRIC_TYPE = "COSINE"  # 度量方式：余弦相似度（bge-m3 输出已归一化，此时 COSINE 与内积等价，取值落在 [-1,1] 便于设阈值）

# ==================== 二、向量模型（本地 bge-m3 权重） ====================
# 用 langchain_huggingface 的 HuggingFaceEmbeddings 直载本地权重：
# 好处是离线可用、没有网络往返和 Ollama 服务依赖；代价是进程首次调用要花几秒加载权重。
EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH", r"D:\Projects\Models\bge-m3")  # 本地模型目录（用 r"" 原始字符串，避免 \P 之类被当成转义符）
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")  # 运行设备：cpu 最稳；有 GPU 可改 "cuda"（需装对应 CUDA 版 torch）
EMBED_DIM = 1024  # bge-m3 输出维度（换模型必须同步改，且重建集合：Milvus 的向量维度在建集合时就固化了）

# ==================== 三、重排模型（本地权重目录） ====================
RERANK_MODEL = os.getenv(
    "RERANK_MODEL",
    r"D:\Projects\Models\BAAI--bge-reranker-large\bge-reranker-large",  # HF 缓存外层多套一层目录（真正权重在嵌套目录里，路径少写一层会加载失败）
)
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cpu")  # 运行设备：有 GPU 可改 "cuda"（重排是逐条推理，GPU 提速明显）

# ==================== 四、大模型（DeepSeek API） ====================
# 在线 API 方式（DeepSeek）：用 langchain_openai.ChatOpenAI，它兼容 OpenAI 协议
LLM_API_KEY = os.getenv("deepseek_api_key1")  # DeepSeek API Key（从环境变量读；取不到是 None，调用时才会报鉴权错）
LLM_BASE_URL = os.getenv("deepseek_base_url")  # 接口地址
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")  # 模型名
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))  # 温度：0 最确定、1 最发散；问答场景 0.3 偏稳定
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))  # 单次回答最大输出 token（限制成本与响应延迟）

# ==================== 五、文本切分与检索参数 ====================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # 单块目标字数：太大召回不精准，太小语义不完整
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))  # 块间重叠字数：防止答案正好被切断在块边界上
TOP_K_RECALL = 10  # 每路召回条数（向量 10 + BM25 10），两路各自召回后再交给 RRF 融合
TOP_K_RERANK = 5  # 重排后默认保留条数，即最终喂给 LLM 的文档数
SCORE_THRESHOLD = 0.3  # 余弦相似度过滤阈值：低于此分的文档不喂给 LLM（宁可少给，也别用无关内容污染回答）

# ==================== 六、FastAPI 服务 ====================
API_HOST = os.getenv("API_HOST", "0.0.0.0")  # 监听地址（0.0.0.0 = 对外开放，局域网可访问；127.0.0.1 只允许本机）
API_PORT = int(os.getenv("API_PORT", "8000"))  # 端口

# ==================== 七、日志 ====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # 日志级别：DEBUG / INFO / WARNING / ERROR（DEBUG 可看到检索命中的原始文本）
LOG_FILE = os.getenv("LOG_FILE", "rag_pdf_qa.log")  # 日志文件路径（相对路径时落在启动进程的工作目录）
