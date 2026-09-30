# -*- coding: utf-8 -*-
"""全局配置：从 .env / 环境变量读取。"""
from pydantic_settings import BaseSettings, SettingsConfigDict
# 解析：pydantic-settings（配置类 + .env 加载）


# 全局配置类：pydantic-settings 从 .env/环境变量加载
class Settings(BaseSettings):
    # 解析：全局配置类
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)
    # 解析：从 .env 加载，字段名大小写不敏感（环境变量优先于 .env）

    # 大模型（OpenAI 兼容）
    llm_base_url: str = "https://api.deepseek.com/v1"
    # 解析：LLM API 地址（DeepSeek 默认；豆包/千问/本地 vLLM 换这里）
    llm_api_key: str = ""
    # 解析：LLM API 密钥
    llm_model: str = "deepseek-chat"
    # 解析：模型名
    llm_timeout: float = 60.0
    # 解析：请求超时（秒）
    llm_mock: bool = False  # 压测开关：true 时 LLM 层返回预设回复
    # 解析：压测开关——为真时对话不发外部 API（压测 S2/S4 用）

    # MySQL
    mysql_host: str = "127.0.0.1"
    # 解析：MySQL 地址
    mysql_port: int = 3306
    # 解析：MySQL 端口
    mysql_user: str = "root"
    # 解析：用户名
    mysql_password: str = ""
    # 解析：密码（.env 里配置）
    mysql_database: str = "rag_roleplay"
    # 解析：库名

    # Redis
    redis_host: str = "127.0.0.1"
    # 解析：Redis 地址
    redis_port: int = 6379
    # 解析：Redis 端口
    redis_password: str = ""
    # 解析：Redis 密码（本地无密码）
    redis_db: int = 0
    # 解析：Redis 库号

    # 会话
    history_max_rounds: int = 10
    # 解析：短期记忆保留轮数
    history_ttl_seconds: int = 604800
    # 解析：历史过期（7 天）
    token_ttl_seconds: int = 604800
    # 解析：登录 token 有效期（7 天）

    # RAGAS 评测 judge（换模型重判用；不填则复用 llm_* 配置）
    judge_base_url: str = ""
    # 解析：judge API 地址（留空回落 llm_base_url）
    judge_api_key: str = ""
    # 解析：judge 密钥
    judge_model: str = ""
    # 解析：judge 模型名

    # RAG
    rag_enabled: bool = True
    # 解析：RAG 全局开关
    milvus_host: str = "127.0.0.1"
    # 解析：Milvus 地址
    milvus_port: int = 19530
    # 解析：Milvus 端口
    embedding_model_path: str = "models/bge-m3"
    # 解析：向量化模型路径（可换成微调后的模型）
    rerank_model_path: str = "models/bge-reranker-v2-m3"
    # 解析：重排模型路径
    rag_top_k: int = 4
    # 解析：重排后注入提示词的条数
    rag_recall_k: int = 30
    # 解析：混合检索召回条数
    query_rewrite: bool = False  # 查询改写扩写（每轮多 1 次 LLM 调用）
    # 解析：查询改写开关（默认关省 API 调用）
    longterm_memory: bool = True  # 长期记忆：对话沉淀到 Milvus 并检索注入
    # 解析：长期记忆开关
    memory_top_k: int = 3  # 每次注入的相关记忆条数
    # 解析：记忆注入条数
    memory_similarity: float = 0.5  # 记忆相似度阈值（低于则过滤）
    # 解析：记忆相似度过滤阈值
    chunk_size: int = 700
    # 解析：分块大小
    chunk_overlap: int = 80
    # 解析：块间重叠
    chunking_mode: str = "sentence"  # sentence / heading / semantic / parent_child
    # 解析：分块模式四选一
    semantic_threshold: float = 0.5  # 语义分块相邻句相似度切分阈值
    # 解析：语义分块阈值
    parent_chunk_size: int = 2400  # 父子块模式：父块大小（子块用 chunk_size）
    # 解析：父子块模式的父块大小
    pdf_remove_watermark: bool = True  # 上传 PDF 自动去水印
    # 解析：PDF 去水印开关


settings = Settings()
# 解析：模块级单例——全项目 `from app.config import settings` 使用
