from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def env_int(name: str, default: int) -> int:
    try:
        return int(env(name, str(default)))
    except ValueError:
        return default


def env_bool(name: str, default: bool = False) -> bool:
    return env(name, str(default)).lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    try:
        return float(env(name, str(default)))
    except ValueError:
        return default


def env_path(name: str, default: str) -> Path:
    return Path(env(name, default))


def runtime_mysql_url() -> str:
    compose_password = env("MYSQL_APP_PASSWORD")
    if compose_password:
        return (
            "mysql+pymysql://law_rag:"
            f"{quote_plus(compose_password)}@127.0.0.1:3306/law_rag"
        )
    return env("MYSQL_URL", env("MYSQL_DOCKER_URL", ""))


@dataclass(frozen=True)
class Settings:
    host: str = env("SERVER_HOST", "0.0.0.0")
    port: int = env_int("SERVER_PORT", 7294)
    debug: bool = env_bool("APP_DEBUG", env_bool("DEBUG_VERBOSE", False))
    expose_docs: bool = env_bool("EXPOSE_DOCS", True)
    cors_allowed_origins: str = env("CORS_ALLOWED_ORIGINS", env("CORS_ORIGINS"))
    cors_allowed_origin_regex: str = env(
        "CORS_ALLOWED_ORIGIN_REGEX",
        env("CORS_ORIGIN_REGEX", r"^https?://(localhost|127\.0\.0\.1|host\.docker\.internal)(:\d+)?$"),
    )
    cors_allow_credentials: bool = env_bool("CORS_ALLOW_CREDENTIALS", True)
    mysql_url: str = env("MYSQL_URL", env("MYSQL_DOCKER_URL", ""))
    redis_url: str = env("REDIS_URL", "redis://127.0.0.1:6379/0")
    milvus_uri: str = env("MILVUS_URI", "http://127.0.0.1:19530")
    milvus_token: str = env("MILVUS_TOKEN", "root:Milvus")
    milvus_database: str = env("MILVUS_DATABASE", "default")
    private_milvus_database: str = env("MILVUS_UPLOAD_DATABASE", "legal_private")
    # milvus_timeout: Milvus 普通 insert/search/query/delete 的最长等待秒数。
    milvus_timeout: int = env_int("MILVUS_TIMEOUT", 30)
    # milvus_load_timeout: collection load 的最长等待秒数，设短一点可避免首次检索卡太久。
    milvus_load_timeout: int = env_int("MILVUS_LOAD_TIMEOUT", 2)
    # milvus_load_failure_cooldown: load 失败后多少秒内不重复尝试，避免每次请求都等超时。
    milvus_load_failure_cooldown: int = env_int("MILVUS_LOAD_FAILURE_COOLDOWN", 120)
    # milvus_flush_timeout: 上传向量后 flush 的最长等待秒数，超时只记 warning，不阻断上传成功。
    milvus_flush_timeout: int = env_int("MILVUS_FLUSH_TIMEOUT", 5)
    upload_dir: str = env("UPLOAD_DIR", "data/uploads")
    processed_dir: str = env("PROCESSED_DIR")
    max_upload_bytes: int = env_int("MAX_UPLOAD_BYTES", env_int("UPLOAD_MAX_BYTES", 20 * 1024 * 1024))
    mineru_enabled: bool = env_bool("MINERU_ENABLED", False)
    mineru_api_token: str = env("MINERU_API_TOKEN")
    mineru_api_base_url: str = env("MINERU_API_BASE_URL", "https://mineru.net/api/v4")
    mineru_model_version: str = env("MINERU_MODEL_VERSION", "vlm")
    mineru_timeout: int = env_int("MINERU_TIMEOUT_SECONDS", 600)
    mineru_poll_interval: int = env_int("MINERU_POLL_INTERVAL_SECONDS", 3)
    # upload_concurrency: 批量上传时同时处理的文件数；越大越快，但会增加 OCR/Embedding/Milvus 压力。
    upload_concurrency: int = env_int("UPLOAD_CONCURRENCY", 2)
    # workspace_async_media_processing: 是否让音频/视频先返回 processing，再由后台继续抽取和索引。
    workspace_async_media_processing: bool = env_bool("WORKSPACE_ASYNC_MEDIA_PROCESSING", True)
    # workspace_async_media_types: 允许后台处理的媒体类型列表，默认只把最慢的 audio、video 延后处理。
    workspace_async_media_types: str = env("WORKSPACE_ASYNC_MEDIA_TYPES", "audio,video")
    # workspace_background_workers: 文件解析、向量化和入库的并发 worker 数。
    workspace_background_workers: int = env_int("WORKSPACE_BACKGROUND_WORKERS", 4)
    workspace_pending_limit: int = env_int("WORKSPACE_PENDING_LIMIT", 1000)
    workspace_use_celery: bool = env_bool("START_CELERY_WORKER", False)
    workspace_compression_enabled: bool = env_bool("WORKSPACE_COMPRESSION_ENABLED", True)
    workspace_compression_threshold: int = env_int("WORKSPACE_COMPRESSION_THRESHOLD", 100)
    enable_ocr: bool = env_bool("ENABLE_OCR", True)
    ocr_engine: str = env("OCR_ENGINE", "auto")
    ocr_language: str = env("OCR_LANGUAGE", "chi_sim+eng")
    ocr_min_text_chars: int = env_int("OCR_MIN_TEXT_CHARS", 80)
    tesseract_cmd: str = env("TESSERACT_CMD")
    tessdata_prefix: str = env("TESSDATA_PREFIX")
    poppler_path: str = env("POPPLER_PATH")
    embedding_model: str = env("EMBEDDING_MODEL", env("SILICONFLOW_EMBEDDING_MODEL", "Pro/BAAI/bge-m3"))
    siliconflow_embedding_base_url: str = env("SILICONFLOW_EMBEDDING_BASE_URL", env("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"))
    embedding_dim: int = env_int("EMBEDDING_DIM", 1024)
    # embedding_timeout: 单次 embedding HTTP 请求最长等待秒数，上传向量化慢时首先看这个值。
    embedding_timeout: int = env_int("SILICONFLOW_EMBEDDING_TIMEOUT", env_int("EMBEDDING_TIMEOUT", 60))
    embedding_connect_timeout: int = env_int("EMBEDDING_CONNECT_TIMEOUT_SECONDS", 5)
    # 网络故障时只做一次短重试，避免旧配置产生 3+6+12+24+48 秒的等待。
    embedding_retry_count: int = env_int("EMBEDDING_RETRY_COUNT", 1)
    embedding_retry_delay: float = env_float("EMBEDDING_RETRY_DELAY_SECONDS", 1.0)
    # 一次请求确认服务不可用后，短时间内直接失败，避免每个用户都重复等待网络超时。
    embedding_failure_cooldown: int = env_int("EMBEDDING_FAILURE_COOLDOWN_SECONDS", 30)
    # embedding_cache_size: 内存 embedding 缓存最多保存多少条文本向量；0 表示关闭缓存。
    embedding_cache_size: int = env_int("EMBEDDING_CACHE_SIZE", 2048)
    # embedding_cache_ttl: embedding 缓存存活秒数；默认 86400 秒，减少重复上传/重复 chunk 的等待。
    embedding_cache_ttl: int = env_int("EMBEDDING_CACHE_TTL_SECONDS", 86400)
    # embedding_batch_size: 单次 embedding 请求最多提交多少个 chunk，用于减少 HTTP 请求次数。
    embedding_batch_size: int = env_int("EMBEDDING_BATCH_SIZE", 32)
    # embedding_batch_max_chars: 单次 embedding 请求最多提交多少字符，用于避免长文本批次超时。
    embedding_batch_max_chars: int = env_int("EMBEDDING_BATCH_MAX_CHARS", 60000)
    reranker_model: str = env("RERANKER_MODEL", env("SILICONFLOW_RERANKER_MODEL", "Pro/BAAI/bge-reranker-v2-m3"))
    siliconflow_reranker_base_url: str = env("SILICONFLOW_RERANKER_BASE_URL", env("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"))
    reranker_top_n: int = env_int("RERANKER_TOP_N", 6)
    reranker_timeout: int = env_int("SILICONFLOW_RERANKER_TIMEOUT", env_int("RERANKER_TIMEOUT", 60))
    reranker_doc_max_chars: int = env_int("RERANKER_DOC_MAX_CHARS", 800)
    retrieval_vector_top_k: int = env_int("PUBLIC_VECTOR_TOP_K", env_int("VECTOR_TOP_K", 3))
    retrieval_priority_top_k: int = env_int("PUBLIC_PRIORITY_TOP_K", 4)
    retrieval_non_priority_top_k: int = env_int("PUBLIC_NON_PRIORITY_TOP_K", 1)
    retrieval_keyword_top_k: int = env_int("PUBLIC_KEYWORD_TOP_K", env_int("KEYWORD_TOP_K", 3))
    retrieval_exact_top_k: int = env_int("PUBLIC_EXACT_TOP_K", env_int("EXACT_TOP_K", 4))
    retrieval_private_top_k: int = env_int("PRIVATE_TOP_K", 4)
    retrieval_web_top_k: int = env_int("WEB_TOP_K", env_int("TAVILY_MAX_RESULTS", 3))
    retrieval_candidate_pool_max: int = env_int("CANDIDATE_POOL_MAX", 30)
    retrieval_rerank_input_max: int = env_int("RERANK_INPUT_MAX", 10)
    retrieval_evidence_limit: int = env_int("EVIDENCE_LIMIT", 8)
    retrieval_rrf_k: int = env_int("RRF_K", 60)
    retrieval_max_query_variants: int = env_int("PUBLIC_QUERY_VARIANTS", 4)
    retrieval_non_priority_query_variants: int = env_int("PUBLIC_NON_PRIORITY_QUERY_VARIANTS", 1)
    retrieval_max_collections: int = env_int("PUBLIC_MAX_COLLECTIONS", 6)
    evidence_max_chars: int = env_int("EVIDENCE_MAX_CHARS", 1200)
    context_max_tokens: int = env_int("CONTEXT_MAX_TOKENS", 24000)
    context_evidence_token_budget: int = env_int("CONTEXT_EVIDENCE_TOKEN_BUDGET", 12000)
    context_evidence_max_chars: int = env_int("CONTEXT_EVIDENCE_MAX_CHARS", 800)
    answer_context_max_chars: int = env_int("ANSWER_CONTEXT_MAX_CHARS", 600)
    intent_max_tokens: int = env_int("INTENT_MAX_TOKENS", 900)
    answer_max_tokens: int = env_int("ANSWER_MAX_TOKENS", 1600)
    no_evidence_answer_max_tokens: int = env_int("NO_EVIDENCE_ANSWER_MAX_TOKENS", env_int("ANSWER_MAX_TOKENS", 1600))
    solution_max_tokens: int = env_int("SOLUTION_MAX_TOKENS", 1200)
    siliconflow_api_key: str = env("SILICONFLOW_API_KEY")
    siliconflow_base_url: str = env("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    multimodal_model: str = env("MULTIMODAL_MODEL", env("SILICONFLOW_MULTIMODAL_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct"))
    multimodal_timeout: int = env_int("MULTIMODAL_TIMEOUT", env_int("SILICONFLOW_MULTIMODAL_TIMEOUT", 180))
    multimodal_max_tokens: int = env_int("MULTIMODAL_MAX_TOKENS", 1600)
    deepseek_api_key: str = env("DEEPSEEK_API_KEY", env("LLM_API_KEY"))
    deepseek_base_url: str = env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = env("DEEPSEEK_MODEL", env("LLM_MODEL", "deepseek-chat"))
    deepseek_temperature: float = env_float("DEEPSEEK_TEMPERATURE", env_float("LLM_TEMPERATURE", 0.2))
    intent_llm_temperature: float = env_float("INTENT_LLM_TEMPERATURE", 0.0)
    answer_llm_temperature: float = env_float("ANSWER_LLM_TEMPERATURE", 0.25)
    no_evidence_llm_temperature: float = env_float("NO_EVIDENCE_LLM_TEMPERATURE", 0.2)
    solution_llm_temperature: float = env_float("SOLUTION_LLM_TEMPERATURE", 0.35)
    deepseek_thinking: bool = env_bool("DEEPSEEK_THINKING", False)
    deepseek_timeout: int = env_int("DEEPSEEK_LLM_TIMEOUT", env_int("LLM_TIMEOUT", 120))
    tavily_api_key: str = env("TAVILY_API_KEY")
    tavily_base_url: str = env("TAVILY_BASE_URL", "https://api.tavily.com")
    tavily_timeout: int = env_int("TAVILY_TIMEOUT", 30)
    tavily_search_depth: str = env("TAVILY_SEARCH_DEPTH", "advanced")
    tavily_max_results: int = env_int("TAVILY_MAX_RESULTS", 5)
    tavily_allowed_domains: str = env("TAVILY_ALLOWED_DOMAINS")
    tavily_blocked_domains: str = env("TAVILY_BLOCKED_DOMAINS")
    tavily_trusted_domains: str = env("TAVILY_TRUSTED_DOMAINS")
    auth_cookie_name: str = env("AUTH_COOKIE_NAME", "lawrag_auth")
    auth_token_ttl: int = env_int("AUTH_TOKEN_TTL_SECONDS", 2592000)
    auth_cookie_secure: bool = env_bool("AUTH_COOKIE_SECURE", False)
    history_ttl: int = env_int("HISTORY_TTL_SECONDS", 604800)
    # short_memory_compress_token_limit: 短期记忆压缩触发预算，超过这个估算 token 就压缩。
    short_memory_compress_token_limit: int = env_int("SHORT_MEMORY_COMPRESS_TOKEN_LIMIT", 8000)
    # long_memory_persist_token_limit: 会话达到这个估算 token 后，自动提取并落库长期记忆。
    long_memory_persist_token_limit: int = env_int("LONG_MEMORY_PERSIST_TOKEN_LIMIT", 80000)
    log_level: str = env("LOG_LEVEL", "INFO")
    log_format: str = env("LOG_FORMAT", "json")
    log_dir: Path = env_path("LOG_DIR", "data/processed/system_logs")
    log_file: str = env("LOG_FILE", "law-rag.log")
    log_max_bytes: int = env_int("LOG_MAX_BYTES", 10 * 1024 * 1024)
    log_backup_count: int = env_int("LOG_BACKUP_COUNT", 7)
    log_request_header: str = env("LOG_REQUEST_HEADER", "X-Request-ID")
    log_max_record_bytes: int = env_int("LOG_MAX_RECORD_BYTES", 1024 * 1024)
    log_queue_max_size: int = env_int("LOG_QUEUE_MAX_SIZE", 10000)
    log_low_confidence_score: float = env_float("LOG_LOW_CONFIDENCE_SCORE", 0.7)
    log_audit_token: str = env("LOG_AUDIT_TOKEN")
    log_audit_users: str = env("LOG_AUDIT_USERS")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        host=env("SERVER_HOST", "0.0.0.0"),
        port=env_int("SERVER_PORT", 7294),
        debug=env_bool("APP_DEBUG", env_bool("DEBUG_VERBOSE", False)),
        expose_docs=env_bool("EXPOSE_DOCS", True),
        mysql_url=runtime_mysql_url(),
        redis_url=env("REDIS_URL", "redis://127.0.0.1:6379/0"),
        milvus_uri=env("MILVUS_URI", "http://127.0.0.1:19530"),
        milvus_token=env("MILVUS_TOKEN", "root:Milvus"),
        milvus_database=env("MILVUS_DATABASE", "default"),
        private_milvus_database=env("MILVUS_UPLOAD_DATABASE", "legal_private"),
        mineru_enabled=env_bool("MINERU_ENABLED", False),
        mineru_api_token=env("MINERU_API_TOKEN"),
        mineru_api_base_url=env("MINERU_API_BASE_URL", "https://mineru.net/api/v4"),
        mineru_model_version=env("MINERU_MODEL_VERSION", "vlm"),
        mineru_timeout=env_int("MINERU_TIMEOUT_SECONDS", 600),
        mineru_poll_interval=env_int("MINERU_POLL_INTERVAL_SECONDS", 3),
        embedding_dim=env_int("EMBEDDING_DIM", 1024),
    )
