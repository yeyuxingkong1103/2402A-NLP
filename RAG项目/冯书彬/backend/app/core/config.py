import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_local_env() -> None:
    # 本地开发可用 .env 保存密钥；已有环境变量优先，避免覆盖部署平台注入的配置。
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_local_env()


@dataclass
class AppSettings:
    # DeepSeek 密钥只用于启动就绪检查，日志和响应中不得输出真实值。
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    # MinerU API Key 仅用于批量解析导入，禁止写入日志或响应。
    MINERU_API_KEY: str = os.getenv("MINERU_API_KEY", "")
    MINERU_BASE_URL: str = os.getenv("MINERU_BASE_URL", "https://mineru.net")
    MINERU_MODEL_VERSION: str = os.getenv("MINERU_MODEL_VERSION", "vlm")
    # 加密主密钥仅用于派生 AES-GCM 密钥，严禁写入日志或响应。
    APP_MASTER_KEY: str = os.getenv("APP_MASTER_KEY", "")
    # HMAC 密钥仅用于稳定摘要，严禁与加密主密钥混用。
    APP_HMAC_KEY: str = os.getenv("APP_HMAC_KEY", "")

    # 模型路径支持部署环境注入；默认路径相对仓库根目录，避免绑定开发机绝对路径。
    BGE_M3_MODEL_PATH: str = os.getenv(
        "BGE_M3_MODEL_PATH",
        str(Path(__file__).resolve().parents[3] / "models" / "bge-m3"),
    )
    BGE_RERANKER_MODEL_PATH: str = os.getenv(
        "BGE_RERANKER_MODEL_PATH",
        str(Path(__file__).resolve().parents[3] / "models" / "bge-reranker-large"),
    )
    # 本地 BGE 模型统一使用 FP16；禁止在代码中静默降级到 CPU 或其他精度。
    MODEL_DTYPE: str = "float16"

    # 消息和并发限制集中在配置层，避免业务代码散落魔法数字。
    MESSAGE_MAX_CHARS: int = 5000
    ACCOUNT_MESSAGES_PER_MINUTE: int = 5
    ACCOUNT_MESSAGES_PER_DAY: int = 100
    ACCOUNT_CONCURRENT_REQUESTS: int = 2
    SYSTEM_CONCURRENT_REQUESTS: int = 4
    SYSTEM_QUEUE_TIMEOUT_SECONDS: int = 30
    ANSWER_TIMEOUT_SECONDS: int = 120
    # 请求控制默认内存实现；生产可切换 Redis 实现跨进程限流和并发控制。
    REQUEST_CONTROL_BACKEND: str = os.getenv("REQUEST_CONTROL_BACKEND", "memory")

    # 检索阈值和认证时效仅暴露配置，不在 Task 1 实现相关业务。
    RERANKER_THRESHOLD: float = 0.5
    # 当前运行环境用于禁用生产固定验证码。
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    # 认证存储默认保持内存实现；生产可显式切换到 sql。
    AUTH_STORE_BACKEND: str = os.getenv("AUTH_STORE_BACKEND", "memory")
    # 数据库连接串仅在 AUTH_STORE_BACKEND=sql 时使用，严禁写入日志。
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    # OTP 后端默认内存，生产可显式切换 Redis 实现跨进程共享。
    OTP_STORE_BACKEND: str = os.getenv("OTP_STORE_BACKEND", "memory")
    REDIS_URL: str = os.getenv("REDIS_URL", "")
    # Milvus 仅保存连接地址和集合名，向量内容由专用向量库管理。
    MILVUS_URI: str = os.getenv("MILVUS_URI", "")
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "legal_material_chunks")
    MILVUS_VECTOR_DIMENSION: int = int(os.getenv("MILVUS_VECTOR_DIMENSION", "1024"))
    # 默认使用 LangChain 兼容编排层；legacy 保留为显式回退，shadow 用于结果对比。
    RAG_FRAMEWORK: str = os.getenv("RAG_FRAMEWORK", "langchain")
    # 应用日志默认输出 INFO，生产可通过环境变量收窄到 WARNING。
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    # Celery 使用 Redis 作为默认 broker/backend，可由生产环境显式覆盖。
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", os.getenv("REDIS_URL", ""))
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", os.getenv("REDIS_URL", ""))
    CELERY_TASK_ALWAYS_EAGER: bool = os.getenv("CELERY_TASK_ALWAYS_EAGER", "false").lower() == "true"
    # 本地浏览器联调可由 FastAPI 同源托管静态前端；生产网关可关闭。
    SERVE_FRONTEND: bool = os.getenv("SERVE_FRONTEND", "true").lower() == "true"
    # 指标端点默认关闭，避免未授权暴露运行信息；生产由内部网关或鉴权代理开放。
    METRICS_ENABLED: bool = os.getenv("METRICS_ENABLED", "false").lower() == "true"
    FRONTEND_DIR: str = os.getenv("FRONTEND_DIR", str(Path(__file__).resolve().parents[3] / "frontend"))
    # 前后端分端口联调默认允许 3010 访问 8010；多个来源用英文逗号分隔。
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "http://127.0.0.1:3010,http://localhost:3010")
    # 本地联调免登录模式仅允许开发环境使用，前端会携带演示身份头对齐接口。
    DEV_AUTH_BYPASS: bool = os.getenv("DEV_AUTH_BYPASS", "true").lower() == "true"
    DEV_AUTH_USER_ID: str = os.getenv("DEV_AUTH_USER_ID", "guest-user")
    DEV_AUTH_ROLES: str = os.getenv("DEV_AUTH_ROLES", "super_admin,content_reviewer")
    # 固定测试手机号只允许开发、测试或内部环境使用。
    TEST_PHONE_NUMBERS: dict[str, str] = field(default_factory=dict)
    ACCESS_TOKEN_MINUTES: int = 120
    REFRESH_TOKEN_DAYS: int = 30
    OTP_TTL_SECONDS: int = 300
    OTP_MAX_ATTEMPTS: int = 5
    OTP_LOCK_SECONDS: int = 900

    def __post_init__(self) -> None:
        # 生产环境禁止开发免登录，即使调用方误传 True 也不能放行。
        if self.ENVIRONMENT.lower() == "production":
            self.DEV_AUTH_BYPASS = False

    @property
    def model_paths(self) -> dict[str, str]:
        # 只返回非敏感模型路径配置，便于健康检查和后续模型加载复用。
        return {
            "bge_m3": self.BGE_M3_MODEL_PATH,
            "bge_reranker": self.BGE_RERANKER_MODEL_PATH,
        }

    @property
    def limits(self) -> dict[str, int]:
        # 只收集整数限制，保持返回类型严格为 dict[str, int]。
        return {
            "message_max_chars": self.MESSAGE_MAX_CHARS,
            "account_messages_per_minute": self.ACCOUNT_MESSAGES_PER_MINUTE,
            "account_messages_per_day": self.ACCOUNT_MESSAGES_PER_DAY,
            "account_concurrent_requests": self.ACCOUNT_CONCURRENT_REQUESTS,
            "system_concurrent_requests": self.SYSTEM_CONCURRENT_REQUESTS,
            "system_queue_timeout_seconds": self.SYSTEM_QUEUE_TIMEOUT_SECONDS,
            "answer_timeout_seconds": self.ANSWER_TIMEOUT_SECONDS,
            "access_token_minutes": self.ACCESS_TOKEN_MINUTES,
            "refresh_token_days": self.REFRESH_TOKEN_DAYS,
            "otp_ttl_seconds": self.OTP_TTL_SECONDS,
            "otp_max_attempts": self.OTP_MAX_ATTEMPTS,
            "otp_lock_seconds": self.OTP_LOCK_SECONDS,
        }


def validate_production_settings(app_settings: AppSettings) -> None:
    # 生产启动前拒绝开发后端、默认凭据和缺失的关键依赖配置。
    if app_settings.ENVIRONMENT.lower() != "production":
        return

    errors: list[str] = []
    if app_settings.AUTH_STORE_BACKEND != "sql":
        errors.append("AUTH_STORE_BACKEND must be sql")
    if app_settings.OTP_STORE_BACKEND != "redis":
        errors.append("OTP_STORE_BACKEND must be redis")
    if app_settings.REQUEST_CONTROL_BACKEND != "redis":
        errors.append("REQUEST_CONTROL_BACKEND must be redis")
    if not app_settings.DATABASE_URL:
        errors.append("DATABASE_URL is required")
    if not app_settings.REDIS_URL:
        errors.append("REDIS_URL is required")
    if not app_settings.MILVUS_URI:
        errors.append("MILVUS_URI is required")
    if not app_settings.CELERY_BROKER_URL:
        errors.append("CELERY_BROKER_URL is required")
    if not app_settings.CELERY_RESULT_BACKEND:
        errors.append("CELERY_RESULT_BACKEND is required")
    if len(app_settings.APP_MASTER_KEY.encode("utf-8")) < 32:
        errors.append("APP_MASTER_KEY must be at least 32 bytes")
    if len(app_settings.APP_HMAC_KEY.encode("utf-8")) < 32:
        errors.append("APP_HMAC_KEY must be at least 32 bytes")
    if not app_settings.DEEPSEEK_API_KEY:
        errors.append("DEEPSEEK_API_KEY is required")
    if app_settings.SERVE_FRONTEND:
        errors.append("SERVE_FRONTEND must be false")
    if not app_settings.CORS_ORIGINS.strip():
        errors.append("CORS_ORIGINS must be explicitly configured")
    if errors:
        raise RuntimeError("invalid production configuration: " + "; ".join(errors))


def get_settings() -> AppSettings:
    # 当前配置无运行期可变状态，返回轻量实例即可。
    return AppSettings()


settings = get_settings()
