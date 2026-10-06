"""全局配置：所有配置项从 .env 读取，禁止硬编码。

设计理念：
- 敏感信息（DB 口令、JWT 密钥、LLM API Key）只允许来自 .env，代码里一律留空字符串。
  这样源码可以安全入库/分享，泄露风险集中在 .env 一个文件上。
- 用 pydantic-settings 而非手写 os.getenv：可自动做类型转换与校验，
  且 IDE 能对 settings.xxx 做补全和静态检查。
- 模块级单例 settings + lru_cache，保证全进程只解析一次配置。
"""
import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

# .env 位置：优先环境变量 PSYCH_ENV_FILE，其次项目根目录
# parents[2] 从 src/core/config.py 向上三级 → 项目根目录（core → src → root）。
_DEFAULT_ROOT = Path(__file__).resolve().parents[2]
# resolve() 把相对路径转成绝对路径；可被 PSYCH_ENV_FILE 覆盖，便于测试注入不同 .env。
ENV_FILE = Path(os.getenv("PSYCH_ENV_FILE", str(_DEFAULT_ROOT / ".env")))


class Settings(BaseSettings):
    """Pydantic 配置模型：字段名小写，对应 .env 中的大写键名（大小写不敏感）。"""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),      # 指定要加载的 .env 文件
        env_file_encoding="utf-8",   # 强制 utf-8，避免中文注释/值在 Windows 下乱码
        extra="ignore",              # .env 里出现模型未声明的键时忽略而非报错（向前兼容）
        case_sensitive=False,        # JWT_SECRET_KEY 与 jwt_secret_key 视为同一项
    )

    # ===== 项目 =====
    project_root: str = str(_DEFAULT_ROOT)
    app_name: str = "基于RAG的心理医生多角色陪伴系统"
    app_version: str = "1.0.0"
    debug: bool = False  # 默认关闭：生产环境若为 True 会暴露堆栈并放开 uvicorn 日志
    log_dir: str = str(_DEFAULT_ROOT / "logs")
    log_level: str = "INFO"
    data_dir: str = str(_DEFAULT_ROOT / "data")

    # ===== 本地模型 =====
    # 全部指向本地磁盘：避免每次启动联网下载/校验模型，离线环境也能跑。
    embedding_model_path: str = "/home/dabaie/models/bge-m3"
    reranker_model_path: str = "/home/dabaie/models/bge-reranker-v2-m3"
    embedding_dim: int = 1024  # bge-m3 输出维度固定 1024，必须与 Milvus 建表维度一致
    embedding_device: str = "cuda:0"  # 无 GPU 时需改成 "cpu"，否则加载即报错
    embedding_batch_size: int = 16  # 显存不足时调小；过大会 OOM
    hf_hub_offline: int = 1  # 1=强制离线（配合下游环境变量生效）
    reranker_device: str = "cuda:0"
    reranker_batch_size: int = 8  # rerank 单条更长，batch 通常小于 embedding
    rerank_top_n: int = 5  # 粗排召回后只保留前 N 条喂给大模型，控制上下文长度

    # ===== MySQL =====
    db_host: str = "127.0.0.1"
    db_port: int = 3307  # 非默认 3306：宿主机/容器端口映射常见做法，改前先确认实例监听端口
    db_name: str = "rag_roleplay"
    db_user: str = "dev"
    db_password: str = ""  # 必须由 .env 提供，代码不落真实口令
    db_pool_size: int = 10  # 常驻连接池大小
    db_max_overflow: int = 20  # 突发时允许临时超出的连接数（上限 = pool_size + max_overflow）
    db_echo: bool = False  # 打开会打印所有 SQL，仅调试时用，生产会严重拖慢

    # ===== Redis =====
    # 用途：短期对话记忆、检索缓存、token 黑名单、限流计数。
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_db: int = 0  # 逻辑库编号，便于不同用途隔离（默认 0）
    redis_password: str = ""

    # ===== Milvus =====
    # 向量库：存人格知识库与用户长期记忆两套 collection。
    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530  # Milvus 默认端口
    milvus_uri: str = "http://127.0.0.1:19530"  # 新版 SDK 用 URI 形式连接
    milvus_collection: str = "persona_knowledge"  # 角色人格/知识切片
    milvus_memory_collection: str = "user_long_term_memory"  # 用户长期记忆

    # ===== RAG =====
    # 父子分块策略：父块（大）用于喂上下文，子块（小）用于精准召回。
    chunk_size: int = 512  # 通用切片长度（字符/近似 token）
    chunk_overlap: int = 80  # 相邻切片重叠，防止关键句被切断在边界处丢语义
    parent_chunk_size: int = 1024
    child_chunk_size: int = 256
    retrieve_top_k: int = 20  # 向量粗排召回条数（再交给 reranker 精排）
    similarity_threshold: float = 0.35  # 低于该相似度的结果丢弃，避免"硬凑"无关知识
    short_term_max_turns: int = 10  # 短期记忆最多保留的对话轮数，超出摘入长期记忆
    short_term_ttl: int = 86400  # 短期记忆 Redis 过期秒数 = 24 小时
    long_term_summary_trigger: int = 20  # 累计多少轮触发一次长期记忆摘要
    query_rewrite_enabled: bool = True  # 用 LLM 改写/补全用户问题以提升召回；失败可回退

    # ===== 文档解析增强（OCR / MinerU）=====
    # 当 PDF 文本层过薄时（< MIN_TEXT_LAYER_CHARS），自动触发 OCR 回退
    ocr_fallback_enabled: bool = True
    ocr_min_text_layer_chars: int = 100       # 文本层字符数低于此阈值则判定为扫描件
    ocr_timeout: int = 1800                   # OCR 单文件超时（秒），30 分钟足够处理 ~900 页

    # MinerU 配置（文本型 PDF 高质量解析 + 版面分析）
    mineru_enabled: bool = True
    mineru_tier: str = "flash"                # flash=纯文本 / standard=OCR+版面（需下载模型）
    mineru_venv_path: str = "/home/dabaie/code/my_project/.venv-mineru/bin/mineru"
    # 说明：MinerU/Paddle 依赖重且与主环境冲突，故用独立 venv 以子进程方式调用。

    # PaddleOCR 配置（扫描件回退）
    paddle_ocr_venv_path: str = "/home/dabaie/code/my_project/.venv-paddle/bin/python"
    paddle_ocr_lang: str = "ch"  # 中文语料，切英文模型会显著掉点

    # WSL 跨平台支持（Windows 上运行时使用，项目本体在 WSL 中可忽略）
    wsl_distribution: str = "Ubuntu-22.04"

    # ===== LLM =====
    llm_provider: str = "openai_compatible"  # 走 OpenAI 兼容协议，便于换供应商
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = ""
    llm_model: str = "deepseek-v4-flash"  # 可选：deepseek-flash / deepseek-v4-pro（以 /models 返回为准）
    llm_thinking: str = "disabled"  # deepseek-flash 为思考型模型：disabled 才能直接返回正文
    llm_rewrite_timeout: float = 3.0  # Query 改写短超时：失败即回退原问题，不拖慢首 token
    llm_temperature: float = 0.7  # 陪伴对话偏自然，故不用 0；事实问答可下调
    llm_max_tokens: int = 2048  # 单次回答上限，防跑飞与高额计费
    llm_timeout: int = 60  # 主对话超时（秒），比改写宽松得多

    # ===== 鉴权 =====
    jwt_secret_key: str = ""  # 必须由 .env 提供，禁止代码默认值
    jwt_algorithm: str = "HS256"  # 对称签名，单服务够用；多服务可换 RS256
    jwt_access_token_expire_minutes: int = 1440  # 1 天：短期令牌，泄露窗口可控
    jwt_refresh_token_expire_minutes: int = 10080  # 7 天：用于静默续期，减少重复登录
    bcrypt_rounds: int = 12  # 12 轮在安全与登录耗时间较平衡（每 +1 耗时翻倍）

    # ===== 安全 =====
    crisis_hotline: str = "12356"  # 全国心理援助热线，自伤/自杀倾向时展示
    emergency_phone: str = "120,110"
    rate_limit_per_minute: int = 30  # 常规接口每用户每分钟上限
    login_rate_limit_per_minute: int = 10  # 未登录用户按 IP 限流，防暴力破解
    retrieval_cache_ttl: int = 300  # 检索结果 Redis 缓存秒数；0=禁用
    cors_allow_origins: str = "http://localhost:3000,http://127.0.0.1:3000"  # 白名单，禁用 "*"

    # ===== 服务 =====
    api_host: str = "0.0.0.0"  # 监听所有网卡，容器内必须如此才能被外部访问
    api_port: int = 8000
    api_workers: int = 1  # 默认单 worker：本地模型显存占用大，多进程会重复加载

    # ===== 初始化管理员 =====
    admin_username: str = "admin"
    admin_password: str = ""  # 必须由 .env 提供

    # ---------- 派生属性 ----------
    @property
    def database_url(self) -> str:
        """拼出 SQLAlchemy 连接串（含库名）。

        charset=utf8mb4 而非 utf8：MySQL 的 utf8 只支持 3 字节，存 emoji/生僻字会报错，
        心理陪伴场景用户常发表情，必须用 utf8mb4。
        """
        return (
            f"mysql+pymysql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}?charset=utf8mb4"
        )

    @property
    def server_url(self) -> str:
        """不带库名的连接串，用于自动建库。"""
        # 库还不存在时无法带库名连接，先用这个 URL 连上 server 执行 CREATE DATABASE。
        return (
            f"mysql+pymysql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/?charset=utf8mb4"
        )

    @property
    def emergency_phones(self) -> List[str]:
        """把逗号分隔的紧急电话串拆成列表，并过滤空项（容忍多余逗号/空格）。"""
        return [p.strip() for p in self.emergency_phone.split(",") if p.strip()]

    @property
    def cors_origins(self) -> List[str]:
        """CORS 白名单（逗号分隔配置）。"""
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    def validate_security(self) -> None:
        """启动期安全校验：密钥/口令缺失或使用占位符时拒绝启动（S3）。

        为什么"拒绝启动"而不是打警告？安全配置出错往往在线上才被利用，
        快速失败（fail-fast）能强制运维在部署阶段就修正，避免带病上线。
        """
        # 显式列出常见占位符：防止有人直接复制 .env.example 里的假值上线。
        weak_secrets = {"", "change_me", "change_me_in_production"}
        if self.jwt_secret_key in weak_secrets:
            raise RuntimeError("JWT_SECRET_KEY 未配置或使用了占位符，请在 .env 中设置（≥32 字符随机串）")
        # HS256 的安全性取决于密钥熵，太短的密钥可被离线暴力破解，故强制 32 字符下限。
        if len(self.jwt_secret_key) < 32:
            raise RuntimeError("JWT_SECRET_KEY 强度不足（<32 字符），请在 .env 中加强")
        if not self.admin_password:
            raise RuntimeError("ADMIN_PASSWORD 未配置，请在 .env 中设置初始管理员密码")

    def ensure_dirs(self) -> None:
        """创建运行期必需目录；exist_ok=True 保证重复调用安全（幂等）。"""
        # 逐个创建：日志、数据根、上传目录（上传目录是 data 的子目录，故用 join 拼接）。
        for d in (self.log_dir, self.data_dir, os.path.join(self.data_dir, "uploads")):
            # parents=True 允许一次创建多级不存在的父目录；已存在时不报错。
            Path(d).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置单例。

    maxsize=1 的 lru_cache 让本函数成为"进程内唯一实例"工厂：
    配置解析只做一次，各处拿到的 settings 完全一致。
    """
    s = Settings()
    # 本地模型必须离线加载，避免联网校验
    # 这里写进 os.environ 是为了让 transformers/huggingface_hub 在 import 阶段也能读到，
    # 它们只认环境变量，不认我们的 Settings 对象。
    os.environ["HF_HUB_OFFLINE"] = str(s.hf_hub_offline)
    os.environ["TRANSFORMERS_OFFLINE"] = str(s.hf_hub_offline)
    # 提前建目录：后续日志/上传等模块直接写文件，不必各自判空。
    s.ensure_dirs()
    return s


# 模块级实例：import 即完成配置加载（副作用），因此不要在 core 包里再导出成循环依赖。
settings = get_settings()