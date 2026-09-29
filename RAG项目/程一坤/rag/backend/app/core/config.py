"""应用配置。

配置来源有两处，优先级从高到低：
1. 进程环境变量（测试注入、容器编排用它）
2. 项目根目录的 .env 文件（本地开发用）

先读文件、再读环境变量，是为了让测试注入的值不被 .env 覆盖 ——
否则本地跑测试时，.env 里的生产值会把测试值顶掉。

模块拆分说明（批次 25-2）：
本文件原 381 行、超过 300 行上限，按"常量 / 读取校验 / 类与实例"三分：

- `app/core/config_defaults.py` ← **默认值常量表**（DEFAULT_* 与必填键清单）
- `app/core/config_loader.py`   ← **读取与校验逻辑**（`build_settings_from_environment`）
- 本文件：`load_environment_file()` + `Settings` 类 + `settings` 实例

两条"必须留在本文件"的东西及原因：
1. `load_environment_file()`：`tests/test_production_smtp_validation.py` 用
   `monkeypatch.setattr("app.core.config.load_environment_file", ...)` 屏蔽真实
   `.env`；被 patch 的名字必须与**调用点**在同一模块命名空间，搬走会让该用例
   静默失效（详见 config_loader 模块 docstring）。
2. `Settings` 类：字段默认值虽来自常量表，但"有哪些配置"的清单必须可见，
   且类上的默认值同时是 `from_environment()` 的取值兜底（`cls.<字段>`）。

字段默认值统一写作 `defaults.DEFAULT_*`（模块别名导入）而非逐个 from-import，
是为了让本文件的"配置清单"一眼到底，而不是先翻 50 行导入列表。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from app.core import config_defaults as defaults
from app.core.config_loader import build_settings_from_environment, parse_env_file

# 兼容再导出：app/retrieval/synonym_expansion.py 与 app/retrieval/assembly.py
# 一直从本模块取 PROJECT_ROOT_DIRECTORY，拆分后保持同一导入路径可用
PROJECT_ROOT_DIRECTORY = defaults.PROJECT_ROOT_DIRECTORY
ENVIRONMENT_FILE_PATH = defaults.ENVIRONMENT_FILE_PATH


# 环境名 → 配置文件名（批次 33 裁决：三套环境配置 .env.development/.env.test/.env.production）。
# ENVIRONMENT 取值必须在此表内，拼写错误直接报错，不允许静默落回 development
ENVIRONMENT_ENV_FILE_NAMES = {
    "development": ".env.development",
    "test": ".env.test",
    "production": ".env.production",
}


def load_environment_file(file_path: Path | None = None) -> int:
    """按 ENVIRONMENT 选择 .env 文件并写入环境变量，返回写入条数。

    文件选择与优先级（批次 33/34 裁决，优先级从高到低）：
    1. 进程环境变量（测试注入 / 容器编排 / run.sh export）——任何文件都盖不掉；
    2. 本机私有 `.env`（批次 34）：**后读并覆盖** `.env.<env>` 刚写入的键——
       三份交付配置全占位符，真实凭据只在私有 .env 里，必须能盖回来；
       私有文件不入交付物、不提交；
    3. `.env.<ENVIRONMENT>` 主文件（ENVIRONMENT_FILE_PATH 同目录）：
        * production：`.env.production` 必须存在，缺失直接抛 RuntimeError——
          不许带着默认值静默起生产服务（fail fast 语义批次 33 起保留）；
        * development / test：主文件可缺，缺失时只读私有 .env（等价旧回退）；
          两份都缺返回 0（默认值可用）。
    - ENVIRONMENT 取值不在 `ENVIRONMENT_ENV_FILE_NAMES` 表内 → RuntimeError。
    - 显式传入 file_path 时只读该文件、不读私有覆盖（既有测试与工具脚本用法）。

    解析与覆盖判定统一在 `config_loader.parse_env_file`（KEY=VALUE 最小解析，
    不引入额外依赖）。
    """
    if file_path is not None:
        loaded, _ = parse_env_file(file_path)
        return loaded

    environment = os.environ.get("ENVIRONMENT", defaults.DEFAULT_ENVIRONMENT)
    file_name = ENVIRONMENT_ENV_FILE_NAMES.get(environment)
    if file_name is None:
        raise RuntimeError(
            f"未知 ENVIRONMENT 取值：{environment!r}"
            f"（允许取值：{'、'.join(sorted(ENVIRONMENT_ENV_FILE_NAMES))}）"
        )
    candidate = ENVIRONMENT_FILE_PATH.parent / file_name
    primary_keys: set[str] = set()
    loaded_count = 0
    if candidate.is_file():
        loaded_count, primary_keys = parse_env_file(candidate)
    elif environment == "production":
        raise RuntimeError(
            f"ENVIRONMENT=production 但缺少配置文件：{candidate}"
            "（生产环境禁止静默回退默认值）"
        )
    # 本机私有覆盖文件（批次 34）：覆盖主文件刚写入的键；进程变量不受影响
    private_file = ENVIRONMENT_FILE_PATH
    if private_file.is_file():
        more, _ = parse_env_file(private_file, override_keys=primary_keys)
        loaded_count += more
    return loaded_count


@dataclass(frozen=True)
class Settings:
    """应用配置类，从环境变量加载配置。

    字段默认值统一取自 `app/core/config_defaults.py` 的 `DEFAULT_*` 常量表；
    每个取值"为什么是这个数"的实测依据写在对应常量上方（改值请连依据一起改）。
    """

    # 应用名称
    app_name: str = defaults.DEFAULT_APP_NAME
    # 运行环境（development/production）
    environment: str = defaults.DEFAULT_ENVIRONMENT
    # 日志级别
    log_level: str = defaults.DEFAULT_LOG_LEVEL
    # Redis 连接串：会话令牌与短期记忆共用
    redis_url: str = defaults.DEFAULT_REDIS_URL
    # 会话令牌有效期（秒）
    session_ttl_seconds: int = defaults.DEFAULT_SESSION_TTL_SECONDS
    # 邮件发送（验证码）。所有环境都使用 SMTPMailer 真发信，凭据全部来自
    # SMTP_* 环境变量；测试通过依赖注入使用 InMemoryMailer
    smtp_host: str | None = None
    smtp_port: int = defaults.DEFAULT_SMTP_PORT
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_from_email: str | None = None
    smtp_use_ssl: bool = defaults.DEFAULT_SMTP_USE_SSL

    # ---------------- 向量化（BGE-M3，走 API）----------------
    # 检索侧与索引侧必须使用同一模型，维度必须与 Milvus 集合一致
    embedding_api_base_url: str = defaults.DEFAULT_EMBEDDING_API_BASE_URL
    embedding_api_key: str = ""
    embedding_model: str = defaults.DEFAULT_EMBEDDING_MODEL
    embedding_dimension: int = defaults.DEFAULT_EMBEDDING_DIMENSION
    embedding_timeout_seconds: float = defaults.DEFAULT_EMBEDDING_TIMEOUT_SECONDS

    # ---------------- 重排（BGE-reranker-v2-m3，走 API）----------------
    reranker_api_base_url: str = defaults.DEFAULT_RERANKER_API_BASE_URL
    reranker_api_key: str = ""
    reranker_model: str = defaults.DEFAULT_RERANKER_MODEL
    reranker_timeout_seconds: float = defaults.DEFAULT_RERANKER_TIMEOUT_SECONDS
    # 拒答阈值：0 表示不启用（仅"零候选"才拒答）；校准依据见常量表
    refusal_min_vector_score: float = defaults.DEFAULT_REFUSAL_MIN_VECTOR_SCORE

    # 召回窗口（批次 10，可调参）：向量 / 关键词两路各召回的条数，
    # 以及 RRF 融合后送重排的候选上限（40/40/40 的由来见常量表注释）
    recall_vector_limit: int = defaults.DEFAULT_RECALL_VECTOR_LIMIT
    recall_keyword_limit: int = defaults.DEFAULT_RECALL_KEYWORD_LIMIT
    rerank_candidate_limit: int = defaults.DEFAULT_RERANK_CANDIDATE_LIMIT

    # 同义术语扩写（批次 13）：只追加进关键词（BM25）路查询，默认关，
    # 开启后检索统计里会多出 synonym_expansion_count，便于对比与一键回退
    synonym_expansion_enabled: bool = defaults.DEFAULT_SYNONYM_EXPANSION_ENABLED
    synonym_table_path: str = defaults.DEFAULT_SYNONYM_TABLE_PATH
    # 扩展词在 BM25 的得分权重（1.0=与原词同权；实测 1.0 最优，见常量表）
    synonym_expansion_weight: float = defaults.DEFAULT_SYNONYM_EXPANSION_WEIGHT

    # ---------------- 大模型（OpenAI 兼容接口）----------------
    # 换供应商只需改这三项，代码不用动
    llm_api_base_url: str = defaults.DEFAULT_LLM_API_BASE_URL
    llm_api_key: str = ""
    llm_model: str = defaults.DEFAULT_LLM_MODEL
    llm_temperature: float = defaults.DEFAULT_LLM_TEMPERATURE
    llm_max_tokens: int = defaults.DEFAULT_LLM_MAX_TOKENS
    llm_timeout_seconds: float = defaults.DEFAULT_LLM_TIMEOUT_SECONDS

    # ---------------- Milvus（向量检索）----------------
    milvus_host: str = defaults.DEFAULT_MILVUS_HOST
    milvus_port: int = defaults.DEFAULT_MILVUS_PORT
    milvus_collection_name: str = defaults.DEFAULT_MILVUS_COLLECTION_NAME
    # 长期记忆独立集合（批次 14）：严禁与知识库集合混用，理由见常量表注释
    milvus_long_term_collection_name: str = (
        defaults.DEFAULT_MILVUS_LONG_TERM_COLLECTION_NAME
    )
    # 记忆去重阈值：新摘要与该用户已有记忆的余弦相似度 ≥ 阈值 → 更新原记录不新增
    long_term_memory_dedup_threshold: float = (
        defaults.DEFAULT_LONG_TERM_MEMORY_DEDUP_THRESHOLD
    )
    # 注入提示词的记忆条数上限（需求 3.6：按相关度取前 5）
    long_term_memory_top_k: int = defaults.DEFAULT_LONG_TERM_MEMORY_TOP_K

    # 会话摘要 / 前情（批次 21）：默认关，关闭时行为与引入前逐字一致
    # （不读、不写、不注入），便于一键回退与对拍；阈值见 summary_policy.py
    session_summary_enabled: bool = defaults.DEFAULT_SESSION_SUMMARY_ENABLED

    # ---------------- PDF 结构化解析（MinerU，批次 22）----------------
    # PDF 通道属于**可选能力**：这两个 key 不进 REQUIRED_IN_PRODUCTION，
    # 未配置时只有 PDF 解析不可用，问答链路不受影响。
    # MinerU 是主路径：本地 PDF → 提交任务 → 轮询 → 取回 Markdown 正文
    mineru_api_base_url: str = defaults.DEFAULT_MINERU_API_BASE_URL
    mineru_api_key: str = ""
    # 轮询间隔与总超时（MinerU 是异步任务，需要轮询到 done/failed）
    mineru_poll_interval_seconds: float = defaults.DEFAULT_MINERU_POLL_INTERVAL_SECONDS
    mineru_max_poll_seconds: float = defaults.DEFAULT_MINERU_MAX_POLL_SECONDS
    # 解析语言与开关：中文法规固定 ch；表格/公式按需
    mineru_language: str = defaults.DEFAULT_MINERU_LANGUAGE
    mineru_enable_table: bool = defaults.DEFAULT_MINERU_ENABLE_TABLE
    mineru_enable_formula: bool = defaults.DEFAULT_MINERU_ENABLE_FORMULA
    # 是否强制 OCR：有文字层的 PDF 设 False（走 MinerU 原生解析，保留版面结构）
    mineru_is_ocr: bool = defaults.DEFAULT_MINERU_IS_OCR

    # ---------------- PDF 兜底文字识别（Qwen-VL，批次 22）----------------
    # MinerU 返回不完整时，用 qwen-vl-ocr 对渲染页图做文字识别
    qwen_vl_api_base_url: str = defaults.DEFAULT_QWEN_VL_API_BASE_URL
    qwen_vl_api_key: str = ""
    qwen_vl_model: str = defaults.DEFAULT_QWEN_VL_MODEL
    qwen_vl_timeout_seconds: float = defaults.DEFAULT_QWEN_VL_TIMEOUT_SECONDS
    # 单次请求最多带几页图（页数上限；真正生效的分批约束是下面的体积上限）
    qwen_vl_max_pages_per_request: int = (
        defaults.DEFAULT_QWEN_VL_MAX_PAGES_PER_REQUEST
    )
    # 单次请求体上限（base64 后字符数）——正确性护栏，实测依据见常量表注释
    qwen_vl_max_request_bytes: int = defaults.DEFAULT_QWEN_VL_MAX_REQUEST_BYTES

    # ---------------- MySQL（取检索命中的条文正文）----------------
    mysql_host: str = defaults.DEFAULT_MYSQL_HOST
    mysql_port: int = defaults.DEFAULT_MYSQL_PORT
    mysql_user: str = ""
    mysql_password: str = ""
    mysql_database: str = defaults.DEFAULT_MYSQL_DATABASE

    # 生产环境必须显式配置的变量名；缺失即启动失败，避免带着空配置上线。
    # SMTP 四键在内：生产环境验证码依赖真发信（SMTPMailer），
    # 缺配置必须启动期就暴露，不能等到首个注册请求才 500
    REQUIRED_IN_PRODUCTION: tuple[str, ...] = field(
        default=defaults.PRODUCTION_REQUIRED_KEYS
    )

    @classmethod
    def from_environment(cls) -> "Settings":
        """加载配置并校验；先加载 .env，再读环境变量。

        本方法是薄包装：`.env` 注入与字段读取/校验分别在
        `load_environment_file()`（同模块）与
        `app.core.config_loader.build_settings_from_environment()`。
        `load_environment_file()` 刻意留在本模块调用，以保住测试对
        `app.core.config.load_environment_file` 的 monkeypatch（见模块 docstring）。
        """
        load_environment_file()
        return build_settings_from_environment(cls)


# 加载配置（启动时立即校验）
settings = Settings.from_environment()
