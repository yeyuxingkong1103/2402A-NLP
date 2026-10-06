"""配置读取与校验逻辑（从 `app.core.config` 拆出，批次 25-2）。

本模块只有一个函数 `build_settings_from_environment()`：把环境变量读进
`Settings`（dataclass）并做生产环境校验。它是原 `Settings.from_environment()`
的方法体，**逐字搬移**，只是换了宿主（模块级函数 + 显式传入 `cls`）。

与 `app/core/config.py` 的分工：
- `config.py`         ：`load_environment_file()`（.env 解析）+ `Settings` 类 +
                        薄包装 `Settings.from_environment()` + `settings` 实例
- `config_loader.py`  ：本模块，读取与校验主体
- `config_defaults.py`：默认值常量表

为什么不把 `load_environment_file()` 也搬进来（看起来更"按职责"）：
`tests/test_production_smtp_validation.py` 用
`monkeypatch.setattr("app.core.config.load_environment_file", lambda *a, **k: 0)`
屏蔽本机真实 `.env`。若本模块自行调用 `load_environment_file`，
patch 只能改 `app.core.config` 这个名字、改不到本模块的引用，
测试就会**静默失效**（真实 `.env` 的 SMTP_* 被读回，"缺配置"场景模拟不出来，
用例反而可能通过）。所以该调用留在 `config.py` 的薄包装里。

本模块刻意不 import `app.core.config`（否则形成循环导入）——
`cls` 由调用方传入，读默认值全靠 `cls.<字段名>`（dataclass 字段默认值
仍是类属性，语义与拆分前一致）。

维护提示：本文件是"环境变量名 → Settings 字段"的唯一映射表。
**新增配置项时三处都要动**：config_defaults 加默认值常量、
config.py 的 Settings 加字段、本文件加一行读取；漏掉最后一步时
该字段会静默保持默认值（不会报错），这也是把映射集中在一个函数里的原因。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def parse_env_file(
    file_path: Path, override_keys: set[str] | None = None
) -> tuple[int, set[str]]:
    """解析单个 env 文件并把键值写入环境变量（批次 34 抽出）。

    参数：
        file_path: env 文件路径
        override_keys: 允许覆盖的键集合；None = 一律不覆盖已存在的键
    返回：
        (实际写入条数, 本次写入的键集合)——键集合供调用方实现
        ".env 私有覆盖文件"语义：后读的文件允许覆盖**前一份文件刚写入**的键，
        但进程里本来就有的环境变量任何文件都盖不掉（进程优先级最高）。
    """
    loaded_count = 0
    written: set[str] = set()
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        # 覆盖判定：键已在进程环境里、且不在"允许覆盖"名单内 → 跳过
        if key in os.environ and (override_keys is None or key not in override_keys):
            continue
        os.environ[key] = value
        written.add(key)
        loaded_count += 1
    return loaded_count, written


def build_settings_from_environment(cls: type) -> Any:
    """加载配置并校验；先加载 .env，再读环境变量。

    参数：
        cls: `Settings` 类本身（dataclass）。用于①读各字段默认值
             （`cls.<字段>`）②构造实例（`cls(...)`）③取生产环境必填键
             （`cls.REQUIRED_IN_PRODUCTION`）。
    返回：
        构造好的 `Settings` 实例。
    异常：
        ValueError: production 环境缺 `REDIS_URL`，或缺
                    `REQUIRED_IN_PRODUCTION` 里的任一项 ——
                    都在启动期抛出，不留到首个请求才 500。
    """
    # 读取顺序：优先环境变量，缺失时回落到 cls.<字段>（即常量表里的默认值）
    environment = os.getenv("ENVIRONMENT", cls.environment)
    redis_url = os.getenv("REDIS_URL", cls.redis_url)

    # 生产环境必须显式配置 Redis 地址，避免误连本机默认实例
    # （本机默认值是 127.0.0.1:6379/0，生产沿用它会连到容器自身的 Redis）
    if environment == "production" and redis_url == cls.redis_url:
        raise ValueError("生产环境必须配置 REDIS_URL 环境变量")

    # 生产环境必须有完整的模型与检索配置；缺任何一项都直接启动失败
    # 校验放在构造之前：宁可启动失败，也不要带着半套配置对外服务
    if environment == "production":
        missing_names = [
            name for name in cls.REQUIRED_IN_PRODUCTION if not os.getenv(name)
        ]
        if missing_names:
            raise ValueError(
                "生产环境缺少必需配置：" + "、".join(missing_names)
            )

    return cls(
        # 基础项：只影响日志与标记；environment=production 才会触发上面的必填校验
        app_name=os.getenv("APP_NAME", cls.app_name),
        environment=environment,
        log_level=os.getenv("LOG_LEVEL", cls.log_level).upper(),
        redis_url=redis_url,
        session_ttl_seconds=int(
            os.getenv("SESSION_TTL_SECONDS", str(cls.session_ttl_seconds))
        ),
        # 邮件：所有环境都使用 SMTPMailer；测试通过依赖注入使用内存 mailer
        smtp_host=os.getenv("SMTP_HOST"),
        smtp_port=int(os.getenv("SMTP_PORT", str(cls.smtp_port))),
        smtp_username=os.getenv("SMTP_USERNAME"),
        smtp_password=os.getenv("SMTP_PASSWORD"),
        smtp_from_email=os.getenv("SMTP_FROM_EMAIL"),
        smtp_use_ssl=os.getenv("SMTP_USE_SSL", "true").strip().lower() != "false",
        # 向量化：换模型或改维度等于换向量空间，必须同时重建索引，
        # 否则"查询向量"与"库内向量"不在同一空间，召回结果无意义
        embedding_api_base_url=os.getenv(
            "EMBEDDING_API_BASE_URL", cls.embedding_api_base_url
        ),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY", cls.embedding_api_key),
        embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
        embedding_dimension=int(
            os.getenv("EMBEDDING_DIMENSION", str(cls.embedding_dimension))
        ),
        embedding_timeout_seconds=float(
            os.getenv("EMBEDDING_TIMEOUT_SECONDS", str(cls.embedding_timeout_seconds))
        ),
        # 重排：只对已召回的候选正文打分，不改变查询本身；调用失败由调用方退回召回序
        reranker_api_base_url=os.getenv(
            "RERANKER_API_BASE_URL", cls.reranker_api_base_url
        ),
        reranker_api_key=os.getenv("RERANKER_API_KEY", cls.reranker_api_key),
        reranker_model=os.getenv("RERANKER_MODEL", cls.reranker_model),
        reranker_timeout_seconds=float(
            os.getenv("RERANKER_TIMEOUT_SECONDS", str(cls.reranker_timeout_seconds))
        ),
        # 拒答阈值（阶段 8.4 校准产物；未配置时用类默认值 0 = 不启用，
        # 即只有"零候选"才拒答，候选分数再低也会交给大模型）
        refusal_min_vector_score=float(
            os.getenv("REFUSAL_MIN_VECTOR_SCORE", str(cls.refusal_min_vector_score))
        ),
        # 召回窗口（批次 10）：三个数依次是"向量召回条数 / 关键词召回条数 /
        # RRF 融合后送重排的上限"，扩容理由见常量表注释
        recall_vector_limit=int(
            os.getenv("RECALL_VECTOR_LIMIT", str(cls.recall_vector_limit))
        ),
        recall_keyword_limit=int(
            os.getenv("RECALL_KEYWORD_LIMIT", str(cls.recall_keyword_limit))
        ),
        rerank_candidate_limit=int(
            os.getenv("RERANK_CANDIDATE_LIMIT", str(cls.rerank_candidate_limit))
        ),
        # 同义术语扩写（批次 13）：只影响 BM25 那一路的查询词，
        # 向量路与重排输入一律不动 —— 所以开关切换不会改变向量召回的候选
        synonym_expansion_enabled=os.getenv(
            "SYNONYM_EXPANSION_ENABLED", str(cls.synonym_expansion_enabled)
        )
        .strip()
        .lower()
        in ("1", "true", "yes", "on"),
        synonym_table_path=os.getenv("SYNONYM_TABLE_PATH", cls.synonym_table_path),
        synonym_expansion_weight=float(
            os.getenv("SYNONYM_EXPANSION_WEIGHT", str(cls.synonym_expansion_weight))
        ),
        # 大模型：OpenAI 兼容协议，换供应商只改 base_url / key / model 三项；
        # 前两项与 model 都在 production 必填清单里
        llm_api_base_url=os.getenv("LLM_API_BASE_URL", cls.llm_api_base_url),
        llm_api_key=os.getenv("LLM_API_KEY", cls.llm_api_key),
        llm_model=os.getenv("LLM_MODEL", cls.llm_model),
        llm_temperature=float(
            os.getenv("LLM_TEMPERATURE", str(cls.llm_temperature))
        ),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", str(cls.llm_max_tokens))),
        llm_timeout_seconds=float(
            os.getenv("LLM_TIMEOUT_SECONDS", str(cls.llm_timeout_seconds))
        ),
        # Milvus：集合名必须与建索引时一致，否则查不到任何东西；
        # 长期记忆用**独立集合**，从存储层杜绝过滤条件写漏导致的跨用户泄漏
        milvus_host=os.getenv("MILVUS_HOST", cls.milvus_host),
        milvus_port=int(os.getenv("MILVUS_PORT", str(cls.milvus_port))),
        milvus_collection_name=os.getenv(
            "MILVUS_COLLECTION_NAME", cls.milvus_collection_name
        ),
        milvus_long_term_collection_name=os.getenv(
            "MILVUS_LONG_TERM_COLLECTION_NAME",
            cls.milvus_long_term_collection_name,
        ),
        # 记忆去重阈值：0.70 的实测依据（同事实 0.74~0.89 / 不同事实 0.54~0.60）
        # 写在 config_defaults 的常量注释里，改动前请连同依据一起复核
        long_term_memory_dedup_threshold=float(
            os.getenv(
                "LONG_TERM_MEMORY_DEDUP_THRESHOLD",
                str(cls.long_term_memory_dedup_threshold),
            )
        ),
        long_term_memory_top_k=int(
            os.getenv("LONG_TERM_MEMORY_TOP_K", str(cls.long_term_memory_top_k))
        ),
        # 会话摘要（批次 21）：默认关；关闭时"不读、不写、不注入"，
        # 行为与引入前逐字一致，因此可随时回退与对拍
        session_summary_enabled=os.getenv(
            "SESSION_SUMMARY_ENABLED", str(cls.session_summary_enabled)
        )
        .strip()
        .lower()
        in ("1", "true", "yes", "on"),
        # MinerU（批次 22）：PDF 结构化解析的主通道；未配 key 只是"PDF 解析不可用"，
        # 问答链路完全不受影响（所以它不在 production 必填清单里）
        mineru_api_base_url=os.getenv("MINERU_API_BASE_URL", cls.mineru_api_base_url),
        mineru_api_key=os.getenv("MINERU_API_KEY", cls.mineru_api_key),
        mineru_poll_interval_seconds=float(
            os.getenv(
                "MINERU_POLL_INTERVAL_SECONDS", str(cls.mineru_poll_interval_seconds)
            )
        ),
        mineru_max_poll_seconds=float(
            os.getenv("MINERU_MAX_POLL_SECONDS", str(cls.mineru_max_poll_seconds))
        ),
        mineru_language=os.getenv("MINERU_LANGUAGE", cls.mineru_language),
        mineru_enable_table=os.getenv(
            "MINERU_ENABLE_TABLE", str(cls.mineru_enable_table)
        )
        .strip()
        .lower()
        in ("1", "true", "yes", "on"),
        mineru_enable_formula=os.getenv(
            "MINERU_ENABLE_FORMULA", str(cls.mineru_enable_formula)
        )
        .strip()
        .lower()
        in ("1", "true", "yes", "on"),
        mineru_is_ocr=os.getenv("MINERU_IS_OCR", str(cls.mineru_is_ocr))
        .strip()
        .lower()
        in ("1", "true", "yes", "on"),
        # Qwen-VL 兜底（批次 22）：MinerU 不完整时对渲染页图做 OCR；
        # max_request_bytes 是**正确性**护栏（超限时服务端只回第一页内容），依据见常量表
        qwen_vl_api_base_url=os.getenv(
            "QWEN_VL_API_BASE_URL", cls.qwen_vl_api_base_url
        ),
        qwen_vl_api_key=os.getenv("QWEN_VL_API_KEY", cls.qwen_vl_api_key),
        qwen_vl_model=os.getenv("QWEN_VL_MODEL", cls.qwen_vl_model),
        qwen_vl_timeout_seconds=float(
            os.getenv("QWEN_VL_TIMEOUT_SECONDS", str(cls.qwen_vl_timeout_seconds))
        ),
        qwen_vl_max_pages_per_request=int(
            os.getenv(
                "QWEN_VL_MAX_PAGES_PER_REQUEST",
                str(cls.qwen_vl_max_pages_per_request),
            )
        ),
        qwen_vl_max_request_bytes=int(
            os.getenv("QWEN_VL_MAX_REQUEST_BYTES", str(cls.qwen_vl_max_request_bytes))
        ),
        # MySQL：只用于"取检索命中的条文正文"（向量库里只有检索文本，没有正文）
        mysql_host=os.getenv("MYSQL_HOST", cls.mysql_host),
        mysql_port=int(os.getenv("MYSQL_PORT", str(cls.mysql_port))),
        mysql_user=os.getenv("MYSQL_USER", cls.mysql_user),
        mysql_password=os.getenv("MYSQL_PASSWORD", cls.mysql_password),
        mysql_database=os.getenv("MYSQL_DATABASE", cls.mysql_database),
    )
