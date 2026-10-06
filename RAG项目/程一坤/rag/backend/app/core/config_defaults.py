"""配置默认值常量表（从 `app.core.config` 拆出，批次 25-2）。

职责边界：**只放常量**，不放任何读取/校验逻辑。
读取与校验（`.env` 解析、生产环境必填项检查）在 `app/core/config_loader.py`，
`Settings` 类与 `settings` 实例在 `app/core/config.py`。

为什么把默认值集中到这里：
1. `app/core/config.py` 原本 381 行、超过 300 行上限；其中很大一部分是
   "字段 = 字面量"与解释这些取值的注释 —— 它们不依赖任何运行时状态，
   天然适合独立成表。
2. 默认值集中后，"这个阈值默认多少"只需看一处，不必在读代码时来回跳。

命名约定：`DEFAULT_<字段名大写>`，与 `Settings` 的字段一一对应。
带 ★ 的常量在注释里写清了取值的实测依据（为什么是这个数字）；
改这些值时请连同依据一起复核。
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------

# 本文件位于 backend/app/core/ 下，向上三层即项目根目录。
# ★ 本常量与 app/core/config.py 同目录，故 parents[3] 的语义不变；
#   若本文件被移动到别的层级，这里必须同步调整。
PROJECT_ROOT_DIRECTORY = Path(__file__).resolve().parents[3]
ENVIRONMENT_FILE_PATH = PROJECT_ROOT_DIRECTORY / ".env"

# --------------------------------------------------------------------------
# 应用基础
# --------------------------------------------------------------------------

DEFAULT_APP_NAME = "legal-rag"
DEFAULT_ENVIRONMENT = "development"
DEFAULT_LOG_LEVEL = "INFO"
# Redis 连接串：会话令牌与短期记忆共用
DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"
DEFAULT_SESSION_TTL_SECONDS = 7200

# --------------------------------------------------------------------------
# 邮件发送（验证码）
# --------------------------------------------------------------------------
# 所有环境都使用 SMTPMailer 真发信，凭据全部来自 SMTP_* 环境变量；
# 测试需要内存邮件器时通过依赖注入提供
DEFAULT_SMTP_PORT = 587
DEFAULT_SMTP_USE_SSL = True

# --------------------------------------------------------------------------
# 向量化（BGE-M3，走 API）
# --------------------------------------------------------------------------
# ★ 检索侧与索引侧必须使用同一模型，维度必须与 Milvus 集合一致
DEFAULT_EMBEDDING_API_BASE_URL = ""
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_DIMENSION = 1024
DEFAULT_EMBEDDING_TIMEOUT_SECONDS = 60.0

# --------------------------------------------------------------------------
# 重排（BGE-reranker-v2-m3，走 API）
# --------------------------------------------------------------------------

DEFAULT_RERANKER_API_BASE_URL = ""
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_TIMEOUT_SECONDS = 60.0

# ★ 拒答阈值：候选最高向量相似度低于该值时判为"查不到"，直接拒答、不调用大模型。
# 取值由阶段 8.4 用评测集校准得出（见 reports/refusal_calibration_*.md）；
# 0 表示不启用阈值（仅"零候选"才拒答）。
DEFAULT_REFUSAL_MIN_VECTOR_SCORE = 0.0

# --------------------------------------------------------------------------
# 召回窗口（批次 10，可调参）
# --------------------------------------------------------------------------
# 向量 / 关键词两路各召回的条数，以及 RRF 融合后送重排的候选上限。
# ★ 批次 10 前为 20/20/20，实测融合窗口偏小（解释（一）第三十二条 ANN 排 #44
#   进不了重排候选，直接丢分），放大到 40/40/40。
DEFAULT_RECALL_VECTOR_LIMIT = 40
DEFAULT_RECALL_KEYWORD_LIMIT = 40
DEFAULT_RERANK_CANDIDATE_LIMIT = 40

# --------------------------------------------------------------------------
# 同义术语扩写（批次 13）
# --------------------------------------------------------------------------
# 把用户口语表述映射成法条用语，**只追加进关键词（BM25）路查询**，
# 向量路与重排输入不动。默认关：术语表落盘 data/（带版本号），
# 开启后检索统计里会多出 synonym_expansion_count，便于评测对比与一键回退。
DEFAULT_SYNONYM_EXPANSION_ENABLED = False
DEFAULT_SYNONYM_TABLE_PATH = "data/legal_synonyms_v1.json"

# ★ 扩展词在 BM25 的得分权重（批次 13-收尾）：1.0=与原词同权，0.5=半权。
# 术语表收窄（v1.1 去掉 df 过高的泛词）后实测 1.0 最优；
# 若未来往表里加泛化用语，可用 <1.0 的权重抑制挤占
# （实测 0.7 会让 asof-031 收益消失）
DEFAULT_SYNONYM_EXPANSION_WEIGHT = 1.0

# --------------------------------------------------------------------------
# 大模型（OpenAI 兼容接口）
# --------------------------------------------------------------------------
# 换供应商只需改这几项，代码不用动
DEFAULT_LLM_API_BASE_URL = ""
DEFAULT_LLM_MODEL = ""
DEFAULT_LLM_TEMPERATURE = 0.2
DEFAULT_LLM_MAX_TOKENS = 2048
DEFAULT_LLM_TIMEOUT_SECONDS = 120.0

# --------------------------------------------------------------------------
# Milvus（向量检索）
# --------------------------------------------------------------------------

DEFAULT_MILVUS_HOST = "127.0.0.1"
DEFAULT_MILVUS_PORT = 19530
DEFAULT_MILVUS_COLLECTION_NAME = "legal_documents"

# ★ 长期记忆独立集合（批次 14）：**严禁**与知识库集合混用，
# 从存储层杜绝过滤条件写漏导致跨用户泄漏（需求 3.6 / 架构文档 9.3）
DEFAULT_MILVUS_LONG_TERM_COLLECTION_NAME = "legal_long_term_memory"

# ★ 记忆去重阈值：新摘要与该用户已有记忆的余弦相似度 ≥ 阈值 → 更新原记录不新增。
# 实测（批次 14，bge-m3）：同事实不同措辞 0.74~0.89，不同事实 0.54~0.60，
# 取 0.70 兼顾两者；样本少（3+3 对），后续扩样复核，若误合率升可上调
DEFAULT_LONG_TERM_MEMORY_DEDUP_THRESHOLD = 0.70
# 注入提示词的记忆条数上限（需求 3.6：按相关度取前 5）
DEFAULT_LONG_TERM_MEMORY_TOP_K = 5

# --------------------------------------------------------------------------
# 会话摘要 / 前情（批次 21）
# --------------------------------------------------------------------------
# 短期记忆窗口满（10 轮）后，被 ltrim 丢弃的更早轮次会被压成一段 ≤300 字的前情，
# 注入提示词让模型知道"本会话聊过什么"。默认 False：开启会改变回答的输入
# （需重跑评测集确认无退化），关闭时行为与引入前逐字一致（不读、不写、不注入），
# 便于一键回退与对拍。触发阈值与上限见 app/memory/summary_policy.py。
DEFAULT_SESSION_SUMMARY_ENABLED = False

# --------------------------------------------------------------------------
# PDF 结构化解析（MinerU，批次 22）
# --------------------------------------------------------------------------
# PDF 通道属于**可选能力**：这两个 key 不进 REQUIRED_IN_PRODUCTION，
# 未配置时只有 PDF 解析不可用，问答链路不受影响。
# MinerU 是主路径：本地 PDF → 提交任务 → 轮询 → 取回 Markdown 正文
DEFAULT_MINERU_API_BASE_URL = "https://mineru.net"
# 轮询间隔与总超时（MinerU 是异步任务，需要轮询到 done/failed）
DEFAULT_MINERU_POLL_INTERVAL_SECONDS = 3.0
DEFAULT_MINERU_MAX_POLL_SECONDS = 300.0
# 解析语言与开关：中文法规固定 ch；表格/公式按需
DEFAULT_MINERU_LANGUAGE = "ch"
DEFAULT_MINERU_ENABLE_TABLE = True
DEFAULT_MINERU_ENABLE_FORMULA = True
# 是否强制 OCR：有文字层的 PDF 设 False（走 MinerU 原生解析，保留版面结构）
DEFAULT_MINERU_IS_OCR = False

# --------------------------------------------------------------------------
# PDF 兜底文字识别（Qwen-VL，批次 22）
# --------------------------------------------------------------------------
# MinerU 返回不完整时，用 qwen-vl-ocr 对渲染页图做文字识别
DEFAULT_QWEN_VL_API_BASE_URL = ""
DEFAULT_QWEN_VL_MODEL = "qwen-vl-ocr"
DEFAULT_QWEN_VL_TIMEOUT_SECONDS = 120.0
# 单次请求最多带几页图（页数上限；真正生效的分批约束是下面的体积上限）
DEFAULT_QWEN_VL_MAX_PAGES_PER_REQUEST = 8

# ★ 单次请求体上限（base64 后字符数）。正确性护栏，不只是省流量：
# 实测 qwen-vl-ocr 请求体超约 2MB 时**只返回第一张图的内容**
# （4 图/1.66MB → 1929 字完整；6 图/2.45MB → 400 字；8 图/3.28MB → 399 字）。
# 默认 1.5MB 留出余量，150 DPI 下约 3~4 页一批。
DEFAULT_QWEN_VL_MAX_REQUEST_BYTES = 1_500_000

# --------------------------------------------------------------------------
# MySQL（取检索命中的条文正文）
# --------------------------------------------------------------------------

DEFAULT_MYSQL_HOST = "127.0.0.1"
DEFAULT_MYSQL_PORT = 3306
DEFAULT_MYSQL_DATABASE = "legal_rag"

# --------------------------------------------------------------------------
# 生产环境必填项
# --------------------------------------------------------------------------
# 生产环境必须显式配置的变量名；缺失即启动失败，避免带着空配置上线。
# SMTP 四键在内：生产环境验证码依赖真发信（SMTPMailer），
# 缺配置必须启动期就暴露，不能等到首个注册请求才 500
PRODUCTION_REQUIRED_KEYS = (
    "EMBEDDING_API_KEY",
    "RERANKER_API_KEY",
    "LLM_API_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "SMTP_HOST",
    "SMTP_USERNAME",
    "SMTP_PASSWORD",
    "SMTP_FROM_EMAIL",
)
