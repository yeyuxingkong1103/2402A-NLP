"""启动期配置加载。

**本模块只在启动期被调用一次**，绝不在请求路径上被调用（FR-022）。
否则运行中改环境变量会导致行为不一致，N4（可复现）失效。

⚠️ 必需项 = **向量检索组**（S9 起转正）。

S7 时这里的必需项是空集，理由写在当时的注释里："本期只有提问通道，不存在
任何会调用外部模型或向量库的代码路径"。S8 加了向量化，S9 加了检索 ——
那个前提已经不成立：现在有真实的外部依赖（Milvus）与真实的模型调用。

转正时点当初就标好了（`REQUIRED_WHEN_RETRIEVAL_LANDS` 的注释写着"转正时机：
检索模块（I-05）接入时"），S9 把它执行了。**调用方一行都没改** ——
`serve.py` 与未来的 `/health` 走的还是 `missing_required()`，这正是当初把
必需项分组的价值。

⚠️ 剩下的转正时点仍标在下方 `REQUIRED_WHEN_*` 常量上：生成模块（I-06）与
紧急判定模块（I-04）接入时，`agicto_api_key` / `llm_base_url` / `llm_model`
随之转正。转正时只需把字段名从后者移到 `REQUIRED_NOW`。
"""

from pathlib import Path

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.chat import (
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_HISTORY_MESSAGES,
    DEFAULT_SESSION_TTL,
)
from backend.retrieve import (
    DEFAULT_ADMIT_RANK,
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    DEFAULT_CANDIDATES,
    DEFAULT_LEXICAL_MIN_COVERAGE,
    DEFAULT_RRF_K,
)

from . import PREAMBLE_ENV

# backend/api/config.py → parents[0]=api, [1]=backend, [2]=仓库根
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

# 会话存储的默认连接串。
#
# ⚠️ 它**不含凭据**（本地 Docker 的默认实例无密码）。若部署时 `REDIS_URL`
#    里带了密码（`redis://:password@host:6379/0`），则本字符串是**唯一**
#    可能内含凭据的配置值 —— 因此 `chat_problems()` 的报错信息 MUST NOT
#    回显它，任何日志同样 MUST NOT（constitution 原则 III）。
DEFAULT_REDIS_URL = "redis://localhost:6379/0"


class ConfigError(Exception):
    """必需配置项缺失。调用方 MUST 以非 0 状态退出，MUST NOT 用默认值兜底。"""


# ---- 必需项分组 ----

# 本期必需：检索组（S9 转正）。
#
# ⚠️ 转正时点比预期来得早/晚不重要 —— 重要的是它**到了**。S7 的注释写着
#    "转正时机：检索模块（I-05）接入时"，而 S9 就是那时。转正后调用方
#    （serve.py、未来的 /health）一行都不用改，这正是当初把它们分组的目的。
REQUIRED_NOW: tuple[str, ...] = (
    "milvus_uri",
    "milvus_collection",
    "embed_model_path",
    "similarity_threshold",
    "top_k",
    # LLM 组（S10 生成接入时转正）。
    #
    # ⚠️ 缺任一项**服务起不来** —— 这是刻意的。生成已经接进提问链路，
    #    没有密钥时服务照常起来、每次提问都静默走拒答，用户看到的是
    #    "知识库没有答案"，而真相是"没配密钥"。那正是 constitution 原则 III
    #    要防的静默降级。
    "agicto_api_key",
    "llm_base_url",
    "llm_model",
)

# 转正时机：仅剩紧急判定模块（I-04）接入时。当前为空。
REQUIRED_WHEN_EMERGENCY_LANDS: tuple[str, ...] = ()

# 转正时机：运行 S2 解析（MinerU）时。由离线管线 CLI 使用，不被后端进程读取。
REQUIRED_WHEN_PIPELINE_LANDS: tuple[str, ...] = (
    "mineru_models_dir",
    "mineru_model_source",
)


class AppConfig(BaseSettings):
    """全部配置项的声明处。

    字段名 == 环境变量名的小写形式（pydantic-settings 大小写不敏感匹配），
    这样变量名与 docs/02 §10 的权威清单可以逐字对齐，不需要额外维护映射。
    """

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        # .env 里会存在大量后续模块的变量，本期不该因它们而报错。
        extra="ignore",
        case_sensitive=False,
    )

    # ---- 本期使用 ----

    medrag_host: str = "127.0.0.1"
    medrag_port: int = 8000

    # ---- 后续模块（本期不读取，仅声明以驱动 missing_required）----

    agicto_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None

    milvus_uri: str | None = None
    milvus_collection: str | None = None
    milvus_token: str | None = None
    embed_model_path: str | None = None
    similarity_threshold: float | None = None
    top_k: int | None = None

    # ---- 检索参数（S9 新增，均有默认值 ⇒ 不进入必需项）----
    #
    # 为什么放在配置里而不是常量里：constitution 的「知识与数据边界」明确要求
    # "检索参数 MUST 在 spec/plan 中显式记录，MUST NOT 依赖库的默认值"。
    # 它们的取值**必须能被追溯与调整**，而调整不该要求改代码。
    #
    # ⚠️ 但"可配置"不等于"该常调"。RRF_K 与 BM25 的 k1/b 是**平滑/饱和常数**，
    #    不是调参旋钮（research R1/R5）—— 调它们之前先怀疑语料与候选集大小。

    retrieval_candidates: int = DEFAULT_CANDIDATES
    rrf_k: int = DEFAULT_RRF_K
    lexical_admit_rank: int = DEFAULT_ADMIT_RANK
    lexical_min_coverage: float = DEFAULT_LEXICAL_MIN_COVERAGE
    bm25_k1: float = DEFAULT_BM25_K1
    bm25_b: float = DEFAULT_BM25_B

    # ---- 多轮对话（S11 新增，均有默认值 ⇒ **不进入必需项**）----
    #
    # ⚠️ `redis_url` **刻意不进 `REQUIRED_NOW`**，这与 `milvus_uri` 的处置
    #    看似矛盾，差别在**依赖的位置**（research.md R13）：
    #
    #      Milvus  → `/ask` 主链路，没有它服务**没有任何可用的功能**
    #      Redis   → 只有 `/chat/*` 这 5 个新接口用它；不可用时其余功能完全正常
    #
    #    让整个服务因为会话功能起不来是过度的。**但该判定附条件**：
    #    `serve.py` MUST 在启动期主动 ping 一次并打印结果 ——
    #    "Redis 连不上"必须在启动时可见，而不是等第一次用户提问才暴露。
    #
    # ⚠️ 三项数值配置的取值理由见 `backend/chat/__init__.py` 的常量注释。
    #    这里只是"可配置"的落点 —— 与检索参数同取向：可配 ≠ 该常调。

    redis_url: str = DEFAULT_REDIS_URL
    chat_session_ttl: int = DEFAULT_SESSION_TTL
    chat_max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    chat_max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES

    mineru_models_dir: str | None = None
    mineru_model_source: str | None = None

    request_timeout_s: float | None = None

    # ---- 故障注入（仅验收用）----

    # 见 specs/006 research.md R8：本期没有紧急判定模块，FR-030（话术必须是首个
    # token）因而没有真实触发条件，靠本项注入一段前置文本来建立断言。
    #
    # ⚠️ 它**必须在启动期经配置读取**，而不是在请求路径上读 os.environ：
    # FR-022 要求配置在启动期一次性加载，请求路径上不得读环境变量。
    # 这条对故障注入同样成立 —— 「只有测试代码才违规」不是理由，
    # 一旦开了口子，后来者无法从代码上分辨哪次读取是合理的。
    test_preamble: str | None = Field(default=None, validation_alias=PREAMBLE_ENV)

    # ---- 校验 ----

    def missing_required(self) -> list[str]:
        """返回缺失的必需项**变量名**列表，空列表表示配置完整。

        MUST NOT 返回或记录变量的**值** —— 报错信息里回显密钥等于把密钥写进
        日志（constitution 原则 III：日志脱敏）。只报变量名。
        """

        return [
            name
            for name in REQUIRED_NOW
            if getattr(self, name, None) in (None, "")
        ]

    def retrieval_problems(self) -> list[str]:
        """检索参数的取值校验。返回问题列表（人话），空列表表示合法。

        ⚠️ 报错信息里 MUST NOT 出现任何配置项的**值**，只出现项名与范围
        （constitution 原则 III：日志脱敏）。`milvus_token` 恰恰是这一组里
        最可能被写错的一项，而它不该因为一次校验失败被打印到终端。
        """

        problems: list[str] = []

        if self.top_k is not None and self.top_k < 1:
            problems.append("top_k 必须 >= 1")

        if self.retrieval_candidates < 1:
            problems.append("retrieval_candidates 必须 >= 1")
        elif self.top_k is not None and self.retrieval_candidates < self.top_k:
            # 候选池比要返回的还少时，融合这一步失去意义 —— 它只是在两路
            # 已经截断的结果里挑，而那正是"混合"要避免的。
            problems.append(
                "retrieval_candidates(%d) 必须 >= top_k(%d)"
                % (self.retrieval_candidates, self.top_k)
            )

        if self.rrf_k < 1:
            problems.append("rrf_k 必须 >= 1")
        if self.lexical_admit_rank < 1:
            problems.append("lexical_admit_rank 必须 >= 1")
        if not 0.0 <= self.lexical_min_coverage <= 1.0:
            # 超过 1 会让关键词准入永远不通过（覆盖率上界是 1），
            # 那等于关键词路被静默关掉 —— 不是"更严格"，是"没在用"。
            problems.append("lexical_min_coverage 必须在 [0, 1] 内")
        if self.bm25_k1 <= 0:
            problems.append("bm25_k1 必须 > 0")
        if not 0.0 <= self.bm25_b <= 1.0:
            problems.append("bm25_b 必须在 [0, 1] 内")

        if self.similarity_threshold is not None and not (
            -1.0 <= self.similarity_threshold <= 1.0
        ):
            # 余弦相似度的值域是 [-1, 1]。越界的阈值不是"更严格/更宽松"，
            # 而是让判定恒真或恒假 —— 那是"看起来在跑、实际没在判"。
            problems.append("similarity_threshold 必须在 [-1, 1] 内（余弦相似度的值域）")

        return problems

    def chat_problems(self) -> list[str]:
        """多轮对话配置的取值校验。返回问题列表（人话），空列表表示合法。

        ⚠️ 与 `retrieval_problems()` 同样，报错信息里 MUST NOT 出现任何配置项的
        **值**（constitution 原则 III）。`redis_url` 恰恰是这一组里唯一可能含
        凭据的一项 —— 它不该因为一次校验失败被打印到终端。
        因此下面只报项名与范围。
        """

        problems: list[str] = []

        if not (self.redis_url or "").strip():
            problems.append("redis_url 不能为空")

        # 三个数值项都必须 > 0：
        #   ttl = 0     → 每个会话一建就过期，对话功能完全不可用
        #   tokens = 0  → 上下文窗口装不下任何东西，模型永远收不到历史
        #   history = 0 → LTRIM 每次把 List 清空，历史永远是空的
        # 三者都不会报错，只会让功能静默地不工作。
        if self.chat_session_ttl <= 0:
            problems.append("chat_session_ttl 必须 > 0")
        if self.chat_max_context_tokens <= 0:
            problems.append("chat_max_context_tokens 必须 > 0")
        if self.chat_max_history_messages <= 0:
            problems.append("chat_max_history_messages 必须 > 0")

        return problems


def load() -> AppConfig:
    """加载配置。必需项缺失或检索参数非法时抛 `ConfigError`。

    MUST NOT 存在"使用默认值继续"的分支 —— 静默降级会让服务以错误配置
    悄悄作答，这比启动失败危险得多（constitution 原则 III）。
    """

    try:
        config = AppConfig()
    except ValidationError as exc:
        # ⚠️ 这一层捕获是补上一个真实的踩坑经历：
        #
        # `.env` 从 `.env.example` 复制后，`TOP_K=<YOUR_TOP_K>` 这样的**占位符
        # 是非空字符串** —— 它通过了"必需项非空"的检查，却在 pydantic 解析
        # `int` / `float` 时炸掉。结果是使用者看到一屏 pydantic 堆栈，
        # 二十行里没有一个字提到"你还有 4 项没填"。
        #
        # 把占位符识别出来单独报，把"该改哪几行"直接说出来。
        raise ConfigError(_explain_validation(exc)) from exc

    missing = config.missing_required()
    if missing:
        raise ConfigError(
            "缺少必需的配置项：" + "、".join(missing) + "（请检查 .env）"
        )

    problems = config.retrieval_problems()
    if problems:
        raise ConfigError(
            "检索参数不合法：" + "；".join(problems) + "（请检查 .env 与 .env.example）"
        )

    problems = config.chat_problems()
    if problems:
        raise ConfigError(
            "多轮对话参数不合法：" + "；".join(problems) + "（请检查 .env 与 .env.example）"
        )

    return config


def _explain_validation(exc: ValidationError) -> str:
    """把 pydantic 的解析错误翻译成"该改 .env 的哪几行"。

    只报**变量名**，MUST NOT 报值（constitution 原则 III：日志脱敏）——
    这也正是本函数不回显 `input_value` 的原因：那里面可能是用户误填进数字字段的
    密钥，或者任何不该出现在终端与日志里的东西。
    """

    fields = [
        str(error["loc"][0]).upper()
        for error in exc.errors()
        if error.get("loc")
    ] or ["<未知字段>"]

    return (
        ".env 有 %d 项无法解析：%s\n"
        "  最常见的原因是直接复制了 .env.example 而没改占位符 ——\n"
        "  `<YOUR_TOP_K>` 这样的字符串在数字字段上会解析失败。\n"
        "  请把这几项填成真实取值（见 .env.example 中各自的说明）。"
        % (len(fields), "、".join(fields))
    )
