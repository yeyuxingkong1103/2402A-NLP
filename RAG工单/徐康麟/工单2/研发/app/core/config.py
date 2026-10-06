"""全局配置管理（工单2）。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 基础设施（对应 设计/接口设计.md §2.1、§8）

设计要点：
1. 所有路径以**项目根 E:\\gao6gongdan\\工单2** 为唯一基准，避免工作目录漂移。
2. 通过环境变量 ``RAG_<SECTION>__<FIELD>`` 覆盖任意配置项（算力云部署无需改代码）。
3. 所有可调参数（含检索加权上限）集中在本文件，``config.py`` 是唯一的"魔法数字"来源。
4. 日志目录按环境事实 6.5 落到 ``部署/日志/``，评估结果落到 ``优化/评估结果/``。

环境变量示例::

    set RAG_LLM__BACKEND=ollama
    set RAG_LLM__MODEL=qwen2.5:3b
    set RAG_EMBEDDING__OLLAMA_MODEL=bge-m3:latest
    set RAG_RETRIEVAL__TOTAL_BOOST_CAP=1.6
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.errors import (  # noqa: F401  （对外统一导出异常类型，便于 `from app.core.config import RAGError`）
    ConfigError,
    IndexNotReadyError,
    LLMUnavailableError,
    PDFParseError,
    RAGError,
    RetrievalError,
    StorageError,
)

# --------------------------------------------------------------------------
# 目录基准（**唯一路径基准**）
# --------------------------------------------------------------------------
# 本文件位于 <root>/研发/app/core/config.py，因此：
#     parents[0]=core  parents[1]=app  parents[2]=研发  parents[3]=项目根
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
#: 源码根（研发）
SOURCE_ROOT: Path = Path(__file__).resolve().parents[2]
#: 语料与索引产物（研发/data）
DATA_ROOT: Path = SOURCE_ROOT / "data"

ENV_PREFIX = "RAG_"


class PathSettings(BaseModel):
    """文件系统路径配置（全部按交付阶段归档）。"""

    project_root: Path = PROJECT_ROOT
    source_root: Path = SOURCE_ROOT
    data_raw: Path = DATA_ROOT / "raw"
    data_processed: Path = DATA_ROOT / "processed"
    data_index: Path = DATA_ROOT / "index"
    #: 评估脚本输出（属「优化」阶段产物）
    eval_results: Path = PROJECT_ROOT / "优化" / "评估结果"
    #: 日志统一落到「部署/日志」（环境事实 6.5）
    logs: Path = PROJECT_ROOT / "部署" / "日志"
    #: 金标准问答与自造用例（测试阶段产物）
    test_data: Path = PROJECT_ROOT / "测试" / "测试数据"
    sqlite_path: Path = DATA_ROOT / "index" / "rag.sqlite3"
    default_pdf: Path = DATA_ROOT / "raw" / "招股说明书1.pdf"


class PDFSettings(BaseModel):
    """PDF 解析配置（pymupdf；pdfplumber 本机不可用，详见环境事实 2.2）。"""

    # 文本块最小字数：低于该长度的行视为噪声（页眉页脚、孤立数字）
    min_line_chars: int = 2
    # 需要剔除的页眉页脚模式（正则），招股书每页都有公司名与 1-1-xxx 页码
    header_footer_patterns: list[str] = Field(
        default_factory=lambda: [
            r"^\s*武汉兴图新科电子股份有限公司\s*招股意向书\s*$",
            r"^\s*1-1-\d+\s*$",
            r"^\s*\d{1,3}\s*$",
        ]
    )
    # 表格提取开关（用 PyMuPDF page.find_tables()，非 pdfplumber）
    extract_tables: bool = True
    # 单页表格识别超时保护：识别失败的页只记 WARNING 并计数，不中断整篇解析
    continue_on_table_error: bool = True
    # 跨页表格合并：列数一致 + 表头结构兼容 + 列宽近似才合并
    merge_cross_page_tables: bool = True
    # 表格 Markdown 单元格字符上限（防止超长单元格污染检索）
    max_cell_chars: int = 200


class ChunkSettings(BaseModel):
    """分块配置（工单 6.2 要求 chunk_size 400~600、overlap 80）。"""

    chunk_size: int = 500
    chunk_overlap: int = 80
    # 表格块单独处理：表格不切分，整表作为一个 chunk（避免破坏行列结构）
    table_as_single_chunk: bool = True
    min_chunk_chars: int = 30
    # 过短段落向同 section 邻块吸附的阈值（减少碎块）
    absorb_min_chars: int = 80
    # 表格块首行摘要（表头 + 首列 + 数值列）长度上限
    table_summary_chars: int = 160
    # 标题栈最大层级（section 形如 A > B > C > D）
    max_heading_levels: int = 4


class EmbeddingSettings(BaseModel):
    """向量化配置（主：Ollama bge-m3 1024 维；降级：sentence-transformers 512 维）。"""

    model_config = ConfigDict(protected_namespaces=())

    # backend: "ollama" 走本机 Ollama /api/embed；"sentence_transformers" 走本地模型
    backend: Literal["ollama", "sentence_transformers", "auto"] = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "bge-m3:latest"
    #: 期望维度；与索引 meta.json 不一致则拒绝加载（不静默混用）
    dimension: int = 1024
    timeout: float = 120.0
    # 降级路径
    model_name: str = "BAAI/bge-small-zh-v1.5"
    local_model_dir: str = "models/bge-small-zh-v1.5"
    fallback_dimension: int = 512
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True
    # 批量编码时单次请求的文本条数（Ollama 单请求多文本更快）
    ollama_batch_size: int = 16

    def resolve_st_model_path(self) -> str:
        """返回降级模型路径：本地目录优先，否则回退模型名（联网下载）。"""
        local = PROJECT_ROOT / self.local_model_dir
        if local.is_dir() and any(local.glob("*.json")):
            return str(local)
        return self.model_name


class RetrievalSettings(BaseModel):
    """检索配置（融合 / 加权 / 降权 / 置信度）。

    定量设计来源：设计/优化方案设计.md §3.1~§3.3。
    """

    vector_top_k: int = 20
    bm25_top_k: int = 20
    #: 合并去重后进入重排的候选数
    fusion_top_k: int = 20
    #: 精排输入池大小 = 融合 top20 ∪ 数值覆盖 top-N（实测：数值型问题的真实证据
    #  在向量与 BM25 上都不占优，必须靠"多值齐备度"单独召回一次；池子扩容后才谈得上"重排"）
    rerank_pool_size: int = 70
    #: 数值型问题额外召回的块数（按多值覆盖度排序）
    value_augment_top_n: int = 60
    value_augment_enabled: bool = True
    #: 精排后送入生成的片段数（工单要求 top5）
    rerank_top_n: int = 5
    #: 两路各自 min-max 归一后再线性融合（不可用未归一分数直接加权）
    vector_weight: float = 0.55
    bm25_weight: float = 0.45
    #: BM25 参数
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    # 表格块加权（问题含表格提示词时用 table_boost_hint）
    table_boost: float = 1.15
    table_boost_hint: float = 1.25
    # 目标数值覆盖加权上限增量：1 + numeric_boost_max * coverage
    numeric_boost_max: float = 0.30
    # 多值百分比块追加系数（块内 ≥3 个百分比）
    multi_percent_boost: float = 1.05
    multi_percent_min_count: int = 3
    # 数字密度微弱修正上限增量
    density_boost_max: float = 0.10
    density_reference: float = 0.05
    # 领域关键词加权上限增量：1 + keyword_boost_max * min(命中权重和/提问权重和, 1)
    keyword_boost_max: float = 0.35
    #: 单块总加权硬上限（防止单一规则把某块抬到不可动摇）
    total_boost_cap: float = 1.6
    # 释义页降权（三段取最小值，不叠乘）
    boilerplate_penalty: float = 0.35
    boilerplate_company_penalty: float = 0.55
    boilerplate_short_penalty: float = 0.85
    boilerplate_penalty_floor: float = 0.30
    #: "前 N 页"释义页判定界
    front_page_cutoff: int = 30
    #: 前 N 页内公司全称出现次数阈值（>= 视为样板页）
    boilerplate_company_hits: int = 3
    #: 前 N 页内短块阈值（< 该长度视为碎片）
    boilerplate_short_chars: int = 120
    # 碎片降权（沿用基线经验）
    fragment_half_penalty: float = 0.55
    fragment_prefix_penalty: float = 0.75
    # 置信度：用**原始余弦**（不用归一化融合分）
    min_confidence_cosine: float = 0.55
    min_relevance_score: float = 0.35
    #: 问题实义词至少命中 1 个才算可答
    min_overlap_terms: int = 1
    #: 多路查询变体合并时相对置信度的平方衰减基数
    variant_decay: float = 0.6
    # ---- 子块（segment）级向量召回：解决长块语义稀释 ----
    # 实测（本工单自测）：证据句在整块（400~600 字符）内的向量余弦仅 0.50，
    # 而该证据句单独编码为 0.69 —— 长块把答案句"稀释"掉了。因此检索侧额外
    # 建立**子块级**向量索引（分块规模仍保持工单要求的 400~600），命中后映射回父块。
    segment_recall_enabled: bool = True
    segment_max_chars: int = 180
    segment_min_chars: int = 20
    segment_max_per_chunk: int = 12
    segment_top_k: int = 40
    #: 子块路径在"向量侧"的权重（与块级路径取较优，加权融合仍为 vector_weight/bm25_weight）
    segment_path_weight: float = 0.85
    # 领域关键词权重表（工单 6.2/6.3 点名词；高频通用词取小权重，判别性强的词取大权重）
    keyword_boost: dict[str, float] = Field(
        default_factory=lambda: {
            "注册资本": 1.50,
            "法定代表人": 1.50,
            "技术标准": 1.50,
            "科技进步奖": 1.50,
            "补充流动资金": 1.50,
            "募集资金": 1.40,
            "募资": 1.40,
            "上游": 1.30,
            "下游": 1.30,
            "主营业务收入": 1.25,
            "收入": 1.20,
            "占比": 1.20,
            "比重": 1.20,
            "供应商": 1.15,
            "客户": 1.05,
            "军用领域": 1.30,
            "报告期内": 1.10,
        }
    )
    #: 表格类提示词（出现即用 table_boost_hint）
    table_hint_words: list[str] = Field(
        default_factory=lambda: [
            "多少", "分别", "金额", "比重", "占比", "万元", "单位", "表格", "合计", "比例",
        ]
    )


class RerankerSettings(BaseModel):
    """重排配置（本机无重排模型权重，默认走规则重排，诚实标注 mode）。"""

    model_config = ConfigDict(protected_namespaces=())

    enabled: bool = True
    #: auto：有本地模型用 model，否则 rule
    mode: Literal["auto", "model", "rule", "off"] = "auto"
    #: 本地重排模型目录（云端挂载 bge-reranker-base 时填写）
    model_path: str = "BAAI/bge-reranker-base"
    budget_ms: int = 500
    # 规则打分权重（合计 1.0）。
    #
    # 【实测修订，与本工单自测数据一致】设计初稿给"融合分归一"只有 0.10，
    # 但实测重排后证据被释义/同主题块挤出 top-5（531 第 3→掉出、543 第 11→掉出），
    # 说明"语义+BM25 融合名次"本身信息量最大。故上调融合项、新增"多值齐备度"项，
    # 并把"表格先验"限制为**表格型问题**（否则 p22/63/64 的释义表与子公司表会把真证据挤掉），
    # 同时把检索期释义/碎片降权系数乘到重排分上。
    weight_keyword: float = 0.25
    weight_numeric: float = 0.15
    #: 多值齐备度（去重金额/百分比是否齐全）——"分别是多少/比重分别是多少"的关键信号
    weight_value: float = 0.15
    weight_type: float = 0.05
    weight_position: float = 0.10
    weight_fusion: float = 0.30
    #: 是否把检索期的释义/碎片降权系数乘到重排分上（默认开）
    apply_retrieval_penalty: bool = True


class LLMSettings(BaseModel):
    """LLM 生成配置（可插拔后端：ollama > openai 兼容 > extractive）。"""

    #: auto / ollama / openai / extractive；RAG_LLM__BACKEND 可覆盖
    backend: Literal["auto", "ollama", "openai", "extractive"] = "auto"
    #: Ollama 原生接口
    ollama_base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:3b"
    #: OpenAI 兼容（vLLM/SGLang）接口
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.1
    top_p: float = 0.8
    max_tokens: int = 512
    #: 探测超时必须短且不重试，否则直接劣化首字延迟（实测基线 57 s）
    probe_timeout: float = 0.5
    probe_max_retries: int = 0
    #: 生成期连接/读取超时（秒）
    connect_timeout: float = 3.0
    read_timeout: float = 60.0
    #: 送入 LLM 的最大片段数（工单要求 top5）
    max_context_chunks: int = 5
    #: 单块送入 prompt 的字符上限（控制 prefill 耗时）
    max_chunk_chars: int = 900
    #: 启动是否预热嵌入与 LLM（把冷启动移出首问）
    warmup: bool = True
    #: 无 LLM 服务时是否降级为抽取式回答
    allow_extractive_fallback: bool = True


class ConversationSettings(BaseModel):
    """多轮对话配置（工单 6.5）。"""

    max_history_rounds: int = 5
    #: 改写当前问题时携带的历史轮数
    rewrite_history_rounds: int = 3


class LanguageSettings(BaseModel):
    """多语言（中/英）问答配置（工单 6.6）。"""

    default_answer_language: Literal["auto", "zh", "en"] = "auto"
    #: 英文提问的语言桥接方式：glossary（术语表，毫秒级确定） / model / raw
    query_bridge: Literal["glossary", "model", "raw"] = "glossary"


class AppSettings(BaseModel):
    """应用级配置与兜底文案（对应 §9 错误码文案）。"""

    # 关闭 pydantic "model_" 保护命名空间告警（本文件使用 model_name/model_path 字段）
    model_config = ConfigDict(protected_namespaces=())

    app_name: str = "基于 PDF 文档的 RAG 问答系统（工单2 优化版）"
    version: str = "2.0.0"
    #: 工单验收：首字返回时间 < 3 秒
    first_token_budget_seconds: float = 3.0
    #: 统一兜底回复
    unknown_answer: str = "不清楚"
    llm_unavailable_answer: str = "服务暂时不可用，请稍后再试"
    page_filter_empty_answer: str = "指定页码未检索到内容"
    log_level: str = "INFO"
    log_json: bool = True
    #: 中间步骤是否回传前端（工单要求：只展示最终答案）
    expose_intermediate_steps: bool = False


class Settings(BaseModel):
    """配置聚合根。"""

    app: AppSettings = Field(default_factory=AppSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    pdf: PDFSettings = Field(default_factory=PDFSettings)
    chunk: ChunkSettings = Field(default_factory=ChunkSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    conversation: ConversationSettings = Field(default_factory=ConversationSettings)
    language: LanguageSettings = Field(default_factory=LanguageSettings)

    def ensure_directories(self) -> None:
        """创建运行所需的全部目录（幂等）。"""
        for path in (
            self.paths.data_raw,
            self.paths.data_processed,
            self.paths.data_index,
            self.paths.eval_results,
            self.paths.logs,
            self.paths.test_data,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def index_dir(self, embedder_slug: str, dimension: int) -> Path:
        """索引按嵌入模型分目录存放，不同维度不可混用。

        例：``研发/data/index/bge-m3-1024/``。
        """
        return self.paths.data_index / f"{embedder_slug}-{dimension}"


def _coerce(current: object, raw: str) -> tuple[bool, object]:
    """把环境变量字符串转换成与默认值同类型；类型不支持时返回 ``(False, None)``。"""
    try:
        if isinstance(current, bool):
            return True, raw.strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(current, int):
            return True, int(raw)
        if isinstance(current, float):
            return True, float(raw)
        if isinstance(current, str):
            return True, raw
    except ValueError:
        return False, None
    return False, None


def _apply_env_overrides(settings: Settings, environ: dict[str, str]) -> Settings:
    """把 ``RAG_SECTION__FIELD`` 形式的环境变量写入配置。

    仅支持两级（section.field）。未知 section/field 与类型不匹配的取值**跳过但不静默**：
    启动时由 ``setup_logging`` 后的日志留痕（见 ``get_settings``）。
    """
    data = settings.model_dump()
    rejected: list[str] = []
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        remainder = key[len(ENV_PREFIX):]
        if "__" not in remainder:
            continue
        section, _, field = remainder.partition("__")
        section, field = section.lower(), field.lower()
        if section not in data or field not in data[section]:
            rejected.append(f"{key}(未知配置项)")
            continue
        current = data[section][field]
        ok, value = _coerce(current, raw)
        if not ok:
            rejected.append(f"{key}(类型不匹配: {type(current).__name__})")
            continue
        data[section][field] = value
    # 被忽略的环境变量记录在模块级容器（pydantic 模型不允许写未声明属性）
    _REJECTED_ENV.clear()
    _REJECTED_ENV.extend(rejected)
    return Settings(**data)


#: 被忽略的环境变量（供启动日志提示）
_REJECTED_ENV: list[str] = []


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局唯一配置实例（进程级缓存）。"""
    settings = Settings()
    settings = _apply_env_overrides(settings, dict(os.environ))
    try:
        settings.ensure_directories()
    except OSError as exc:  # 目录不可创建属于配置级错误
        raise ConfigError(f"运行目录不可创建: {exc}", detail={"paths": str(settings.paths.project_root)}) from exc
    return settings


def reload_settings() -> Settings:
    """清除缓存并重新加载配置（测试用）。"""
    get_settings.cache_clear()
    return get_settings()


def rejected_env_keys(settings: Settings | None = None) -> tuple[str, ...]:
    """返回被忽略的环境变量键（供启动日志提示，避免"覆盖了却没生效"的困惑）。"""
    return tuple(_REJECTED_ENV)


# 便捷别名（进程启动即加载配置）
settings = get_settings()

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
