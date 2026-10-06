"""全局配置管理。

设计要点：
1. 所有路径都以项目根目录为基准，避免工作目录不同导致的路径漂移。
2. 通过环境变量覆盖任意配置项（前缀 ``RAG_``），便于算力云部署时无需改代码。
3. 所有可调参数集中在此，``config.py`` 是唯一的“魔法数字”来源。

环境变量示例::

    set RAG_LLM__BASE_URL=http://127.0.0.1:8000/v1
    set RAG_RETRIEVAL__TOP_K=20
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# 项目根目录（**唯一的路径基准**）
# --------------------------------------------------------------------------
# 目录结构按交付阶段分为 设计 / 研发 / 测试 / 优化 / 部署 五个一级目录，
# 本文件位于 <root>/研发/app/core/config.py，因此：
#     parents[0] = core   parents[1] = app
#     parents[2] = 研发   parents[3] = 项目根
#
# 运行期数据（data/、logs/、models/）刻意保留在**项目根**，
# 与所处阶段无关：索引与日志是运行产物，不属于任何单一阶段目录，
# 放在根目录也便于算力云上挂载数据盘。
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
#: 源码所在目录（研发），提示词等随代码一起发布
SOURCE_ROOT: Path = Path(__file__).resolve().parents[2]
#: 语料分块、索引等运行产物
DATA_ROOT: Path = PROJECT_ROOT / "data"
#: 本地模型（嵌入 / 语音 / 翻译）
MODELS_ROOT: Path = PROJECT_ROOT / "models"
#: 评估报告输出目录（属「优化」阶段产物，按交付分类归档）
EVAL_RESULTS_ROOT: Path = PROJECT_ROOT / "优化" / "评估结果" / "eval_results"

ENV_PREFIX = "RAG_"


class PathSettings(BaseModel):
    """文件系统路径配置。"""

    project_root: Path = PROJECT_ROOT
    source_root: Path = SOURCE_ROOT
    data_raw: Path = DATA_ROOT / "raw"
    data_processed: Path = DATA_ROOT / "processed"
    data_index: Path = DATA_ROOT / "index"
    data_eval: Path = DATA_ROOT / "eval"
    # 评估报告按交付分类放到「优化/评估结果」，
    # 通过环境变量 RAG_PATHS__EVAL_RESULTS 可覆盖回 data/ 下。
    eval_results: Path = EVAL_RESULTS_ROOT
    logs: Path = PROJECT_ROOT / "logs"
    sqlite_path: Path = DATA_ROOT / "index" / "rag.sqlite3"
    default_pdf: Path = DATA_ROOT / "raw" / "招股说明书1.pdf"


class PDFSettings(BaseModel):
    """PDF 解析配置。"""

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
    # 表格提取开关：pdfplumber 较慢，默认只对疑似含表格的页启用
    extract_tables: bool = True
    # 表格提取页数上限（0 表示不限制）
    max_table_pages: int = 0


class ChunkSettings(BaseModel):
    """分块配置（工单要求 chunk_size 500~800，overlap 100）。"""

    chunk_size: int = 700
    chunk_overlap: int = 100
    # 表格块单独处理：表格不切分，整表作为一个 chunk（避免破坏行列结构）
    table_as_single_chunk: bool = True
    min_chunk_chars: int = 30


class EmbeddingSettings(BaseModel):
    """向量化配置。"""

    # 默认使用本地模型；无模型时自动降级为 hash 向量（离线可跑通）
    # 本地优先：若 models/bge-small-zh-v1.5 存在则直接用，避免联网下载
    model_name: str = "BAAI/bge-small-zh-v1.5"
    # 本地模型目录（相对项目根目录）；存在时优先于 model_name 使用
    local_model_dir: str = "models/bge-small-zh-v1.5"
    # 可选：BAAI/bge-large-zh-v1.5、BAAI/bge-m3
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True
    # 降级 hash 向量维度（仅在模型不可用时使用）
    fallback_dim: int = 512
    # 向量库后端：
    #   "auto"   —— 优先 Chroma，不可用或异常时回退 numpy
    #   "numpy"  —— 纯 numpy 精确检索（默认，无外部服务、无索引损坏风险）
    #   "chroma" —— Chroma 持久化向量库（工单指定方案，适合更大规模）
    backend: str = "numpy"

    def resolve_model_path(self) -> str:
        """返回实际可用的模型路径：本地目录优先，否则回退到模型名（联网）。"""
        local = PROJECT_ROOT / self.local_model_dir
        if local.is_dir() and any(local.glob("*.json")):
            return str(local)
        return self.model_name


class RetrievalSettings(BaseModel):
    """检索配置。"""

    vector_top_k: int = 10
    bm25_top_k: int = 10
    # 合并去重后进入重排的候选数
    fusion_top_k: int = 20
    # 最终交给生成器的片段数。
    # 说明：招股书前 30 页含大量“释义/基本用语”样板段，会与真实证据争夺名额；
    # 取 8 可以保证在样板被降权后，真正的证据段仍能进入提示词。
    rerank_top_n: int = 8
    # 混合检索权重：final = vector_weight * vec_score + bm25_weight * bm25_score
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    # 是否启用重排模型（bge-reranker-base）
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
    # 表格片段加权：招股书的关键财务数据大多位于表格中
    table_boost: float = 1.25
    # 领域关键词加权表（招股书特攻优化）
    keyword_boost: dict[str, float] = Field(
        default_factory=lambda: {
            "收入": 1.5,
            "占比": 1.5,
            "比重": 1.5,
            "注册资本": 1.8,
            "法定代表人": 1.8,
            "募集资金": 1.6,
            "补充流动资金": 1.6,
            "上游": 1.5,
            "下游": 1.4,
            "供应商": 1.4,
            "客户": 1.3,
            "技术标准": 1.6,
            "科技进步奖": 1.6,
            "军用领域": 1.5,
            "主营业务收入": 1.5,
        }
    )
    # 判定“检索结果是否足以回答”的**原始余弦相似度**阈值。
    #
    # 为什么用原始余弦而不是融合分数：融合分数会被归一化到 1.0，
    # 导致任何问题（哪怕完全无关）看起来都“很有把握”。
    # 实测本语料：10 个工单问题的最高余弦为 0.737~0.822，
    # 而“今天天气怎么样”这类无关问题只有 0.329~0.391，
    # 取 0.55 有充足的安全间隔，同时也给未来的相关问法留出空间。
    min_confidence_cosine: float = 0.55
    # 融合分数的相对阈值（作为辅助判据，防止余弦偶发偏高）
    min_relevance_score: float = 0.35
    # 检索结果的“可答性”检查：问题里的实义词至少要有这么多个出现在证据片段中。
    #
    # 为什么需要它：多轮追问里若出现语料没有的议题（例如“那知识产权呢？”），
    # 改写后会退化为“上一轮问题 + 追问”，检索自然命中上一轮主题、余弦也很高，
    # 于是把上一个问题的答案错答给新问题。数量改为 0 可关闭该校验。
    min_overlap_terms: int = 1


class LLMSettings(BaseModel):
    """LLM 生成配置（对接 vLLM / SGLang 的 OpenAI 兼容接口）。"""

    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    model: str = "Qwen2.5-7B-Instruct-AWQ"
    temperature: float = 0.1
    top_p: float = 0.8
    max_tokens: int = 512
    # 连接与首字超时（秒）
    connect_timeout: float = 3.0
    read_timeout: float = 60.0
    # 可用性探测（models.list）的超时与重试次数。
    #
    # 这两个值直接决定“没有 LLM 服务时会不会卡住”：
    # OpenAI SDK 默认 max_retries=2，一次探测会发起 3 次连接；
    # 若每次 connect_timeout=3 秒，探测就要 ~9 秒才返回失败，
    # 而它发生在**生成答案之前**，会让首字延迟从毫秒级劣化到 9 秒。
    # 因此探测使用极短超时 + 不重试，快速失败后走抽取式回答。
    # 0.5 秒对本机/同内网服务足够（正常服务响应在 10ms 量级）。
    probe_timeout: float = 0.5
    probe_max_retries: int = 0
    # 正式生成时的重试次数（生成失败才需要重试）
    max_retries: int = 1
    # 送入 LLM 的最大片段数（0 表示用检索层的 rerank_top_n）。
    #
    # 为什么可调：提示词长度直接决定 **prefill 耗时**。实测 8 个片段
    # （约 1900 token）在 CPU 上 prefill 需要 ~18 秒，3 个片段只需数秒；
    # GPU + vLLM 上 8 个片段约 0.2~0.4 秒，因此云端保持默认（0）即可。
    # 本机用 CPU 联调服务时可设为 3，让首字延迟落到可接受范围。
    max_context_chunks: int = 0
    # 无 LLM 服务时是否降级为“抽取式回答”
    allow_extractive_fallback: bool = True


class ConversationSettings(BaseModel):
    """多轮对话配置。"""

    max_history_rounds: int = 5
    # 改写当前问题时携带的历史轮数
    rewrite_history_rounds: int = 3


class AppSettings(BaseModel):
    """应用级配置。"""

    app_name: str = "基于 PDF 文档的 RAG 问答系统"
    version: str = "1.0.0"
    # 工单验收：首字返回时间 < 3 秒
    first_token_budget_seconds: float = 3.0
    # 统一兜底回复
    unknown_answer: str = "不清楚"
    log_level: str = "INFO"
    log_json: bool = True
    # 中间步骤是否回传前端（工单要求：只展示最终答案）
    expose_intermediate_steps: bool = False


class ASRSettings(BaseModel):
    """语音识别（ASR）配置。

    工单追加要求：支持语音输入。中文与英文都要能识别。
    """

    # auto = 优先 OpenAI 兼容接口（省显存），否则本地 Whisper；也可强制 "api"/"local"
    backend: str = "auto"
    # OpenAI 兼容转写接口。vLLM 起 Whisper 服务后填 http://127.0.0.1:8001/v1
    base_url: str = "http://127.0.0.1:8001/v1"
    api_key: str = "EMPTY"
    model: str = "openai/whisper-large-v3"
    # 本地 Whisper 模型目录（相对项目根目录）；留空则自动扫描 models/whisper*
    local_model_dir: str = "models/whisper-tiny"
    # 单条音频的转写超时（秒）
    timeout: float = 60.0
    # 录音最长时长（秒），超过则提示用户说短一点
    max_audio_seconds: int = 120


class LanguageSettings(BaseModel):
    """多语言（中/英）问答配置。

    工单追加要求：支持中文与英文问答。
    """

    # 本地小模型目录（Qwen3-0.6B 等），供 "model" 桥接与英文润色使用
    translation_model_dir: str = "models/Qwen3-0.6B"

    # 默认回答语言：auto = 跟随提问语言；也可强制 "zh" / "en"
    default_answer_language: str = "auto"
    # 英文提问的语言桥接方式：
    #   "glossary" —— 领域术语表 + 专有名词映射（**默认**：毫秒级、完全确定、
    #                 不会把小模型译错公司名的问题带进检索）
    #   "model"    —— 本地小模型翻译（质量上限更高，但 CPU 上 1~10 秒且不稳定）
    #   "auto"     —— 有本地模型就用 model，否则 glossary
    query_bridge: str = "glossary"
    # 是否允许用本地小模型生成英文答案（关闭则用模板化英文回答）
    allow_local_translation: bool = True
    # 翻译超时（秒）
    translation_timeout: float = 30.0
    # 单次翻译最大生成 token 数
    translation_max_tokens: int = 96


class Settings(BaseModel):
    """配置聚合根。"""

    app: AppSettings = Field(default_factory=AppSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    pdf: PDFSettings = Field(default_factory=PDFSettings)
    chunk: ChunkSettings = Field(default_factory=ChunkSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    conversation: ConversationSettings = Field(default_factory=ConversationSettings)
    asr: ASRSettings = Field(default_factory=ASRSettings)
    language: LanguageSettings = Field(default_factory=LanguageSettings)

    def ensure_directories(self) -> None:
        """创建运行所需的全部目录（幂等）。"""
        for path in (
            self.paths.data_raw,
            self.paths.data_processed,
            self.paths.data_index,
            self.paths.data_eval,
            self.paths.eval_results,
            self.paths.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _apply_env_overrides(settings: Settings, environ: dict[str, str]) -> Settings:
    """把 ``RAG_SECTION__FIELD`` 形式的环境变量写入配置。

    仅支持两级（section.field），值统一按字符串解析后交给 pydantic 转换。
    """
    data = settings.model_dump()
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        remainder = key[len(ENV_PREFIX) :]
        if "__" not in remainder:
            continue
        section, _, field = remainder.partition("__")
        section, field = section.lower(), field.lower()
        if section not in data or field not in data[section]:
            continue
        current = data[section][field]
        try:
            if isinstance(current, bool):
                value: object = raw.strip().lower() in {"1", "true", "yes", "on"}
            elif isinstance(current, int):
                value = int(raw)
            elif isinstance(current, float):
                value = float(raw)
            else:
                value = raw
        except ValueError:
            continue
        data[section][field] = value
    return Settings(**data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局唯一配置实例（带缓存）。"""
    settings = Settings()
    settings = _apply_env_overrides(settings, dict(os.environ))
    settings.ensure_directories()
    return settings


def reload_settings() -> Settings:
    """清除缓存并重新加载配置（测试与运行时切换用）。"""
    get_settings.cache_clear()
    return get_settings()


# 便捷别名
settings = get_settings()

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
