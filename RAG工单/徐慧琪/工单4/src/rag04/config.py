# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""配置中心：双模式开关、模型名、路径、并发参数与硬约束阈值。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

WORK_ORDER_ID = "人工智能NLP-RAG-图像内容解析及检索优化"

# 项目根目录：本文件位于 <root>/src/rag04/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALID_MODES = ("baseline_03", "full_04")


@dataclass(frozen=True)
class Settings:
    """全局配置。baseline_03 代表工单01/02/03的能力，full_04 代表本工单。"""

    pipeline_mode: str = "full_04"

    # --- 路径 ---
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    qdrant_path: Path = PROJECT_ROOT / "data" / "qdrant"
    models_dir: Path = PROJECT_ROOT / "data" / "models"
    fig_cache_dir: Path = PROJECT_ROOT / "data" / "figures"
    vlm_cache_dir: Path = PROJECT_ROOT / "data" / "vlm_cache"
    log_dir: Path = PROJECT_ROOT / "logs"
    reports_dir: Path = PROJECT_ROOT / "docs" / "reports"

    # --- 文档 ---
    corpus: tuple[str, ...] = ("招股说明书1.pdf", "招股说明书2.pdf")

    # --- 图像解析硬约束（见 Global Constraints）---
    render_dpi: int = 210
    crop_margin: float = 0.15
    vlm_max_tokens: int = 2500
    vlm_model: str = "deepseek-v4-flash-vision-exp"
    clip_repo: str = "openai/clip-vit-base-patch32"
    clip_hf_mirror: str = "https://hf-mirror.com"
    clip_dim: int = 512
    fig_min_area: float = 9000.0      # 图区候选最小面积(pt^2)
    fig_gap: float = 12.0             # 绘图矩形聚类间距(pt)

    # --- 检索 ---
    embed_model: str = "bge-m3"
    embed_dim: int = 1024
    embed_batch: int = 16
    # 必须用 IPv4 字面量，不能写 localhost：Windows 上 getaddrinfo("localhost")
    # 先返回 ::1，而 Ollama 只监听 IPv4，每次 /api/embed 都要等 IPv6 连接失败再
    # 回退。实测（未复用连接的 requests.post）：localhost 2205/2269ms、
    # 127.0.0.1 229/229/211ms —— 每次嵌入固定多付约 2 秒，直接吃掉 3 秒预算。
    ollama_url: str = "http://127.0.0.1:11434"
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_top_n: int = 30
    final_top_k: int = 5
    rrf_k: int = 60
    # RC3（免重建修复）：图像指向时 CLIP 路的召回条数上限。融合权重 2.0 ×
    # boost 1.5 下第 r 张图得分 = 3/(rrf_k+r)，文本理论最高 = 2/(rrf_k+1)；
    # 只要 CLIP 返回满 30 条，30 个重排名额会被图像全部占满（实测 id5/id6：
    # top-30 = 30 图 / 0 文本）。收紧到 10 后图像仍居前（10 席），但
    # 其余 20 席留给文本/表格，金标文本块（原融合 31/32 名）得以进入重排。
    clip_max_k: int = 10
    # RC1（免重建修复）：样板过滤后的 dense 深挖条数。页眉在 top-30 可占 27~30
    # 席（实测 id1/id34/id543），只取 k_each 再过滤会把真实候选一并截掉；
    # 深挖到 500 才能让金标块补位（id34 答案块 dense 328 名）。
    # 实测 embedded Qdrant 开销：k=30→91.9ms、k=500→100.3ms（+8ms/查询），
    # 可忽略；仅当样板过滤启用时才深挖，baseline/无过滤路径开销不变。
    boilerplate_fetch_k: int = 500

    # --- 生成 ---
    llm_model: str = "deepseek-v4-flash"
    llm_fallback_model: str = "qwen2.5:3b"
    llm_timeout_s: float = 30.0
    llm_max_concurrency: int = 8
    llm_max_tokens: int = 1500

    # --- 评估 ---
    k_report: tuple[int, ...] = (3, 5, 10)
    answer_acc_threshold: float = 0.80

    # --- 性能硬指标 ---
    # 端到端问答 ≤3 秒（工单硬约束）。此处不做任何强制：/api/ask 只如实回报
    # within_budget，超预算不截断、不缓存、不重试 —— 缺口由 /health 与压测报告
    # 暴露（实测 p50 ≈ 3.9s，见 docs/reports/eval_full_04_rc2.json）。
    latency_budget_ms: float = 3000.0

    # --- 分块 ---
    fixed_chunk_size: int = 512        # baseline_03 用
    semantic_min_chars: int = 500      # full_04 用
    semantic_max_chars: int = 800
    chunk_overlap_ratio: float = 0.15

    # --- 双模式开关（由 __post_init__ 按 mode 覆写）---
    use_figures: bool = True
    use_tables: bool = True
    use_hybrid: bool = True
    use_rerank: bool = True
    use_clip_retrieval: bool = True

    def __post_init__(self) -> None:
        if self.pipeline_mode not in VALID_MODES:
            raise ValueError(
                f"pipeline_mode 必须是 {VALID_MODES} 之一，收到 {self.pipeline_mode!r}"
            )
        if self.pipeline_mode == "baseline_03":
            object.__setattr__(self, "use_figures", False)
            object.__setattr__(self, "use_tables", False)
            object.__setattr__(self, "use_hybrid", False)
            object.__setattr__(self, "use_rerank", False)
            object.__setattr__(self, "use_clip_retrieval", False)


def get_settings(mode: str | None = None) -> Settings:
    """按模式取配置。mode=None 时读环境变量 RAG04_MODE，默认 full_04。"""
    import os

    return Settings(pipeline_mode=mode or os.environ.get("RAG04_MODE", "full_04"))


def latency_verdict(ms: float, budget_ms: float) -> tuple[float, bool]:
    """时延达标判定与上报口径的唯一实现（接口 / 界面 / 压测三处共用，勿各自比较）。

    先把实测值四舍五入到 1 位小数（界面与接口的展示精度），再与预算比较：
    3000.04 ms 显示为 3000.0 ms，判定必须同为「达标」，否则同一份测量会渲染成
    「响应时间 3000.0 ms，超出 3000.0 ms 预算」的自相矛盾。边界含等号
    （``ms == budget`` 视为达标），超预算不截断、不缓存、不重试。

    返回 ``(展示用毫秒值, 是否在预算内)``。除展示外不改变任何判定语义：
    只做「先定精度、后比较」，让三个界面报出同一个结论。
    """
    shown = round(float(ms), 1)
    return shown, shown <= float(budget_ms)


def ensure_dirs(s: Settings) -> None:
    """建齐所有运行期目录（幂等）。"""
    for d in (
        s.data_dir, s.qdrant_path, s.models_dir, s.fig_cache_dir,
        s.vlm_cache_dir, s.log_dir, s.reports_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
