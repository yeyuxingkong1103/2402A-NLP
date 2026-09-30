# -*- coding: utf-8 -*-
"""
全局配置模块

设计约定：
    1. 所有可调参数集中在本文件定义，通过项目根目录的 .env 覆盖；
    2. 业务代码一律 ``from backend.config import settings``，
       不直接读取 os.environ，避免配置散落各处；
    3. 代码中不出现硬编码的路径、密钥、模型名。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录：backend/config.py 的上两级
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """服务全局配置。字段名与 .env 中的变量名一一对应（大小写不敏感）。"""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ==================== 一、应用服务 ====================
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_title: str = "计算机专业知识库 RAG 问答助手"

    # ==================== 二、数据路径 ====================
    source_docs_dir: str = "data/source_docs"
    parsed_dir: str = "data/parsed"
    avatars_dir: str = "data/avatars"   # 用户自选头像（从相册上传的图片）
    logs_dir: str = "logs"
    eval_dir: str = "eval"

    # ==================== 三、Milvus ====================
    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    milvus_collection: str = "cs_kb_chunks"
    milvus_dim: int = 1024
    milvus_metric_type: str = "COSINE"
    milvus_hnsw_m: int = 16
    milvus_hnsw_ef_construction: int = 200

    # ==================== 四、MySQL ====================
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "cs_rag"
    mysql_charset: str = "utf8mb4"

    # ==================== 五、Redis ====================
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0
    session_ttl_seconds: int = 1800
    session_max_turns: int = 5
    cache_ttl_seconds: int = 3600

    # ==================== 六、生成模型 ====================
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    llm_temperature: float = 0.1
    # 生成最大 token 数。
    # 注意：deepseek-v4-flash 是**推理模型**，会先输出 reasoning_content（推理过程）
    # 再输出 content（正式答复），两者共用本预算。1024 在多处实测中不够用
    # （129 次生成有 10 次触顶，并产生过空答案），先提高到 4096。
    # 2026-09-21 二次上调到 8192：4096 下「召回 5 条片段」的复杂问题仍会因推理占满
    # 预算而截断（前端表现为「答案生成服务暂时不可用」），提到 8192 后实测正常。
    llm_max_tokens: int = 8192
    llm_timeout: int = 60

    # ==================== 七、视觉模型 ====================
    vision_enabled: bool = False
    vision_api_key: str = ""
    vision_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    vision_model: str = "qwen-vl-plus"
    vision_timeout: int = 120

    # ==================== 八、本地模型 ====================
    bge_m3_path: str = "D:/桌面/模型/bge-m3"
    bge_reranker_path: str = "D:/桌面/模型/bge-reranker-v2-m3"
    embed_device: str = "cpu"
    embed_batch_size: int = 8
    embed_max_length: int = 8192

    # ==================== 九、MinerU ====================
    mineru_backend: str = "pipeline"
    # 必须用 ocr：实测 txt 模式会丢失行首字符（"GB/T 11457" 被解析成 "/"），
    # 导致标准编号检索失效，详见 .env.example 中的说明
    mineru_method: str = "ocr"
    mineru_lang: str = "ch"
    mineru_timeout: int = 3600
    mineru_fallback_enabled: bool = True

    # ==================== 十、分块 ====================
    chunk_size: int = 500
    chunk_overlap: int = 100
    chunk_max_chars: int = 2000

    # ==================== 十一、检索 ====================
    retrieve_top_k: int = 5
    retrieve_candidate_k: int = 20
    retrieve_score_threshold: float = 0.3
    # 说明：hybrid_sparse_weight 是「按分数加权融合」设想的遗留配置。
    # V2 采用 RRF（基于名次）融合，不使用权重，故该配置项当前**无任何代码读取**。
    # 保留字段是为了不破坏既有 .env 的兼容性；详见技术决策记录 ADR-019。
    hybrid_sparse_weight: float = 0.3
    # RRF 平滑常数 k，融合公式 score = Σ 1/(k + rank)
    rrf_k: int = 60

    # ==================== 十二、链路版本 ====================
    # 服务端使用的检索链路版本：
    #   v1 = 朴素稠密检索（V1 基线）
    #   v2 = 稠密 + 稀疏 + RRF 混合检索（V2）
    #   v3 = 重排 + 查询改写（V3）
    # 若某一版本出现异常，改回 v1 即可回退，无需改动任何代码。
    pipeline_version: str = "v2"

    # ==================== 十三、重排（V3 主链路）====================
    # 流程：稠密20 + 稀疏20 → RRF → 候选 RERANK_CANDIDATE_K 个 → 重排 → top_k
    rerank_enabled: bool = True
    # 送入重排的候选数量。实测重排 CPU 耗时约 0.42 秒/对（max_length=256），
    # 10 个候选约 4.2 秒；该值是「效果」与「N3 ≤8 秒」之间的平衡点（见 ADR-023）
    rerank_candidate_k: int = 10
    # 重排输入的最大 token 长度。实测 512→0.78s/对、256→0.39s/对、128→0.21s/对
    rerank_max_length: int = 256
    rerank_batch_size: int = 8
    rerank_timeout: int = 30

    # ==================== 十四、查询改写（对照实验分支）====================
    # 默认关闭：实测改写耗时 3.2~4.4 秒，开启后端到端约 10.5 秒，超出 N3（见 ADR-025）
    query_rewrite_enabled: bool = False
    query_rewrite_max_queries: int = 1
    query_rewrite_timeout: int = 20

    # ==================== 十五、日志 ====================
    log_level: str = "INFO"
    log_file: str = "rag_service.log"
    log_max_bytes: int = 10 * 1024 * 1024
    log_backup_count: int = 10
    log_to_console: bool = True

    # ==================== 十六、评测 ====================
    eval_top_k_list: str = "1,3,5,10"
    ragas_judge_model: str = "deepseek-v4-flash"
    ragas_enabled: bool = True

    # ----------------------------------------------------------------
    # 校验
    # ----------------------------------------------------------------

    @field_validator("chunk_size")
    @classmethod
    def _check_chunk_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("CHUNK_SIZE 必须为正整数")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def _check_overlap(cls, v: int, info) -> int:
        if v < 0:
            raise ValueError("CHUNK_OVERLAP 不能为负数")
        return v

    @field_validator("retrieve_top_k", "retrieve_candidate_k")
    @classmethod
    def _check_topk(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("检索 top-k 必须为正整数")
        return v

    # ----------------------------------------------------------------
    # 路径（统一解析为绝对路径，避免工作目录不同导致找不到文件）
    # ----------------------------------------------------------------

    def _abs(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    @property
    def source_docs_path(self) -> Path:
        """知识库源 PDF 目录"""
        return self._abs(self.source_docs_dir)

    @property
    def parsed_path(self) -> Path:
        """MinerU 解析产物目录"""
        return self._abs(self.parsed_dir)

    @property
    def avatars_path(self) -> Path:
        """用户自选头像图片目录（从相册上传的图片落在这里）"""
        return self._abs(self.avatars_dir)

    @property
    def logs_path(self) -> Path:
        """日志目录"""
        return self._abs(self.logs_dir)

    @property
    def eval_path(self) -> Path:
        """评测目录"""
        return self._abs(self.eval_dir)

    @property
    def eval_result_path(self) -> Path:
        """评测结果输出目录"""
        return self.eval_path / "eval_result"

    @property
    def frontend_path(self) -> Path:
        """前端静态资源目录"""
        return PROJECT_ROOT / "frontend"

    @property
    def log_file_path(self) -> Path:
        """日志文件完整路径"""
        return self.logs_path / self.log_file

    # ----------------------------------------------------------------
    # 派生配置
    # ----------------------------------------------------------------

    @property
    def top_k_list(self) -> List[int]:
        """解析 EVAL_TOP_K_LIST，例如 "1,3,5,10" -> [1, 3, 5, 10]"""
        result: List[int] = []
        for item in str(self.eval_top_k_list).split(","):
            item = item.strip()
            if item.isdigit():
                result.append(int(item))
        return sorted(set(result)) or [1, 3, 5, 10]

    @property
    def mysql_dsn(self) -> str:
        """MySQL 连接串（不含库名，用于建库）"""
        return (
            f"mysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/?charset={self.mysql_charset}"
        )

    def ensure_directories(self) -> None:
        """确保运行期需要的目录都存在（幂等）"""
        for path in (
            self.source_docs_path,
            self.parsed_path,
            self.logs_path,
            self.eval_result_path,
            self.avatars_path,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取配置单例（进程内只解析一次 .env）"""
    return Settings()


# 全局配置对象。业务代码统一使用：from backend.config import settings
settings = get_settings()


if __name__ == "__main__":
    # 直接运行本文件可打印当前生效配置，便于排查配置问题：
    #     python -m backend.config
    import json

    print("项目根目录:", PROJECT_ROOT)
    print(json.dumps(
        settings.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        default=str,
    ))
