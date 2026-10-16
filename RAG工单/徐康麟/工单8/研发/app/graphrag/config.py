# -*- coding: utf-8 -*-
"""工单8 Graph RAG 集中配置（路径 / 阈值 / 环境变量）。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

职责：
    * 冻结本工单的**绝对路径契约**：语料 raw 在本仓库，**索引直接引用工单7 的绝对路径**
      （设计取舍：工单7 已完成 8847 块的解析/嵌入，重复解析与重复嵌入是纯浪费，
      且 8/9/10/11/13 号工单共享同一份 CCF 年报语料与同一套 bge-m3 向量）；
    * **所有判断阈值集中在此**（正则置信度、跳数、top_k、图谱扩展权重、抽样规模、
      Ollama 超时…），不散落在业务代码里硬编码；
    * 环境变量覆盖：``GRAPHRAG_<段>__<字段>``；未识别的 ``GRAPHRAG_*`` 写 WARN 而不静默忽略。

依赖纪律：本模块只 import 标准库；**禁止** eager import torch/faiss/jieba（保持可被 pytest 快速 import）。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

#: PDF 封面「工单编号」字段原文（所有源码头部注释必须含此串）
WORK_ORDER = "人工智能NLP-RAG-基于Graph RAG 实现金融问答"

#: 仓库根：研发/app/graphrag/config.py → parents[3] = 工单8
REPO_ROOT: Path = Path(__file__).resolve().parents[3]

#: 工单7 仓库根（**只读**引用：索引与冻结评估结果都取自这里，绝不写入）
BASE_REPO: Path = REPO_ROOT.parent / "工单7"
#: 工单7 已建好的索引目录（8847 块 = 文本 7325 + 表格 1417 + 图像 105）
BASE_INDEX_DIR: Path = BASE_REPO / "研发" / "data" / "index"
#: 工单7 的分块产物（图谱写抽时的输入；本工单**不重新解析 PDF**）
BASE_PROCESSED_DIR: Path = BASE_REPO / "研发" / "data" / "processed"
#: 工单7 冻结的评估结果（用于「比对 07 工单的测试结果」验收项）
BASE_EVAL_SUMMARY: Path = BASE_REPO / "测试" / "测试数据" / "评估汇总.json"
BASE_EVAL_JSONL: Path = BASE_REPO / "测试" / "测试数据" / "评估结果.jsonl"

DATA_DIR: Path = REPO_ROOT / "研发" / "data"
RAW_DIR: Path = DATA_DIR / "raw"
GRAPH_DIR: Path = DATA_DIR / "graph"
CONFIG_DIR: Path = REPO_ROOT / "研发" / "config"
TEST_DATA_DIR: Path = REPO_ROOT / "测试" / "测试数据"
TEST_SHOT_DIR: Path = REPO_ROOT / "测试" / "测试截图"
OPT_DIR: Path = REPO_ROOT / "优化"
OPT_EVAL_DIR: Path = OPT_DIR / "评估结果"
DEPLOY_DIR: Path = REPO_ROOT / "部署"
LOG_DIR: Path = DEPLOY_DIR / "日志"
DEPLOY_CONFIG_DIR: Path = DEPLOY_DIR / "配置"

#: 本工单自己的测试问题集副本（口径来源：工单7 研发/评估/测试问题集.jsonl，17 题）
QUESTION_SET: Path = DATA_DIR / "测试问题集.jsonl"

#: 图谱产物文件名（固定契约，UI / 测试 / 部署脚本都按此读取）
GRAPH_FILES = {
    "entities": "entities.jsonl",
    "relations": "relations.jsonl",
    "graph": "graph.json",
    "graphml": "graph.graphml",
    "svg": "graph.svg",
    "html": "graph.html",
    "build_stats": "图谱构建统计.json",
    "llm_supplement": "llm_关系补抽.jsonl",
}

# ---------------------------------------------------------------------------
# 阈值：图谱抽取
# ---------------------------------------------------------------------------
DEFAULT_EXTRACT_SETTINGS: dict[str, Any] = {
    # 扫描范围：只抽文本块（表格块以数字为主，噪声大；如需全量可置 true）
    "include_table_chunks": False,
    # 单块正文最大扫描字符数（超长块截断，避免正则退化为 O(n²)）
    "max_scan_chars": 4000,
    # 正则置信度打分（写进实体/关系，供下游过滤与调参；不做硬编码魔数）
    "confidence": {
        "org_fullname": 0.95,      # 带「股份有限公司/有限公司」后缀
        "org_short": 0.80,         # 词典/文件名派生的简称
        "institution": 0.90,
        "person_title": 0.88,      # 「董事长李建红」这类带职务
        "person_bare": 0.55,       # 仅靠 百家姓+2~3字（弱）
        "metric": 0.92,
        "amount": 0.90,
        "ratio": 0.90,
        "date": 0.85,
        "product": 0.70,
        "relation_pattern": 0.86,  # 正则三元组
        "relation_cooccur": 0.55,  # 同句共现推导（弱）
        "relation_llm": 0.75,      # Ollama 补抽
    },
    # 每个实体最多保留多少个「提及块」（控制图谱规模与扩展召回成本）
    "mentions_per_entity": 6,
    # 高频噪声剪枝：实体出现的块数 > max_doc_ratio × 扫描块数 ⇒ 剪除（公司/机构类豁免）
    "max_doc_ratio": 0.08,
    # 关系抽取的最小实体长度 / 一句最多产出多少三元组
    "min_entity_len": 2,
    "max_relations_per_sentence": 6,
    # 实体名长度上限（防正则贪婪吃掉整段）
    "max_entity_len": 30,
    # 无信息量比例：数值 ≤ 该值且为整数（0% / 1% / 5% / 10% / 1倍）⇒ 剪除。
    # 依据：这类比例是财务表格的通用刻度，几乎出现在每一页，度极大但零信息量，
    # 两跳扩展会把整张图拉进来（实测 `0%` / `5%` 曾进入度数 Top-20 并存于可视化图中心）。
    "uninformative_ratio_max": 10.0,
}

# ---------------------------------------------------------------------------
# 阈值：Graph RAG 检索
# ---------------------------------------------------------------------------
DEFAULT_RETRIEVAL_SETTINGS: dict[str, Any] = {
    # 基础召回（沿用 工单7 的 RRF 混合检索：向量 + BM25）
    "top_k": 10,
    "base_k": 10,
    # 检索缓存：评估必须关（否则第二次问同一题量到的是缓存命中时间，不是真实检索时间）
    "use_cache": False,
    # 图谱扩展：跳数（1 跳 = 命中实体直接相连的实体；2 跳 = 再外扩一层）
    "hop": 2,
    "max_expand_nodes": 40,     # 单次查询最多遍历的邻居节点数（性能闸门）
    "max_graph_chunks": 15,     # 图谱通道最多贡献多少候选块
    # 图谱通道块的基础分（与 RRF 分数同量级；RRF 单通道 rank1 ≈ 1/61 ≈ 0.0164）
    "graph_seed_score": 0.055,  # 命中实体自身提及的块
    "graph_hop1_score": 0.035,  # 一跳邻居提及的块
    "graph_hop2_score": 0.022,  # 两跳邻居提及的块
    # 图谱独有候选**插入到 Top-K 的哪些名次**（1-based）。
    # ⚠️ 默认 **空（不占名额）** 的实测依据：9 份年报混排时，图谱两跳会走到**别家公司**的年报
    #   （实测 Q07「平安银行营业收入/不良率」：图谱独有块把 `中国平安 p7`、`招商银行 p7`、
    #   `国泰君安 p4` 顶进第 2/5/9 名，把真正含答案的 `p19/p44` 表格挤出 Top-10，
    #   recall@5 与答案质量同时下降）。
    #   因此默认只让图谱块**追加**在 Top-K 之后（`graph_augment_after_topk`），
    #   基础排名完全不动；需要做「图谱占名额」的对照实验时再显式打开（如 "2,5,9"）。
    "graph_slots": "",
    # 图谱块追加位置上限（不占名额时的补充条数）
    "graph_augment_max": 5,
    # **跨语料闸门**：问句点名了某家公司（corpus_hint）时，图谱扩展只保留该语料自己的块。
    # 依据：年报之间「母公司/子公司」互相提及，跨语料扩展必然引入别家公司数据。
    "graph_corpus_filter": True,
    # 实体命中：问句 token 与实体名的匹配方式（精确包含 / 别名）
    "min_entity_name_len": 2,
    # 融合后进入生成阶段的证据条数。
    # 取 5（而不是 3）的实测依据：数值题的同一个指标在不同页有**不同口径的表述**
    # （实测 Q07：「营业收入 137,958 百万元 · 18.2%」在 p19 的表格里，
    # 「1,379.58 亿元 · 18.2%」「净利润 281.95 亿元」在 p5 的董事长致辞里，
    # 「拨备覆盖率 183.12%」在 p44），只给 3 条会漏掉口径完整的那一条；
    # 草稿打分本身有句子级门槛，多给证据不会引入噪声。
    "prompt_top_k": 5,
    # 进生成阶段的证据选块规则（见 engine.select_evidence 的实测依据）
    "evidence_max_per_page": 2,      # 同一页最多几块进入生成阶段（防止同页表格霸占证据）
    "evidence_prefer_text": True,    # 文本块优先于表格块（表格仍留在检索结果里）
    # 证据页补全：把命中页的同类块补进生成上下文（枚举完整性）。
    # ⚠️ 默认 **0（关闭）** 的实测依据：本工单语料是 9 家公司混排，
    #   同页补全会把**别家公司/别年度的表格**塞进证据（实测 Q07 补进 p20/p44/p95 表格，
    #   与 Q07 的 6 个关键数字毫无关系，反而把答案污染）。数值题的证据本来就在
    #   表格块里、且已经在 Top-K 内（实测 p19-t1-0 直接排第 1），无需按页补全。
    "page_context_extra": 0,
    # 图谱通道置信度下限（低于该值的实体不参与扩展）
    "min_confidence": 0.55,
}

# ---------------------------------------------------------------------------
# 阈值：Ollama 关系补抽（抽样，控制耗时）
# ---------------------------------------------------------------------------
DEFAULT_LLM_SETTINGS: dict[str, Any] = {
    "enabled": True,
    "base_url": "http://127.0.0.1:11434",
    "model": "qwen2.5:3b",
    "sample_chunks": 500,       # 抽样块数（工单要求 300~800）
    "sample_seed": 20260108,    # 固定随机种子 ⇒ 可复现
    "max_chars_per_chunk": 320, # 送入模型的正文截断（控制 prompt 长度）
    "timeout_s": 45.0,          # 单块超时
    "num_ctx": 2048,
    "max_relations_per_chunk": 3,
    "retry": 0,                 # 失败不重试（抽样统计，重试只会拉长耗时）
    "concurrency": 1,           # Ollama 单实例串行最稳
    # 模型驻留策略。权衡（实测）：
    #   * `0s`（每次请求后立刻卸载）最省内存，但每次调用都要重新加载模型 —— 实测单题端到端
    #     从约 1.5 s 涨到约 5.4 s（多出的 3.9 s 全是重复加载），SLA 达标率直接归零；
    #   * `120s`（会话内驻留、闲置 2 分钟自动卸载）在**单模型**前提下内存占用与 `0s` 同阶
    #     （qwen2.5:3b 约 2.2 GB），且**不会**出现「qwen2.5:3b + bge-m3 同时驻留」——
    #     嵌入模型由检索层按需加载，评估脚本在收尾时显式卸载生成模型。
    # 结论：默认 120s；需要极致省内存（如与其它重活并行）再把环境变量改回 `0s`。
    "keep_alive": "120s",
}

# ---------------------------------------------------------------------------
# 阈值：可视化
# ---------------------------------------------------------------------------
DEFAULT_VIZ_SETTINGS: dict[str, Any] = {
    "top_nodes": 120,           # 力导向绘制的最小节点数（工单要求 ≥120）
    "canvas": [2400, 1600],
    "iterations": 320,          # 力导向迭代轮数
    "repulsion": 12000.0,
    "spring": 0.012,
    "damping": 0.86,
    "label_min_degree": 2,      # 度数 ≥ 该值的节点才画标签（防糊成一团）
    "max_edges": 400,           # 最多绘制多少条边
    "min_edges": 120,           # 边数下限：不足时自动放宽节点名额（见 visualize.select_subgraph）
    "random_seed": 42,
    "legend": True,
    "node_radius_min": 7,
    "node_radius_max": 26,
}


def _env(prefix: str) -> Mapping[str, str]:
    return {k: v for k, v in os.environ.items() if k.startswith(prefix)}


def _apply_env_overrides(base: Mapping[str, Any], prefix: str, *, logger: Any = None) -> dict[str, Any]:
    """``GRAPHRAG_<SECTION>__<KEY>=value`` 覆盖默认阈值；未识别的键写 WARN。"""
    out = dict(base)
    unknown: list[str] = []
    for key, raw in _env(prefix).items():
        field_name = key[len(prefix):].lower()
        if field_name not in out:
            unknown.append(key)
            continue
        default = out[field_name]
        text = str(raw).strip()
        try:
            if isinstance(default, bool):
                out[field_name] = text.lower() in {"1", "true", "yes", "on", "y"}
            elif isinstance(default, int):
                out[field_name] = int(text)
            elif isinstance(default, float):
                out[field_name] = float(text)
            elif isinstance(default, list):
                out[field_name] = [int(x) for x in text.split(",") if x.strip()]
            else:
                out[field_name] = text
        except (TypeError, ValueError):
            unknown.append(f"{key}(取值非法:{text!r})")
    if unknown and logger is not None:
        try:
            logger.log_event("graphrag.unknown_env", level="WARNING", keys=sorted(unknown))
        except Exception:  # noqa: BLE001 —— 日志不可用不得阻断配置
            pass
    return out


@dataclass(slots=True)
class GraphRAGConfig:
    """全量配置（阈值全部可被环境变量覆盖，见 ``from_env``）。"""

    repo_root: Path
    base_repo: Path
    index_dir: Path
    processed_dir: Path
    graph_dir: Path
    raw_dir: Path
    question_set: Path
    log_dir: Path
    base_eval_summary: Path
    base_eval_jsonl: Path
    extract: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_EXTRACT_SETTINGS))
    retrieval: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_RETRIEVAL_SETTINGS))
    llm: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_LLM_SETTINGS))
    viz: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_VIZ_SETTINGS))

    # -- 构造 ---------------------------------------------------------------
    @classmethod
    def from_env(cls, *, logger: Any = None) -> "GraphRAGConfig":
        index_dir = Path(os.environ.get("GRAPHRAG_INDEX_DIR") or BASE_INDEX_DIR)
        processed_dir = Path(os.environ.get("GRAPHRAG_PROCESSED_DIR") or BASE_PROCESSED_DIR)
        graph_dir = Path(os.environ.get("GRAPHRAG_GRAPH_DIR") or GRAPH_DIR)
        return cls(
            repo_root=REPO_ROOT,
            base_repo=BASE_REPO,
            index_dir=index_dir,
            processed_dir=processed_dir,
            graph_dir=graph_dir,
            raw_dir=RAW_DIR,
            question_set=QUESTION_SET,
            log_dir=Path(os.environ.get("RAG_LOG__DIR") or LOG_DIR),
            base_eval_summary=BASE_EVAL_SUMMARY,
            base_eval_jsonl=BASE_EVAL_JSONL,
            extract=_apply_env_overrides(DEFAULT_EXTRACT_SETTINGS, "GRAPHRAG_EXTRACT__", logger=logger),
            retrieval=_apply_env_overrides(DEFAULT_RETRIEVAL_SETTINGS, "GRAPHRAG_RETRIEVAL__", logger=logger),
            llm=_apply_env_overrides(DEFAULT_LLM_SETTINGS, "GRAPHRAG_LLM__", logger=logger),
            viz=_apply_env_overrides(DEFAULT_VIZ_SETTINGS, "GRAPHRAG_VIZ__", logger=logger),
        )

    # -- 便捷访问 -----------------------------------------------------------
    def chunks_indexed(self) -> Path:
        """工单7 索引内的分块文件（行序 == vectors.npy 行序）。"""
        return self.index_dir / "bge-m3_1024" / "chunks_indexed.jsonl"

    def graph_file(self, key: str) -> Path:
        return self.graph_dir / GRAPH_FILES[key]

    def chunks_source(self) -> Path:
        """图谱抽取的输入：优先工单7 索引内的 chunks_indexed.jsonl，缺失则回落 processed/chunks.jsonl。"""
        primary = self.chunks_indexed()
        if primary.is_file():
            return primary
        return self.processed_dir / "chunks.jsonl"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("repo_root", "base_repo", "index_dir", "processed_dir", "graph_dir", "raw_dir",
                    "question_set", "log_dir", "base_eval_summary", "base_eval_jsonl"):
            data[key] = str(data[key])
        return data

    def fingerprint(self) -> str:
        """配置指纹（进缓存键：阈值一变，缓存自动失效）。"""
        blob = json.dumps({"r": self.retrieval, "e": self.extract}, ensure_ascii=False, sort_keys=True)
        import hashlib

        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


_CONFIG_CACHE: GraphRAGConfig | None = None


def get_config(*, refresh: bool = False, logger: Any = None) -> GraphRAGConfig:
    """进程级配置缓存。"""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None or refresh:
        _CONFIG_CACHE = GraphRAGConfig.from_env(logger=logger)
    return _CONFIG_CACHE


def ensure_dirs(cfg: GraphRAGConfig | None = None) -> dict[str, str]:
    """确保本工单所有输出目录存在（**不创建也不修改工单7 的任何路径**）。"""
    config = cfg or get_config()
    made: dict[str, str] = {}
    for name, path in (("graph_dir", config.graph_dir), ("log_dir", config.log_dir),
                       ("test_data", TEST_DATA_DIR), ("test_shot", TEST_SHOT_DIR),
                       ("opt_eval", OPT_EVAL_DIR), ("deploy_config", DEPLOY_CONFIG_DIR)):
        try:
            path.mkdir(parents=True, exist_ok=True)
            made[name] = str(path)
        except OSError as exc:  # noqa: BLE001 —— 目录不可建时如实返回，不静默崩
            made[name] = f"ERROR: {type(exc).__name__}: {exc}"
    return made
