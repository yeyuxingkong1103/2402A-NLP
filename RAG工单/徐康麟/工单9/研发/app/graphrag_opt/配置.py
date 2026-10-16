# -*- coding: utf-8 -*-
"""工单9 优化配置：三层面优化开关与阈值（**唯一真源**，全部可被配置/环境变量覆盖）。

工单编号：人工智能NLP-RAG-Graph RAG 优化任务

设计纪律（与 `_共享约定.md` §4.3 一致）：
    * **任何判断阈值都不许硬编码在业务代码里** —— 全部集中在本模块的 ``DEFAULTS``；
    * 覆盖顺序（后者优先）：``DEFAULTS`` → ``部署/配置/优化配置.json`` → 环境变量 ``GRAPHRAGOPT_<段>__<键>``；
    * 未识别的键**写 WARNING 而不是静默忽略**（避免「以为改了其实没生效」）；
    * 本模块只 import 标准库，保证 pytest 能快速 import（禁止 eager import torch/faiss/jieba）。

三层面优化的开关分组：
    * ``extract``  —— **实体/关系构建层**：别名归并、关系方向与类型规范化、噪声剪枝、置信度过滤；
    * ``retrieval``—— **检索层**：查询改写与实体链接、跳数与图剪枝、图特征×向量/BM25 融合权重、
                      重排候选裁剪、同页上下文扩展；
    * ``generation``—— **生成层**：图谱三元组结构化注入、引用约束、拒答护栏、模板化快路径。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

WORK_ORDER = "人工智能NLP-RAG-Graph RAG 优化任务"

#: 仓库根：研发/app/graphrag_opt/配置.py → parents[3] = 工单9
REPO_ROOT: Path = Path(__file__).resolve().parents[3]

#: 工单8 仓库根（**只读**：图谱原始产物与基线配置从这取，绝不写入）
BASE_REPO: Path = REPO_ROOT.parent / "工单8"
BASE_GRAPH_DIR: Path = BASE_REPO / "研发" / "data" / "graph"
#: 工单7 索引（**只读**，8847 块；本工单不重新解析、不重新嵌入）
BASE_INDEX_DIR: Path = REPO_ROOT.parent / "工单7" / "研发" / "data" / "index" / "bge-m3_1024"

#: 本工单目录契约
DATA_DIR: Path = REPO_ROOT / "研发" / "data"
OPT_GRAPH_DIR: Path = DATA_DIR / "图谱优化后"
EVAL_DIR: Path = REPO_ROOT / "研发" / "评估"
TEST_DATA_DIR: Path = REPO_ROOT / "测试" / "测试数据"
TEST_SHOT_DIR: Path = REPO_ROOT / "测试" / "测试截图"
OPT_DIR: Path = REPO_ROOT / "优化"
OPT_EVAL_DIR: Path = OPT_DIR / "评估结果"
DEPLOY_DIR: Path = REPO_ROOT / "部署"
LOG_DIR: Path = DEPLOY_DIR / "日志"
DEPLOY_CONFIG_DIR: Path = DEPLOY_DIR / "配置"
DEPLOY_SCRIPT_DIR: Path = DEPLOY_DIR / "脚本"

#: 优化配置文件（**人可读、可改**；改了立即生效，不需要动代码）
CONFIG_FILE: Path = DEPLOY_CONFIG_DIR / "优化配置.json"
#: 参考答案集（本工单自己从 sample_questions.pdf 整理 + 与 17 题集核对）
REFERENCE_SET: Path = EVAL_DIR / "参考答案集.jsonl"

# ===========================================================================
# 默认阈值
# ===========================================================================
DEFAULTS: dict[str, dict[str, Any]] = {
    # ------------------------------------------------------------------
    # ① 实体/关系构建层
    # ------------------------------------------------------------------
    "extract": {
        # 是否启用优化后图谱（False = 完全回退到 工单8 原始图谱，用于「优化前」对照）
        "enabled": True,
        # --- 别名归并 ---
        "alias_merge": True,
        # 公司/机构名后缀（归并时剥离，用于判定「同一主体的全称/简称」）
        "org_suffixes": ["股份有限公司", "有限责任公司", "有限公司", "集团公司", "集团", "公司"],
        # 人名前缀归并：A 是 B 的前缀、长度差 ≤ 该值、且 B 的提及数 ≥ A×ratio ⇒ A 并入 B
        # 依据（实测）：工单8 图谱里同时存在「何肖」(count=1) 与「何肖锋」——
        # PDF 抽取在换行处把姓名切断，不归并会分裂成两个弱节点，实体链接时两条都不命中。
        "person_prefix_len_diff": 1,
        "person_prefix_count_ratio": 0.3,
        # 数值型实体（比例/金额/日期）的归一化写法（去掉空格、统一「年度」→「年」）
        "normalize_numeric_names": True,
        # --- 关系方向与类型规范化 ---
        "relation_canonical": True,      # 同义关系名归并到 schema 规范名
        "relation_direction_fix": True,  # head/tail 类型反了 ⇒ 自动反转（并计数）
        "drop_invalid_relation": True,   # 不在 schema 内的关系 ⇒ 剪除
        "drop_self_loop": True,          # 自环（head == tail）⇒ 剪除
        "dedupe_edges": True,            # 同 (head, relation, tail) 去重（取最大置信度）
        # --- 噪声剪枝 ---
        "uninformative_ratio_max": 10.0, # 数值 ≤ 该值的整数比例（0%/5%/10%）是表格通用刻度 ⇒ 剪除
        "prune_types": ["date", "ratio", "amount"],  # 度极大时优先剪这些「通用刻度」类型
        # 通用刻度判据用实体的 **count（真实出现次数）**，不是保留的提及块数——
        # 依据（实测）：工单8 的「2020年度」count=593，但落盘时 mentions_per_entity 只留 6 条，
        # 若用提及块数判，593 → 6，永远低于阈值 ⇒ 噪声剪枝一条也剪不掉（实测「通用刻度剪除数=0」）。
        "max_doc_ratio": 0.05,           # count > ratio × 扫描块数 ⇒ 判为通用刻度
        "min_absolute_count": 200,       # 且 count 至少达到该绝对值（防小语料误伤）
        # 抽取碎片剪枝：正则从长句里切出的「伪实体」（如「之后深发展吸收合并原平安银行」）
        "prune_fragments": True,
        "fragment_max_len": 16,          # 公司/机构名超过该长度 ⇒ 判为碎片
        # 句级连词/指示词：只要出现就可判碎片（正常机构名不会带这些词）
        "fragment_markers": ["之后", "同受", "其中", "上述", "以下", "根据", "但是", "并且",
                             "或者", "以及", "分别", "同时", "由于", "如果", "本集团", "本公司"],
        # 并列连词：**单独出现不算碎片**（「薪酬与考核委员会」「审计与风险管理委员会」是真实机构），
        # 只有「名字很长」或「出现 ≥2 次」才判碎片。
        # 依据（实测）：第一版把「与/和/及」一律当碎片标记，误剪掉 774 个实体里
        # 包括「审计与风险管理委员会」这类**真实机构**，伤到图谱质量。
        "fragment_joiners": ["和", "与", "及"],
        "fragment_joiner_min_len": 12,   # 含并列连词且长度 ≥ 该值 ⇒ 碎片
        # --- 置信度过滤 ---
        # ⚠️ 边置信度下限必须与 工单8 对齐（0.55）：
        #   实测把 min_edge_confidence 提到 0.60 会剪掉 2628/4861 = 54% 的边
        #   （工单8 的 ``rule_cooccur`` 同句共现边置信度正好是 0.55），
        #   平均度数从 0.99 掉到 0.65 —— 图反而更碎，图谱通道召回能力下降。
        #   正确做法：**保留边**，把剪枝压力放在节点侧（孤立节点 + 通用刻度 + 碎片）。
        "min_confidence": 0.60,          # 实体置信度下限（工单8 为 0.55）
        "min_edge_confidence": 0.55,     # 边置信度下限（与工单8 对齐，不下调边的保留率）
        "mentions_per_entity": 4,        # 每实体保留的提及块数（工单8 为 6；越小扩展越省时）
        "min_entity_len": 2,
        "max_degree": 400,               # 度数上限（超过判为枢纽噪声）
    },
    # ------------------------------------------------------------------
    # ② 检索层
    # ------------------------------------------------------------------
    "retrieval": {
        "enabled": True,
        # --- 查询改写与实体链接 ---
        "query_rewrite": True,
        "entity_linking": True,
        "link_min_len": 2,               # 实体链接的最短实体名长度
        "link_limit": 12,                # 单次查询最多链接多少个实体
        "rewrite_max_terms": 8,          # 查询改写最多追加多少个字段词/别名
        # --- 图谱扩展：跳数与剪枝 ---
        "hop": 2,
        "hop_when_linked": 1,            # 已明确锁定语料时降到 1 跳（防跨语料污染）
        "max_expand_nodes": 32,          # 单次查询最多遍历的邻居节点数
        "neighbor_prune": True,          # 邻居按「边置信度 × 度数倒数」排序后截断
        "neighbor_top_k": 12,            # 每跳每种子最多保留的邻居数
        "min_confidence": 0.60,
        "max_graph_chunks": 20,
        "graph_corpus_filter": True,     # 问句点名语料时只保留该语料的块
        # --- 融合权重（图特征 × 向量/BM25）---
        # 工单8 的做法是「base_score + graph_bonus」直接相加，两者量纲差 2 个数量级
        # （base ≈ 4.x，graph_bonus ∈ [0.022, 0.06]）⇒ 图谱通道形同虚设
        # （实测：17 题里 graph_only 进 Top5 的题数 = 0）。
        # 本工单改为 **各自 min-max 归一化后加权**，权重和 = 1，可解释、可调、可审计。
        "fusion_weights": {
            "base": 0.62,                # 向量+BM25（RRF+重排）归一化分
            "graph": 0.23,               # 图结构分（跳数衰减 × 实体热度）归一化分
            "feature": 0.15,             # 图特征（实体命中率 / 跳数衰减 / 块内命中实体数）
        },
        "feature_weights": {"entity_hit": 0.5, "hop_decay": 0.3, "density": 0.2},
        "base_k": 20,                    # 基础召回候选数（比最终 top_k 大，供裁剪）
        "top_k": 10,
        "rerank_trim": 14,               # 融合后保留多少条进入「重排」（候选裁剪）
        "rerank_lexical_weight": 0.35,   # 重排打分里「问句实词命中率」的权重
        # --- 同页上下文扩展（提升上下文召回的关键手段，**只用查询侧信息**）---
        "page_expand": True,
        "page_expand_pages": 4,          # 对融合后 Top-N 命中的**不同页**做扩展
        "page_expand_max_per_page": 8,   # 每页最多补几块（同页块几乎必然相关 ⇒ 对精度友好）
        "page_expand_max_total": 16,     # 扩展块总数上限
        "page_expand_require_corpus": True,  # 只在「问句点名语料」内扩展（防跨公司污染）
        # --- 字段锚点召回（**上下文召回的主要来源**）---
        # 依据（实测 _缺失分析）：仅靠 Top-K，数值题的答案块常常进不了上下文
        # （问「营业收入/净利润/不良贷款率/拨备覆盖率」时，p19 的营业收入进了、
        #  p44 的拨备覆盖率进不来）⇒ 召回卡在 0.77。按问句里的**每个财务字段词**定点捞块，
        #  保证「问到的每个字段都有对应证据」。
        "field_probe": True,
        "field_probe_per_field": 2,      # 每个探针词最多补几块
        "field_probe_max": 10,           # 补块总数上限
        "field_probe_max_terms": 10,     # 单次查询最多用几个探针词（控制耗时）
        "field_probe_page_bonus": 0.5,   # 候选块落在「已命中页 ±window」时的加分（页一致优先）
        "field_probe_page_window": 1,    # 页一致窗口（±N 页）
        # 锚点页兄弟块扩展（把字段锚点命中页的其余块一并取回）。
        # ⚠️ 默认 **关闭** 的实测依据：打开后邮储 p18 的双块确实都进来了（Q13 召回 0.667→0.833），
        #   但**全局**精度从 0.8025 掉到 0.7437、召回也从 0.8739 掉到 0.8655 ——
        #   因为锚点页里也有「锚点只因一个词命中、整页其实无关」的情况，兄弟块被一并带进噪声。
        #   净收益为负，故默认关闭，仅作为可开启的对照组保留（配置化，随时可复现该实验）。
        "probe_expand": False,
        # --- 证据选取（进生成阶段）---
        "evidence_k": 14,                # 生成阶段证据块条数（工单8 为 5）
        "evidence_max_per_page": 4,
        "evidence_prefer_text": True,
        "evidence_rank_weight": 0.6,     # 证据排序里「检索名次」与「问句实词命中」的权重
        "evidence_diversify_corpus": True,  # 候选池跨多份语料时按语料轮转（跨文档题必需）
        # **焦点页簇补全**：把「融合后 Top-N 命中所在页」的全部块优先放进上下文。
        # 依据（实测 _页内块 诊断）：国泰君安 p4 的答案要点在 p4-x1、招商银行 p5 的 4 个要点在 p5-x1，
        # 它们都因「问句实词命中少、加权分低」而被挤出上下文 —— 同页兄弟块互相挤掉。
        "evidence_focus_pages": 4,       # 锁定前几个「焦点页」
        "evidence_focus_scan": 12,       # 从融合后前多少条里收集焦点页
        "evidence_focus_per_page": 6,    # 每个焦点页最多取几块
        "evidence_focus_diversify_corpus": True,  # 跨语料题：每个语料最多 1 个焦点页
        # **通道配额**：保证焦点页与字段锚点都能分到名额（否则前者会吃光全部名额）。
        "evidence_quota_focus": 7,       # 焦点页簇最多占几个名额
        "evidence_quota_probe": 5,       # 字段锚点块最多占几个名额
        # 多语言配额：英文提问时向量/BM25 跨语对齐不可靠，改用更可靠的**字段锚点**为主。
        "evidence_quota_focus_english": 4,
        "evidence_quota_probe_english": 8,
    },
    # ------------------------------------------------------------------
    # ③ 生成层
    # ------------------------------------------------------------------
    "generation": {
        "enabled": True,
        # --- 图谱三元组结构化注入 ---
        "inject_triples": True,
        "max_triples": 12,               # prompt 里最多注入多少条三元组
        "triple_max_chars": 1200,        # 三元组区块的字符上限
        # --- 引用约束 ---
        "citation_required": True,       # 答案必须标 [块N]
        "citation_min": 1,
        # --- 拒答护栏 ---
        "refusal_guard": True,
        "refusal_min_lexical_hits": 1,   # 证据与问句实词的重叠数下限（低于则拒答）
        "refusal_min_evidence": 1,       # 证据块条数下限
        # --- 模板化落地生成（快路径；SLA 的关键）---
        # 依据（实测）：工单8 全量 LLM 生成平均 3947 ms，端到端 SLA 达标率仅 29.41%，
        # 瓶颈 100% 在生成层。快路径用「证据抽取 + 结构化模板 + 引用」直接产出答案，
        # 生成耗时 ≈ 0 ms；只有草稿置信度不足时才调 LLM。
        "template_first": True,
        "template_min_chars": 30,        # 草稿长度下限
        "template_min_numeric": 1,       # 数值题草稿至少含几个数值原子
        "template_max_chars": 900,
        # --- LLM 慢路径（仅在快路径不达标时使用）---
        "llm_fallback": True,
        "llm_max_tokens": 200,           # 工单8 为 380 ⇒ 输出越长越慢
        "llm_num_ctx": 1536,             # 工单8 为 2048 ⇒ 上下文越小 prefill 越快
        "llm_evidence_chars": 220,       # 每条证据进 prompt 的截断长度（工单8 为 420）
        "llm_temperature": 0.1,
        "llm_timeout_s": 20.0,
        "keep_alive": "0s",              # 资源纪律（§5.5 规则 11）：用完立刻卸载
    },
    # ------------------------------------------------------------------
    # ④ 评估
    # ------------------------------------------------------------------
    "eval": {
        "context_precision_target": 0.80,
        "context_recall_target": 0.90,
        "sla_ms": 3000.0,
        "answer_len_min": 20.0,
        "answer_len_max": 800.0,
        "relevancy_keyword_weight": 0.7,
        "stability_repeat": 50,          # 稳定性压测：连续查询次数
        "thread_workers": 2,             # 并发上限（**资源纪律：只允许 I/O 型，≤2**）
    },
}


# ---------------------------------------------------------------------------
# 覆盖：配置文件 + 环境变量
# ---------------------------------------------------------------------------
def _apply_env_overrides(base: dict[str, Any], prefix: str, *, unknown: list[str]) -> dict[str, Any]:
    """``GRAPHRAGOPT_<段>__<键>=value`` 覆盖（段名大写）。"""
    out = dict(base)
    for key, raw in os.environ.items():
        if not key.startswith(prefix):
            continue
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
            else:
                out[field_name] = text
        except (TypeError, ValueError):
            unknown.append(f"{key}(取值非法:{text!r})")
    return out


def _deep_merge(base: dict[str, Any], patch: Mapping[str, Any], *, unknown: list[str],
                prefix: str = "") -> dict[str, Any]:
    """递归合并配置补丁；未识别的键记入 ``unknown``（不静默忽略）。"""
    out = dict(base)
    for key, value in (patch or {}).items():
        full = f"{prefix}{key}"
        if key not in out:
            unknown.append(full)
            continue
        if isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value, unknown=unknown, prefix=f"{full}.")
        else:
            out[key] = value
    return out


@dataclass(slots=True)
class OptConfig:
    """工单9 优化配置（含三层面开关与阈值 + 路径契约）。"""

    repo_root: Path
    base_repo: Path
    base_graph_dir: Path
    index_dir: Path
    opt_graph_dir: Path
    reference_set: Path
    log_dir: Path
    config_file: Path
    extract: dict[str, Any] = field(default_factory=lambda: json.loads(json.dumps(DEFAULTS["extract"])))
    retrieval: dict[str, Any] = field(default_factory=lambda: json.loads(json.dumps(DEFAULTS["retrieval"])))
    generation: dict[str, Any] = field(default_factory=lambda: json.loads(json.dumps(DEFAULTS["generation"])))
    eval: dict[str, Any] = field(default_factory=lambda: json.loads(json.dumps(DEFAULTS["eval"])))
    unknown_keys: list[str] = field(default_factory=list)
    config_source: str = "defaults"

    # -- 构造 ---------------------------------------------------------------
    @classmethod
    def from_env(cls, *, logger: Any = None, config_file: Path | None = None) -> "OptConfig":
        unknown: list[str] = []
        sections = {name: json.loads(json.dumps(value)) for name, value in DEFAULTS.items()}
        source = "defaults"
        path = Path(config_file) if config_file else CONFIG_FILE
        if path.is_file():
            try:
                patch = json.loads(path.read_text(encoding="utf-8"))
                for name, value in (patch or {}).items():
                    if name not in sections:
                        unknown.append(name)
                        continue
                    if isinstance(value, Mapping):
                        sections[name] = _deep_merge(sections[name], value, unknown=unknown,
                                                     prefix=f"{name}.")
                    else:
                        sections[name] = value
                source = f"defaults + {path.name}"
            except Exception as exc:  # noqa: BLE001 —— 配置损坏不得让系统起不来
                unknown.append(f"{path.name}(解析失败:{type(exc).__name__}: {exc})")
        for name in list(sections):
            sections[name] = _apply_env_overrides(sections[name], f"GRAPHRAGOPT_{name.upper()}__",
                                                  unknown=unknown)
        cfg = cls(
            repo_root=REPO_ROOT, base_repo=BASE_REPO, base_graph_dir=BASE_GRAPH_DIR,
            index_dir=Path(os.environ.get("GRAPHRAGOPT_INDEX_DIR") or BASE_INDEX_DIR),
            opt_graph_dir=Path(os.environ.get("GRAPHRAGOPT_GRAPH_DIR") or OPT_GRAPH_DIR),
            reference_set=REFERENCE_SET,
            log_dir=Path(os.environ.get("RAG_LOG__DIR") or LOG_DIR),
            config_file=path, unknown_keys=unknown, config_source=source,
            extract=sections["extract"], retrieval=sections["retrieval"],
            generation=sections["generation"], eval=sections["eval"],
        )
        if unknown and logger is not None:
            try:
                logger.log_event("optconfig.unknown_keys", level="WARNING", keys=sorted(unknown))
            except Exception:  # noqa: BLE001 —— 日志不可用不得阻断配置
                pass
        return cfg

    # -- 便捷访问 -----------------------------------------------------------
    def fingerprint(self) -> str:
        """配置指纹（进缓存键 / 进报告，保证「数字可追溯到配置」）。"""
        import hashlib

        blob = json.dumps({k: getattr(self, k) for k in ("extract", "retrieval", "generation")},
                          ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_order": WORK_ORDER,
            "config_source": self.config_source,
            "config_file": str(self.config_file),
            "unknown_keys": self.unknown_keys,
            "fingerprint": self.fingerprint(),
            "paths": {"repo_root": str(self.repo_root), "base_repo": str(self.base_repo),
                      "base_graph_dir": str(self.base_graph_dir), "index_dir": str(self.index_dir),
                      "opt_graph_dir": str(self.opt_graph_dir), "reference_set": str(self.reference_set),
                      "log_dir": str(self.log_dir)},
            "extract": self.extract, "retrieval": self.retrieval,
            "generation": self.generation, "eval": self.eval,
        }

    def dump(self, path: Path | None = None) -> Path:
        """把**生效的**配置快照落盘（验收要能追溯每个阈值）。"""
        target = Path(path) if path else DEPLOY_CONFIG_DIR / "优化配置快照.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target


_CONFIG_CACHE: OptConfig | None = None


def get_config(*, refresh: bool = False, logger: Any = None,
               config_file: Path | None = None) -> OptConfig:
    """进程级配置缓存。"""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None or refresh:
        _CONFIG_CACHE = OptConfig.from_env(logger=logger, config_file=config_file)
    return _CONFIG_CACHE


def ensure_dirs(cfg: OptConfig | None = None) -> dict[str, str]:
    """确保本工单输出目录存在（**不创建也不修改 工单7/工单8 的任何路径**）。"""
    config = cfg or get_config()
    made: dict[str, str] = {}
    for name, path in (("opt_graph_dir", config.opt_graph_dir), ("log_dir", config.log_dir),
                       ("test_data", TEST_DATA_DIR), ("test_shot", TEST_SHOT_DIR),
                       ("opt_eval", OPT_EVAL_DIR), ("deploy_config", DEPLOY_CONFIG_DIR),
                       ("deploy_script", DEPLOY_SCRIPT_DIR)):
        try:
            path.mkdir(parents=True, exist_ok=True)
            made[name] = str(path)
        except OSError as exc:  # noqa: BLE001 —— 目录不可建时如实返回，不静默崩
            made[name] = f"ERROR: {type(exc).__name__}: {exc}"
    return made


def write_default_config_file(cfg: OptConfig | None = None) -> Path:
    """把默认配置写成**可直接改**的 ``部署/配置/优化配置.json``（交付要求：阈值配置化）。"""
    config = cfg or get_config()
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_说明": [
            "工单9 Graph RAG 三层面优化配置。改这里立即生效（不需要动代码）。",
            "覆盖优先级：本文件 > 代码默认值 DEFAULTS；环境变量 GRAPHRAGOPT_<段>__<键> 优先级最高。",
            "任何阈值都不许硬编码在业务代码里 —— 统一放这里。",
            "把 extract.enabled / retrieval.enabled / generation.enabled 置 false 可单独关闭某一层优化（做消融）。",
        ],
        "extract": DEFAULTS["extract"],
        "retrieval": DEFAULTS["retrieval"],
        "generation": DEFAULTS["generation"],
        "eval": DEFAULTS["eval"],
    }
    CONFIG_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return CONFIG_FILE
