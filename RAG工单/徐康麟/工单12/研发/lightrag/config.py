# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""配置中心：所有阈值/路径/模型名一律来自配置，代码内不得硬编码。

配置来源优先级（后者覆盖前者）：
  1. 内置默认值（本文件 DEFAULT_CONFIG）
  2. 配置文件：部署/配置/lightrag.json（可用 --config 指定别的路径）
  3. 环境变量：LIGHTRAG__<段>__<字段>（例如 LIGHTRAG__RETRIEVAL__TOP_K=10）

设计要点：
  * ``load_config()`` 返回带点号访问的 ``Cfg`` 包装对象，避免到处写 dict["a"]["b"]。
  * 任何取值都经过 ``_coerce`` 做类型兜底，配置写错不裸崩。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

# 仓库根：研发/lightrag/config.py → parents[2] = 工单12
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

WORK_ORDER = "人工智能NLP-RAG 项目-LightRAG 优化"

# ---------------------------------------------------------------------------
# 内置默认配置
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict[str, Any] = {
    "work_order": WORK_ORDER,
    "paths": {
        "repo_root": str(REPO_ROOT),
        "raw_dir": str(REPO_ROOT / "研发" / "data" / "raw"),
        "processed_dir": str(REPO_ROOT / "研发" / "data" / "processed"),
        "index_dir": str(REPO_ROOT / "研发" / "data" / "index"),
        "graph_dir": str(REPO_ROOT / "研发" / "data" / "graph"),
        "images_dir": str(REPO_ROOT / "研发" / "data" / "images"),
        "cache_dir": str(REPO_ROOT / "研发" / "data" / "cache"),
        "eval_dir": str(REPO_ROOT / "研发" / "评估"),
        "optimize_dir": str(REPO_ROOT / "优化" / "评估结果"),
        "test_data_dir": str(REPO_ROOT / "测试" / "测试数据"),
        "screenshot_dir": str(REPO_ROOT / "测试" / "测试截图"),
        "log_dir": str(REPO_ROOT / "部署" / "日志"),
        "config_file": str(REPO_ROOT / "部署" / "配置" / "lightrag.json"),
    },
    "llm": {
        "base_url": "http://127.0.0.1:11434",
        # 生成主力 / 图谱抽取多用一个稍大的模型
        "gen_model": "qwen2.5:3b",
        "extract_model": "qwen2.5:3b",
        "embed_model": "bge-m3:latest",
        "embed_dim": 1024,          # 实测 bge-m3 = 1024 维
        "temperature": 0.0,
        "extract_temperature": 0.0,
        "max_tokens": 900,
        "extract_max_tokens": 1200,
        "request_timeout_s": 600,
        "embed_timeout_s": 300,
        "embed_batch_size": 16,
        "keep_alive": "30m",
        "max_retries": 2,
        "retry_sleep_s": 1.0,
    },
    "chunk": {
        "size": 700,           # 目标块字符数
        "overlap": 120,        # 重叠字符
        "min_size": 80,        # 小于此长度的块丢弃
        "max_table_chars": 1600,
        "drop_below_pages": 0,  # 保留全部页
    },
    "image": {
        "enable": True,
        # 水印瓦片实测为 143×127（每页 15 个），故下限取 200×150 直接滤掉水印
        "min_width": 200,
        "min_height": 150,
        "min_pixels": 60000,
        "max_aspect": 6.0,
        "phash_hamming": 6,            # 感知哈希汉明距离 ≤ 此值判为重复图
        "max_per_page": 3,
        "caption_radius_chars": 260,   # 图注关联：图块附近文本窗口
        "max_images": 600,             # 抽图总量上限（控制体积）
        "render_dpi": 120,
        "save_images": False,          # 图像按「图页/图注关联+文本描述」处理，默认不落盘
    },
    "graph": {
        "sample_size": 400,            # 抽样抽取的块数上限
        "sample_seed": 20250826,
        "min_chunk_chars": 150,        # 信息密度下限
        "density_bonus_types": ["table", "image"],  # 表格/图块加权
        "batch_chunks": 1,             # 每次给 LLM 的块数
        "alias_similarity": 0.82,      # 实体别名归并阈值（字符级相似度）
        "alias_min_len": 3,
        "merge_relation_duplicates": True,
        "relation_min_confidence": 0.0,
        "save_raw_extractions": True,
        "extract_cache": True,
    },
    "retrieval": {
        "top_k": 8,                 # 最终返回块数
        "local_top_k": 12,          # 局部（向量）通道候选
        "global_top_k": 12,         # 全局（图谱扩展）通道候选
        "bm25_top_k": 12,           # 纯 RAG 基线 BM25 通道候选
        "bm25_k1": 1.5,             # BM25 词频饱和参数
        "bm25_b": 0.75,             # BM25 长度归一参数
        "bm25_use_jieba": True,
        "rrf_k": 60,                # RRF 常数
        # 融合权重（LightRAG 双层）
        "local_weight": 0.55,
        "global_weight": 0.45,
        # 纯 RAG 基线融合权重（向量 + BM25）
        "rag_vector_weight": 0.5,
        "rag_bm25_weight": 0.5,
        "graph_hops": 1,            # 全局检索展开跳数
        "graph_max_hops": 3,
        "graph_expand_top_entities": 8,   # 全局检索命中实体数
        "graph_max_relations": 60,        # 沿关系收集的上限
        "min_vector_similarity": 0.15,    # 低于此相似度的向量命中丢弃
        "relation_confidence_weight": 0.35,  # 全局通道分数里关系置信度占比
        "graph_score_norm": True,
    },
    "routing": {
        # 关键词分流：实体词典命中 → 局部；抽象度启发式 → 全局
        "abstract_suffixes": [
            "行业地位", "市场地位", "竞争格局", "竞争地位", "募集资金用途", "募集资金",
            "行业发展", "发展前景", "经营模式", "业务模式", "盈利模式", "风险因素",
            "上游", "下游", "产业链", "主要产品", "主营业务", "财务状况", "经营业绩",
            "股权结构", "组织结构", "治理结构", "技术标准", "核心技术", "行业趋势",
            "市场规模", "应用结构", "采购模式", "销售模式", "关联交易", "同业竞争",
            "战略规划", "募投项目", "研发投入", "人才", "客户结构", "供应商",
        ],
        "abstract_char_hints": [
            "情况", "结构", "模式", "地位", "格局", "趋势", "前景", "用途", "构成",
            "有哪些", "哪些", "如何", "为什么", "影响", "优势", "特点", "政策",
        ],
        "whitelist_local": ["发行股数", "注册资本", "法定代表人", "持股比例", "销售处", "销售部"],
        "entity_dict_from_graph": True,
        "min_entity_len": 2,
        "max_entity_len": 40,
        "global_min_keywords": 1,
        "route_fallback": "both",   # 分不出时两路都走：local | global | both
    },
    "answer": {
        "top_k_in_prompt": 6,
        "max_context_chars": 5200,
        "temperature": 0.1,
        "max_tokens": 700,
        "enable_citation": True,
    },
    "eval": {
        "question_file": str(REPO_ROOT / "研发" / "评估" / "测试问题集.jsonl"),
        "top_k": 8,
        "faithfulness_judge": "keyword",   # keyword = 无 LLM 打分，关键词/数字覆盖率近似
        "context_precision_threshold": 0.5,  # 判定「相关块」的相似度/命中阈值
        "answer_relevancy_min_len": 4,
        "save_generated_answers": True,
        "backend_list": ["rag", "lightrag"],
    },
    "viz": {
        "max_nodes": 60,
        "max_edges": 120,
        "figsize": [18, 12],
        "dpi": 140,
        "layout_iterations": 120,
    },
    "server": {
        "host": "127.0.0.1",
        "port": 8612,
    },
}

# 环境变量前缀
ENV_PREFIX = "LIGHTRAG__"


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并配置；override 中的 None 视为「不覆盖」。"""
    out = dict(base)
    for key, value in override.items():
        if value is None:
            continue
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


def _coerce(raw: str) -> Any:
    """环境变量字符串 → Python 值；失败则回退原字符串。"""
    low = raw.strip()
    if low.lower() in ("true", "false"):
        return low.lower() == "true"
    if low.lower() in ("null", "none"):
        return None
    try:
        return int(low)
    except ValueError:
        pass
    try:
        return float(low)
    except ValueError:
        pass
    if low.startswith(("[", "{")):
        try:
            return json.loads(low)
        except json.JSONDecodeError:
            return raw
    return raw


def env_overrides(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """扫描 LIGHTRAG__A__B=value 形式的环境变量。"""
    env = environ if environ is not None else os.environ
    out: dict[str, Any] = {}
    for key, value in env.items():
        if not key.startswith(ENV_PREFIX):
            continue
        parts = [p.lower() for p in key[len(ENV_PREFIX):].split("__") if p]
        if not parts:
            continue
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(value)
    return out


class Cfg:
    """带点号访问的配置包装：cfg.retrieval.top_k / cfg.path("graph_dir")。"""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError as exc:  # 明确报错，避免静默取到 None
            raise AttributeError(f"配置缺少字段：{name}") from exc
        if isinstance(value, Mapping):
            return Cfg(value)
        return value

    def __contains__(self, name: str) -> bool:
        return name in self._data

    def get(self, name: str, default: Any = None) -> Any:
        value = self._data.get(name, default)
        return Cfg(value) if isinstance(value, Mapping) else value

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._data, ensure_ascii=False))

    def path(self, name: str) -> Path:
        """取 paths.<name> 并确保目录存在。"""
        raw = self._data["paths"][name]
        target = Path(raw)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def path_str(self, name: str) -> str:
        return str(self._data["paths"][name])

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"Cfg(keys={sorted(self._data)})"


def load_config(config_file: str | os.PathLike[str] | None = None,
                *, environ: Mapping[str, str] | None = None) -> Cfg:
    """加载配置：默认值 → 配置文件 → 环境变量。配置文件缺失/损坏只告警不崩。"""
    data = _deep_merge({}, DEFAULT_CONFIG)

    candidates: list[Path] = []
    if config_file:
        candidates.append(Path(config_file))
    candidates.append(Path(str(DEFAULT_CONFIG["paths"]["config_file"])))
    for cand in candidates:
        if cand.is_file():
            try:
                loaded = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(loaded, Mapping):
                    data = _deep_merge(data, loaded)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[WARN] 配置文件读取失败，忽略：{cand} ({type(exc).__name__}: {exc})")
            break

    data = _deep_merge(data, env_overrides(environ))
    return Cfg(data)


def dump_effective_config(cfg: Cfg, target: Path | str) -> Path:
    """把生效配置落盘，便于审计「本次跑数用的是哪套阈值」。"""
    out = Path(target)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


CONFIG = load_config()
