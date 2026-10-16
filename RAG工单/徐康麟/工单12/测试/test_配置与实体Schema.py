# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""t4 用例：配置加载 + 实体/关系 Schema（离线，不调 LLM）。

对应验收点：设计\\验收标准.md 1.1（类型体系）/ S5（阈值配置化）/ S6（错误兜底）。
"""
from __future__ import annotations

import json
from pathlib import Path

from lightrag import entity_schema as es
from lightrag.config import DEFAULT_CONFIG, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = REPO_ROOT / "部署" / "配置" / "lightrag.json"


def test_配置_部署配置文件存在且真被加载():
    """磁盘上的阈值必须真的进了内存配置（文件联动，不是抄文档）。"""
    assert CONFIG_FILE.is_file(), f"缺少生效配置：{CONFIG_FILE}"
    on_disk = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    cfg = load_config()
    assert cfg.retrieval.top_k == on_disk["retrieval"]["top_k"]
    assert cfg.graph.sample_size == on_disk["graph"]["sample_size"]
    assert cfg.llm.gen_model == on_disk["llm"]["gen_model"]
    assert Path(cfg.path_str("config_file")) == CONFIG_FILE


def test_配置_环境变量可覆盖且类型自动吸附():
    cfg = load_config(environ={"LIGHTRAG__RETRIEVAL__TOP_K": "10",
                               "LIGHTRAG__GRAPH__SAMPLE_SIZE": "5"})
    assert cfg.retrieval.top_k == 10
    assert cfg.graph.sample_size == 5
    # 未注入的环境变量不影响：默认加载仍是配置文件里的值
    default_top_k = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))["retrieval"]["top_k"]
    assert load_config().retrieval.top_k == default_top_k


def test_配置_损坏文件只告警不裸崩(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{这不是合法 JSON", encoding="utf-8")
    cfg = load_config(broken)
    assert cfg.graph.sample_size == DEFAULT_CONFIG["graph"]["sample_size"]


def test_Schema_实体类型17类且说明非空():
    assert len(es.ENTITY_TYPES) == 17, f"实体类型应为 17 类，实际 {len(es.ENTITY_TYPES)}"
    assert len(set(es.all_entity_types())) == 17
    assert all(v.strip() for v in es.ENTITY_TYPES.values())


def test_Schema_关系类型16类且说明非空():
    assert len(es.RELATION_TYPES) == 16, f"关系类型应为 16 类，实际 {len(es.RELATION_TYPES)}"
    assert len(set(es.all_relation_types())) == 16
    assert all(v.strip() for v in es.RELATION_TYPES.values())


def test_Schema_关系约束覆盖全部关系类型():
    assert set(es.RELATION_CONSTRAINTS) == set(es.RELATION_TYPES), "有关系未定义头尾约束"
    for rel, (heads, tails) in es.RELATION_CONSTRAINTS.items():
        assert heads and tails, f"{rel} 的约束集合为空"


def test_Schema_非法类型吸附回合法集():
    assert es.coerce_type("组织") == "机构"
    assert es.coerce_type("人名") == "自然人"
    assert es.coerce_type("标准") == "专利与技术标准"
    assert es.coerce_type("不存在的类型", kind="relation") in es.RELATION_TYPES
    assert es.coerce_type("", kind="entity") in es.ENTITY_TYPES


def test_Schema_指代别名归并到规范名():
    assert es.resolve_alias("本公司") == "发行人"
    assert es.resolve_alias(" 力源信息 ") == "武汉力源信息技术股份有限公司"
    assert es.resolve_alias("武汉力源信息技术有限公司") == "武汉力源信息技术股份有限公司"
    assert es.resolve_alias("随意名字") == "随意名字"


def test_Schema_关系头尾相容校验命中与拒绝():
    assert es.relation_allowed("募投", "公司", "募投项目") is True
    assert es.relation_allowed("控股", "股东", "公司") is True
    assert es.relation_allowed("控股", "产品", "行业") is False
    # 未登记组合宽松放行（不误杀真关系）
    assert es.relation_allowed("未登记关系", "产品", "行业") is True
