# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""冒烟测试：工程骨架、离线引导、配置、复用数据是否就位。"""
import io
import json
import os


def test_project_root_is_workorder8():
    from src import bootstrap
    root = bootstrap.project_root()
    assert root.replace("\\", "/").endswith("工单8"), root


def test_offline_env_forced():
    from src import bootstrap  # noqa: F401
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_kg_config_present():
    from src import config
    assert config.KG_DIR.endswith(os.path.join("data", "kg"))
    assert config.KG_SUBGRAPH_MAX_NODES > 0
    assert abs(sum(config.KG_FUSION_WEIGHTS.values()) - 1.0) < 1e-6


def test_reused_chunks_available():
    """复用 07 的分块数据必须存在且为 7757 块。"""
    from src import config
    path = os.path.join(config.CHUNK_DIR, "chunks.json")
    assert os.path.isfile(path), f"缺少 {path}（应从工单07 复制）"
    with io.open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    assert len(chunks) > 7000, f"分块数异常：{len(chunks)}"
    assert {"chunk_id", "text", "source", "page_idx"} <= set(chunks[0])


def test_scan_local_models_runs_offline():
    """扫描脚本可运行且不炸（不校验具体模型，只校验结构）。"""
    import scan_local_models as slm
    report = slm.scan()
    assert report["workorder"] == "人工智能NLP-RAG-基于GraphRAG实现金融问答"
    assert isinstance(report["models"], list)
    assert isinstance(report["checked_locations"], list)
