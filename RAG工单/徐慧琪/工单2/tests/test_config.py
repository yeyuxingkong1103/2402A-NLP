# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化 —— 配置自检"""
import os


def test_workorder_id():
    from src import config
    assert config.WORKORDER_ID == "人工智能NLP-RAG-基于PDF文档的问答系统优化"


def test_model_paths_local_and_exist():
    from src import config
    assert os.path.isdir(config.EMBEDDING_MODEL_PATH)
    assert os.path.isdir(config.RERANKER_MODEL_PATH)
    assert os.path.isdir(config.MINERU_VLM_MODELS)


def test_dirs_created():
    from src import config
    for d in (config.DATA_DIR, config.PARSED_DIR, config.CHUNK_DIR,
              config.QDRANT_DIR, config.EVAL_DIR):
        assert os.path.isdir(d)


def test_baseline_paths():
    from src import config
    assert os.path.isdir(config.BASELINE_QDRANT_DIR)
    assert os.path.isfile(config.BASELINE_GT_FILE)


def test_offline_env_set_by_bootstrap():
    """bootstrap 必须把 HF 切到离线，保证任何遗漏 local_files_only 的调用不会联网。"""
    from src import config  # noqa: F401
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
