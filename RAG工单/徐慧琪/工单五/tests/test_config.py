# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Query理解优化
#          （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
"""配置自检。

继承自工单3（人工智能NLP-RAG-PDF文档的表格解析及检索优化）的基线测试，
断言已按工单5 的值更新。
"""
import os

from src import config  # 工单5 追加的用例在函数内直接引用 config


def test_multi_document_config():
    """工单3：多文档配置必须包含两份招股说明书，且路由别名双向可查。"""
    import os
    from src import config
    assert [os.path.basename(p) for p in config.SOURCE_PDFS] == [
        "招股说明书1.pdf", "招股说明书2.pdf"]
    assert set(config.DOC_ALIASES) == {"招股说明书1.pdf", "招股说明书2.pdf"}
    assert "武汉力源" in config.DOC_ALIASES["招股说明书2.pdf"]
    assert "兴图新科" in config.DOC_ALIASES["招股说明书1.pdf"]


def test_table_config_present():
    """工单3：表格链路的关键开关与阈值必须存在。"""
    from src import config
    assert config.TABLE_ENABLED is True
    assert config.TABLE_ROW_MAX_CHARS > 0
    assert 0 < config.TABLE_QUALITY_PASS <= 1
    # 工单5 配置分叉：本工单用自己的 collection（工单3 的 zhaogu_v3_table 只读）
    assert config.COLLECTION_OPT == "zhaogu_v5_mt"


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


def test_workorder_id_is_v5():
    assert config.WORKORDER_ID == "人工智能NLP-RAG-Query理解优化"
    assert config.WORKORDER_ID_PDF == "人工智能NLP-RAG-Query 理解优化任务"


def test_v5_collection_differs_from_v3():
    """工单5 必须用自己的 collection，否则与工单3 互相覆盖。"""
    assert config.COLLECTION_OPT != config.GY3_COLLECTION


def test_multiturn_defaults():
    assert config.DIALOGUE_HISTORY_WINDOW == 3
    assert config.REWRITE_MODEL_JUDGE.endswith("0.5b")
    assert config.REWRITE_MODEL_GEN.endswith("3b")
    assert config.REWRITE_MAX_CHARS > 0


def test_gy3_root_points_to_sibling_dir():
    import os
    assert os.path.isdir(config.GY3_ROOT), "工单3 基线目录必须存在（只读引用）"
    assert os.path.basename(config.GY3_ROOT) == "工单3"
