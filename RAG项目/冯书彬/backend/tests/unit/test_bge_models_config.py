from backend.app.core.config import settings


def test_model_paths_are_local():
    assert settings.BGE_M3_MODEL_PATH.endswith("models\\bge-m3")
    assert settings.BGE_RERANKER_MODEL_PATH.endswith("models\\bge-reranker-large")


def test_model_precision_is_fp16():
    assert settings.MODEL_DTYPE == "float16"
