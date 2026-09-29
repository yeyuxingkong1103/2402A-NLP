"""批次 22：PDF 解析相关配置接线单测（app/core/config.py）。

批次 22 前 .env 里的 MINERU_* / QWEN_VL_* 共 13 个键**无人读取**（改了不生效也不报错）。
本批次把字段接上，这里逐项校验：默认值、环境变量覆盖、bool/数值解析，
以及"PDF 是可选能力，不能进生产必需项"。
"""

from app.core.config import Settings


def test_pdf_fields_have_defaults():
    settings = Settings()

    assert settings.mineru_api_base_url == "https://mineru.net"
    assert settings.mineru_api_key == ""
    assert settings.mineru_poll_interval_seconds == 3.0
    assert settings.mineru_max_poll_seconds == 300.0
    assert settings.mineru_language == "ch"
    assert settings.mineru_enable_table is True
    assert settings.mineru_enable_formula is True
    assert settings.mineru_is_ocr is False
    assert settings.qwen_vl_api_base_url == ""
    assert settings.qwen_vl_api_key == ""
    assert settings.qwen_vl_model == "qwen-vl-ocr"
    assert settings.qwen_vl_timeout_seconds == 120.0
    assert settings.qwen_vl_max_pages_per_request == 8
    assert settings.qwen_vl_max_request_bytes == 1_500_000


def test_pdf_fields_read_from_environment(monkeypatch):
    monkeypatch.setenv("MINERU_API_BASE_URL", "https://mineru.example")
    monkeypatch.setenv("MINERU_POLL_INTERVAL_SECONDS", "1.5")
    monkeypatch.setenv("MINERU_MAX_POLL_SECONDS", "60")
    monkeypatch.setenv("MINERU_LANGUAGE", "en")
    monkeypatch.setenv("MINERU_ENABLE_TABLE", "false")
    monkeypatch.setenv("MINERU_ENABLE_FORMULA", "0")
    monkeypatch.setenv("MINERU_IS_OCR", "true")
    monkeypatch.setenv("QWEN_VL_API_BASE_URL", "https://qwen.example/v1")
    monkeypatch.setenv("QWEN_VL_MODEL", "qwen-vl-ocr-latest")
    monkeypatch.setenv("QWEN_VL_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("QWEN_VL_MAX_PAGES_PER_REQUEST", "2")
    monkeypatch.setenv("QWEN_VL_MAX_REQUEST_BYTES", "900000")

    settings = Settings.from_environment()

    assert settings.mineru_api_base_url == "https://mineru.example"
    assert settings.mineru_poll_interval_seconds == 1.5
    assert settings.mineru_max_poll_seconds == 60.0
    assert settings.mineru_language == "en"
    assert settings.mineru_enable_table is False
    assert settings.mineru_enable_formula is False
    assert settings.mineru_is_ocr is True
    assert settings.qwen_vl_api_base_url == "https://qwen.example/v1"
    assert settings.qwen_vl_model == "qwen-vl-ocr-latest"
    assert settings.qwen_vl_timeout_seconds == 45.0
    assert settings.qwen_vl_max_pages_per_request == 2
    assert settings.qwen_vl_max_request_bytes == 900000


def test_bool_flag_accepts_common_truthy_spellings(monkeypatch):
    for truthy in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("MINERU_IS_OCR", truthy)
        assert Settings.from_environment().mineru_is_ocr is True, truthy
    for falsy in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("MINERU_IS_OCR", falsy)
        assert Settings.from_environment().mineru_is_ocr is False, falsy


def test_pdf_keys_are_not_required_in_production():
    """PDF 是可选能力：未配置时只是 PDF 不可用，不该让服务起不来。"""
    required = Settings.REQUIRED_IN_PRODUCTION
    for name in (
        "MINERU_API_KEY",
        "MINERU_API_BASE_URL",
        "QWEN_VL_API_KEY",
        "QWEN_VL_API_BASE_URL",
        "QWEN_VL_MODEL",
    ):
        assert name not in required, name


def test_settings_singleton_exposes_pdf_fields():
    """模块级 settings 也必须带上这些字段（供装配函数读取）。"""
    from app.core import config

    assert hasattr(config.settings, "mineru_api_base_url")
    assert hasattr(config.settings, "qwen_vl_max_pages_per_request")
