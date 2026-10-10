# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rag04.config import get_settings, PROJECT_ROOT
from rag04.schema import FigureBlock
from rag04.ingest.vlparser import (
    build_figure_prompt, vlm_describe, ocr_fallback,
    clip_encode_image, clip_encode_text, parse_figures,
)

GY2 = PROJECT_ROOT / "招股说明书2.pdf"


def _settings(tmp_path):
    """把 VLM 缓存隔离到 tmp_path。

    默认 settings.vlm_cache_dir 指向项目 data/vlm_cache；多个用例共用同一张
    **确定性**假图（60x40 纯白）与空题注时，缓存键完全相同，后一个用例会直接
    读前一个用例的缓存而绕过被测路径（实测：OCR 兜底用例读到「第二次成功」）。
    """
    from dataclasses import replace
    return replace(get_settings(), vlm_cache_dir=tmp_path / "vlm_cache")


def test_prompt_demands_complete_enumeration_and_hierarchy():
    p = build_figure_prompt("公司组织结构图")
    # 必须要求完整枚举，这是 6 个销售处不漏答的关键
    assert "完整" in p
    assert "逐条" in p or "列出" in p
    # 必须要求保留层级
    assert "层级" in p or "上级" in p or "下属" in p
    # 必须包含题注上下文
    assert "公司组织结构图" in p


def test_prompt_forbids_guessing():
    p = build_figure_prompt()
    assert "不要推测" in p or "未显示" in p or "无法识别" in p


def _fake_fig(tmp_path) -> FigureBlock:
    from PIL import Image
    img = Image.new("RGB", (60, 40), "white")
    p = tmp_path / "f.png"
    img.save(p)
    return FigureBlock(doc_id="d", page=1, bbox=(0, 0, 10, 10),
                       image_path=str(p), caption="测试图")


def test_vlm_describe_passes_max_tokens_floor(tmp_path):
    """硬约束：max_tokens 必须 ≥ 2000，否则答案被思维链挤空。"""
    f = _fake_fig(tmp_path)
    captured = {}

    class FakeCompletions:
        def create(self, **kw):
            captured.update(kw)
            m = MagicMock()
            m.choices = [MagicMock()]
            m.choices[0].finish_reason = "stop"
            m.choices[0].message.content = "描述内容"
            return m

    client = MagicMock()
    client.chat.completions = FakeCompletions()
    s = _settings(tmp_path)
    out = vlm_describe(f.image_path, f.caption, s, client=client)

    assert out == "描述内容"
    assert captured["max_tokens"] >= 2000
    assert captured["model"] == s.vlm_model


def test_vlm_describe_retries_on_length_truncation(tmp_path):
    """finish_reason=length 说明被截断，必须重试而不是接受空答案。"""
    f = _fake_fig(tmp_path)
    calls = {"n": 0}

    class FakeCompletions:
        def create(self, **kw):
            calls["n"] += 1
            m = MagicMock()
            m.choices = [MagicMock()]
            if calls["n"] == 1:
                m.choices[0].finish_reason = "length"
                m.choices[0].message.content = ""
            else:
                m.choices[0].finish_reason = "stop"
                m.choices[0].message.content = "第二次成功"
            return m

    client = MagicMock()
    client.chat.completions = FakeCompletions()
    out = vlm_describe(f.image_path, "", _settings(tmp_path), client=client, retries=3)
    assert out == "第二次成功"
    assert calls["n"] == 2


def test_vlm_describe_falls_back_to_ocr_on_total_failure(tmp_path, monkeypatch):
    f = _fake_fig(tmp_path)

    class BoomCompletions:
        def create(self, **kw):
            raise RuntimeError("api down")

    client = MagicMock()
    client.chat.completions = BoomCompletions()
    monkeypatch.setattr("rag04.ingest.vlparser.ocr_fallback", lambda p: "OCR兜底文本")

    out = vlm_describe(f.image_path, "", _settings(tmp_path), client=client, retries=2)
    assert out == "OCR兜底文本"


def test_vlm_describe_disables_thinking_and_falls_back(tmp_path):
    """实测：不关思维链时 reasoning 独占 4000~8600 tokens，答案截断为空；
    必须显式 thinking=disabled；服务端不认该参数时剥离后重试。"""
    f = _fake_fig(tmp_path)
    seen = []

    class FakeCompletions:
        def create(self, **kw):
            seen.append(kw)
            if len(seen) == 1:
                raise RuntimeError("Unsupported parameter: thinking")
            m = MagicMock()
            m.choices = [MagicMock()]
            m.choices[0].finish_reason = "stop"
            m.choices[0].message.content = "关闭思维链后的完整答案"
            return m

    client = MagicMock()
    client.chat.completions = FakeCompletions()
    out = vlm_describe(f.image_path, "", _settings(tmp_path), client=client, retries=3)

    assert seen[0]["extra_body"]["thinking"]["type"] == "disabled"
    assert "extra_body" not in seen[1]
    assert out == "关闭思维链后的完整答案"


def _bad_request_without_thinking_word():
    """真实 openai.BadRequestError(400)，报文只提 extra_body、不含 thinking 字样。"""
    import httpx
    from openai import BadRequestError

    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return BadRequestError("Unrecognized request argument supplied: extra_body",
                           response=response, body=None)


class _Fake400(Exception):
    """非 openai 类型的 4xx：仅带 status_code，报文同样不含 thinking。"""
    status_code = 400


def _ok_response(text):
    m = MagicMock()
    m.choices = [MagicMock()]
    m.choices[0].finish_reason = "stop"
    m.choices[0].message.content = text
    return m


def test_vlm_describe_strips_thinking_on_400_without_word(tmp_path):
    """400 报文只提 extra_body、不提 thinking 时也必须剥离重试。

    多数 OpenAI 兼容服务端对未知参数的措辞是 Unrecognized request argument /
    extra fields not permitted，永远不会出现 thinking 字样；只测子串会漏判，
    导致重试全部发送同一个非法请求、静默退化成 OCR。
    """
    err = _bad_request_without_thinking_word()
    assert "thinking" not in str(err).lower(), "测试前提：异常文本不含 thinking"

    f = _fake_fig(tmp_path)
    seen = []

    class FakeCompletions:
        def create(self, **kw):
            seen.append(kw)
            if len(seen) == 1:
                raise err
            return _ok_response("剥离 thinking 后的答案")

    client = MagicMock()
    client.chat.completions = FakeCompletions()
    out = vlm_describe(f.image_path, "", _settings(tmp_path), client=client, retries=3)

    assert "extra_body" in seen[0], "首次请求必须带 thinking（DeepSeek 靠它出答案）"
    assert "extra_body" not in seen[1], "400 后必须剥离 thinking 重试"
    assert out == "剥离 thinking 后的答案"
    assert len(seen) == 2


def test_vlm_describe_strip_retry_does_not_consume_slot_with_retries_1(tmp_path):
    """retries=1 时剥离重试不得占用重试配额，否则无 thinking 的请求永远发不出去。"""
    f = _fake_fig(tmp_path)
    seen = []

    class FakeCompletions:
        def create(self, **kw):
            seen.append(kw)
            if len(seen) == 1:
                raise _Fake400("Unrecognized request argument supplied: extra_body")
            return _ok_response("retries=1 也拿到了答案")

    client = MagicMock()
    client.chat.completions = FakeCompletions()
    out = vlm_describe(f.image_path, "", _settings(tmp_path), client=client, retries=1)

    assert len(seen) == 2, "400 后即使 retries=1 也必须再发一次不带 thinking 的请求"
    assert "extra_body" in seen[0]
    assert "extra_body" not in seen[1]
    assert out == "retries=1 也拿到了答案"


def test_vlm_describe_keeps_thinking_when_backend_accepts(tmp_path):
    """支持 thinking 的后端（DeepSeek）必须继续收到该参数——它是 3 秒延迟与
    非空答案的关键，不能被兼容分支误剥离。"""
    f = _fake_fig(tmp_path)
    seen = []

    class FakeCompletions:
        def create(self, **kw):
            seen.append(kw)
            return _ok_response("支持 thinking 的正常描述")

    client = MagicMock()
    client.chat.completions = FakeCompletions()
    out = vlm_describe(f.image_path, "", _settings(tmp_path), client=client)

    assert out == "支持 thinking 的正常描述"
    assert len(seen) == 1
    assert seen[0]["extra_body"]["thinking"]["type"] == "disabled"


def test_vlm_describe_reports_ocr_fallback_via_callback(tmp_path, monkeypatch):
    """OCR 兜底必须可观测：返回 OCR 文本的同时回调 on_fallback。"""
    f = _fake_fig(tmp_path)

    class BoomCompletions:
        def create(self, **kw):
            raise RuntimeError("api down")

    client = MagicMock()
    client.chat.completions = BoomCompletions()
    monkeypatch.setattr("rag04.ingest.vlparser.ocr_fallback", lambda p: "OCR兜底文本")
    seen = []

    out = vlm_describe(f.image_path, "回调题注", _settings(tmp_path), client=client,
                       retries=1, on_fallback=seen.append)

    assert out == "OCR兜底文本"
    assert seen == ["OCR兜底文本"]


def test_vlm_result_cached_by_image_hash_and_caption(tmp_path):
    """全局约束：同一 (图片内容, 题注) 命中缓存不重复调 API；题注变化则重算。"""
    f = _fake_fig(tmp_path)
    s = _settings(tmp_path)
    calls = {"n": 0}

    class FakeCompletions:
        def create(self, **kw):
            calls["n"] += 1
            m = MagicMock()
            m.choices = [MagicMock()]
            m.choices[0].finish_reason = "stop"
            m.choices[0].message.content = f"描述{calls['n']}"
            return m

    client = MagicMock()
    client.chat.completions = FakeCompletions()
    a = vlm_describe(f.image_path, "题注", s, client=client)
    b = vlm_describe(f.image_path, "题注", s, client=client)
    c = vlm_describe(f.image_path, "另一题注", s, client=client)

    assert a == b == "描述1"
    assert c == "描述2"
    assert calls["n"] == 2


def test_parse_figures_marks_warning_on_failure(tmp_path, monkeypatch):
    f = _fake_fig(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("vlm down")

    monkeypatch.setattr("rag04.ingest.vlparser.vlm_describe", boom)
    monkeypatch.setattr("rag04.ingest.vlparser.clip_encode_image", lambda p, s: [0.0] * 512)
    out = parse_figures([f], get_settings())
    assert out[0].parse_warning, "失败必须留痕，不能静默"


def test_parse_figures_marks_warning_when_ocr_fallback_used(tmp_path, monkeypatch):
    """VLM 失败走 OCR 兜底时 FigureBlock 必须留痕，不能与真实描述混淆。"""
    f = _fake_fig(tmp_path)

    def fake_vlm(image_path, caption, settings, **kw):
        kw["on_fallback"]("OCR兜底文本")
        return "OCR兜底文本"

    monkeypatch.setattr("rag04.ingest.vlparser.vlm_describe", fake_vlm)
    monkeypatch.setattr("rag04.ingest.vlparser.clip_encode_image", lambda p, s: [0.0] * 512)
    out = parse_figures([f], get_settings())

    assert out[0].description == "OCR兜底文本"
    assert out[0].parse_warning, "OCR 兜底必须留痕，不能静默降级"
    assert "OCR" in out[0].parse_warning


def test_parse_figures_marks_warning_on_empty_description(tmp_path, monkeypatch):
    """VLM 与 OCR 都无输出时也必须留痕（空描述 + 空 warning 属于静默失败）。"""
    f = _fake_fig(tmp_path)
    monkeypatch.setattr("rag04.ingest.vlparser.vlm_describe", lambda *a, **k: "")
    monkeypatch.setattr("rag04.ingest.vlparser.clip_encode_image", lambda p, s: [0.0] * 512)
    out = parse_figures([f], get_settings())
    assert out[0].parse_warning, "空描述必须留痕"


@pytest.mark.integration
def test_clip_encoders_produce_512_dim(tmp_path):
    s = get_settings()
    f = _fake_fig(tmp_path)
    iv = clip_encode_image(f.image_path, s)
    tv = clip_encode_text("组织结构图 销售部", s)
    assert len(iv) == 512
    assert len(tv) == 512
    assert any(abs(x) > 1e-6 for x in iv)
