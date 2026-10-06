"""精简入口和多模态读取测试，不连接真实数据库或模型 API。"""
import importlib
import json
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from tests.test_load import make_pdf
from tests.test_pipeline import _write_sample_public_data

reader = importlib.import_module("data_pipeline.load")
entry = importlib.import_module("data_pipeline.main")


def test_main_defaults_to_processing_without_database(tmp_path, monkeypatch):
    source = tmp_path / "public"
    output = tmp_path / "processed"
    _write_sample_public_data(source)
    monkeypatch.setattr("sys.argv", ["main", "--data-dir", str(source), "--output-dir", str(output)])
    def forbidden(*args, **kwargs):
        pytest.fail("默认处理不能连接数据库或初始化模型")
    monkeypatch.setattr(entry, "ensure_milvus_database", forbidden)
    monkeypatch.setattr(entry, "ModelGateway", forbidden)
    assert entry.main() == 0
    rows = json.loads((output / "civil_code_articles.json").read_text(encoding="utf-8"))
    assert rows[0]["article_number"] == 1
    assert "embedding" not in rows[0]
    assert len(list(output.glob("civil_*.json"))) == 8


def test_main_refuses_two_database_modes(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main", "--append", "--rebuild"])
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2


def test_single_public_file_uses_same_legal_record_fields(tmp_path, monkeypatch):
    source = tmp_path / "law.txt"
    source.write_text("第一条 测试条文。\n第二条 另一条测试条文。", encoding="utf-8")
    output = tmp_path / "processed"
    monkeypatch.setattr("sys.argv", ["main", "--file", str(source), "--collection", "civil_code_articles",
                                     "--output-dir", str(output)])
    assert entry.main() == 0
    rows = json.loads((output / "civil_code_articles.json").read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert rows[0]["article_content"] == "第一条 测试条文。"
    assert rows[0]["id"]
    assert rows[0]["embedding_text"]
    assert "embedding" not in rows[0]


def test_single_unstructured_file_does_not_save_fake_legal_records(tmp_path):
    pipeline = importlib.import_module("data_pipeline.index")
    source = tmp_path / "law.txt"
    source.write_text("无法匹配条文的普通文字", encoding="utf-8")
    with pytest.raises(ValueError, match="记录"):
        pipeline.process_public_file(source, collection="civil_code_articles", output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_legacy_command_uses_same_single_file_entry(tmp_path, monkeypatch):
    legacy = importlib.import_module("data_pipeline.public_main")
    source = tmp_path / "law.txt"
    source.write_text("第一条 测试条文。", encoding="utf-8")
    output = tmp_path / "processed"
    monkeypatch.setattr("sys.argv", ["public_main", "--file", str(source), "--collection", "civil_code_articles",
                                     "--output-dir", str(output)])
    assert legacy.main() == 0
    assert json.loads((output / "civil_code_articles.json").read_text(encoding="utf-8"))[0]["article_number"] == 1


def test_single_file_default_output_does_not_replace_full_build(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "law.txt"
    source.write_text("第一条 测试条文。", encoding="utf-8")
    full = tmp_path / "data/processed"
    full.mkdir(parents=True)
    previous = full / "civil_code_articles.json"
    previous.write_text('[{"id":"previous-full-build"}]', encoding="utf-8")
    report = full / "quality_report.json"
    report.write_text('{"previous":true}', encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["main", "--file", str(source), "--collection", "civil_code_articles"])
    assert entry.main() == 0
    assert json.loads(previous.read_text(encoding="utf-8")) == [{"id": "previous-full-build"}]
    assert json.loads(report.read_text(encoding="utf-8")) == {"previous": True}
    assert list((full / "previews").rglob("civil_code_articles.json"))


def test_single_file_cannot_append_without_cross_collection_checks(tmp_path, monkeypatch):
    source = tmp_path / "law.txt"
    source.write_text("第一条 测试条文。", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["main", "--file", str(source), "--collection", "civil_code_articles", "--append",
                                     "--output-dir", str(tmp_path / "out")])
    def forbidden(*args):
        pytest.fail("单份预览不能尝试入库")
    monkeypatch.setattr(entry, "ensure_milvus_database", forbidden)
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()


def test_vision_pdf_retains_real_page_and_marks_vision(tmp_path, monkeypatch):
    source = tmp_path / "scan.pdf"
    make_pdf(source, [("", True), ("", False)])
    monkeypatch.setattr(reader, "read_image_with_vision", lambda image: "第一条 测试原文", raising=False)
    result = reader.extract_content(source, pdf_method="vision")
    assert result.status == "success"
    assert result.pages[0]["text"] == "第一条 测试原文"
    assert result.pages[0]["page"] == 1
    assert result.pages[0]["method"] == "vision"
    assert result.pages[1]["text"] == ""
    assert result.vision_used is True
    assert result.ocr_used is False


def test_vision_failure_is_not_saved_as_complete_text(tmp_path, monkeypatch):
    source = tmp_path / "scan.pdf"
    make_pdf(source, [("", True)])
    def failed(image):
        raise RuntimeError("测试模型不可用")
    monkeypatch.setattr(reader, "read_image_with_vision", failed, raising=False)
    with pytest.raises(RuntimeError, match="完整"):
        reader.load(source, pdf_method="vision")


def test_vision_uses_existing_configuration_and_original_text_prompt(monkeypatch):
    config = importlib.import_module("backend.app.config")
    settings = SimpleNamespace(siliconflow_api_key="test-key", siliconflow_base_url="https://example.test/v1",
                               multimodal_model="test-vision", multimodal_max_tokens=1600, multimodal_timeout=180)
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    def respond(url, **options):
        assert url == "https://example.test/v1/chat/completions"
        assert options["json"]["model"] == "test-vision"
        content = options["json"]["messages"][0]["content"]
        assert "不要总结" in content[0]["text"]
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "choices": [{"finish_reason": "stop", "message": {"content": "第一条 原文"}}]})
    monkeypatch.setattr(httpx, "post", respond)
    assert reader.read_image_with_vision(Image.new("RGB", (20, 20))) == "第一条 原文"


@pytest.mark.parametrize("reason,text", [("length", "未完成的条文"), ("stop", ""), ("stop", "内容\ufffd")])
def test_vision_refuses_truncated_empty_or_corrupted_text(monkeypatch, reason, text):
    config = importlib.import_module("backend.app.config")
    settings = SimpleNamespace(siliconflow_api_key="test-key", siliconflow_base_url="https://example.test/v1",
                               multimodal_model="test-vision", multimodal_max_tokens=1600, multimodal_timeout=180)
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: httpx.Response(200,
        request=httpx.Request("POST", url), json={"choices": [{"finish_reason": reason, "message": {"content": text}}]}))
    with pytest.raises(RuntimeError):
        reader.read_image_with_vision(Image.new("RGB", (20, 20)))


@pytest.mark.parametrize("ocr_text", ["", "第一条\ufffd", "【看不清】"])
def test_auto_vision_repairs_incomplete_ocr(tmp_path, monkeypatch, ocr_text):
    source = tmp_path / "scan.png"
    Image.new("RGB", (40, 40)).save(source)
    monkeypatch.setattr(reader, "ocr_image", lambda image: (ocr_text, "rapidocr"))
    monkeypatch.setattr(reader, "read_image_with_vision", lambda image: "第一条 完整原文。")
    result = reader.extract_content(source, pdf_method="auto-vision")
    assert result.status == "success"
    assert result.text == "第一条 完整原文。"
    assert result.ocr_used is True
    assert result.vision_used is True


def test_auto_vision_does_not_call_cloud_for_good_ocr(tmp_path, monkeypatch):
    source = tmp_path / "scan.png"
    Image.new("RGB", (40, 40)).save(source)
    monkeypatch.setattr(reader, "ocr_image", lambda image: ("第一条 原文。", "rapidocr"))
    def forbidden(image):
        pytest.fail("OCR成功时不能调用云端")
    monkeypatch.setattr(reader, "read_image_with_vision", forbidden)
    result = reader.extract_content(source, pdf_method="auto-vision")
    assert result.text == "第一条 原文。"
    assert result.vision_used is False


def test_default_local_reader_refuses_corrupted_ocr_without_upload(tmp_path, monkeypatch):
    source = tmp_path / "scan.png"
    Image.new("RGB", (40, 40)).save(source)
    monkeypatch.setattr(reader, "ocr_image", lambda image: ("条文\ufffd", "rapidocr"))
    def forbidden(image):
        pytest.fail("没有选择云端补读，不能上传图片")
    monkeypatch.setattr(reader, "read_image_with_vision", forbidden)
    with pytest.raises(RuntimeError, match="完整"):
        reader.load(source)


def test_auto_vision_repairs_pdf_after_ocr_error(tmp_path, monkeypatch):
    source = tmp_path / "scan.pdf"
    make_pdf(source, [("Article one", False), ("", True)])
    def failed(image):
        raise RuntimeError("OCR不可用")
    monkeypatch.setattr(reader, "ocr_image", failed)
    monkeypatch.setattr(reader, "read_image_with_vision", lambda image: "第二条 扫描原文。")
    result = reader.extract_content(source, pdf_method="auto-vision")
    assert result.status == "success"
    assert result.pages[0]["text"] == "Article one"
    assert result.pages[1]["text"] == "第二条 扫描原文。"
    assert result.ocr_used is True
    assert result.vision_used is True


def test_auto_vision_failure_does_not_save_partial_legal_records(tmp_path, monkeypatch):
    pipeline = importlib.import_module("data_pipeline.index")
    source = tmp_path / "scan.png"
    Image.new("RGB", (40, 40)).save(source)
    monkeypatch.setattr(reader, "ocr_image", lambda image: ("第一条\ufffd", "rapidocr"))
    def failed(image):
        raise RuntimeError("云端失败")
    monkeypatch.setattr(reader, "read_image_with_vision", failed)
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="完整"):
        pipeline.process_public_file(source, collection="civil_code_articles", output_dir=output, pdf_method="auto-vision")
    assert not output.exists()


def test_main_accepts_explicit_ocr_then_vision_mode(tmp_path, monkeypatch):
    source = tmp_path / "scan.png"
    Image.new("RGB", (40, 40)).save(source)
    monkeypatch.setattr(reader, "ocr_image", lambda image: ("第一条 原文。", "rapidocr"))
    output = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["main", "--file", str(source), "--collection", "civil_code_articles",
                                     "--pdf-method", "auto-vision", "--output-dir", str(output)])
    assert entry.main() == 0
    rows = json.loads((output / "civil_code_articles.json").read_text(encoding="utf-8"))
    assert rows[0]["article_content"] == "第一条 原文。"


@pytest.mark.parametrize("vector", [[], [float("nan"), 1], [float("inf"), 1], ["1", 2], [True, 2]])
def test_embedding_refuses_invalid_vector_values(vector):
    embedding = importlib.import_module("data_pipeline.embedding")
    model = SimpleNamespace(embed=lambda texts: [vector])
    with pytest.raises(ValueError, match="向量"):
        embedding.embed([{"text": "法律原文"}], model)


@pytest.mark.parametrize("vectors", [[], [[1, 2], [3, 4]], [[1]]])
def test_embedding_checks_response_count_and_dimension(vectors):
    embedding = importlib.import_module("data_pipeline.embedding")
    model = SimpleNamespace(embed=lambda texts: vectors)
    with pytest.raises(ValueError, match="向量"):
        embedding.embed([{"text": "法律原文"}], model, expected_dimension=2)


@pytest.mark.parametrize("mode", ["append_public_collections", "rebuild_public_collections"])
def test_public_writer_validates_records_before_connecting(mode, monkeypatch):
    pipeline = importlib.import_module("data_pipeline.index")
    def forbidden(*args, **kwargs):
        pytest.fail("坏记录不应触发数据库连接")
    monkeypatch.setattr(pipeline, "_milvus_client", forbidden)
    model = SimpleNamespace(embed=lambda texts: [[1, 2] for text in texts])
    with pytest.raises(ValueError, match="空正文"):
        getattr(pipeline, mode)(collections={"civil_code_articles": [{"id": "article-1", "article_content": ""}]},
                               model=model, embedding_dimension=2)


@pytest.mark.parametrize("mode", ["append_public_collections", "rebuild_public_collections"])
def test_public_writer_requires_embedding_model_before_connecting(mode, monkeypatch):
    pipeline = importlib.import_module("data_pipeline.index")
    def forbidden(*args, **kwargs):
        pytest.fail("没有嵌入模型不应触发数据库连接")
    monkeypatch.setattr(pipeline, "_milvus_client", forbidden)
    with pytest.raises(ValueError, match="模型"):
        getattr(pipeline, mode)(collections={"civil_code_articles": [{"id": "article-1", "article_content": "第一条 原文。"}]},
                               embedding_dimension=2)


def test_public_payload_does_not_silently_truncate_legal_content():
    pipeline = importlib.import_module("data_pipeline.index")
    row = {"id": "article-1", "article_content": "法" * 22000}
    model = SimpleNamespace(embed=lambda texts: [[1, 2] for text in texts])
    with pytest.raises(ValueError, match="长度"):
        pipeline._public_rows([row], model, 2, "civil_code_articles")


@pytest.mark.parametrize("mode", ["append_public_collections", "rebuild_public_collections"])
def test_overlong_public_records_stop_before_model_or_database(mode, monkeypatch):
    pipeline = importlib.import_module("data_pipeline.index")
    def forbidden(*args, **kwargs):
        pytest.fail("过长记录应在模型和数据库操作之前被拦住")
    monkeypatch.setattr(pipeline, "_milvus_client", forbidden)
    model = SimpleNamespace(embed=forbidden)
    with pytest.raises(ValueError, match="长度"):
        getattr(pipeline, mode)(collections={"civil_code_articles": [{"id": "article-1", "article_content": "法" * 22000}]},
                               model=model, embedding_dimension=2)


class PublicWriteClient:
    """模拟Milvus的外部边界；真实记录校验、向量和入库组装仍由项目代码执行。"""
    def __init__(self, dimension=2):
        self.dimension = dimension
        self.rows = {"civil_code_articles": [{"id": "old", "article_content": "旧原文。", "embedding": [1, 2]}]}
        self.writes = []

    def has_collection(self, name):
        return name in self.rows

    def describe_collection(self, name):
        return {"fields": [
            {"name": "id", "is_primary": True, "params": {"max_length": 128}},
            {"name": "embedding", "params": {"dim": self.dimension}},
            {"name": "article_content", "params": {"max_length": 65535}},
        ]}

    def prepare_index_params(self):
        return SimpleNamespace(add_index=lambda *args, **kwargs: None)

    def create_collection(self, name, **kwargs):
        self.writes.append(("create", name))
        self.rows[name] = []

    def insert(self, name, rows):
        self.writes.append(("insert", name))
        self.rows[name].extend(rows)

    def flush(self, name):
        self.writes.append(("flush", name))

    def get_collection_stats(self, name):
        return {"row_count": len(self.rows[name])}

    def drop_collection(self, name):
        self.writes.append(("drop", name))
        del self.rows[name]

    def rename_collection(self, old, new):
        self.writes.append(("rename", old, new))
        self.rows[new] = self.rows.pop(old)


def test_append_refuses_existing_collection_with_wrong_dimension():
    pipeline = importlib.import_module("data_pipeline.index")
    client = PublicWriteClient(dimension=3)
    model = SimpleNamespace(embed=lambda texts: [[1, 2] for text in texts])
    with pytest.raises(ValueError, match="维度"):
        pipeline.append_public_collections(collections={"civil_code_articles": [{"id": "new", "article_content": "第一条 原文。"}]},
                                           model=model, client=client, embedding_dimension=2)
    assert client.writes == []
    assert client.rows["civil_code_articles"][0]["id"] == "old"


@pytest.mark.parametrize("mode", ["append_public_collections", "rebuild_public_collections"])
def test_invalid_vectors_do_not_mutate_public_collections(mode):
    pipeline = importlib.import_module("data_pipeline.index")
    client = PublicWriteClient()
    model = SimpleNamespace(embed=lambda texts: [[float("nan"), 2] for text in texts])
    with pytest.raises(ValueError, match="向量"):
        getattr(pipeline, mode)(collections={"civil_code_articles": [{"id": "new", "article_content": "第一条 原文。"}]},
                               model=model, client=client, embedding_dimension=2)
    assert client.writes == []
    assert client.rows["civil_code_articles"][0]["id"] == "old"


@pytest.mark.parametrize("mode", ["append_public_collections", "rebuild_public_collections"])
def test_public_writer_preserves_legal_id_and_body(mode):
    pipeline = importlib.import_module("data_pipeline.index")
    client = PublicWriteClient()
    model = SimpleNamespace(embed=lambda texts: [[1, 2] for text in texts])
    result = getattr(pipeline, mode)(collections={"civil_code_articles": [{
        "id": "civil_code_articles_article_1", "article_number": 1, "article_content": "第一条 原文。"}]},
        model=model, client=client, embedding_dimension=2)
    row = client.rows["civil_code_articles"][-1]
    assert row["id"] == "civil_code_articles_article_1"
    assert row["article_content"] == "第一条 原文。"
    assert row["article_number"] == 1
    assert row["embedding"] == [1, 2]
    assert set(client.rows) == {"civil_code_articles"}
    assert result["civil_code_articles"] == (2 if mode == "append_public_collections" else 1)


def test_collection_creation_error_does_not_create_unknown_schema():
    pipeline = importlib.import_module("data_pipeline.index")
    client = PublicWriteClient()
    def broken(name, **kwargs):
        if kwargs:
            raise TypeError("字段定义失败")
        client.rows[name] = []
    client.create_collection = broken
    with pytest.raises(TypeError, match="字段"):
        pipeline._ensure_public_collection(client, "civil_code_articles__candidate", 2)
    assert set(client.rows) == {"civil_code_articles"}


def test_main_checks_overlong_records_before_database_initialization(tmp_path, monkeypatch):
    source = tmp_path / "public"
    _write_sample_public_data(source)
    law = source / "civil_code_articles/civil_code_articles.txt"
    law.write_text("第一条 " + "法" * 22000, encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["main", "--data-dir", str(source), "--output-dir", str(tmp_path / "out"), "--append"])
    def forbidden(*args, **kwargs):
        pytest.fail("坏资料不应触发数据库初始化或模型初始化")
    monkeypatch.setattr(entry, "ensure_milvus_database", forbidden)
    monkeypatch.setattr(entry, "ModelGateway", forbidden)
    with pytest.raises(ValueError, match="长度"):
        entry.main()


def test_mixed_page_short_ocr_result_triggers_vision(tmp_path, monkeypatch):
    source = tmp_path / "mixed.pdf"
    original = "A long legal text. " * 8
    make_pdf(source, [(original, True)])
    monkeypatch.setattr(reader, "ocr_image", lambda image: ("short", "rapidocr"))
    monkeypatch.setattr(reader, "read_image_with_vision", lambda image: original + "Image text.")
    result = reader.extract_content(source, pdf_method="auto-vision")
    assert result.status == "success"
    assert "Image text." in result.pages[0]["text"]
    assert result.pages[0]["method"] == "vision"


def test_mixed_page_short_vision_result_is_not_complete(tmp_path, monkeypatch):
    source = tmp_path / "mixed.pdf"
    make_pdf(source, [("A long legal text. " * 8, True)])
    monkeypatch.setattr(reader, "ocr_image", lambda image: ("short", "rapidocr"))
    monkeypatch.setattr(reader, "read_image_with_vision", lambda image: "too short")
    with pytest.raises(RuntimeError, match="完整"):
        reader.load(source, pdf_method="auto-vision")


@pytest.mark.parametrize("row", [
    {"id": 123, "article_content": "第一条 原文。"},
    {"id": "article-1", "article_content": ["第一条 原文。"]},
])
@pytest.mark.parametrize("mode", ["append_public_collections", "rebuild_public_collections"])
def test_public_field_types_are_checked_before_connecting(row, mode, monkeypatch):
    pipeline = importlib.import_module("data_pipeline.index")
    def forbidden(*args, **kwargs):
        pytest.fail("主键或正文类型错误不应触发模型和数据库")
    monkeypatch.setattr(pipeline, "_milvus_client", forbidden)
    with pytest.raises(ValueError, match="字符串"):
        getattr(pipeline, mode)(collections={"civil_code_articles": [row]},
                               model=SimpleNamespace(embed=forbidden), embedding_dimension=2)


def test_shared_image_reader_can_explicitly_use_vision_without_ocr(monkeypatch):
    # 图片和PDF共用这个选择步骤；明确选vision时必须跳过OCR。
    def forbidden(image):
        pytest.fail("直接多模态模式不能调用OCR")
    monkeypatch.setattr(reader, "ocr_image", forbidden)
    monkeypatch.setattr(reader, "read_image_with_vision", lambda image: "第一条 原文。")
    with Image.new("RGB", (40, 40)) as image:
        text, method = reader.read_scanned_image(image, use_vision=True)
    assert text == "第一条 原文。"
    assert method == "vision"


def test_rebuild_failed_insert_only_removes_candidate_collection():
    pipeline = importlib.import_module("data_pipeline.index")
    client = PublicWriteClient()
    def failed_insert(name, rows):
        raise RuntimeError("插入失败")
    client.insert = failed_insert
    model = SimpleNamespace(embed=lambda texts: [[1, 2] for text in texts])
    with pytest.raises(RuntimeError, match="插入"):
        pipeline.rebuild_public_collections(collections={"civil_code_articles": [{"id": "new", "article_content": "第一条 原文。"}]},
                                            model=model, client=client, embedding_dimension=2)
    assert set(client.rows) == {"civil_code_articles"}
    assert client.rows["civil_code_articles"][0]["id"] == "old"


def test_rebuild_wrong_candidate_count_preserves_original_collection():
    pipeline = importlib.import_module("data_pipeline.index")
    client = PublicWriteClient()
    client.get_collection_stats = lambda name: {"row_count": 0}
    model = SimpleNamespace(embed=lambda texts: [[1, 2] for text in texts])
    with pytest.raises(RuntimeError, match="行数"):
        pipeline.rebuild_public_collections(collections={"civil_code_articles": [{"id": "new", "article_content": "第一条 原文。"}]},
                                            model=model, client=client, embedding_dimension=2)
    assert set(client.rows) == {"civil_code_articles"}
    assert client.rows["civil_code_articles"][0]["id"] == "old"


@pytest.mark.parametrize("fails", [False, True])
def test_rendered_pdf_image_is_closed_even_when_ocr_fails(tmp_path, monkeypatch, fails):
    source = tmp_path / "scan.pdf"
    make_pdf(source, [("", True)])
    picture = Image.new("RGB", (40, 40))
    monkeypatch.setattr(reader, "render_pdf_page", lambda path, page: picture)
    def read(image):
        if fails:
            raise RuntimeError("OCR失败")
        return "第一条 原文。", "rapidocr"
    monkeypatch.setattr(reader, "ocr_image", read)
    reader.extract_content(source)
    # 文件句柄关闭不等于图片像素内存关闭；原实现会调用image.close()。
    with pytest.raises(ValueError, match="closed"):
        picture.getpixel((0, 0))
