from main import describe_image, make_image_record


def test_image_record_keeps_page_path_and_ocr(tmp_path):
    path = tmp_path / "chart.png"
    path.write_bytes(b"not-a-real-image")
    record = make_image_record("report.pdf", 6, path, ocr_text="净利润趋势")
    assert record.page == 6 and record.path == str(path)
    assert record.text == "净利润趋势"


def test_missing_visual_backend_does_not_fake_description(tmp_path):
    path = tmp_path / "chart.png"
    path.write_bytes(b"data")
    assert describe_image(path) == ""
