from src.edu_rag_ingest.ingestion.cleaner import clean_text


def test_clean_text_removes_repeated_lines_and_extra_blank_lines():
    raw = "页眉\n第一行\n页眉\n第二行\n页眉\n\n\n第三行"

    cleaned = clean_text(raw)

    assert "页眉" not in cleaned
    assert "\n\n\n" not in cleaned
    assert "第一行" in cleaned
    assert "第二行" in cleaned
