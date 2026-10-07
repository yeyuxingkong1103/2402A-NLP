from main import markdown_table_to_text, index_table


def test_table_conversion_and_metadata():
    text = markdown_table_to_text("| 指标 | 2024 |\n|---|---|\n| 营收 | 100 |")
    assert "指标" in text and "营收" in text and "2024" in text
    record = index_table("report.pdf", 8, "| 指标 | 数值 |\n|---|---|\n| 毛利率 | 20% |")
    assert record.page == 8 and record.source == "report.pdf"
