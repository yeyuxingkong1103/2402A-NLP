from common.models import DocumentChunk
from main import answer_question


def test_basic_answer_keeps_evidence():
    answer = answer_question("营业收入是多少", [DocumentChunk(source="report.pdf", page=2, content="营业收入为 100 亿元")])
    assert "100" in answer.text
    assert answer.citations[0].page == 2
