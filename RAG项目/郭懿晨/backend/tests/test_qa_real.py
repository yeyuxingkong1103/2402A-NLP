from backend.app.models import Citation
from backend.app.qa import FALLBACK_ANSWER, QaService, build_answer_prompt, build_fallback_response


class FakeUrlResponse:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.payload.encode("utf-8")


class FakeRequestModule:
    def __init__(self) -> None:
        self.calls = []

    def Request(self, url, data, headers, method):
        self.calls.append({"url": url, "data": data, "headers": headers, "method": method})
        return {"url": url, "data": data, "headers": headers, "method": method}

    def urlopen(self, req, timeout):
        self.calls.append({"req": req, "timeout": timeout})
        return FakeUrlResponse('{"message": {"content": "根据证据回答。"}}')


def test_qa_service_returns_fallback_when_no_citations():
    response = build_fallback_response()

    assert response.answer == FALLBACK_ANSWER
    assert response.citations == []
    assert response.fallback is True


def test_build_answer_prompt_contains_citation_context():
    citation = Citation(
        document_id="doc-1",
        file_name="a.pdf",
        page=5,
        category="正文",
        text="真实依据",
    )

    prompt = build_answer_prompt("问题是什么", [citation])

    assert "a.pdf 第 5 页" in prompt
    assert "只基于以下证据回答" in prompt


def test_qa_service_uses_ollama_chat(monkeypatch):
    fake_request = FakeRequestModule()
    monkeypatch.setattr("backend.app.qa.request", fake_request)

    service = QaService(model="deepseek-r1:7b", base_url="http://localhost:11434")
    response = service.answer(
        "问题是什么",
        [
            Citation(
                document_id="doc-1",
                file_name="a.pdf",
                page=5,
                category="正文",
                text="真实依据",
            )
        ],
    )

    assert response.answer == "根据证据回答。"
    assert response.fallback is False
    assert fake_request.calls[0]["method"] == "POST"
    assert fake_request.calls[0]["url"] == "http://localhost:11434/api/chat"
