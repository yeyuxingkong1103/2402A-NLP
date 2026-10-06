import json
from urllib import error, request

from backend.app.models import ChatResponse, Citation
from backend.app.vector_store import SearchResult


FALLBACK_ANSWER = "知识库中未找到相关依据。"


def build_fallback_response() -> ChatResponse:
    """构建无依据兜底响应。"""
    return ChatResponse(answer=FALLBACK_ANSWER, citations=[], fallback=True)


def build_citations(results: list[SearchResult], file_name_by_document_id: dict[str, str]) -> list[Citation]:
    """从检索结果构建引用。"""
    return [
        Citation(
            document_id=result.chunk.document_id,
            file_name=file_name_by_document_id.get(result.chunk.document_id, result.chunk.document_id),
            page=result.chunk.page,
            category=result.chunk.category,
            text=result.chunk.text,
            score=float(result.score),
        )
        for result in results
    ]


def build_answer_prompt(question: str, citations: list[Citation]) -> str:
    """构建带引用约束的提示词。"""
    evidence = "\n\n".join(
        f"[{index}] {citation.file_name} 第 {citation.page} 页 / {citation.category}\n{citation.text}"
        for index, citation in enumerate(citations, start=1)
    )
    return (
        "你是严谨的 RAG 问答助手。只基于以下证据回答。"
        "直接回答问题本身，不要输出“知识库中找到相关依据”这类套话。"
        "如果证据不足，只回答：知识库中未找到相关依据。"
        "答案末尾单独列出引用编号，例如：引用编号：[1][2]。\n\n"
        f"问题：{question}\n\n证据：\n{evidence}\n\n答案："
    )


def _call_ollama_chat(base_url: str, model: str, messages: list[dict[str, str]]) -> str:
    """向本地 Ollama 发送 chat 请求。"""
    payload = json.dumps({"model": model, "messages": messages, "stream": False}).encode("utf-8")
    chat_url = f"{base_url.rstrip('/')}/api/chat"
    chat_request = request.Request(
        chat_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(chat_request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"Ollama 调用失败: {exc}") from exc

    message = body.get("message", {}) if isinstance(body, dict) else {}
    return str(message.get("content", "")).strip()


class QaService:
    """检索问答服务。"""

    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url

    def answer(self, question: str, citations: list[Citation]) -> ChatResponse:
        """基于引用生成答案。"""
        if not citations:
            return build_fallback_response()
        prompt = build_answer_prompt(question, citations)
        answer = _call_ollama_chat(
            self.base_url,
            self.model,
            [
                {"role": "system", "content": "你必须只基于提供的证据回答。"},
                {"role": "user", "content": prompt},
            ],
        )
        if not answer:
            return build_fallback_response()
        return ChatResponse(answer=answer, citations=citations, fallback=False)
