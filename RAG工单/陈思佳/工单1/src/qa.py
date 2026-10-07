from dataclasses import dataclass
from typing import Callable, Iterable

from rag_core import Chunk
from retrieval import BM25Retriever, SearchResult


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    sources: tuple[dict[str, str | int | None], ...]


class LocalRAG:
    """Retrieval-only QA baseline; an LLM can be injected through answerer."""

    def __init__(self, chunks: Iterable[Chunk], answerer: Callable[[str, str], str] | None = None):
        self.retriever = BM25Retriever(chunks)
        self.answerer = answerer or self._extractive_answer

    def ask(self, question: str, top_k: int = 5) -> Answer:
        results = self.retriever.search(question, top_k=top_k)
        context = "\n\n".join(result.chunk.text for result in results)
        answer = self.answerer(question, context)
        sources = tuple(
            {"source": result.chunk.source, "page": result.chunk.page, "score": round(result.score, 4)}
            for result in results
        )
        return Answer(question=question, answer=answer, sources=sources)

    @staticmethod
    def _extractive_answer(question: str, context: str) -> str:
        if not context:
            return "未检索到相关内容。"
        sentences = [part.strip() for part in context.replace("\n", "").split("。") if part.strip()]
        question_tokens = set(retrieve_tokens(question))
        ranked = sorted(sentences, key=lambda sentence: len(question_tokens & set(retrieve_tokens(sentence))), reverse=True)
        return "。".join(ranked[:3]) + ("。" if ranked else "")


def retrieve_tokens(text: str) -> list[str]:
    import re
    return [token.lower() for token in re.findall(r"[一-鿿]|[A-Za-z0-9_]+", text)]
