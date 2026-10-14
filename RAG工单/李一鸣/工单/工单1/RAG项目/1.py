"""Work order 01: a baseline PDF question-answering RAG pipeline."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Chunk, citation, dense_search, make_chunks


def extract_pdf_pages(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Extract one record per PDF page. PyMuPDF is intentionally optional."""
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install PyMuPDF or replace this loader with your PDF parser") from exc
    pages = []
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            pages.append({"source": str(pdf_path), "page": page_number, "text": page.get_text("text")})
    return pages


class PdfRagQA:
    def __init__(self, chunks: list[Chunk], llm: Any | None = None):
        self.chunks = chunks
        self.llm = llm

    def retrieve(self, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        return [{"chunk": item.to_dict(), "score": score} for item, score in dense_search(question, self.chunks, top_k)]

    def build_prompt(self, question: str, hits: list[dict[str, Any]]) -> str:
        context = "\n\n".join(
            f"[{index}] {hit['chunk']['text']}\nSource: {hit['chunk']['source']} p.{hit['chunk']['page']}"
            for index, hit in enumerate(hits, start=1)
        )
        return (
            "You answer only from the supplied PDF context. If the context is insufficient, say so. "
            "Cite sources as [n].\n\nQuestion: " + question + "\n\nContext:\n" + context
        )

    def answer(self, question: str, top_k: int = 5) -> dict[str, Any]:
        hits = self.retrieve(question, top_k)
        prompt = self.build_prompt(question, hits)
        if self.llm is None:
            answer = "\n".join(hit["chunk"]["text"] for hit in hits[:2]) or "No relevant evidence found."
        else:
            answer = self.llm.invoke(prompt)
        return {
            "question": question,
            "answer": answer,
            "citations": [citation(Chunk(**hit["chunk"])) for hit in hits],
            "prompt": prompt,
        }


def build_index(pdf_path: str | Path, chunk_size: int = 900, overlap: int = 120) -> PdfRagQA:
    pages = extract_pdf_pages(pdf_path)
    return PdfRagQA(make_chunks(pages, chunk_size=chunk_size, overlap=overlap))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("question")
    args = parser.parse_args()
    print(json.dumps(build_index(args.pdf).answer(args.question), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
