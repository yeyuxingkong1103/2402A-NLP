"""人工智能NLP-RAG-基于PDF文档的问答系统。

Baseline pipeline: PDF pages -> chunks -> embeddings -> retrieval -> cited answer.
The model client is injected so the indexing layer is independent from a vendor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_rag import dense_search, make_chunks, reciprocal_rank_fusion, write_json


class ChatModel(Protocol):
    def invoke(self, prompt: str) -> str: ...


def load_pdf(path: str | Path) -> list[dict[str, Any]]:
    import pymupdf

    rows = []
    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, 1):
            rows.append({"source": str(path), "page": page_number, "text": page.get_text("text", sort=True)})
    return rows


class PdfRagQA:
    def __init__(self, chunks, llm: ChatModel | None = None):
        self.chunks = chunks
        self.llm = llm

    def retrieve(self, question: str, top_k: int = 5):
        return reciprocal_rank_fusion(dense_search(question, self.chunks, top_k * 2), top_k=top_k)

    def build_prompt(self, question: str, hits) -> str:
        context = "\n\n".join(
            f"[{index}] {chunk.text}\n来源：{chunk.source} 第{chunk.page}页"
            for index, (chunk, _) in enumerate(hits, 1)
        )
        return (
            "你是严谨的文档问答助手。只能依据给定上下文回答；证据不足时明确说明。"
            "答案中使用[编号]标记引用。\n\n问题：" + question + "\n\n上下文：\n" + context
        )

    def answer(self, question: str, top_k: int = 5) -> dict[str, Any]:
        hits = self.retrieve(question, top_k)
        prompt = self.build_prompt(question, hits)
        answer = self.llm.invoke(prompt) if self.llm else "\n".join(chunk.text for chunk, _ in hits[:2])
        return {
            "question": question,
            "answer": answer,
            "citations": [
                {"source": chunk.source, "page": chunk.page, "chunk_id": chunk.chunk_id, "score": score}
                for chunk, score in hits
            ],
        }


def build_index(pdf_path: str, output_path: str | None = None) -> PdfRagQA:
    rag = PdfRagQA(make_chunks(load_pdf(pdf_path), chunk_size=900, overlap=120))
    if output_path:
        write_json(output_path, [chunk.to_dict() for chunk in rag.chunks])
    return rag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("question")
    parser.add_argument("--index-output")
    args = parser.parse_args()
    print(json.dumps(build_index(args.pdf, args.index_output).answer(args.question), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
