"""工单 01：基础问答。"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import Settings
from common.index import DocumentIndex
from common.llm import answer_with_backend
from common.models import Answer, DocumentChunk
from common.parsing import chunk_pages, read_document


def safe_write(text: str, stream=None) -> None:
    target = stream or sys.stdout
    encoding = getattr(target, "encoding", None) or "utf-8"
    line = text + "\n"
    try:
        line.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        line = line.encode(encoding, errors="backslashreplace").decode(encoding, errors="replace")
    try:
        target.write(line)
    except UnicodeEncodeError:
        target.write(line.encode(encoding, errors="backslashreplace").decode(encoding, errors="replace"))


def configure_stdout() -> None:
    stream = sys.stdout
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def build_index(input_path: str | Path, index_path: str | Path | None = None) -> DocumentIndex:
    source = Path(input_path)
    index = DocumentIndex()
    for chunk in chunk_pages(read_document(source)):
        index.add_chunk(DocumentChunk(source=source.name, page=chunk.page, content=chunk.content))
    if index_path is not None:
        index.save(index_path)
    return index


def answer_question(question: str, chunks: list[DocumentChunk], settings: Settings | None = None) -> Answer:
    index = DocumentIndex()
    for chunk in chunks:
        index.add_chunk(chunk)
    results = index.search(question, top_k=(settings or Settings.from_env()).rag_top_k)
    return answer_with_backend(question, results, settings)


def main(argv: list[str] | None = None) -> int:
    import argparse

    configure_stdout()
    parser = argparse.ArgumentParser(description="基础文档问答")
    parser.add_argument("question")
    parser.add_argument("input_path")
    args = parser.parse_args(argv)
    try:
        index = build_index(args.input_path)
        answer = answer_question(args.question, index.chunks)
    except Exception as exc:
        safe_write(f"错误：{exc}", sys.stderr)
        return 1
    safe_write(answer.text)
    for citation in answer.citations:
        safe_write(f"来源：{citation.source} 第{citation.page}页")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
