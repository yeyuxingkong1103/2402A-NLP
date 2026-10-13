from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import Document, build_graph, chunk_text, graph_search, hybrid_search, lexical_search, read_text


def load_documents(paths: list[str]) -> list[Document]:
    result = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"}]
        else:
            files = [path]
        for file in files:
            result.extend(chunk_text(read_text(file), str(file)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--query", required=True)
    parser.add_argument("--method", choices=["lexical", "hybrid", "graph"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    docs = load_documents(args.paths)
    if args.method == "lexical":
        rows = lexical_search(args.query, docs, args.top_k)
    elif args.method == "graph":
        rows = graph_search(args.query, build_graph(docs), docs, args.top_k)
    else:
        rows = hybrid_search(args.query, docs, args.top_k)
    output = json.dumps({"query": args.query, "method": args.method, "results": rows}, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
