from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "共享"))
from rag_workbench.cli import load_documents
from rag_workbench.core import hybrid_search

def main():
    p = argparse.ArgumentParser(); p.add_argument("--input", required=True); p.add_argument("--query", required=True); p.add_argument("--top-k", type=int, default=5); p.add_argument("--output", type=Path)
    a = p.parse_args(); docs = load_documents([a.input]); result = {"query": a.query, "method": "hybrid", "results": hybrid_search(a.query, docs, a.top_k)}
    text = json.dumps(result, ensure_ascii=False, indent=2); print(text)
    if a.output: a.output.write_text(text, encoding="utf-8")
if __name__ == "__main__": main()
