import argparse, csv, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ragkit import demo_documents, load_documents, save, search

parser = argparse.ArgumentParser(description="工单3：PDF 表格解析与检索")
parser.add_argument("--docs", nargs="*"); parser.add_argument("--query", default="发行募集资金项目")
parser.add_argument("--demo", action="store_true"); args = parser.parse_args()
docs = load_documents(args.docs) if args.docs else demo_documents()
rows = []
for doc in docs:
    for line in doc["text"].splitlines():
        cells = [cell.strip() for cell in re.split(r"\s{2,}|\t|[|]", line) if cell.strip()]
        if len(cells) >= 2: rows.append({"source": doc["source"], "page": doc["page"], "cells": cells})
table_docs = docs + [{"source": row["source"], "page": row["page"], "chunk": 0, "text": " | ".join(row["cells"])} for row in rows]
result = {"tables": rows, "results": search(args.query, table_docs, 5)}
save(Path(__file__).parent / "outputs/tables_and_search.json", result)
print(f"解析表格行 {len(rows)} 条，返回 {len(result['results'])} 条检索结果。")
