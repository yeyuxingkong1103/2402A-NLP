import argparse, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ragkit import answer, demo_documents, load_documents, save, search

def rewrite(query):
    aliases = {"这家公司": "公司", "干什么": "主营业务", "钱用哪": "募集资金用途", "谁管": "法定代表人"}
    rewritten = query
    for source, target in aliases.items(): rewritten = rewritten.replace(source, target)
    parts = [part.strip() for part in re.split(r"[，,；;]|以及|并且|和", rewritten) if part.strip()]
    return rewritten, parts

parser = argparse.ArgumentParser(description="工单5：Query 理解与多轮优化")
parser.add_argument("--docs", nargs="*"); parser.add_argument("--query", default="这家公司干什么，钱用哪？")
parser.add_argument("--demo", action="store_true"); args = parser.parse_args()
docs = load_documents(args.docs) if args.docs else demo_documents(); rewritten, subqueries = rewrite(args.query)
answers = [{"subquery": item, "answer": answer(item, search(item, docs, 4))} for item in subqueries]
payload = {"original": args.query, "rewritten": rewritten, "intent": "复合信息查询", "subqueries": answers}
save(Path(__file__).parent / "outputs/query_understanding.json", payload); print(json.dumps(payload, ensure_ascii=False, indent=2))
