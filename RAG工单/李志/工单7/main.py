import argparse, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ragkit import answer, demo_documents, load_documents, save, search, timed, tokens

questions = [
    ("公司的主营业务是什么？", "电子元器件 代理分销"), ("销售部由哪些部门构成？", "大客户 区域 销售支持"),
    ("募集资金用于什么？", "研发中心 营销网络 流动资金"), ("法定代表人是谁？", "赵佳生"),
    ("有哪些风险因素？", "供应链 市场竞争 汇率 技术"), ("是否有研发中心项目？", "研发中心"),
    ("是否建设营销网络？", "营销网络"), ("公司是否提供技术服务？", "技术服务"),
    ("资金是否补充流动资金？", "流动资金"), ("风险是否包含汇率变化？", "汇率变化")]

parser = argparse.ArgumentParser(description="工单7：10题功能测试与评估")
parser.add_argument("--docs", nargs="*"); parser.add_argument("--questions-pdf"); parser.add_argument("--demo", action="store_true"); args = parser.parse_args()
if args.questions_pdf:
    from pypdf import PdfReader
    raw = "\n".join((page.extract_text() or "") for page in PdfReader(args.questions_pdf).pages)
    parsed = re.findall(r"问题[：:]\s*(.+?)(?:参考?答案[：:]|答案[：:])\s*(.+?)(?=\n\s*问题[：:]|\Z)", raw, re.S)
    extracted = [(re.sub(r"\s+", " ", q).strip(), re.sub(r"\s+", " ", a).strip()) for q, a in parsed]
    questions = extracted + questions[:max(0, 10 - len(extracted))]
docs = load_documents(args.docs) if args.docs else demo_documents(); details = []
search("索引预热", docs, 1)
for question, expected in questions:
    results, elapsed = timed(lambda: search(question, docs, 4)); response = answer(question, results)
    wanted = set(tokens(expected)); hit = len(wanted & set(tokens(response))) / max(1, len(wanted))
    details.append({"question": question, "expected": expected, "answer": response, "recall": round(hit, 3), "elapsed_ms": elapsed})
report = {"count": len(details), "sample_pdf_questions": len(extracted) if args.questions_pdf else 0,
          "average_recall": round(sum(x["recall"] for x in details) / len(details), 3),
          "under_3s_rate": sum(x["elapsed_ms"] < 3000 for x in details) / len(details), "details": details}
save(Path(__file__).parent / "outputs/evaluation.json", report); print(json.dumps(report, ensure_ascii=False, indent=2))
