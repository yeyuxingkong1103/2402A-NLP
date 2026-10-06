# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""RAG 评估体系：LLM-as-Judge + 引用率，并与纯 LLM 做对比

选用 LLM-as-Judge 方案（RAG 评估的主流做法之一），两项核心指标：
  grounded  答案的事实性内容能否在检索到的上下文中找到依据（衡量有没有编造）
  relevant  答案是否正面回答了问题
对 RAG 答案与纯 LLM 答案用同一套裁判、同一份上下文打分，保证对比公平。
RAG 答案额外统计页码引用率。
"""
import json
import re

from config import TEST_QUESTIONS
from rag_engine import RAGEngine, chat

JUDGE_PROMPT = """你是 RAG 评估裁判。请依据给定上下文，对回答打分。

【问题】
{question}

【检索到的上下文】
{context}

【待评估回答】
{answer}

只输出一行 JSON，不要任何多余文字：
{{"grounded": 0, "relevant": 0, "reason": "一句话理由"}}

grounded 取 1 表示回答中的事实性内容都能在上下文中找到依据，0 表示存在上下文里没有的内容。
relevant 取 1 表示回答正面回答了问题，0 表示答非所问。
上下文为空时，grounded 一律取 0。"""

_PAGE_CITE = re.compile(r"第\s*\d+\s*页|page\s*\d+|p\.?\s*\d+", re.I)


def _judge(question, context, answer):
    # max_tokens 不能给小：裁判同样走推理模型，额度被 reasoning 吃光就返回空串。
    raw = chat(JUDGE_PROMPT.format(question=question, context=context or "（无）",
                                   answer=answer), temperature=0.0, max_tokens=2048)
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return {"grounded": 0, "relevant": 0, "reason": "裁判输出无法解析"}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"grounded": 0, "relevant": 0, "reason": "裁判输出无法解析"}
    return {
        "grounded": int(bool(data.get("grounded", 0))),
        "relevant": int(bool(data.get("relevant", 0))),
        "reason": str(data.get("reason", ""))[:120],
    }


def evaluate(engine, questions=None):
    """逐题跑 RAG 与纯 LLM，返回明细行。"""
    rows = []
    for item in questions or TEST_QUESTIONS:
        question = item["question"]
        rag = engine.answer(question)
        plain = engine.answer_without_rag(question)
        context = "\n\n".join(
            f"[第{c['page']}页] {c['text']}" for c, _ in rag["contexts"]
        )
        rows.append({
            "id": item["id"],
            "question": question,
            "pages": [c["page"] for c, _ in rag["contexts"]],
            "rag_answer": rag["answer"],
            "rag_elapsed": rag["elapsed"],
            "rag_cited": bool(_PAGE_CITE.search(rag["answer"])),
            "rag_score": _judge(question, context, rag["answer"]),
            "plain_answer": plain["answer"],
            "plain_elapsed": plain["elapsed"],
            "plain_cited": bool(_PAGE_CITE.search(plain["answer"])),
            "plain_score": _judge(question, context, plain["answer"]),
        })
        print(f"[评估] ID {item['id']} 完成", flush=True)
    return rows


def _avg(rows, key, field):
    return sum(r[key][field] for r in rows) / len(rows) if rows else 0.0


def report(rows):
    """返回可直接写进文档的评估报告文本。"""
    n = len(rows)
    lines = [
        "=" * 68,
        "RAG 评估报告（LLM-as-Judge）",
        "=" * 68,
        f"题目数：{n}",
        "",
        f"{'指标':<22}{'RAG（BGE-M3）':<20}{'纯 LLM':<16}",
        "-" * 68,
        f"{'grounded（有据可依）':<20}{_avg(rows,'rag_score','grounded'):<22.0%}"
        f"{_avg(rows,'plain_score','grounded'):<16.0%}",
        f"{'relevant（答到点上）':<20}{_avg(rows,'rag_score','relevant'):<22.0%}"
        f"{_avg(rows,'plain_score','relevant'):<16.0%}",
        f"{'页码引用率':<22}{sum(r['rag_cited'] for r in rows)/n:<22.0%}"
        f"{sum(r['plain_cited'] for r in rows)/n:<16.0%}",
        f"{'平均响应耗时(s)':<19}{sum(r['rag_elapsed'] for r in rows)/n:<22.2f}"
        f"{sum(r['plain_elapsed'] for r in rows)/n:<16.2f}",
        "-" * 68,
        "",
        "逐题明细：",
    ]
    for r in rows:
        lines += [
            f"\n[ID {r['id']}] {r['question']}",
            f"  检索页码 {r['pages']}",
            f"  RAG   {r['rag_elapsed']:.2f}s 有据={r['rag_score']['grounded']} "
            f"切题={r['rag_score']['relevant']} 引用={'是' if r['rag_cited'] else '否'}"
            f"  | {r['rag_score']['reason']}",
            f"  纯LLM {r['plain_elapsed']:.2f}s 有据={r['plain_score']['grounded']} "
            f"切题={r['plain_score']['relevant']}"
            f"  | {r['plain_score']['reason']}",
            f"  RAG 回答：{r['rag_answer'][:120]}",
            f"  纯LLM回答：{r['plain_answer'][:120]}",
        ]
    return "\n".join(lines)


def main(pdf_path):
    """命令行跑一遍完整评估：python evaluator.py [PDF路径]"""
    from pathlib import Path

    from document import load_pdf, build_chunks
    from vector_store import BGEM3VectorStore

    pages, tables = load_pdf(pdf_path)
    store = BGEM3VectorStore()
    store.build(build_chunks(pages, tables))

    rows = evaluate(RAGEngine(store))
    text = report(rows)
    print(text)
    out = Path(__file__).parent / "eval_result.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已写入 {out}")
    return rows


if __name__ == "__main__":
    import sys

    DEFAULT_PDF = r"D:\BW\RAG 工单\附件\招股说明书1.pdf"
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF)
