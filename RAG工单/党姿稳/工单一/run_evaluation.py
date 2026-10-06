# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
评估脚本：对10个验收问题进行 RAG vs 纯LLM 的对比测试，并生成评估报告
"""
import os
import json
import time
from datetime import datetime
from pdf_parser import build_chunks, load_chunks
from vector_store import VectorStore
from rag_chain import rag_answer, llm_only_answer
from config import EVAL_QUESTIONS, CHUNKS_FILE

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_all():
    """对所有验收问题进行测试，返回结果列表"""
    # 初始化知识库
    if os.path.exists(CHUNKS_FILE):
        chunks = load_chunks()
    else:
        chunks = build_chunks()
    store = VectorStore(chunks)

    results = []
    for i, item in enumerate(EVAL_QUESTIONS, 1):
        qid = item["id"]
        question = item["question"]
        print(f"\n{'='*60}")
        print(f"[{i}/{len(EVAL_QUESTIONS)}] 问题{qid}: {question}")
        print(f"{'='*60}")

        # RAG 回答
        rag_ans, contexts, rag_time = rag_answer(question, store)
        print(f"  RAG回答 ({rag_time:.2f}s): {rag_ans[:150]}...")

        # 纯 LLM 回答
        llm_ans, llm_time = llm_only_answer(question)
        print(f"  纯LLM回答 ({llm_time:.2f}s): {llm_ans[:150]}...")

        results.append({
            "id": qid,
            "question": question,
            "rag_answer": rag_ans,
            "rag_time": round(rag_time, 2),
            "llm_answer": llm_ans,
            "llm_time": round(llm_time, 2),
            "retrieved_contexts": [
                {"text": c[:300], "score": round(s, 4)} for c, s in contexts
            ],
        })

    return results


def save_report(results):
    """保存评估报告为 JSON 和 Markdown"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存 JSON
    json_path = os.path.join(OUTPUT_DIR, f"eval_report_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 生成 Markdown 报告
    md_path = os.path.join(OUTPUT_DIR, f"eval_report_{timestamp}.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# RAG问答系统评估报告\n\n")
        f.write(f"**工单编号：** 人工智能NLP-RAG-基于PDF文档的问答系统\n")
        f.write(f"**评估时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**测试问题数：** {len(results)}\n\n")
        f.write("## 对比分析\n\n")

        # 统计平均响应时间
        rag_times = [r["rag_time"] for r in results]
        llm_times = [r["llm_time"] for r in results]
        f.write(f"- RAG 平均响应时间: {sum(rag_times)/len(rag_times):.2f}s\n")
        f.write(f"- 纯LLM 平均响应时间: {sum(llm_times)/len(llm_times):.2f}s\n\n")

        f.write("## 详细结果\n\n")
        for r in results:
            f.write(f"### 问题 {r['id']}\n")
            f.write(f"**问题：** {r['question']}\n\n")
            f.write(f"**🤖 RAG回答** ({r['rag_time']}s):\n{r['rag_answer']}\n\n")
            f.write(f"**💬 纯LLM回答** ({r['llm_time']}s):\n{r['llm_answer']}\n\n")
            f.write(f"**📚 检索片段：**\n")
            for i, ctx in enumerate(r["retrieved_contexts"], 1):
                f.write(f"- 片段{i} (相似度{ctx['score']}): {ctx['text']}\n")
            f.write("\n---\n\n")

    print(f"\n✅ 评估报告已保存:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")
    return json_path, md_path


if __name__ == "__main__":
    print("=" * 60)
    print("RAG问答系统 - 验收问题评估")
    print("=" * 60)

    results = evaluate_all()
    save_report(results)

    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)
