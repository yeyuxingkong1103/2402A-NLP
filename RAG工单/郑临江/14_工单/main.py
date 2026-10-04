# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失工单
主程序：构建知识库 → 对 6 个测试问题检索 → 输出答案与精确度。
"""
import time

import config
from queue import enqueue_parse
from task_executor import TaskExecutor
from retriever import retrieve, rerank


def answer(query, kb):
    results = retrieve(kb, query)
    results = rerank(results, query)
    return results[0][2]["text"] if results else ""


def evaluate():
    # 触发解析并消费任务（对应 do_handle_task 完整流程）
    enqueue_parse(config.TEST_PDF, parser_id="knowledge_graph")
    ex = TaskExecutor()
    n = ex.consume(count=10)
    print(f"[main] 已索引 {n} 个文本块")

    correct = 0
    for i, q in enumerate(config.TEST_QUESTIONS, 1):
        t0 = time.perf_counter()
        ans = answer(q["question"], ex.kb)
        elapsed = time.perf_counter() - t0
        hit = any(k in ans for k in q["answer"][:6].split(".")[:1]) or \
              (q["answer"].replace(" ", "")[:8] in ans.replace(" ", ""))
        ok = q["answer"][:10].replace(" ", "") in ans.replace(" ", "") or \
             q["answer"].split(".")[0] in ans
        if ok:
            correct += 1
        print(f"[main] 问题{i} 命中={ok} 耗时={elapsed * 1000:.1f}ms")
        print(f"        标准答案: {q['answer']}")

    print(f"\n===== 测试精度: {correct}/{len(config.TEST_QUESTIONS)} "
          f"= {correct / len(config.TEST_QUESTIONS) * 100:.1f}% =====")
    return correct


if __name__ == "__main__":
    evaluate()
