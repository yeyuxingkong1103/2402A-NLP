# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-优化技术图纸与文本的跨模态检索流程工单
主程序：优化前（仅文本检索）vs 优化后（跨模态检索）对比 6 个测试问题。
"""
import config
from parser import extract_pdf, build_blocks, find_figure_block
from query_understanding import detect_visual_reference, rewrite_query, is_visual_query
from retriever import vector_search, hybrid_retrieve
from reranker import rerank
from prompt import build_prompt


def answer_before(blocks, query):
    """优化前：仅文本向量检索（故障基线）。"""
    results = vector_search(blocks, query, top_k=config.TOP_K)
    return [b for _, i, b in results]


def answer_after(blocks, query):
    """优化后：查询理解 + 多路召回融合 + 跨模态重排。"""
    ref = detect_visual_reference(query)
    fig_block = None
    if ref and ref.get("figure"):
        fig_block = find_figure_block(blocks, ref["figure"])
    enhanced = rewrite_query(query, fig_block) if is_visual_query(query) else None
    results = hybrid_retrieve(blocks, query, enhanced)
    return rerank(results, query)


def hit(target, answer_blocks):
    """判断 Top 结果是否包含目标图纸引用或答案关键词。"""
    for b in answer_blocks:
        if any(k in b.get("image_desc", "") + b.get("text", "") for k in target[:4]):
            return True
    return False


def main():
    pages = extract_pdf(config.TEST_PDF)
    blocks = build_blocks(pages)
    print(f"[main] 构建图文块 {len(blocks)} 个（含图纸图文块）")

    before_ok = after_ok = 0
    print("\n===== 优化前后对比 =====")
    for i, q in enumerate(config.TEST_QUESTIONS, 1):
        b = answer_before(blocks, q["question"])
        a = answer_after(blocks, q["question"])
        b_hit = hit(q["answer"], b)
        a_hit = hit(q["answer"], a)
        before_ok += b_hit
        after_ok += a_hit
        print(f"问题{i}({q['kind']}): 优化前={'√' if b_hit else '×'} "
              f"优化后={'√' if a_hit else '×'} | 答案:{q['answer']}")
        if q["kind"] == "image":
            top1 = a[0] if a else {}
            print(f"        Top-1 图文块: 第{top1.get('page','?')}页 "
                  f"图{top1.get('figures','?')} is_image={top1.get('is_image')}")

    print(f"\n===== 准确率：优化前 {before_ok}/{len(config.TEST_QUESTIONS)} "
          f"= {before_ok/len(config.TEST_QUESTIONS)*100:.0f}%  "
          f"优化后 {after_ok}/{len(config.TEST_QUESTIONS)} "
          f"= {after_ok/len(config.TEST_QUESTIONS)*100:.0f}% =====")

    # 展示优化后的 Prompt 示例
    q = config.TEST_QUESTIONS[2]["question"]
    ctx = answer_after(blocks, q)[:3]
    print("\n===== 优化后 Prompt 示例 =====")
    print(build_prompt(q, ctx))


if __name__ == "__main__":
    main()
