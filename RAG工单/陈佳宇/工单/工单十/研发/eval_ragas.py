# -*- coding: utf-8 -*-
# eval_ragas.py 工单9 评估脚本
def calculate_context_recall(retrieved_contexts: list[str], reference_answers: list[str]) -> float:
    """
    context_recall：参考答案需要的信息，有多少被检索上下文覆盖
    """
    total_ref_tokens = 0
    matched_tokens = 0
    for ref_ans in reference_answers:
        ref_words = set(ref_ans.replace("，", "").replace("。", "").split())
        total_ref_tokens += len(ref_words)
        for ctx in retrieved_contexts:
            ctx_words = set(ctx.replace("，", "").replace("。", "").split())
            matched_tokens += len(ref_words & ctx_words)
    if total_ref_tokens == 0:
        return 0.0
    return round(matched_tokens / total_ref_tokens, 4)


def calculate_context_precision(retrieved_contexts: list[str], reference_answers: list[str]) -> float:
    """
    context_precision：检索出来的上下文里面，真正相关的内容占比
    """
    relevant = 0
    total = len(retrieved_contexts)
    for ctx in retrieved_contexts:
        for ref in reference_answers:
            if ctx in ref or ref in ctx:
                relevant +=1
                break
    if total ==0:
        return 0.0
    return round(relevant / total,4)


if __name__ == "__main__":
    # 评测样例，可直接在main.py调用
    sample_questions = [
        "武汉力源信息技术股份有限公司组织结构图中,销售部有几个部门构成,其中大客户销售部有几个销售处构成?",
        "公司主要竞争对手是谁？",
        "汽车电子业务规划是什么？"
    ]
    sample_reference = [
        "销售部包含大客户销售部、中小客户销售部；大客户销售部下辖华北、华东、华南、西南4个销售处",
        "竞争对手是中电港、文晔科技",
        "汽车电子业务是未来重点方向，计划加大研发投入，拓展国内车企客户"
    ]
    retrieved = [
        "销售部包含大客户销售部、中小客户销售部",
        "大客户销售部下辖华北销售处、华东销售处、华南销售处、西南销售处",
        "武汉力源信息技术股份有限公司竞争对手为中电港、文晔科技",
        "汽车电子业务是公司未来重点发展方向，计划加大研发投入，拓展国内车企客户。"
    ]
    recall = calculate_context_recall(retrieved, sample_reference)
    precision = calculate_context_precision(retrieved, sample_reference)
    print(f"==== 工单9 RAG评估指标 ====")
    print(f"context_recall(上下文召回): {recall}")
    print(f"context_precision(上下文精度): {precision}")
