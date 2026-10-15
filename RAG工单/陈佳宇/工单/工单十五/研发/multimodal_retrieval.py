# -*- coding: utf-8 -*-
# multimodal_retrieval.py 工单15 跨模态检索核心模块
import re
from config import FIGURE_REF_PATTERN,TOP_K,RRF_ALPHA_QUERY,RRF_ALPHA_AUGMENT

def detect_figure_reference(query:str):
    """
    检测用户问题中图纸引用；返回是否包含图纸引用，提取页码、图号
    返回: has_figure(bool), page_no, figure_no
    """
    match = FIGURE_REF_PATTERN.search(query)
    if not match:
        return False, None, None
    page_no = match.group(1)
    fig_no = match.group(2) or match.group(3)
    return True, page_no, fig_no

def build_augmented_query(origin_query:str, page_no, figure_no):
    """生成图纸增强检索query"""
    aug_parts = [origin_query]
    if page_no and figure_no:
        aug_parts.append(f"参考第{page_no}页图{figure_no}图纸，部件编号、部件位置关系、装置结构图")
    elif figure_no:
        aug_parts.append(f"参考图{figure_no}图纸，部件编号、部件位置关系、装置结构图")
    return " ".join(aug_parts)

def reciprocal_rank_fusion(list_a:list, list_b:list, alpha_a=60, alpha_b=60):
    """
    RRF 秩融合两路召回结果
    list_a / list_b: [{"score":float,"content":str,"source":str},...]
    """
    rank_dict = {}
    for rank, item in enumerate(list_a):
        score = 1.0/(rank + alpha_a)
        key = item["content"]
        rank_dict[key] = rank_dict.get(key,0.0) + score
    for rank, item in enumerate(list_b):
        score = 1.0/(rank + alpha_b)
        key = item["content"]
        rank_dict[key] = rank_dict.get(key,0.0) + score
    # 重排序
    combined = []
    content_map = {}
    for it in list_a+list_b:
        content_map[it["content"]] = it
    for content, rrf_score in rank_dict.items():
        origin_item = content_map[content]
        combined.append({**origin_item, "rrf_score":rrf_score})
    combined.sort(key=lambda x:x["rrf_score"], reverse=True)
    return combined

def two_path_recall(ragflow_api_func, origin_query:str, kb_id:str):
    """
    两路召回：原始query、图纸增强query；没有图纸引用直接退化为单路
    ragflow_api_func:调用ragflow检索接口函数
    """
    has_fig, page, fig = detect_figure_reference(origin_query)
    recall_origin = ragflow_api_func(origin_query, kb_id, top_k=TOP_K)
    if not has_fig:
        # 没有图纸引用，直接返回原始支路
        return recall_origin

    aug_query = build_augmented_query(origin_query, page, fig)
    recall_aug = ragflow_api_func(aug_query, kb_id, top_k=TOP_K)
    fused_result = reciprocal_rank_fusion(
        recall_origin, recall_aug,
        alpha_a=RRF_ALPHA_QUERY,
        alpha_b=RRF_ALPHA_AUGMENT
    )
    return fused_result

def multimodal_rerank(candidate_list:list):
    """
    简易图文重排：提升包含【图X、第XX页】片段的打分优先级
    """
    out = []
    for item in candidate_list:
        s = item["content"]
        bonus = 0.0
        if "图" in s:
            bonus += 0.25
        if "部件" in s and any(str(i) in s for i in range(1,30)):
            bonus +=0.15
        new_score = item.get("rrf_score",item.get("score",0)) + bonus
        out.append({**item,"rerank_score":new_score})
    out.sort(key=lambda x:x["rerank_score"],reverse=True)
    return out
