# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-LightRAG 优化任务
# 模块职责：提供多路召回融合（RRF）与结果去冗余（MMR）两个通用算法，供 RAG 与 LightRAG 共用

import numpy as np                                              # 导入 numpy 用于向量运算
import config                                                   # 导入本工单配置模块
from logger import LOGGER                                       # 导入统一日志对象


def rrf_fuse(rank_lists, weights=None, k=None):                 # 定义 RRF 融合函数
    """倒数排名融合：把多路召回结果合并为一个排序，score = Σ w / (k + rank)。"""
    k = k or config.RRF_K                                       # 取配置中的平滑常数
    weights = weights or [1.0] * len(rank_lists)                # 未指定权重时各路等权
    scores = {}                                                 # 文档下标 -> 融合得分
    for w, ranked in zip(weights, rank_lists):                  # 遍历每一路召回结果
        for rank, idx in enumerate(ranked):                     # 遍历该路的排序位置
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank + 1.0)  # 累加倒数排名得分
    fused = sorted(scores.items(), key=lambda x: -x[1])         # 按融合得分降序排列
    return [idx for idx, _ in fused], scores                    # 返回排序结果与得分字典


def mmr_dedup(cand_idx, cand_scores, vecs, top_k, lam=None):    # 定义 MMR 去冗余函数
    """最大边际相关：在相关性与多样性之间折中，剔除近重复块，返回最终选中的行号列表。"""
    lam = config.MMR_LAMBDA if lam is None else lam             # 取配置中的相关性权重
    if not cand_idx:                                            # 候选为空直接返回
        return []                                               # 返回空列表
    max_s = max(cand_scores.values()) if cand_scores else 1.0    # 取最高分用于归一化
    min_s = min(cand_scores.values()) if cand_scores else 0.0    # 取最低分用于归一化
    span = (max_s - min_s) or 1.0                               # 归一化分母（防除零）
    rel = {i: (cand_scores.get(i, 0.0) - min_s) / span for i in cand_idx}  # 归一化相关性得分
    selected = []                                               # 已选中的行号
    pool = list(cand_idx)                                       # 待选池
    while pool and len(selected) < top_k:                       # 直到选够或候选耗尽
        best, best_val = None, None                             # 本轮最优候选与得分
        for i in pool:                                          # 遍历候选池
            if not selected:                                    # 首个位置只看相关性
                val = rel[i]                                    # 直接用归一化相关性
            else:                                               # 后续位置扣除与已选的最大相似度
                sims = vecs[selected].dot(vecs[i])              # 与所有已选块的余弦相似度
                val = lam * rel[i] - (1.0 - lam) * float(np.max(sims))  # MMR 目标函数
            if best_val is None or val > best_val:              # 记录更优候选
                best, best_val = i, val                         # 更新最优
        selected.append(best)                                   # 选中本轮最优
        pool.remove(best)                                       # 从候选池移除
    return selected                                             # 返回去冗余后的行号列表


def retrieve_summary(tag, n_before, n_after):                   # 定义检索摘要日志函数
    """输出去冗余前后的候选数量，便于观察 MMR 的实际作用。"""
    LOGGER.info("%s 融合候选 %d → MMR 后 %d", tag, n_before, n_after)  # 打印一行摘要
