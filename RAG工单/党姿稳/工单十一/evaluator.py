# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-Embedding 模型微调任务
# 模块职责：工单步骤 6「创建评估器」——用检索指标（Recall@K / MRR / nDCG）量化嵌入模型的检索效果

import math                                                     # 导入 math 用于 nDCG 的对数折扣
import numpy as np                                              # 导入 numpy 用于向量相似度计算
import embedder                                                 # 导入嵌入模型编码模块
import config                                                   # 导入本工单配置模块


class RetrievalEvaluator(object):                               # 定义检索效果评估器类
    """在给定语料候选池上评估嵌入模型的检索质量，支持微调前后对比。"""

    def __init__(self, chunks, chunk_vecs, ks=None, ndcg_k=None):  # 构造方法
        self.chunks = chunks                                    # 保存候选语料块列表
        self.vecs = chunk_vecs                                  # 保存语料向量矩阵（行号与 chunks 对齐）
        self.ks = ks or config.EVAL_KS                         # 评估使用的 K 值列表
        self.ndcg_k = ndcg_k or config.NDCG_K                   # nDCG 截断位置
        self.id2idx = {c["id"]: i for i, c in enumerate(chunks)}  # 建立块 ID 到行号的映射

    def _rank_of(self, query_vec, pos_id):                      # 定义正确块排名计算函数
        """返回正确语料块在该查询下的相似度排名（从 0 开始）；未命中语料时返回 -1。"""
        pos_idx = self.id2idx.get(pos_id)                       # 定位正确块的行号
        if pos_idx is None:                                     # 正确块不在候选池中
            return -1                                           # 返回 -1 表示无法评估
        scores = self.vecs.dot(query_vec)                       # 计算与全部候选块的余弦相似度
        return int(np.sum(scores > scores[pos_idx]))            # 统计比正确块分数高的块数量即为排名

    def evaluate(self, model, eval_pairs, tag="model"):         # 定义评估主函数
        """对给定模型与评估集执行检索评估，返回汇总指标与逐题明细。"""
        questions = [p["question"] for p in eval_pairs]         # 收集全部评估问题
        q_vecs = embedder.encode(model, questions,              # 编码问题（查询侧加 m3e 官方前缀）
                                 batch_size=config.EVAL_BATCH,  # 指定批大小
                                 is_query=True)                 # 标记为查询侧
        ranks = []                                              # 每个问题的正确块排名
        details = []                                            # 逐题明细，便于人工核验
        for p, qv in zip(eval_pairs, q_vecs):                   # 遍历评估问题与查询向量
            r = self._rank_of(qv, p["chunk_id"])                # 计算正确块排名
            ranks.append(r)                                     # 收集排名
            details.append({"question": p["question"],          # 问题原文
                            "source": p["source"],              # 正确来源
                            "page": p["page"],                  # 正确页码
                            "rank": r})                         # 正确块排名（0 表示排第一）
        valid = [r for r in ranks if r >= 0]                    # 过滤无法评估的样本
        n = len(valid)                                          # 有效样本数
        metrics = {}                                            # 初始化指标字典
        for k in self.ks:                                       # 遍历每个 K 值
            hit = sum(1 for r in valid if r < k)                # 统计排名小于 K 的命中数
            metrics["recall@%d" % k] = round(hit / float(n), 4) if n else 0.0  # 计算 Recall@K
        metrics["mrr"] = round(sum(1.0 / (r + 1) for r in valid) / float(n), 4) if n else 0.0  # 计算 MRR
        ndcg = 0.0                                              # nDCG 累计值
        for r in valid:                                         # 遍历每个正确块排名
            if r < self.ndcg_k:                                 # 排名在截断范围内才计分
                ndcg += 1.0 / math.log2(r + 2)                  # 单个相关文档的 DCG 增益
        metrics["ndcg@%d" % self.ndcg_k] = round(ndcg / float(n), 4) if n else 0.0  # 计算 nDCG
        metrics["mean_rank"] = round(sum(valid) / float(n), 2) if n else -1.0  # 平均排名（越小越好）
        metrics["top%d_hit" % max(self.ks)] = sum(1 for r in valid if r < max(self.ks))  # 最大 K 的命中数
        metrics["eval_size"] = n                                # 有效评估样本数
        print("[信息] %s 评估结果：%s" % (tag, self._fmt(metrics)))  # 输出指标摘要
        return {"tag": tag, "metrics": metrics, "details": details}  # 返回结构化结果

    @staticmethod                                               # 定义为静态方法
    def _fmt(metrics):                                          # 定义指标格式化函数
        """把指标字典格式化为便于阅读的一行字符串。"""
        keys = ["recall@1", "recall@5", "recall@10", "mrr",    # 依次展示核心指标
                "ndcg@%d" % config.NDCG_K, "mean_rank"]         # 平均排名与 nDCG
        return " | ".join("%s=%s" % (k, metrics.get(k))         # 逐项拼接
                          for k in keys if k in metrics)        # 跳过不存在的键


def compare(before, after):                                     # 定义微调前后对比函数
    """对比两次评估的指标，返回逐指标差值表与达标结论。"""
    rows = []                                                   # 初始化对比行列表
    keys = [k for k in before["metrics"]                       # 取两版共有的指标键
            if k in after["metrics"] and k not in ("eval_size", "top%d_hit" % max(config.EVAL_KS))]  # 排除计数类
    for k in keys:                                              # 遍历每个指标
        b = before["metrics"][k]                                # 微调前的值
        a = after["metrics"][k]                                 # 微调后的值
        rows.append({"metric": k,                               # 指标名
                     "before": b,                               # 微调前
                     "after": a,                                # 微调后
                     "delta": round(a - b, 4),                  # 绝对变化
                     "better": _is_better(k, b, a)})            # 是否变好
    improved = sum(1 for r in rows if r["better"])              # 统计改善的指标数
    print("[信息] 对比完成：%d/%d 项指标改善" % (improved, len(rows)))  # 输出对比结论
    return {"rows": rows,                                       # 逐指标对比行
            "improved": improved,                               # 改善指标数
            "total": len(rows)}                                 # 指标总数


def _is_better(metric, before, after):                          # 定义「是否变好」判定函数
    """按指标语义判断变化方向：平均排名越小越好，其余指标越大越好。"""
    if metric == "mean_rank":                                   # 平均排名为负向指标
        return after < before                                   # 变小才算改善
    return after > before                                       # 其余指标变大才算改善
