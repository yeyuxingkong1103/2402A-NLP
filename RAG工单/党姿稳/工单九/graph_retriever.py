# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 模块职责：图谱侧检索——实体链接、邻域扩展、关联文本块与关系三元组的召回

import re                                                       # 导入正则用于实体名清洗
import config                                                   # 导入本工单配置模块
from bm25 import tokenize                                       # 复用统一分词函数做实体匹配
from timing import LOGGER                                       # 导入统一日志对象


class GraphRetriever(object):                                   # 定义图谱检索器类
    """基于知识图谱做实体链接与邻域扩展，为生成阶段补充结构化证据。"""

    def __init__(self, graph, chunks):                          # 构造方法：接收图谱数据与全部文本块
        self.graph = graph or {"entities": {}, "relations": []}  # 容错：图谱为空时使用空结构
        self.entities = self.graph.get("entities", {})           # 实体表：名称 -> {type, chunks, count}
        self.relations = self.graph.get("relations", [])         # 关系三元组列表
        self.id2chunk = {c["id"]: c for c in chunks}            # 块 ID -> 块字典，便于快速定位
        self.adj = {}                                           # 邻接表：实体 -> [(关系名, 邻居实体), ...]
        self._build_adj()                                       # 构建邻接表

    def _build_adj(self):                                       # 定义邻接表构建方法
        """由关系三元组构建无向邻接表，便于双向扩展。"""
        for r in self.relations:                                # 遍历全部关系
            h, t, rel = r.get("head"), r.get("tail"), r.get("relation", "RELATED")  # 取出头/尾/关系名
            if not h or not t:                                  # 头尾缺失的脏数据跳过
                continue                                        # 继续下一条
            self.adj.setdefault(h, []).append((rel, t))         # 记录 head -> tail 的出边
            self.adj.setdefault(t, []).append((rel, h))         # 记录 tail -> head 的入边（无向化）

    # -------------------------------------------------------------------
    # 一、实体链接（把用户问题映射到图谱中的实体）
    # -------------------------------------------------------------------
    def link_entities(self, query, top_n=8):                     # 定义实体链接方法
        """采用『字符串包含 + 分词重叠』打分，返回得分最高的若干实体名。"""
        query = query or ""                                     # 容错：空查询转为空串
        q_tokens = set(tokenize(query))                          # 查询分词结果集合
        scores = {}                                             # 实体名 -> 匹配得分
        for name, info in self.entities.items():                # 遍历图谱全部实体
            if not name:                                        # 空名跳过
                continue                                        # 继续下一个
            score = 0.0                                         # 初始得分
            if name in query:                                   # 实体名整体出现在问题中
                score += 3.0 + min(len(name), 10) * 0.1         # 长实体匹配给予更高权重
            elif len(name) >= 2 and query in name:               # 问题整体出现在实体名中（短问题场景）
                score += 1.0                                    # 给予基础分
            overlap = q_tokens & set(tokenize(name))            # 与实体名的分词重叠
            if overlap:                                         # 有重叠时累加得分
                score += 1.2 * len(overlap) / max(1, len(q_tokens))  # 按重叠比例加权
            score += min(info.get("count", 0), 5) * 0.05        # 高频实体给予轻微加权（更可能是主题）
            if score > 0:                                       # 仅保留有匹配的实体
                scores[name] = score                            # 记录得分
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)  # 按得分降序
        top = [n for n, _ in ranked[:top_n]]                    # 取前 top_n 个实体名
        if top:                                                 # 命中了实体
            LOGGER.info("图谱实体链接：%s", "、".join(top[:5]))   # 输出链接结果（最多展示 5 个）
        return top                                              # 返回链接到的实体名列表

    # -------------------------------------------------------------------
    # 二、邻域扩展（沿着关系找到间接相关实体）
    # -------------------------------------------------------------------
    def expand(self, seeds, hop=None):                          # 定义邻域扩展方法
        """从种子实体出发做 hop 跳扩展，返回扩展后的实体名列表。"""
        hop = config.GRAPH_HOP if hop is None else hop          # 取配置中的默认跳数
        visited = set(seeds)                                    # 已访问实体集合（初始为种子）
        frontier = list(seeds)                                  # 当前边界实体
        for _ in range(hop):                                    # 逐跳扩展
            nxt = []                                            # 下一层边界
            for e in frontier:                                  # 遍历当前边界
                for _rel, nb in self.adj.get(e, []):            # 遍历邻居
                    if nb not in visited:                       # 未访问过则加入
                        visited.add(nb)                         # 标记已访问
                        nxt.append(nb)                          # 加入下一层
            frontier = nxt                                      # 更新边界
            if not frontier:                                    # 边界为空则提前结束
                break                                           # 跳出循环
        return list(visited)                                    # 返回全部相关实体

    # -------------------------------------------------------------------
    # 三、检索入口：实体证据块 + 关系三元组
    # -------------------------------------------------------------------
    def retrieve(self, query, top_n_entities=8, max_chunks=6, hop=None):  # 定义图谱检索入口
        """返回 (关联文本块列表, 关系三元组文本列表)。"""
        seeds = self.link_entities(query, top_n=top_n_entities)  # 先做实体链接
        if not seeds:                                           # 未链接到任何实体
            return [], []                                       # 返回空结果，由混合检索兜底
        related = self.expand(seeds, hop=hop)                   # 邻域扩展得到相关实体集合
        hit_count = {}                                          # 块 ID -> 被命中次数
        for e in related:                                       # 遍历相关实体
            weight = 2 if e in seeds else 1                     # 种子实体权重高于扩展实体
            for cid in self.entities.get(e, {}).get("chunks", []):  # 遍历该实体关联的块
                hit_count[cid] = hit_count.get(cid, 0) + weight  # 累加命中权重
        ranked = sorted(hit_count.items(), key=lambda x: x[1], reverse=True)  # 按权重降序
        chunk_objs = [self.id2chunk[cid] for cid, _ in ranked if cid in self.id2chunk]  # 映射为块字典
        triples = self._collect_triples(seeds, related)         # 收集关系三元组文本
        return chunk_objs[:max_chunks], triples                 # 返回证据块与三元组

    def _collect_triples(self, seeds, related, max_triples=12):  # 定义三元组收集方法
        """挑选与种子实体直接相连的关系，渲染成可读文本供生成阶段引用。"""
        seed_set = set(seeds)                                   # 种子实体集合
        result = []                                             # 结果列表
        for r in self.relations:                                # 遍历全部关系
            h, t = r.get("head"), r.get("tail")                 # 取头尾实体
            if h in seed_set or t in seed_set:                  # 只要有一端是种子实体就保留（最相关）
                result.append("%s --%s--> %s" %                 # 渲染为『头 --关系--> 尾』
                              (h, r.get("relation", "RELATED"), t))  # 填入具体三元组
            if len(result) >= max_triples:                      # 达到上限即停止
                break                                           # 跳出循环
        return result                                           # 返回三元组文本列表
