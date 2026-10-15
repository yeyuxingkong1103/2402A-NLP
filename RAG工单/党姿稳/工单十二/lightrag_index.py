# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-LightRAG 优化任务
# 模块职责：LightRAG 索引侧——把知识图谱中的实体编码为向量，支持关键词到实体的语义链接

import os                                                       # 导入 os 用于缓存文件判断
import json                                                     # 导入 json 用于实体清单读写
import numpy as np                                              # 导入 numpy 用于向量矩阵处理
import config                                                   # 导入本工单配置模块
import kg_schema                                                # 导入实体名归一化工具
import embedder                                                 # 导入嵌入编码模块
from logger import LOGGER                                       # 导入统一日志对象

ENTITY_LIST_FILE = os.path.join(config.DATA_DIR, "entity_names.json")  # 实体名称清单缓存路径


def _lexical_bonus(key, name):                                  # 定义字面命中加权函数
    """关键词与实体名存在字面关系时给予加权，把精确同名实体拉到相似度排序前面。

    图谱中存在大量共享发行人全称前缀的实体（如「武汉兴图新科电子股份有限公司章程」），
    纯向量相似度会让这类长名称排在真正的发行人实体之前，进而把局部检索引向无关语料块。
    """
    if not key or not name:                                     # 任一为空则不加权
        return 0.0                                              # 返回 0
    if key == name:                                             # 关键词与实体名完全相同
        return 0.25                                             # 最高加权（精确命中）
    if len(key) >= 4 and (key in name or name in key):          # 长度足够且存在包含关系
        return 0.15                                             # 次高加权（部分命中）
    return 0.0                                                  # 无字面关系时不加权


def entity_text(rec):                                           # 定义实体向量化文本构造函数
    """把实体名、类型与关系描述拼成一段文本，作为实体向量与图谱检索的语义载体。"""
    parts = [rec.get("name", "")]                               # 实体名（权重最高，置于最前）
    if rec.get("type"):                                         # 有类型时加入
        parts.append("类型：" + rec["type"])                     # 追加类型说明
    if rec.get("desc"):                                         # 有描述时加入
        parts.append(rec["desc"])                               # 追加实体描述
    rels = rec.get("rel_text") or []                            # 该实体参与的关系文本（由图谱侧预计算）
    if rels:                                                    # 有关系时加入
        parts.append("；".join(rels[:6]))                        # 最多拼接 6 条关系，避免过长
    return "。".join(parts)                                     # 用小句号拼接为一段完整文本


def _attach_rel_text(graph):                                    # 定义关系文本挂载函数
    """把每个实体参与的关系渲染为短句挂到实体记录上，供实体向量与图谱检索使用。"""
    for rec in graph["entities"].values():                      # 初始化所有实体的关系文本列表
        rec["rel_text"] = []                                    # 默认空列表
    for r in graph["relations"]:                                # 遍历全部关系
        text = "%s%s%s" % (r["head"], r["relation"], r["tail"])  # 渲染关系短句（如「甲公司控股乙公司」）
        if r.get("values"):                                     # 关系带数值时补充（如持股比例）
            text += "（%s）" % "、".join(r["values"][:3])         # 最多补充 3 个取值
        for side in ("head", "tail"):                           # 关系两端实体都要挂上该关系文本
            rec = graph["entities"].get(r[side])                # 取出该侧实体记录
            if rec is not None and text not in rec["rel_text"]:  # 去重后追加
                rec["rel_text"].append(text)                    # 追加关系短句
    return graph                                                # 返回补充后的图谱


class EntityIndex(object):                                      # 定义实体索引类
    """实体向量索引：支持把查询关键词链接到图谱中的实体节点。"""

    def __init__(self, graph, names, vecs):                     # 构造方法
        self.graph = graph                                      # 知识图谱（实体字典 + 关系列表）
        self.names = names                                      # 实体主名列表（行号与向量矩阵对齐）
        self.vecs = vecs                                        # 实体向量矩阵
        self.name2row = {n: i for i, n in enumerate(names)}     # 实体名到向量行号的映射
        self.norm2row = {kg_schema.normalize_name(n): i for i, n in enumerate(names)}  # 归一化名到行号（查精确同名）
        self.norm_names = [kg_schema.normalize_name(n) for n in names]  # 归一化实体名（与向量行对齐）
        self.adj = _build_adjacency(graph)                      # 实体邻接表（用于图谱扩散）

    def match(self, keywords, top_k=None):                      # 定义关键词到实体的链接方法
        """把一组关键词逐一编码并在实体向量中检索，返回 [(实体名, 相似度得分), ...] 降序。

        得分 = 余弦相似度 + 字面命中加权；完全同名的实体无条件纳入候选，不受 top_k 截断影响。
        """
        top_k = top_k or config.ENTITY_VEC_TOP_K                # 每个关键词匹配的实体数量上限
        if not keywords or not self.names:                      # 无关键词或无实体
            return []                                           # 返回空列表
        qv = embedder.encode(embedder.load_model(), keywords, batch_size=16)  # 编码关键词（文档侧口径，不加前缀）
        best = {}                                               # 实体名 -> 最高得分
        for ki, row in enumerate(qv):                           # 遍历每个关键词向量
            key = kg_schema.normalize_name(keywords[ki])        # 当前关键词的归一化形式
            sims = self.vecs.dot(row)                           # 与全部实体计算余弦相似度
            pool = [int(i) for i in np.argsort(-sims)[:top_k]]  # 取相似度最高的若干实体行号
            exact = self.norm2row.get(key)                      # 查是否存在完全同名的实体
            if exact is not None and exact not in pool:         # 同名实体被相似度截断时补入候选
                pool.append(exact)                              # 追加到候选池
            for i in pool:                                      # 遍历候选实体
                s = float(sims[i]) + _lexical_bonus(key, self.norm_names[i])  # 相似度叠加字面命中加权
                name = self.names[i]                            # 取实体名
                if name not in best or s > best[name]:          # 保留该实体的最高得分
                    best[name] = s                              # 更新得分
        return sorted(best.items(), key=lambda x: -x[1])        # 按得分降序返回

    def expand(self, names, hops=None, limit=None):             # 定义图谱邻域扩散方法
        """沿实体关系做多跳扩散，返回 [(实体名, 距起始实体的跳数), ...]。"""
        hops = config.GRAPH_HOP if hops is None else hops       # 取配置的扩散跳数
        limit = config.GRAPH_MAX_NEIGHBORS if limit is None else limit  # 邻居数量上限
        seen = {}                                               # 实体名 -> 最小跳数
        frontier = []                                           # 当前层的实体名列表
        for n in names:                                         # 初始化起始实体
            if n in self.graph["entities"] and n not in seen:   # 排除不在图谱中的名称
                seen[n] = 0                                     # 起始实体跳数为 0
                frontier.append(n)                              # 加入当前层
        for h in range(1, hops + 1):                            # 逐层扩散
            nxt = []                                            # 下一层实体
            for n in frontier:                                  # 遍历当前层
                for nb in self.adj.get(n, []):                  # 遍历邻居
                    if nb in seen:                              # 已访问过则跳过
                        continue                                # 继续下一个邻居
                    seen[nb] = h                                # 记录跳数
                    nxt.append(nb)                              # 加入下一层
                    if len(seen) >= limit:                      # 达到邻居上限即停止
                        break                                   # 跳出邻居循环
                if len(seen) >= limit:                          # 达到上限
                    break                                       # 跳出当前层循环
            frontier = nxt                                      # 进入下一层
            if not frontier or len(seen) >= limit:              # 无新节点或已达上限
                break                                           # 结束扩散
        return sorted(seen.items(), key=lambda x: (x[1], x[0]))  # 按跳数升序返回

    def chunks_of(self, names):                                 # 定义实体关联块收集方法
        """按给定实体名顺序收集其关联的语料块行号，保持顺序去重。"""
        out = []                                                # 结果行号列表
        seen = set()                                            # 去重集合
        for n in names:                                         # 遍历实体名
            rec = self.graph["entities"].get(n)                 # 取实体记录
            if not rec:                                         # 实体不存在则跳过
                continue                                        # 继续下一个
            for cid in rec.get("chunks", []):                   # 遍历该实体的关联块编号
                if cid not in seen:                             # 未收集过才加入
                    seen.add(cid)                               # 记录已收集
                    out.append(cid)                             # 追加块编号
        return out                                              # 返回块编号列表（与语料行号一致）


def _build_adjacency(graph):                                    # 定义邻接表构建函数
    """把关系列表转换为无向邻接表，用于图谱多跳扩散。"""
    adj = {}                                                    # 实体名 -> 邻居列表
    for r in graph["relations"]:                                # 遍历全部关系
        h, t = r["head"], r["tail"]                             # 取关系两端实体
        adj.setdefault(h, [])                                   # 初始化头实体邻居表
        adj.setdefault(t, [])                                   # 初始化尾实体邻居表
        if t not in adj[h]:                                     # 避免了自环与重复
            adj[h].append(t)                                    # 头实体的邻居加入尾实体
        if h not in adj[t]:                                     # 避免对称重复
            adj[t].append(h)                                    # 尾实体的邻居加入头实体
    return adj                                                  # 返回邻接表


def build_or_load_entity_index(model, graph, force=False):       # 定义实体索引构建函数
    """构建实体向量索引：编码实体文本并缓存向量与名称清单。"""
    graph = _attach_rel_text(graph)                             # 先把关系文本挂到实体上
    if (not force and os.path.isfile(config.ENTITY_VEC_FILE)    # 向量缓存存在
            and os.path.isfile(ENTITY_LIST_FILE)):              # 且名称清单缓存存在
        with open(ENTITY_LIST_FILE, "r", encoding="utf-8") as f:  # 以 UTF-8 打开名称清单
            names = json.load(f)                                # 读取实体名列表
        vecs = np.load(config.ENTITY_VEC_FILE)                  # 读取实体向量矩阵
        if len(names) == vecs.shape[0] == len(graph["entities"]):  # 三者数量一致才可用
            LOGGER.info("命中实体向量缓存：%d 个实体", len(names))  # 输出命中信息
            return EntityIndex(graph, names, vecs)              # 组装索引返回
        LOGGER.warning("实体向量缓存与图谱不一致，重建")          # 输出告警
    names = list(graph["entities"].keys())                      # 取全部实体主名（顺序固定）
    texts = [entity_text(graph["entities"][n]) for n in names]  # 构造实体向量化文本
    LOGGER.info("开始编码 %d 个实体", len(names))                # 输出开始提示
    vecs = embedder.encode(model, texts, batch_size=64, show=True)  # 编码实体文本
    config.ensure_dirs()                                        # 确保数据目录存在
    np.save(config.ENTITY_VEC_FILE, vecs)                       # 缓存实体向量
    with open(ENTITY_LIST_FILE, "w", encoding="utf-8") as f:    # 以 UTF-8 写名称清单
        json.dump(names, f, ensure_ascii=False)                 # 序列化实体名列表
    LOGGER.info("实体向量构建完成：%d 个实体 %s", len(names), str(vecs.shape))  # 输出完成日志
    return EntityIndex(graph, names, vecs)                      # 组装索引返回
