# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 模块职责：选取建图文本块、调用 LLM 抽取实体关系、归一化并保存知识图谱（含优化前后两版）

import os                                                       # 导入 os 用于缓存路径判断
import json                                                     # 导入 json 用于图谱缓存读写
import numpy as np                                              # 导入 numpy 用于相关性排序取块
import config                                                   # 导入本工单配置模块
import llm                                                      # 导入 LLM 封装（含重试与缓存）
import graph_schema                                             # 导入实体关系体系与提示词
from timing import LOGGER                                       # 导入统一日志对象


def select_build_chunks(chunks, vectors, questions, limit=None):  # 定义建图块选择函数
    """混合采样：优先取与验收问题相关的块，再用均匀采样补足覆盖面。"""
    limit = limit or config.GRAPH_BUILD_LIMIT                   # 取配置中的建图上限
    if not chunks or limit >= len(chunks):                      # 块数不足上限时直接全用
        return chunks                                           # 返回全部块
    picked = {}                                                 # 已选块：下标 -> 块
    from embedder import encode_query                           # 延迟导入避免无谓的模型加载
    for q in questions:                                         # 遍历验收问题
        qv = encode_query(q["question"])                        # 编码问题文本
        scores = vectors.dot(qv)                                # 计算与全部块的相关性
        for i in np.argsort(-scores)[:8]:                       # 每个问题取最相关的 8 个块
            picked[int(i)] = chunks[int(i)]                     # 记录到已选集合（自动去重）
    step = max(1, len(chunks) // max(1, limit - len(picked)))   # 计算均匀采样的步长
    for i in range(0, len(chunks), step):                       # 按步长均匀遍历全部块
        picked.setdefault(i, chunks[i])                         # 补充覆盖不同文件与章节的块
        if len(picked) >= limit:                                # 达到上限即停止
            break                                               # 跳出循环
    result = [picked[i] for i in sorted(picked.keys())][:limit]  # 按下标排序并截断到上限
    LOGGER.info("建图块选择完成：%d 块", len(result))            # 输出统计日志
    return result                                               # 返回选中的块列表


def _chunk_text_for_prompt(batch):                              # 定义批次文本拼装函数
    """把一批文本块拼成带编号的提示词文本，编号用于回溯来源。"""
    parts = []                                                  # 初始化片段列表
    for i, c in enumerate(batch):                               # 遍历批次中的块
        parts.append("[片段%d] 来源：%s 第%d页\n%s" %            # 拼装编号、来源与正文
                     (i + 1, c["source"], c["page"], c["text"]))  # 填入块的定位信息与内容
    return "\n\n".join(parts)                                   # 用空行分隔各片段


def extract_once(batch, optimized=True):                        # 定义单批抽取函数
    """对一批文本块调用 LLM 抽取实体关系；失败或解析异常时返回空结构（容错）。"""
    text = _chunk_text_for_prompt(batch)                        # 拼装批次文本
    if optimized:                                               # 使用优化后提示词
        system, user = graph_schema.optimized_prompt(text)      # 取带类型约束的提示词
    else:                                                       # 使用优化前基线提示词
        system, user = graph_schema.baseline_prompt(text)       # 取宽泛提示词
    obj = llm.chat_json(system, user, temperature=0.0, max_tokens=2500)  # 调用 LLM 并要求 JSON
    if not isinstance(obj, dict):                               # 结果不是字典（解析失败）时降级
        return {"entities": [], "relations": []}                # 返回空结构保证流程不中断
    ents = obj.get("entities") or []                            # 取出实体列表（空值兜底）
    rels = obj.get("relations") or []                           # 取出关系列表（空值兜底）
    if optimized:                                               # 优化后做类型与字段合法性过滤
        ents = [e for e in ents if isinstance(e, dict)            # 仅保留字典形态的实体
                and e.get("name") and e.get("type") in graph_schema.ENTITY_TYPES]  # 类型必须在体系内
        rels = [r for r in rels if isinstance(r, dict)            # 仅保留字典形态的关系
                and r.get("head") and r.get("tail")               # 头尾实体必须存在
                and r.get("relation") in graph_schema.RELATION_TYPES]  # 关系名必须在体系内
    return {"entities": ents, "relations": rels}                # 返回过滤后的结构


def build_graph(chunks, vectors, questions, optimized=True, limit=None, use_cache=True):  # 定义建图主函数
    """构建知识图谱；优化版与基线版分别缓存到不同文件。"""
    cache = _cache_path(optimized)                              # 根据版本决定缓存文件路径
    if use_cache and os.path.isfile(cache):                     # 缓存存在时直接读取
        graph = _load(cache)                                    # 载入缓存图谱
        if graph:                                               # 载入成功
            LOGGER.info("命中图谱缓存（%s）：%d 实体 / %d 关系",  # 输出缓存命中日志
                        "优化后" if optimized else "优化前",      # 标注版本
                        len(graph["entities"]), len(graph["relations"]))  # 输出规模
            return graph                                        # 返回缓存图谱
    use = select_build_chunks(chunks, vectors, questions, limit)  # 选取建图块
    ents = {}                                                   # 实体聚合表：名称 -> {type, chunks, count}
    rels = []                                                   # 关系三元组列表
    seen_rel = set()                                            # 关系去重集合
    batch_size = config.GRAPH_EXTRACT_BATCH                     # 每批处理的块数
    for start in range(0, len(use), batch_size):                # 按批遍历
        batch = use[start:start + batch_size]                   # 取出当前批
        out = extract_once(batch, optimized=optimized)          # 调用抽取
        bid = [c["id"] for c in batch]                          # 当前批的块 ID 列表
        for e in out["entities"]:                               # 合并实体
            name = str(e.get("name", "")).strip()               # 归一化实体名（去空白）
            if not name:                                        # 名称为空则跳过
                continue                                        # 继续下一个实体
            rec = ents.setdefault(name, {"type": e.get("type", "ORG"),  # 已存在则复用记录
                                         "chunks": [], "count": 0})  # 初始化块列表与计数
            rec["chunks"] = sorted(set(rec["chunks"] + bid))    # 合并来源块并去重
            rec["count"] += 1                                   # 出现次数加一
        for r in out["relations"]:                              # 合并关系
            head, rel, tail = (str(r.get("head") or "").strip(),  # 取头实体名（缺字段时兜底为空）
                               str(r.get("relation") or "").strip(),  # 取关系名
                               str(r.get("tail") or "").strip())  # 取尾实体名
            if not (head and rel and tail):                     # 三要素缺一即视为无效关系
                continue                                        # 跳过该条（基线版未做类型过滤，此处兜底）
            key = (head, rel, tail)                             # 用头/关系/尾构成去重键
            if key in seen_rel:                                 # 已存在则跳过
                continue                                        # 继续下一条
            seen_rel.add(key)                                   # 记录已见关系
            r = dict(r)                                         # 复制一份避免修改原对象
            r["chunks"] = bid                                   # 记录该关系来源的块
            rels.append(r)                                      # 追加到关系列表
        LOGGER.info("图谱抽取进度：%d/%d 批", start // batch_size + 1,  # 输出进度（当前批序号）
                    (len(use) + batch_size - 1) // batch_size)  # 总批数
    graph = {"entities": ents, "relations": rels,               # 组装图谱数据结构
             "optimized": optimized,                            # 标记是否优化版
             "n_chunks": len(use)}                              # 记录建图所用块数
    try:                                                        # 尝试写入缓存
        json.dump(graph, open(cache, "w", encoding="utf-8"), ensure_ascii=False)  # 序列化图谱
        LOGGER.info("图谱已缓存：%s（%d 实体 / %d 关系）", cache,  # 输出完成日志
                    len(ents), len(rels))                       # 输出规模统计
    except Exception as e:                                      # 写入失败时容错
        LOGGER.warning("图谱缓存写入失败：%r", e)                # 仅告警
    return graph                                                # 返回图谱数据


def _cache_path(optimized):                                     # 定义缓存路径计算函数
    base = config.GRAPH_CACHE_FILE                              # 取基础缓存路径
    return base if optimized else base.replace(".json", "_baseline.json")  # 基线版使用不同文件名


def _load(path):                                                # 定义图谱缓存加载函数
    try:                                                        # 尝试读取
        return json.load(open(path, "r", encoding="utf-8"))     # 反序列化并返回
    except Exception as e:                                      # 读取失败时容错
        LOGGER.warning("图谱缓存读取失败：%r", e)                # 输出告警
        return None                                             # 返回 None 触发重建
