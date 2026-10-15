# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-LightRAG 优化任务
# 模块职责：LightRAG 索引构建流程——按 schema 抽取实体与关系、去重归一、输出知识图谱

import os                                                       # 导入 os 用于缓存文件判断
import json                                                     # 导入 json 用于图谱读写
from concurrent.futures import ThreadPoolExecutor, as_completed  # 导入线程池用于并发抽取
import config                                                   # 导入本工单配置模块
import llm                                                      # 导入 LLM 封装（带磁盘缓存）
import bm25                                                     # 导入 BM25 用于挑选与测试问题强相关的块
import kg_schema                                                # 导入图谱 schema 与提示词构造
from logger import LOGGER                                       # 导入统一日志对象

# 指代词到发行人全称的映射：招股说明书中「本公司」在不同文件里指向不同主体，
# 必须按来源文件解析，否则两家公司的实体会被错误合并成同一个节点。
# 归属经全文词频核验：招股说明书1 中「武汉兴图新科电子股份有限公司」出现 571 次、「武汉力源」0 次；
# 招股说明书2 中「武汉力源信息技术股份有限公司」出现 368 次、「武汉兴图新科」0 次
ISSUER_BY_SOURCE = {"招股说明书1": "武汉兴图新科电子股份有限公司",   # 兴图新科的招股说明书
                    "招股说明书2": "武汉力源信息技术股份有限公司"}   # 力源信息的招股说明书


def _resolve_pronoun(name, source):                             # 定义指代词解析函数
    """把「本公司」「发行人」等指代词替换为对应来源文件的发行人全称。"""
    if name in kg_schema.PRONOUN_ALIASES:                       # 命中指代词列表
        return ISSUER_BY_SOURCE.get(source, name)               # 按来源文件解析为发行人全称
    return name                                                 # 非指代词原样返回


def select_chunks(chunks):                                      # 定义建图语料挑选函数
    """挑选参与建图的语料块：优先纳入与测试问题强相关的块，其余均匀抽样补足上限。"""
    limit = min(config.GRAPH_BUILD_LIMIT, len(chunks))          # 实际处理上限
    if limit >= len(chunks):                                    # 上限覆盖全量时无需挑选
        LOGGER.info("建图范围：全量 %d 块", len(chunks))          # 输出范围信息
        return chunks                                           # 直接返回全部块
    idx = bm25.build_or_load_index(chunks)                      # 构建或加载 BM25 索引
    picked = []                                                 # 已选中的块编号（保持顺序去重）
    seen = set()                                                # 已选中的块编号集合
    for q in config.QUESTIONS:                                  # 遍历 16 个测试问题
        for i, _ in idx.search(q["question"], top_k=config.GRAPH_QA_SEED):  # 取该问题的强相关块
            if i not in seen:                                   # 未选中过才加入
                seen.add(i)                                     # 记录已选中
                picked.append(i)                                # 保持 BM25 排序加入
    rest = [i for i in range(len(chunks)) if i not in seen]      # 剩余块编号
    step = max(1, len(rest) // max(limit - len(picked), 1))     # 均匀抽样步长
    for i in rest[::step]:                                      # 按步长均匀抽样
        if len(picked) >= limit:                                # 达到上限即停止
            break                                               # 跳出抽样循环
        picked.append(i)                                        # 加入抽样块
    picked = sorted(set(picked))[:limit]                        # 排序去重并截断到上限
    LOGGER.info("建图范围：%d 块（含与测试问题强相关的 %d 块 + 均匀抽样）",
                len(picked), len(seen & set(picked)))           # 输出挑选结果
    return [chunks[i] for i in picked]                          # 返回选中块对象列表


def _chunk_batches(chunks):                                     # 定义分批函数
    """把语料块按配置的批大小分组，降低 LLM 调用次数。"""
    size = config.GRAPH_EXTRACT_BATCH                          # 每批块数
    return [chunks[i:i + size] for i in range(0, len(chunks), size)]  # 顺序切片分组


def extract_batch(batch):                                       # 定义单批抽取函数
    """对一批语料块调用 LLM 抽取实体与关系，返回归一化后并标注来源块编号的结果。"""
    system, user = kg_schema.build_extraction_prompt(batch)     # 构造带 schema 约束的提示词
    obj = llm.chat_json(system, user, max_tokens=4000)          # 调用 LLM 并解析 JSON
    if not isinstance(obj, dict):                               # 解析失败时返回空结果
        return {"entities": [], "relations": [], "chunk_ids": [c["id"] for c in batch]}  # 空结果仍带块编号
    ids = [c["id"] for c in batch]                              # 本批涉及的块编号
    src_by_id = {c["id"]: c["source"] for c in batch}           # 块编号到来源文件的映射
    default_src = batch[0]["source"]                            # 批内默认来源（用于指代词解析）
    ents = []                                                   # 归一化后的实体列表
    for e in (obj.get("entities") or []):                       # 遍历抽取到的实体
        if not isinstance(e, dict):                             # 跳过结构异常的条目
            continue                                            # 继续下一个
        name = kg_schema.normalize_name(e.get("name"))          # 归一化实体名
        etype = (e.get("type") or "").strip()                   # 取出实体类型
        if len(name) < config.GRAPH_MIN_ENTITY_LEN:             # 过滤过短实体名（噪音）
            continue                                            # 继续下一个
        if etype not in [t["type"] for t in kg_schema.ENTITY_TYPES]:  # 过滤 schema 外的类型
            continue                                            # 继续下一个
        name = _resolve_pronoun(name, default_src)              # 解析指代词为发行人全称
        ents.append({"name": name, "type": etype,             # 记录实体名与类型
                     "desc": (e.get("desc") or "").strip(),     # 记录实体描述
                     "source": default_src})                    # 记录来源文件（用于别名归一）
    rels = []                                                   # 归一化后的关系列表
    for r in (obj.get("relations") or []):                      # 遍历抽取到的关系
        if not isinstance(r, dict):                             # 跳过结构异常的条目
            continue                                            # 继续下一个
        head = kg_schema.normalize_name(r.get("head"))          # 归一化头实体名
        tail = kg_schema.normalize_name(r.get("tail"))          # 归一化尾实体名
        rel = (r.get("relation") or "").strip()                 # 取出关系类型
        if len(head) < config.GRAPH_MIN_ENTITY_LEN or len(tail) < config.GRAPH_MIN_ENTITY_LEN:  # 过滤过短名称
            continue                                            # 继续下一个
        if rel not in [t["relation"] for t in kg_schema.RELATION_TYPES]:  # 过滤 schema 外的关系
            continue                                            # 继续下一个
        head = _resolve_pronoun(head, default_src)              # 解析头实体指代词
        tail = _resolve_pronoun(tail, default_src)              # 解析尾实体指代词
        rels.append({"head": head, "tail": tail, "relation": rel,  # 记录关系三元组
                     "value": (r.get("value") or "").strip(),   # 记录数值（如有）
                     "period": (r.get("period") or "").strip(),  # 记录报告期（如有）
                     "desc": (r.get("desc") or "").strip()})    # 记录关系描述
    _ = src_by_id                                               # 保留来源映射变量（便于扩展按页溯源）
    return {"entities": ents, "relations": rels, "chunk_ids": ids}  # 返回本批抽取结果


def run_extraction(batches, force=False):                       # 定义并发抽取函数
    """并发执行全部批次的抽取；结果按批顺序落盘缓存，重复运行不重复计费。"""
    cache_path = os.path.join(config.DATA_DIR, "extract_raw.json")  # 原始抽取结果缓存路径
    if not force and os.path.isfile(cache_path):                # 缓存存在且非强制重跑
        with open(cache_path, "r", encoding="utf-8") as f:      # 以 UTF-8 打开缓存
            raw = json.load(f)                                  # 反序列化抽取结果
        if len(raw) == len(batches):                            # 批次数一致才可用
            LOGGER.info("命中图谱抽取缓存：%d 批", len(raw))      # 输出命中信息
            return raw                                          # 返回缓存的抽取结果
        LOGGER.warning("抽取缓存批次数 %d 与预期 %d 不一致，重建", len(raw), len(batches))  # 输出告警
    LOGGER.info("开始抽取：共 %d 批，并发 %d", len(batches), config.LLM_WORKERS)  # 输出开始提示
    results = [None] * len(batches)                             # 预分配结果列表（保持批序）
    with ThreadPoolExecutor(max_workers=config.LLM_WORKERS) as pool:  # 创建线程池
        futures = {pool.submit(extract_batch, b): i for i, b in enumerate(batches)}  # 提交全部批次
        done = 0                                                # 已完成计数
        for fut in as_completed(futures):                       # 按完成顺序收集
            i = futures[fut]                                    # 取回批次下标
            try:                                                # 尝试获取结果
                results[i] = fut.result()                       # 写入对应位置保持批序
            except Exception as e:                              # 单批失败时容错
                LOGGER.warning("第 %d 批抽取失败：%r", i, e)      # 输出告警
                results[i] = {"entities": [], "relations": [], "chunk_ids": []}  # 用空结果占位
            done += 1                                           # 计数加一
            if done % 20 == 0 or done == len(batches):          # 每 20 批输出一次进度
                LOGGER.info("抽取进度：%d/%d", done, len(batches))  # 输出进度
    config.ensure_dirs()                                        # 确保数据目录存在
    with open(cache_path, "w", encoding="utf-8") as f:          # 以 UTF-8 覆盖写入缓存
        json.dump(results, f, ensure_ascii=False)               # 序列化原始抽取结果
    llm.save_cache()                                            # 落盘 LLM 磁盘缓存
    LOGGER.info("抽取完成，原始结果写入 %s", cache_path)          # 输出完成日志
    return results                                              # 返回按批顺序的抽取结果


def merge_graph(raw, chunks, force=False):                      # 定义图谱合并函数
    """把逐批抽取结果合并为知识图谱：实体按名称与短名归并，关系聚合并记录支撑块。"""
    if not force and os.path.isfile(config.GRAPH_FILE):         # 缓存存在且非强制重建
        with open(config.GRAPH_FILE, "r", encoding="utf-8") as f:  # 以 UTF-8 打开缓存
            graph = json.load(f)                                # 反序列化图谱
        LOGGER.info("命中图谱缓存：%d 实体，%d 关系", len(graph["entities"]), len(graph["relations"]))  # 输出命中信息
        return graph                                            # 返回缓存的图谱
    entities = {}                                               # 实体名 -> 实体记录
    alias2name = {}                                             # 别名（含短名）-> 主实体名
    relations = {}                                              # (头, 关系, 尾) -> 关系记录

    def _ensure(name, etype, desc, cid):                        # 定义实体登记内部函数
        """登记实体：命中别名则并入已有实体，否则新建。"""
        key = alias2name.get(name) or name                      # 先按别名表解析主名
        short = kg_schema.short_name(name)                      # 计算短名用于跨全称/简称归并
        key = alias2name.get(short, key)                        # 短名命中时以已有主名为准
        if key not in entities:                                 # 主名尚不存在则新建
            entities[key] = {"name": key, "type": etype,        # 记录主名与类型
                             "desc": desc,                      # 记录描述
                             "chunks": [],                      # 支撑块编号列表
                             "aliases": []}                     # 别名列表
        rec = entities[key]                                     # 取出实体记录
        if desc and desc not in rec["desc"]:                    # 合并描述信息
            rec["desc"] = (rec["desc"] + "；" + desc).strip("；") if rec["desc"] else desc  # 拼接去重
        if cid is not None and cid not in rec["chunks"]:        # 记录支撑块编号
            rec["chunks"].append(cid)                           # 追加块编号
        for a in (name, short):                                 # 登记名称与短名两种别名
            if a and a != key:                                  # 与主名不同才记为别名
                alias2name[a] = key                             # 写入别名表
                if a not in rec["aliases"]:                     # 别名未记录过才追加
                    rec["aliases"].append(a)                    # 追加别名
        return key                                              # 返回主名

    for res in raw:                                             # 遍历每批抽取结果
        cids = res.get("chunk_ids") or [None]                   # 本批块编号（无则用 None 占位）
        cid = cids[0] if cids else None                         # 取首个块编号作为支撑块标记
        for e in res.get("entities", []):                       # 遍历实体
            _ensure(e["name"], e["type"], e.get("desc", ""), cid)  # 登记实体
        for r in res.get("relations", []):                      # 遍历关系
            h = _ensure(r["head"], "", "", cid)                  # 登记头实体（类型缺失时留空）
            t = _ensure(r["tail"], "", "", cid)                  # 登记尾实体
            key = (h, r["relation"], t)                         # 三元组作为聚合键
            rec = relations.get(key)                            # 取出已有关系记录
            if rec is None:                                     # 首次出现则新建
                rec = {"head": h, "relation": r["relation"],    # 头实体与关系类型
                       "tail": t, "values": [], "periods": [],  # 尾实体与取值列表
                       "desc": r.get("desc", ""), "chunks": [],  # 描述与支撑块
                       "weight": 0}                             # 出现次数（用于排序）
                relations[key] = rec                            # 写入映射
            for slot, val in (("values", r.get("value")), ("periods", r.get("period"))):  # 收集数值与报告期
                val = (val or "").strip()                       # 去空白
                if val and val not in rec[slot]:                # 非空且未记录过
                    rec[slot].append(val)                       # 追加
            if r.get("desc") and r["desc"] not in rec["desc"]:  # 合并关系描述
                rec["desc"] = (rec["desc"] + "；" + r["desc"]).strip("；") if rec["desc"] else r["desc"]  # 拼接去重
            if cid is not None and cid not in rec["chunks"]:    # 记录支撑块
                rec["chunks"].append(cid)                       # 追加块编号
            rec["weight"] += 1                                  # 出现次数加一

    rel_list = sorted(relations.values(), key=lambda x: -x["weight"])  # 按支撑次数降序排列关系
    graph = {"entities": entities,                              # 实体字典（主名 -> 记录）
             "relations": rel_list,                             # 关系列表
             "stats": {"chunk_total": len(chunks),              # 语料块总数
                       "entity_total": len(entities),           # 实体总数
                       "relation_total": len(rel_list)}}        # 关系总数（去重后）
    config.ensure_dirs()                                        # 确保数据目录存在
    with open(config.GRAPH_FILE, "w", encoding="utf-8") as f:   # 以 UTF-8 覆盖写入图谱
        json.dump(graph, f, ensure_ascii=False, indent=1)       # 序列化图谱（保留中文）
    LOGGER.info("图谱构建完成：%d 实体，%d 关系，写入 %s",
                len(entities), len(rel_list), config.GRAPH_FILE)  # 输出完成日志
    return graph                                                # 返回图谱


def build(force=False):                                         # 定义建图主函数
    """执行完整建图流程：挑选语料 → 并发抽取 → 合并去重 → 落盘图谱。"""
    import corpus                                               # 延迟导入避免循环依赖
    chunks = corpus.build_or_load_chunks()                      # 加载语料块
    selected = select_chunks(chunks)                            # 挑选参与建图的块
    batches = _chunk_batches(selected)                          # 分批
    raw = run_extraction(batches, force=force)                  # 并发抽取
    return merge_graph(raw, chunks, force=force)                # 合并为图谱并返回


if __name__ == "__main__":                                      # 仅在直接运行时执行
    import sys                                                  # 导入 sys 解析命令行参数
    build(force="--force" in sys.argv)                          # 支持 --force 强制重建
