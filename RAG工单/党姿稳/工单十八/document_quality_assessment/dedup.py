# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-18-实现文档质量评估Skill并集成至智能体工作流
# 模块职责：两层重复检测。MD5 精确匹配可直接确认完全相同的文件；SimHash 只能给出高相似度
#           候选，必须人工确认，因此输出的是“待确认的版本冲突列表”，每条都带相似片段证据。

import os                                                          # 路径与文件名校验

from . import simhash as simhash_mod                               # 指纹与汉明距离


def exact_groups(records):
    """MD5 精确重复：返回 {md5: [记录...]} 中重复的组。"""
    buckets = {}                                                   # MD5 → 记录列表
    for r in records:                                              # 逐记录
        md5 = r.get("md5")                                         # 文件指纹
        if not md5:                                                # 读取失败的文件不参与
            continue
        buckets.setdefault(md5, []).append(r)
    groups = []                                                    # 重复组
    for md5, items in buckets.items():                             # 逐指纹
        if len(items) < 2:                                         # 只有一个文件不算重复
            continue
        items = sorted(items, key=lambda r: r["path"])             # 稳定排序
        groups.append({"md5": md5,                                 # 完全相同文件的指纹
                       "count": len(items),                        # 文件个数
                       "size_bytes": items[0].get("size_bytes", 0),  # 单份体积
                       "wasted_bytes": items[0].get("size_bytes", 0) * (len(items) - 1),
                       "files": [{"path": r["path"], "name": r["name"],
                                  "size_bytes": r.get("size_bytes", 0)} for r in items]})
    groups.sort(key=lambda g: (-g["count"], g["md5"]))             # 数量多的排前面
    return groups


def near_pairs(records, cfg, limit=200):
    """SimHash 近似重复：先分桶取候选，再算汉明距离，返回待确认的版本冲突列表。"""
    if not cfg.get("dedup.simhash_enabled", True):                 # 开关关闭则不做
        return []
    th = cfg.get("dedup.hamming_threshold", 3)                     # 汉明距离阈值
    band_n = cfg.get("dedup.band_count", 4)                        # 分桶段数
    ctx_chars = cfg.get("dedup.max_context_chars", 120)            # 证据片段长度

    pool = [r for r in records if r.get("simhash") is not None]    # 有指纹的文档
    buckets = {}                                                   # (段号, 段值) → 记录下标
    for idx, r in enumerate(pool):                                 # 逐文档
        for band in simhash_mod.bands(r["simhash"], band_n):       # 逐段入桶
            buckets.setdefault(band, []).append(idx)               # 同段同值才可能近似
    seen, pairs = set(), []                                        # 去重集合与结果
    for key, idxs in buckets.items():                              # 遍历桶
        if len(idxs) < 2:                                          # 桶内不足两个无候选
            continue
        for a in range(len(idxs)):                                 # 桶内两两比较（桶很小）
            for b in range(a + 1, len(idxs)):                      # 组合
                i, j = idxs[a], idxs[b]                            # 下标
                pair_key = (min(i, j), max(i, j))                  # 无序去重键
                if pair_key in seen:                               # 已比较过
                    continue
                seen.add(pair_key)
                ri, rj = pool[i], pool[j]                          # 参与比较的两条记录
                dist = simhash_mod.hamming(ri["simhash"], rj["simhash"])
                if dist > th:                                      # 超出阈值不近似
                    continue
                pairs.append(_pair_item(ri, rj, dist, ctx_chars))  # 组装证据
    pairs.sort(key=lambda p: (p["hamming_distance"], -p["similarity"]))  # 越像越靠前
    return pairs[:limit]                                           # 截断，避免报告过大


def _pair_item(ri, rj, dist, ctx_chars):
    """组装一条版本冲突候选：含双方路径、相似度、汉明距离与各自文本片段。"""
    return {"hamming_distance": dist,                              # 汉明距离
            "similarity": simhash_mod.similarity(ri["simhash"], rj["simhash"]),
            "file_a": {"path": ri["path"], "name": ri["name"],     # 文件 A
                       "text_chars": ri.get("text_chars", 0),
                       "size_bytes": ri.get("size_bytes", 0),
                       "page_count": ri.get("page_count"),
                       "excerpt": (ri.get("text_sample") or "")[:ctx_chars]},
            "file_b": {"path": rj["path"], "name": rj["name"],     # 文件 B
                       "text_chars": rj.get("text_chars", 0),
                       "size_bytes": rj.get("size_bytes", 0),
                       "page_count": rj.get("page_count"),
                       "excerpt": (rj.get("text_sample") or "")[:ctx_chars]},
            "same_name": os.path.basename(ri["path"]) == os.path.basename(rj["path"])}


def compute(records, cfg, limit=200):
    """主入口：返回精确重复组与近似重复候选。"""
    return {"exact_groups": exact_groups(records) if cfg.get("dedup.md5_enabled", True) else [],
            "near_pairs": near_pairs(records, cfg, limit=limit),
            "md5_enabled": bool(cfg.get("dedup.md5_enabled", True)),
            "simhash_enabled": bool(cfg.get("dedup.simhash_enabled", True)),
            "hamming_threshold": cfg.get("dedup.hamming_threshold", 3)}
