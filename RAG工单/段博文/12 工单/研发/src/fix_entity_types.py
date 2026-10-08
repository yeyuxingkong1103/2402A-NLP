# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-LightRAG优化
"""
实体类型回填修复脚本。

问题：
    LightRAG 1.5.7 持久化的 graphml 中节点 entity_type 全部为 UNKNOWN
    （类型仅存在于 LLM 抽取的原始输出，未随图谱写入），导致验收标准 1
    要求的"准确实体类型"无法在图谱中体现。

修复：
    从 LLM 响应缓存（kv_store_llm_response_cache.json）的每条实体抽取
    结果中解析 "entity<|#|>名称<|#|>类型<|#|>描述"，按实体名聚合
    （多 chunk 抽到同名实体时类型取众数），回填 graphml 节点的
    entity_type 属性并保存。

用法：python fix_entity_types.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

from config import WORK_DIR, RESULT_DIR
from logger import get_logger

logger = get_logger(__name__)

DELIM = "<|#|>"
GRAPHML = WORK_DIR / "graph_chunk_entity_relation.graphml"
CACHE = WORK_DIR / "kv_store_llm_response_cache.json"


def parse_entity_types() -> dict:
    """从 LLM 缓存解析 实体名 -> 类型众数 映射。"""
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    votes = defaultdict(Counter)

    n_records = 0
    for item in cache.values():
        text = item.get("return", "") if isinstance(item, dict) else str(item)
        if "entity" + DELIM not in text:
            continue
        n_records += 1
        for line in text.splitlines():
            parts = line.split(DELIM)
            # 格式：entity | 名称 | 类型 | 描述
            if len(parts) >= 3 and parts[0].strip() == "entity":
                # 兼容 LLM 偶发使用 "<|>" 变体分隔符：字段只取其前段
                name = parts[1].split("<|>")[0].strip()
                etype = parts[2].split("<|>")[0].strip()
                if name and etype:
                    votes[name][etype] += 1

    mapping = {name: counter.most_common(1)[0][0]
               for name, counter in votes.items()}
    logger.info(f"解析了 {n_records} 条抽取缓存，得到 {len(mapping)} 个实体的类型")
    return mapping


def fix_graph() -> dict:
    """回填 graphml 节点类型并保存。"""
    mapping = parse_entity_types()
    g = nx.read_graphml(GRAPHML)

    fixed, missing = 0, []
    for node in g.nodes():
        if node in mapping:
            g.nodes[node]["entity_type"] = mapping[node]
            fixed += 1
        else:
            missing.append(node)

    # 名称变体（简称/全称差异）做包含匹配：取最长匹配键的类型
    map_keys = sorted(mapping, key=len, reverse=True)
    fuzzy = []
    still = []
    for node in missing:
        hit = None
        for k in map_keys:
            if len(k) >= 4 and (k in node or node in k):
                hit = k
                break
        if hit:
            g.nodes[node]["entity_type"] = mapping[hit]
            fixed += 1
            fuzzy.append(node)
        else:
            still.append(node)

    nx.write_graphml(g, GRAPHML)
    type_dist = Counter(d["entity_type"] for _, d in g.nodes(data=True))
    stats = {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "fixed": fixed,
        "fuzzy_fixed": len(fuzzy),
        "unmatched": len(still),
        "entity_type_dist": dict(type_dist.most_common()),
    }
    (RESULT_DIR / "entity_type_fix_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main():
    print(json.dumps(fix_graph(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
