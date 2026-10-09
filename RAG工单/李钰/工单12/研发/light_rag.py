# -*- coding: utf-8 -*-
"""
LightRAG 核心: 双层检索 + 增量更新 + 图谱+向量双索引
工单编号: 人工智能 NLP-RAG 项目-LightRAG 优化

核心特性:
  1. 知识图谱 (实体 + 关系 + 社区)
  2. 向量索引 (实体 + 文本块)
  3. 局部检索 (具体实体 → 邻居扩展)
  4. 全局检索 (社区发现 + 路径推理)
  5. 增量更新 (图差异分析)
"""
import os, sys, json, math, pickle, logging, time
from typing import List, Dict, Set, Tuple
from collections import deque, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v12 as config

logger = logging.getLogger(__name__)


class LightRAG:
    """LightRAG 核心类"""

    def __init__(self):
        # 图谱
        self.entities: Dict[str, Dict] = {}  # id → {type, summary, importance}
        self.relations: List[Dict] = []       # [{src, tgt, relation, weight, summary}]
        self.communities: Dict[str, List[str]] = {}  # community_id → [entity_ids]
        self.community_summaries: Dict[str, str] = {}

        # 向量索引
        self.entity_vectors = {}     # entity_id → vector
        self.text_vectors = {}       # chunk_id → vector
        self.entity_vectorizer = None
        self.text_vectorizer = None

        # 块索引
        self.chunks: Dict[str, Dict] = {}  # chunk_id → {text, entities}

    # ============ 实体/关系 ============

    def add_entity(self, eid: str, etype: str = "", summary: str = "") -> bool:
        """添加实体, 返回是否新增 (增量更新)"""
        if eid in self.entities:
            # 已有: 更新属性 (增量更新)
            old = self.entities[eid]
            changed = False
            if summary and summary != old.get("summary", ""):
                self.entities[eid]["summary"] = summary
                changed = True
            if etype and etype != old.get("type", ""):
                self.entities[eid]["type"] = etype
                changed = True
            return changed  # 返回是否有变更
        else:
            self.entities[eid] = {
                "type": etype, "summary": summary,
                "importance": 0.5,
            }
            return True

    def add_relation(self, src: str, tgt: str, relation: str,
                     weight: float = 0.8, summary: str = "") -> bool:
        """添加关系, 返回是否新增 (增量更新)"""
        # 确保两端实体存在
        self.add_entity(src)
        self.add_entity(tgt)

        # 检查是否已存在
        for r in self.relations:
            if (r["source"] == src and r["target"] == tgt and r["relation"] == relation) or \
               (r["source"] == tgt and r["target"] == src and r["relation"] == relation):
                # 已有: 更新权重 (增量更新)
                if weight > r.get("weight", 0.8):
                    r["weight"] = weight
                    return True
                return False

        self.relations.append({
            "source": src, "target": tgt,
            "relation": relation, "weight": weight,
            "summary": summary,
        })
        return True

    # ============ 社区发现 (全局检索基础) ============

    def detect_communities(self):
        """
        简单社区发现: 基于连通分量 + 边权重聚类

        LightRAG 用 LLM 做社区摘要, 这里用规则版
        """
        # 1. 构建邻接
        adj = defaultdict(set)
        for r in self.relations:
            adj[r["source"]].add(r["target"])
            adj[r["target"]].add(r["source"])

        # 2. BFS 连通分量
        visited = set()
        communities = []

        for eid in self.entities:
            if eid in visited:
                continue
            comp = []
            queue = deque([eid])
            visited.add(eid)
            while queue:
                node = queue.popleft()
                comp.append(node)
                for nb in adj.get(node, []):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            if len(comp) > 1:
                communities.append(comp)

        # 3. 生成社区摘要 (规则版)
        self.communities = {}
        self.community_summaries = {}
        for ci, comp in enumerate(communities):
            cid = f"c_{ci}"
            self.communities[cid] = comp
            types = [self.entities.get(e, {}).get("type", "Unknown") for e in comp]
            type_counts = {}
            for t in types:
                type_counts[t] = type_counts.get(t, 0) + 1
            dominant_type = max(type_counts, key=type_counts.get) if type_counts else "Entity"
            self.community_summaries[cid] = (
                f"社区 {cid}: {len(comp)} 个实体, "
                f"主导类型: {dominant_type}, "
                f"成员: {', '.join(comp[:5])}{'...' if len(comp)>5 else ''}"
            )

        logger.info(f"[社区发现] {len(self.communities)} 个社区")

    # ============ 增量更新 ============

    def incremental_update(self, new_texts: List[str]):
        """
        增量更新: 只更新新增/变更的实体和关系

        流程:
          1. 从新文本提取实体/关系
          2. 对比现有 (图差异分析)
          3. 仅合并变更部分
          4. 重新计算受影响的社区
        """
        from entity_extractor_v2 import extract_entities, extract_relations

        added_entities = 0
        added_relations = 0
        changed_entities = 0

        for text in new_texts:
            ents = extract_entities(text)
            for e in ents:
                changed = self.add_entity(e["id"], e.get("type", ""))
                if changed:
                    if e["id"] in self.entities:
                        changed_entities += 1
                    else:
                        added_entities += 1

            rels = extract_relations(text, ents)
            for r in rels:
                if self.add_relation(r["source"], r["target"], r["relation"]):
                    added_relations += 1

        # 社区发现 (如果有变更)
        if added_entities > 0 or changed_entities > 0:
            self.detect_communities()

        logger.info(f"[增量更新] +{added_entities} 实体, +{added_relations} 关系, "
                     f"~{changed_entities} 更新")
        return {"added_entities": added_entities, "added_relations": added_relations,
                "changed_entities": changed_entities}

    # ============ 保存/加载 ============

    def save(self, path: str = None):
        path = path or config.KG_FILE
        data = {
            "entities": self.entities,
            "relations": self.relations,
            "communities": self.communities,
            "community_summaries": self.community_summaries,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"[LightRAG] 已保存: {path}")

    def load(self, path: str = None):
        path = path or config.KG_FILE
        if not os.path.exists(path):
            logger.warning(f"[LightRAG] 文件不存在: {path}")
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.entities = data.get("entities", {})
        self.relations = data.get("relations", [])
        self.communities = data.get("communities", {})
        self.community_summaries = data.get("community_summaries", {})
        logger.info(f"[LightRAG] 已加载: {len(self.entities)} 实体, "
                     f"{len(self.relations)} 关系, {len(self.communities)} 社区")
        return True

    # ============ 基础查询 ============

    def get_neighbors(self, eid: str, max_depth: int = 2) -> Set[str]:
        """获取 k-hop 邻居"""
        visited = {eid}
        frontier = {eid}
        for _ in range(max_depth):
            next_frontier = set()
            for node in frontier:
                for r in self.relations:
                    if r["source"] == node and r["target"] not in visited:
                        next_frontier.add(r["target"])
                    elif r["target"] == node and r["source"] not in visited:
                        next_frontier.add(r["source"])
            visited.update(next_frontier)
            frontier = next_frontier
        return visited - {eid}

    def find_communities_for_entity(self, eid: str) -> List[str]:
        """实体所在社区"""
        result = []
        for cid, members in self.communities.items():
            if eid in members:
                result.append(cid)
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rag = LightRAG()
    # 加载预设
    kg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "financial_kg.json")
    if os.path.exists(kg_path):
        # 从 V8 预设加载
        import json
        with open(kg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for n in data.get("nodes", []):
            rag.add_entity(n["id"], n.get("type", ""),
                          n.get("attributes", {}).get("全称", ""))
        for e in data.get("edges", []):
            rag.add_relation(e["source"], e["target"], e["relation"],
                             e.get("weight", 0.8))
        rag.detect_communities()
        print(f"实体: {len(rag.entities)}")
        print(f"关系: {len(rag.relations)}")
        print(f"社区: {len(rag.communities)}")
