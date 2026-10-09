# -*- coding: utf-8 -*-
"""
双层检索器 - LightRAG 核心
工单编号: 人工智能 NLP-RAG 项目-LightRAG 优化

局部检索 (Local): 具体实体 → 邻居扩展 → 精确匹配
全局检索 (Global): 社区发现 → 概念推理 → 高层摘要
"""
import os, sys, json, math, logging
from typing import List, Dict, Set, Tuple
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v12 as config
from light_rag import LightRAG

logger = logging.getLogger(__name__)


class DualRetriever:
    """双层检索器"""

    def __init__(self, rag: LightRAG):
        self.rag = rag

    # ============ 关键词分类 ============

    def classify_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        将问题关键词分为局部/全局

        局部: 具体实体 (公司名、人名、产品名、数值)
        全局: 概念/主题 (行业增长、技术标准、关联方)
        """
        try:
            import jieba
            tokens = [t.strip() for t in jieba.cut(query) if len(t.strip()) >= 2]
        except ImportError:
            tokens = [c for c in query if c.strip()]

        local_kws = []
        global_kws = []

        # 实体匹配 → 局部
        for tok in tokens:
            if tok in self.rag.entities:
                local_kws.append(tok)
                continue
            # 尝试模糊匹配
            matched = [eid for eid in self.rag.entities if tok in eid or eid in tok]
            if matched:
                local_kws.append(matched[0])
                continue

        # 剩余 → 全局 (概念关键词)
        global_kws = [tok for tok in tokens if tok not in local_kws]

        # 如果没有局部关键词, 强制用问题核心实体
        if not local_kws:
            # 尝试匹配所有实体
            for eid in self.rag.entities:
                for tok in tokens:
                    if tok in eid or eid in tok:
                        local_kws.append(eid)
                        break

        logger.info(f"[关键词分类] 局部={local_kws}, 全局={global_kws}")
        return {"local": local_kws, "global": global_kws}

    # ============ 局部检索 ============

    def local_retrieve(self, query: str, local_kws: List[str],
                        top_k: int = None) -> List[Dict]:
        """
        局部检索: 具体实体 → 邻居扩展 → 文本块

        1. 从局部关键词定位实体
        2. 扩展 k-hop 邻居
        3. 收集关联文本块
        """
        top_k = top_k or config.LOCAL_TOP_K
        results = []
        seen = set()

        for kw in local_kws:
            # 直接匹配实体
            matched = kw if kw in self.rag.entities else None
            if not matched:
                matched = next((eid for eid in self.rag.entities
                               if kw in eid or eid in kw), None)
            if not matched:
                continue

            # 收集邻居 (含关系)
            neighbors = self.rag.get_neighbors(matched, config.LOCAL_MAX_DEPTH)

            # 添加邻居节点
            for nid in neighbors:
                if nid not in seen:
                    seen.add(nid)
                    node = self.rag.entities.get(nid, {})
                    results.append({
                        "entity": nid,
                        "type": node.get("type", ""),
                        "summary": node.get("summary", ""),
                        "keywords": [kw],
                        "tier": "local_entity",
                    })

            # 添加关系 (含方向)
            for r in self.rag.relations:
                if r["source"] == matched or r["target"] == matched:
                    rel_key = f"{r['source']}|{r['relation']}|{r['target']}"
                    if rel_key not in seen:
                        seen.add(rel_key)
                        results.append({
                            "path": rel_key,
                            "source": r["source"],
                            "target": r["target"],
                            "relation": r["relation"],
                            "weight": r.get("weight", 0.8),
                            "tier": "local_relation",
                        })

        # 排序: 先关系 (更精确), 后实体
        results.sort(key=lambda x: 0 if "relation" in x.get("tier", "") else 1)
        return results[:top_k * 2]  # 局部可以多返回

    # ============ 全局检索 ============

    def global_retrieve(self, query: str, global_kws: List[str],
                         local_kws: List[str], top_k: int = None) -> List[Dict]:
        """
        全局检索: 社区发现 → 概念推理

        1. 基于局部关键词找到社区
        2. 基于全局关键词匹配社区摘要
        3. 扩展到相邻社区
        """
        top_k = top_k or config.GLOBAL_TOP_K
        results = []
        seen = set()

        # 1. 从局部关键词定位社区
        target_communities = set()
        for kw in local_kws:
            communities = self.rag.find_communities_for_entity(kw)
            target_communities.update(communities)

        # 2. 从全局关键词匹配社区摘要
        for cid, summary in self.rag.community_summaries.items():
            for kw in global_kws:
                if kw in summary and cid not in target_communities:
                    target_communities.add(cid)

        # 3. 收集社区信息
        for cid in list(target_communities)[:config.COMMUNITY_TOP_K]:
            if cid in seen:
                continue
            seen.add(cid)
            members = self.rag.communities.get(cid, [])
            summary = self.rag.community_summaries.get(cid, "")

            results.append({
                "community_id": cid,
                "summary": summary,
                "members": members[:10],  # 限制数量
                "member_count": len(members),
                "tier": "global_community",
            })

        # 4. 路径推理: 多实体之间的路径
        if len(local_kws) >= 2:
            for i in range(len(local_kws)):
                for j in range(i + 1, len(local_kws)):
                    paths = self._find_paths(local_kws[i], local_kws[j], max_depth=config.GLOBAL_MAX_DEPTH)
                    for p in paths[:2]:
                        path_key = "|".join(p["path"])
                        if path_key not in seen:
                            seen.add(path_key)
                            results.append({
                                "path": p["path"],
                                "relations": p["relations"],
                                "score": p["score"],
                                "tier": "global_path",
                            })

        return results[:top_k]

    def _find_paths(self, start: str, end: str, max_depth: int = 3) -> List[Dict]:
        """BFS 查找两实体间路径"""
        if start not in self.rag.entities or end not in self.rag.entities:
            return []

        paths = []
        queue = deque([(start, [start], [], 0.0)])
        visited = {start}

        while queue and len(paths) < 5:
            node, path, rels, score = queue.popleft()
            if len(path) - 1 >= max_depth:
                continue

            for r in self.rag.relations:
                next_node = None
                rel = r["relation"]
                if r["source"] == node and r["target"] not in path:
                    next_node = r["target"]
                elif r["target"] == node and r["source"] not in path:
                    next_node = r["source"]

                if next_node:
                    new_path = path + [next_node]
                    new_rels = rels + [rel]
                    new_score = (score + r.get("weight", 0.8)) / (len(new_path) - 1)

                    if next_node == end:
                        paths.append({
                            "path": new_path,
                            "relations": new_rels,
                            "score": round(new_score, 4),
                        })
                    elif next_node not in visited:
                        visited.add(next_node)
                        queue.append((next_node, new_path, new_rels, new_score))

        paths.sort(key=lambda x: x["score"], reverse=True)
        return paths

    # ============ 融合检索 ============

    def search(self, query: str) -> Dict:
        """
        双层检索完整流程

        Returns:
            {
                "local_results": [...],
                "global_results": [...],
                "all_contexts": [...],  # 供 LLM 的上下文
                "local_kws": [...],
                "global_kws": [...],
            }
        """
        # 1. 关键词分类
        kws = self.classify_keywords(query)
        local_kws = kws["local"]
        global_kws = kws["global"]

        # 2. 局部检索
        local_results = self.local_retrieve(query, local_kws)

        # 3. 全局检索
        global_results = self.global_retrieve(query, global_kws, local_kws)

        # 4. 生成 LLM 上下文
        all_contexts = []

        # 局部上下文 (优先, 更精确)
        for r in local_results:
            if r.get("tier") == "local_relation":
                all_contexts.append(
                    f"【局部关系】{r['source']} --[{r['relation']}]--> {r['target']} "
                    f"(权重={r.get('weight', 0.8)})"
                )
            elif r.get("tier") == "local_entity":
                all_contexts.append(
                    f"【局部实体】{r['entity']} (类型:{r.get('type', '未知')}) "
                    f"{r.get('summary', '')}"
                )

        # 全局上下文
        for r in global_results:
            if r.get("tier") == "global_community":
                all_contexts.append(
                    f"【全局社区】{r['summary']} "
                    f"(成员数:{r.get('member_count', 0)})"
                )
            elif r.get("tier") == "global_path":
                path_str = " → ".join([f"[{n}]" for n in r["path"]])
                rels_str = " → ".join(r["relations"])
                all_contexts.append(
                    f"【全局路径】{path_str} | 关系: {rels_str} | 置信度: {r.get('score', 0)}"
                )

        return {
            "local_results": local_results,
            "global_results": global_results,
            "all_contexts": all_contexts,
            "local_kws": local_kws,
            "global_kws": global_kws,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rag = LightRAG()
    kg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "financial_kg.json")
    if os.path.exists(kg_path):
        with open(kg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for n in data.get("nodes", []):
            rag.add_entity(n["id"], n.get("type", ""))
        for e in data.get("edges", []):
            rag.add_relation(e["source"], e["target"], e["relation"])
        rag.detect_communities()

    retriever = DualRetriever(rag)
    queries = [
        "武汉力源的控股股东是谁?",
        "销售部有几个下属?",
        "IC市场增长最快的行业?",
    ]
    for q in queries:
        print(f"\n=== {q} ===")
        result = retriever.search(q)
        print(f"  局部关键词: {result['local_kws']}")
        print(f"  全局关键词: {result['global_kws']}")
        print(f"  局部结果: {len(result['local_results'])}")
        print(f"  全局结果: {len(result['global_results'])}")
        for ctx in result["all_contexts"][:3]:
            print(f"  {ctx[:70]}...")
