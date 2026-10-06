# -*- coding: utf-8 -*-
"""
知识图谱 - NetworkX 实现 (轻量)
工单编号: 人工智能 NLP-RAG-基于 Graph RAG 实现金融问答

如果安装了 networkx 则使用, 否则降级为内置轻量实现
"""
import json
import os
import logging
from typing import List, Dict, Set, Optional
from collections import deque

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logger.warning("未安装 networkx, 使用内置轻量图实现")


class SimpleGraph:
    """内置轻量图 (当 networkx 不可用时)"""

    def __init__(self):
        self.nodes: Dict[str, Dict] = {}  # id → {type, attributes}
        self.adj_out: Dict[str, List[Tuple[str, str, float]]] = {}  # src → [(tgt, relation, weight)]
        self.adj_in: Dict[str, List[Tuple[str, str, float]]] = {}

    def add_node(self, nid: str, ntype: str = "", attributes: Dict = None):
        self.nodes[nid] = {"type": ntype, "attributes": attributes or {}}
        if nid not in self.adj_out:
            self.adj_out[nid] = []
        if nid not in self.adj_in:
            self.adj_in[nid] = []

    def add_edge(self, src: str, tgt: str, relation: str, weight: float = 0.8):
        if src not in self.adj_out:
            self.adj_out[src] = []
        if tgt not in self.adj_in:
            self.adj_in[tgt] = []
        self.adj_out[src].append((tgt, relation, weight))
        self.adj_in[tgt].append((src, relation, weight))

    def neighbors(self, nid: str, direction: str = "both") -> List[Tuple[str, str, float]]:
        result = []
        if direction in ("out", "both") and nid in self.adj_out:
            result.extend(self.adj_out[nid])
        if direction in ("in", "both") and nid in self.adj_in:
            result.extend(self.adj_in[nid])
        return result

    def bfs(self, start: str, max_hops: int = 2) -> Set[str]:
        """BFS 子图遍历"""
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, hop = queue.popleft()
            if hop >= max_hops:
                continue
            for neighbor, _, _ in self.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, hop + 1))
        return visited


class KnowledgeGraph:
    """知识图谱封装"""

    def __init__(self):
        if HAS_NETWORKX:
            self.graph = nx.Graph()
            self.type = "networkx"
        else:
            self.graph = SimpleGraph()
            self.type = "simple"

    def load_from_json(self, path: str):
        """从 JSON 加载图谱"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.load_from_dict(data)

    def load_from_dict(self, data: Dict):
        """从字典加载图谱"""
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        for node in nodes:
            nid = node.get("id", "")
            if not nid:
                continue
            ntype = node.get("type", "")
            attrs = node.get("attributes", {})
            if self.type == "networkx":
                self.graph.add_node(nid, type=ntype, **attrs)
            else:
                self.graph.add_node(nid, ntype, attrs)

        for edge in edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            rel = edge.get("relation", "")
            w = edge.get("weight", 0.8)
            if not src or not tgt:
                continue
            if self.type == "networkx":
                self.graph.add_edge(src, tgt, relation=rel, weight=w)
            else:
                self.graph.add_edge(src, tgt, rel, w)

        logger.info(f"[知识图谱] 加载完成: {len(nodes)} 节点, {len(edges)} 边")

    def get_node(self, nid: str) -> Optional[Dict]:
        """获取节点"""
        if self.type == "networkx":
            if nid in self.graph.nodes:
                data = self.graph.nodes[nid]
                return {"id": nid, **data}
            return None
        else:
            return self.graph.nodes.get(nid)

    def search_nodes(self, query: str, top_k: int = 5) -> List[str]:
        """搜索节点 (关键词匹配)"""
        results = []
        query_lower = query.lower()
        if self.type == "networkx":
            for nid in self.graph.nodes:
                if query_lower in str(nid).lower() or str(nid).lower() in query_lower:
                    results.append(nid)
        else:
            for nid in self.graph.nodes:
                if query_lower in str(nid).lower() or str(nid).lower() in query_lower:
                    results.append(nid)
        return results[:top_k]

    def get_subgraph(self, center_nodes: List[str], max_hops: int = 2) -> Dict:
        """
        获取中心节点的 k-hop 子图

        Returns:
            {"nodes": [...], "edges": [...]}
        """
        visited = set()
        all_edges = []

        for nid in center_nodes:
            if self.type == "networkx":
                sub_nodes = set()
                frontier = {nid}
                sub_nodes.add(nid)
                for hop in range(max_hops):
                    next_frontier = set()
                    for node in frontier:
                        for neighbor in self.graph.neighbors(node):
                            if neighbor not in sub_nodes:
                                sub_nodes.add(neighbor)
                                next_frontier.add(neighbor)
                visited.update(sub_nodes)

                sub_edges = []
                for n1, n2, data in self.graph.edges(data=True):
                    if n1 in visited and n2 in visited:
                        sub_edges.append({
                            "source": n1, "target": n2,
                            "relation": data.get("relation", ""),
                            "weight": data.get("weight", 0.8),
                        })
                all_edges.extend(sub_edges)
            else:
                nodes = self.graph.bfs(nid, max_hops)
                visited.update(nodes)
                # 收集边
                for n in nodes:
                    for tgt, rel, w in self.graph.neighbors(n):
                        if tgt in visited:
                            all_edges.append({
                                "source": n, "target": tgt,
                                "relation": rel, "weight": w,
                            })

        # 去重边
        edge_keys = set()
        unique_edges = []
        for e in all_edges:
            key = f"{e['source']}|{e['target']}|{e['relation']}"
            if key not in edge_keys:
                edge_keys.add(key)
                unique_edges.append(e)

        return {
            "nodes": list(visited),
            "edges": unique_edges,
            "center_nodes": center_nodes,
        }

    def get_neighbors(self, nid: str) -> List[Dict]:
        """获取节点的直接邻居"""
        neighbors = []
        if self.type == "networkx":
            for neighbor, data in self.graph.adj[nid].items():
                neighbors.append({
                    "node": neighbor,
                    "relation": data.get("relation", ""),
                    "weight": data.get("weight", 0.8),
                })
        else:
            for tgt, rel, w in self.graph.neighbors(nid):
                neighbors.append({"node": tgt, "relation": rel, "weight": w})
        return neighbors

    def to_visualization_data(self, subgraph: Dict = None) -> Dict:
        """转换为前端可视化格式 (D3 / ECharts)"""
        if subgraph is None:
            # 返回全部
            if self.type == "networkx":
                nodes = [{"id": n, "type": d.get("type", ""),
                          "attributes": {k: v for k, v in d.items() if k != "type"}}
                         for n, d in self.graph.nodes(data=True)]
                edges = [{"source": n1, "target": n2,
                          "relation": d.get("relation", ""), "weight": d.get("weight", 0.8)}
                         for n1, n2, d in self.graph.edges(data=True)]
            else:
                nodes = [{"id": nid, "type": d.get("type", ""),
                          "attributes": d.get("attributes", {})}
                         for nid, d in self.graph.nodes.items()]
                edges = []
                seen = set()
                for src, lst in self.graph.adj_out.items():
                    for tgt, rel, w in lst:
                        key = f"{src}|{tgt}"
                        if key not in seen:
                            seen.add(key)
                            edges.append({"source": src, "target": tgt,
                                          "relation": rel, "weight": w})
        else:
            nodes = [{"id": n, "type": "", "attributes": {}} for n in subgraph["nodes"]]
            edges = subgraph["edges"]

        return {
            "nodes": nodes,
            "edges": edges,
            "categories": [
                {"name": "Company"}, {"name": "Person"}, {"name": "Product"},
                {"name": "Standard"}, {"name": "Financial"}, {"name": "Industry"},
            ],
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kg = KnowledgeGraph()
    kg.load_from_json(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "financial_kg.json"))

    print(f"\n节点数: {len(kg.graph.nodes if kg.type == 'networkx' else kg.graph.nodes)}")

    # 搜索
    matches = kg.search_nodes("武汉力源")
    print(f"搜索 '武汉力源': {matches}")

    # 子图
    sg = kg.get_subgraph(["武汉力源"], max_hops=2)
    print(f"武汉力源 2-hop 子图: {len(sg['nodes'])} 节点, {len(sg['edges'])} 边")

    for node in sg["nodes"][:5]:
        print(f"  - {node}")
