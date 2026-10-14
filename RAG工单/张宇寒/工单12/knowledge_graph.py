"""小型、可解释、可增量更新的 LightRAG 知识图谱。"""

from __future__ import annotations

import json
import re
from pathlib import Path


TYPE_HINTS = {
    "有限公司": "公司", "股份": "公司", "销售部": "部门", "销售处": "销售处",
    "中心": "项目/机构", "平台": "项目/系统", "行业": "行业", "领域": "领域",
    "工程": "工程", "规范": "标准", "%": "比例", "万元": "金额", "亿元": "金额",
}


def infer_type(name: str) -> str:
    if re.fullmatch(r"\d{2,}(?:,\d{3})*(?:\.\d+)?(?:万元|亿元|万股)|\d+(?:\.\d+)?%", name):
        return "金融指标"
    for token, kind in TYPE_HINTS.items():
        if token in name:
            return kind
    if 2 <= len(name) <= 4 and name.endswith(("明", "克")):
        return "人物"
    return "实体"


class KnowledgeGraph:
    def __init__(self, nodes=None, edges=None):
        self.nodes = {n["name"]: dict(n) for n in (nodes or [])}
        self.edges = list(edges or [])
        self._edge_keys = {(e["source"], e["relation"], e["target"]) for e in self.edges}

    def add_triples(self, triples, source_id: str):
        before_nodes, before_edges = len(self.nodes), len(self.edges)
        for source, relation, target in triples:
            source, relation, target = source.strip(), relation.strip(), target.strip()
            if not source or not relation or not target:
                continue
            self.nodes.setdefault(source, {"name": source, "type": infer_type(source), "sources": []})
            self.nodes.setdefault(target, {"name": target, "type": infer_type(target), "sources": []})
            for name in (source, target):
                if source_id not in self.nodes[name]["sources"]:
                    self.nodes[name]["sources"].append(source_id)
            key = (source, relation, target)
            if key not in self._edge_keys:
                self.edges.append({"source": source, "relation": relation, "target": target, "sources": [source_id]})
                self._edge_keys.add(key)
            else:
                edge = next(e for e in self.edges if (e["source"], e["relation"], e["target"]) == key)
                if source_id not in edge["sources"]:
                    edge["sources"].append(source_id)
        return {"nodes_added": len(self.nodes) - before_nodes, "edges_added": len(self.edges) - before_edges}

    def neighbors(self, name: str, depth: int = 1):
        seen, frontier = {name}, {name}
        for _ in range(max(1, depth)):
            next_frontier = set()
            for edge in self.edges:
                if edge["source"] in frontier:
                    next_frontier.add(edge["target"])
                if edge["target"] in frontier:
                    next_frontier.add(edge["source"])
            next_frontier -= seen
            seen |= next_frontier
            frontier = next_frontier
        return [self.nodes[n] for n in seen if n != name and n in self.nodes]

    def paths_for(self, names: list[str]):
        selected = set(names)
        return [e for e in self.edges if e["source"] in selected or e["target"] in selected]

    def to_dict(self):
        return {"nodes": list(self.nodes.values()), "edges": self.edges}

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["nodes"], data["edges"])


SEED_TRIPLES = [
    ("武汉力源信息技术股份有限公司", "HAS_DEPARTMENT", "销售部"),
    ("销售部", "HAS_DEPARTMENT", "渠道销售部"), ("销售部", "HAS_DEPARTMENT", "电话及网络销售部"),
    ("销售部", "HAS_DEPARTMENT", "大客户销售部"), ("销售部", "HAS_DEPARTMENT", "国际贸易部"),
    *[("大客户销售部", "HAS_SALES_OFFICE", f"{city}销售处") for city in ["北京", "深圳", "广州", "成都", "珠海", "武汉"]],
    ("2008年中国IC市场", "FASTEST_GROWTH", "汽车电子行业"), ("汽车电子行业", "HAS_GROWTH_RATE", "14.0%"),
    ("2008年中国IC市场", "NEGATIVE_GROWTH", "IC卡行业"), ("IC卡行业", "HAS_GROWTH_RATE", "-2.0%"),
    ("武汉力源信息技术股份有限公司", "ISSUES_SHARES", "1,670万股"), ("1,670万股", "AFTER_ISSUE_RATIO", "25.04%"),
    *[("武汉力源信息技术股份有限公司", "INVESTS_IN", x) for x in ["仓储及物流中心", "研发中心", "电子商务平台", "扩充产品种类和数量", "营运资金"]],
    ("赵马克", "CONTROLS", "武汉力源信息技术股份有限公司"), ("赵马克", "HAS_SHARE_RATIO", "42.35%"),
    *[(x, "ASSOCIATED_WITH", "武汉力源信息技术股份有限公司") for x in ["融冰投资", "武汉博润", "上海博润", "听音投资", "联众聚源", "力源贸易", "普芯达"]],
    ("武汉兴图新科电子股份有限公司", "PARTICIPATED_IN_STANDARD", "某视频技术规范1.0"),
    ("武汉兴图新科电子股份有限公司", "IMPORTANT_SUPPLIER_IN", "军队视频指挥领域"),
    *[("电子信息行业", "HAS_DOWNSTREAM", x) for x in ["军队", "政府机关", "能源行业"]],
    *[(x, "UPSTREAM_OF", "电子信息行业") for x in ["电子元器件制造企业", "金属壳体及机箱机柜制造企业"]],
    ("武汉兴图新科电子股份有限公司", "PARTICIPATED_IN", "某情报、指挥、控制与通信网络一体化工程"),
    ("某情报、指挥、控制与通信网络一体化工程", "WON_AWARD", "国家科技进步一等奖"),
    ("武汉兴图新科电子股份有限公司", "HAS_REGISTERED_CAPITAL", "5,520万元"),
    ("武汉兴图新科电子股份有限公司", "LEGAL_REPRESENTATIVE", "程家明"),
    ("武汉兴图新科电子股份有限公司", "USES_PROCEEDS_FOR_LIQUIDITY", "15,000.00万元"),
    ("15,000.00万元", "EQUALS", "1.5亿元"),
]

