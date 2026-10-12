# -*- coding: utf-8 -*-
"""
工单十五：图像知识库（跨模态检索的核心增强）
将专利图纸（图1/图2/图3）的结构化描述与图号、页码绑定，
供检索器在检测到视觉引用时做强化召回。

数据来源：Qwen2-VL 对图纸页面的结构化解析（图号、部件、位置关系）
"""
import json
import os
from typing import Dict, Optional

# 默认内置图知识（CN100347506C 块状散料分配装置）
# 由 Qwen2-VL 解析第11页图3后生成，可被 data/image_knowledge.json 覆盖
_DEFAULT_IMAGE_KB: Dict[str, Dict] = {
    "fig_3": {
        "fig_no": 3,
        "page": 11,
        "caption": "图3 块状散料分配装置结构示意图",
        "image_path": "./data/pages_hd/page_11.png",
        "description": (
            "图3展示了一种块状散料分配装置的整体结构。装置包含：顶部的料斗/入口部件14，"
            "物料从部件14进入后向下流经部件13；部件13位于部件12的内部（套装在导管12之中）；"
            "部件12为输送导管，沿倾斜方向布置；部件10为外部槽体/壳体；"
            "部件11位于装置顶部，是用于调节链条位置的张紧/调节装置。"
            "链条作为分散装置的核心组件贯穿输送路径。"
        ),
        "components": {
            "10": "外部槽体/壳体（倾斜布置）",
            "11": "链条位置调节部件（位于装置顶部，操作它可调整链条位置）",
            "12": "输送导管",
            "13": "位于部件12内部的内套管/导向件（散料从14进入后经过13）",
            "14": "顶部入口料斗（位于整个装置的顶部，散料由此进入）",
        },
        "relations": [
            "部件13位于部件12（导管）的内部",
            "部件14位于整个装置的顶部",
            "散料从部件14进入后，下一步经过部件13",
            "操作部件11可以调整链条的位置",
            "链条是分散装置包含的核心组件",
        ],
    }
}


class ImageKnowledgeBase:
    """图号 → 结构化描述 的映射，支持查询时按图号/页码/部件检索"""

    def __init__(self, json_path: str = "./data/image_knowledge.json"):
        self.kb: Dict[str, Dict] = dict(_DEFAULT_IMAGE_KB)
        if os.path.isfile(json_path):
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self.kb.update(data)
            except Exception as e:
                print(f"[ImageKB] 加载 {json_path} 失败: {e}")

    def get(self, fig_no: Optional[int] = None, page: Optional[int] = None) -> Optional[Dict]:
        if fig_no is not None:
            return self.kb.get(f"fig_{fig_no}")
        if page is not None:
            for v in self.kb.values():
                if v.get("page") == page:
                    return v
        return None

    def get_description(self, fig_no: Optional[int] = None, page: Optional[int] = None) -> str:
        info = self.get(fig_no, page)
        if not info:
            return ""
        comps = "；".join(f"部件{k}={v}" for k, v in info.get("components", {}).items())
        rels = "；".join(info.get("relations", []))
        return f"【{info.get('caption','')}】{info.get('description','')}\n部件：{comps}\n关系：{rels}"

    def all_descriptions(self) -> str:
        return "\n\n".join(self.get_description(fig_no=int(k.split("_")[1])) for k in self.kb)

    def save(self, json_path: str = "./data/image_knowledge.json"):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.kb, f, ensure_ascii=False, indent=2)
