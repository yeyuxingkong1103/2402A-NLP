# -*- coding: utf-8 -*-
"""
图像跨模态检索器
工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化

核心逻辑:
    1. 判定问题是否为图像查询 (含图像关键词)
    2. 预设描述检索: query关键词 ∩ 图像keywords → 打分
    3. CLIP 检索 (可选): 文本-图像向量相似度
    4. 融合排序 → Top-K 图像
"""
import os
import json
import logging
import numpy as np
from typing import List, Dict, Optional

import config_v4 as config
import image_understanding

logger = logging.getLogger(__name__)


def is_image_query(query: str) -> bool:
    """判定是否为图像查询"""
    hits = [kw for kw in config.IMAGE_QUERY_KEYWORDS if kw in query]
    if hits:
        logger.info(f"[图像路由] 问题含关键词: {hits}")
        return True
    return False


class ImageRetriever:
    """图像检索器"""

    def __init__(self):
        self.images: List[Dict] = []

    def load_images(self, images: List[Dict]):
        self.images = images
        logger.info(f"[图像检索] 加载 {len(images)} 条图像数据")

    def _score_preset(self, query: str, query_tokens: List[str],
                      img: Dict) -> float:
        """对预设图像描述打分"""
        keywords = img.get("keywords", [])
        description = img.get("description", "")
        img_type = img.get("image_type", "")

        # 1. 关键词匹配 (权重 0.5)
        kw_hits = sum(1 for t in query_tokens if t in keywords or t in description)
        kw_score = min(kw_hits / max(len(query_tokens), 1), 1.0)

        # 2. 图像类型匹配 (权重 0.2)
        type_bonus = 0.0
        for t in query_tokens:
            if t in img_type or t in query:
                type_bonus = 0.2
                break

        # 3. 结构化数据关键词 (权重 0.3)
        struct_text = json.dumps(img.get("structured_data", {}), ensure_ascii=False)
        struct_hits = sum(1 for t in query_tokens if t in struct_text)
        struct_score = min(struct_hits / max(len(query_tokens), 1), 1.0)

        return 0.5 * kw_score + 0.2 * type_bonus + 0.3 * struct_score

    def _score_clip(self, query: str, img: Dict) -> float:
        """CLIP 相似度 (可选)"""
        if not config.CLIP_AVAILABLE:
            return 0.0
        path = img.get("path")
        if not path or not os.path.exists(path):
            return 0.0
        return image_understanding.clip_similarity(query, path)

    def search(self, query: str, company_filter: str = None,
               top_k: int = 2) -> List[Dict]:
        """
        图像检索

        Returns:
            List[{"image", "score", "hit_keywords"}]
        """
        try:
            import jieba
            query_tokens = [t for t in jieba.cut(query)
                           if len(t.strip()) > 1]
        except ImportError:
            query_tokens = list(query)

        candidates = self.images
        if company_filter:
            candidates = [i for i in candidates
                          if i.get("company", "") == company_filter]

        scored = []
        for img in candidates:
            preset_score = self._score_preset(query, query_tokens, img)
            clip_score = self._score_clip(query, img)
            # 融合: 预设为主, CLIP 加权
            final = 0.7 * preset_score + 0.3 * clip_score
            if final > 0.1:
                hit_kw = [t for t in query_tokens
                         if t in img.get("keywords", []) or t in img.get("description", "")]
                scored.append({
                    "image": img,
                    "score": round(final, 4),
                    "preset_score": round(preset_score, 4),
                    "clip_score": round(clip_score, 4),
                    "hit_keywords": hit_kw,
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def format_for_llm(self, results: List[Dict]) -> str:
        """格式化图像检索结果供 LLM 使用"""
        parts = []
        for i, r in enumerate(results):
            img = r["image"]
            desc = img.get("description", "")
            sd = img.get("structured_data", {})
            sd_text = json.dumps(sd, ensure_ascii=False, indent=2) if sd else ""
            part = (
                f"【图像 {i+1}: {img.get('image_type', '未知')}】\n"
                f"描述: {desc}\n"
                f"结构化数据:\n{sd_text}"
            )
            parts.append(part)
        return "\n\n".join(parts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    retriever = ImageRetriever()
    imgs = image_understanding.load_preset_images()
    retriever.load_images(imgs)

    tests = [
        "武汉力源信息技术股份有限公司组织结构图中销售部有几个部门构成?",
        "2008年中国IC市场应用结构与增长图中增长率最快的是哪个行业?",
    ]
    for q in tests:
        print(f"\n问题: {q}")
        results = retriever.search(q, top_k=2)
        for r in results:
            img = r["image"]
            print(f"  → [{img['image_id']}] score={r['score']}, "
                  f"hit={r['hit_keywords']}")
