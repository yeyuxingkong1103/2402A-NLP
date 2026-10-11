# -*- coding: utf-8 -*-
"""DocumentQualityAssessmentSkill —— 配置加载（所有阈值可配置，不硬编码）"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List

try:
    import yaml
except Exception:  # yaml 可选
    yaml = None


@dataclass
class Config:
    # 1) 格式分布
    extensions: List[str] = field(default_factory=lambda: [".pdf", ".docx", ".doc", ".md", ".txt", ".xlsx", ".pptx"])

    # 2) PDF 页面类型识别
    pdf_min_chars_per_page: int = 20        # 每页字符数 < 该值 → 视为扫描页
    scan_page_ratio: float = 0.70           # 扫描页占比 > 该值 → 整篇「扫描型」
    text_page_ratio: float = 0.70           # 文字页占比 > 该值 → 整篇「文字型」，其余「混合型」

    # 3) 长度分布
    length_quantiles: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 0.9, 0.99])

    # 4) 重复检测
    simhash_bits: int = 64
    simhash_shingle: int = 3                # 字符 shingle 大小
    simhash_hamming_threshold: int = 3      # ≤ 该汉明距离 → 判为高相似「待确认」

    # 5) 敏感信息检测
    enable_phone: bool = True
    enable_email: bool = True
    enable_idcard: bool = True
    enable_bankcard: bool = False           # 默认关闭（误报率高）
    context_chars: int = 20                 # 每条命中附带的上下文长度

    # 6) 文档分类标签体系
    classify: bool = True

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        cfg = cls()
        if path and os.path.exists(path) and yaml is not None:
            data = yaml.safe_load(open(path, encoding="utf-8")) or {}
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)
