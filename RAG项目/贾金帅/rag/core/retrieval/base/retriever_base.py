# -*- coding: utf-8 -*-
"""检索引擎基类与统一结果定义。

所有检索引擎（向量 / 词法 / 图）都继承 :class:`BaseRetriever`，
并实现统一的 ``retrieve(query, top_k, **kwargs)`` 接口，便于检索路由
与融合模块无差别地消费多路结果。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RetrievalResult:
    """单条检索结果。

    Attributes
    ----------
    content : dict
        检索内容，推荐字段：name（主键，用于融合去重）、title、document、
        indication、ingredient、manufacturer 等。
    score : float
        引擎内部得分（越大越相关）。
    source : str
        来源引擎标识（vector / graph）。
    rank : int
        引擎内部排名（1 起）。
    reason : str
        匹配理由（供界面展示）。
    raw : Any
        原始数据（文档块 / 图节点等），可选。
    evidence_key : str | None
        物理证据粒度身份（chunk:id / 批准文号 / 药品记录唯一键）。默认 None 时
        由 ``key`` 回退到 content.fusion_key / name。
    product_key : str | None
        产品粒度身份（厂家+规格），用于跨引擎按产品聚合（预留）。
    entity_key : str | None
        实体粒度身份（规范药名），用于跨引擎按实体聚合（预留）。
    """

    content: dict[str, Any]
    score: float
    source: str
    rank: int = 1
    reason: str = ""
    raw: Any = None
    evidence_key: Optional[str] = None
    product_key: Optional[str] = None
    entity_key: Optional[str] = None

    @property
    def key(self) -> str:
        """融合去重主键：优先显式 evidence_key，其次 content.fusion_key，再次 name。"""
        if self.evidence_key:
            return str(self.evidence_key)
        fusion_key = self.content.get("fusion_key")
        if fusion_key:
            return str(fusion_key)
        name = self.content.get("name")
        if name:
            return str(name)
        return str(self.content)


@dataclass
class BaseRetriever(ABC):
    """检索引擎抽象基类。"""

    name: str = "base"
    top_k: int = 5

    @abstractmethod
    def retrieve(self, query: str, top_k: Optional[int] = None,
                 **kwargs) -> list[RetrievalResult]:
        """统一检索接口。

        Parameters
        ----------
        query : str
            原始自然语言查询。
        top_k : int | None
            返回条数上限；None 使用实例默认值。
        **kwargs
            引擎特定参数。

        Returns
        -------
        list[RetrievalResult]
            按相关性降序的检索结果。
        """
        raise NotImplementedError

    def is_available(self) -> bool:
        """引擎是否可用（依赖缺失 / 数据缺失时返回 False）。"""
        return True

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"<{type(self).__name__} name={self.name!r} available={self.is_available()}>"
