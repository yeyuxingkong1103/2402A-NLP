"""Query rewriting, expansion, and intent recognition for RAG.

This module improves retrieval by:
1. Rewriting unclear queries
2. Expanding queries with synonyms and related terms
3. Identifying user intent
4. Extracting key entities
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class QueryAnalysis:
    """Analysis result of a user query."""

    original_query: str
    rewritten_query: str | None
    expanded_queries: list[str]
    intent: str
    entities: dict[str, list[str]]
    language: str
    confidence: float


class QueryRewriter:
    """Rewrite and expand user queries for better retrieval."""

    def __init__(self, llm_client: Any | None = None) -> None:
        self.llm_client = llm_client
        self._synonym_rules = self._load_synonym_rules()
        self._intent_patterns = self._load_intent_patterns()

    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze user query and return comprehensive analysis."""
        # Detect language
        language = self._detect_language(query)

        # Extract entities
        entities = self._extract_entities(query)

        # Identify intent
        intent = self._identify_intent(query)

        # Rewrite query if needed
        rewritten = self._rewrite_query(query)

        # Expand query
        expanded = self._expand_query(rewritten or query)

        return QueryAnalysis(
            original_query=query,
            rewritten_query=rewritten,
            expanded_queries=expanded,
            intent=intent,
            entities=entities,
            language=language,
            confidence=0.8,
        )

    def _detect_language(self, query: str) -> str:
        """Detect query language (simple heuristic)."""
        # Check for Chinese characters
        chinese_chars = len(re.findall(r'[一-鿿]', query))
        total_chars = len(query.replace(' ', ''))

        if total_chars == 0:
            return "unknown"

        chinese_ratio = chinese_chars / total_chars
        return "zh" if chinese_ratio > 0.3 else "en"

    def _extract_entities(self, query: str) -> dict[str, list[str]]:
        """Extract entities from query using rule-based approach."""
        entities: dict[str, list[str]] = {
            "product": [],
            "action": [],
            "time": [],
            "number": [],
        }

        # Extract numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', query)
        entities["number"] = numbers

        # Extract time expressions
        time_patterns = [
            r'\d+天',
            r'\d+个?工作日',
            r'\d+[年月日]',
            r'昨天|今天|明天',
        ]
        for pattern in time_patterns:
            matches = re.findall(pattern, query)
            entities["time"].extend(matches)

        # Extract common product terms
        product_keywords = ['商品', '产品', '订单', '货物', '物流', '快递']
        for keyword in product_keywords:
            if keyword in query:
                entities["product"].append(keyword)

        # Extract action verbs
        action_keywords = ['退货', '换货', '退款', '发货', '查询', '申请', '取消']
        for keyword in action_keywords:
            if keyword in query:
                entities["action"].append(keyword)

        return {k: v for k, v in entities.items() if v}

    def _identify_intent(self, query: str) -> str:
        """Identify user intent from query."""
        for intent, patterns in self._intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return intent

        return "general_inquiry"

    def _rewrite_query(self, query: str) -> str | None:
        """Rewrite query for better clarity (rule-based)."""
        # Remove filler words
        fillers = ['请问', '我想问', '能不能', '可不可以', '麻烦', '谢谢']
        rewritten = query
        for filler in fillers:
            rewritten = rewritten.replace(filler, '')

        # Normalize punctuation
        rewritten = re.sub(r'[？?]+', '？', rewritten)
        rewritten = re.sub(r'[！!]+', '！', rewritten)

        # Remove extra spaces
        rewritten = re.sub(r'\s+', ' ', rewritten).strip()

        # Only return if actually changed
        if rewritten != query and len(rewritten) > 0:
            return rewritten

        return None

    def _expand_query(self, query: str) -> list[str]:
        """Expand query with synonyms and related terms."""
        expanded = [query]

        # Apply synonym rules
        for term, synonyms in self._synonym_rules.items():
            if term in query:
                for synonym in synonyms:
                    expanded_query = query.replace(term, synonym)
                    if expanded_query not in expanded:
                        expanded.append(expanded_query)

        # Limit expansions
        return expanded[:5]

    def _load_synonym_rules(self) -> dict[str, list[str]]:
        """Load synonym rules for query expansion."""
        return {
            "退货": ["退回", "退换", "退掉"],
            "换货": ["更换", "调换"],
            "退款": ["退钱", "返款", "返钱"],
            "发货": ["出货", "寄货", "寄出"],
            "物流": ["快递", "配送", "运输"],
            "多久": ["多长时间", "需要多少天", "要等多久"],
            "怎么": ["如何", "怎样"],
            "可以": ["能不能", "是否可以"],
        }

    def _load_intent_patterns(self) -> dict[str, list[str]]:
        """Load intent patterns for classification."""
        return {
            "refund_inquiry": [
                r'退货|退款|退钱|返款',
                r'不想要|不要了',
            ],
            "exchange_inquiry": [
                r'换货|更换|调换',
                r'换.*型号|换.*颜色',
            ],
            "shipping_inquiry": [
                r'物流|快递|配送|发货|到货',
                r'什么时候.*到|多久.*送到',
            ],
            "warranty_inquiry": [
                r'保修|维修|质保',
                r'坏了|故障|问题',
            ],
            "order_status": [
                r'订单.*状态|查询.*订单',
                r'到哪里了|处理.*情况',
            ],
            "product_inquiry": [
                r'产品|商品|型号',
                r'有.*吗|是否.*有',
            ],
        }

    def rewrite_with_llm(self, query: str, context: str | None = None) -> str:
        """Use LLM to rewrite query (advanced, requires LLM)."""
        if not self.llm_client:
            return query

        prompt = f"""请将用户的问题改写得更加清晰、完整。保持原意，但使其更适合知识库检索。

原问题：{query}

改写要求：
1. 补充缺失的上下文
2. 使用更准确的术语
3. 保持简洁
4. 只输出改写后的问题，不要解释

改写后的问题："""

        try:
            from app.llm.client import Message

            messages = [Message(role="user", content=prompt)]
            result = self.llm_client.generate(messages, temperature=0.3, max_tokens=200)
            rewritten = result.content.strip()

            # Validate rewritten query
            if len(rewritten) > 10 and len(rewritten) < 200:
                logger.info(f"LLM rewrote query: '{query}' -> '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.warning(f"LLM rewrite failed: {e}")

        return query


class HybridQueryBuilder:
    """Build queries for hybrid search (vector + keyword)."""

    def __init__(self, query_rewriter: QueryRewriter | None = None) -> None:
        self.query_rewriter = query_rewriter or QueryRewriter()

    def build(self, query: str) -> dict[str, Any]:
        """Build hybrid search queries."""
        analysis = self.query_rewriter.analyze(query)

        # Extract keywords for BM25
        keywords = self._extract_keywords(analysis.original_query)

        # Build filter conditions
        filters = self._build_filters(analysis)

        return {
            "vector_query": analysis.rewritten_query or analysis.original_query,
            "expanded_queries": analysis.expanded_queries,
            "bm25_keywords": keywords,
            "intent": analysis.intent,
            "entities": analysis.entities,
            "filters": filters,
            "language": analysis.language,
        }

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract keywords for BM25 search."""
        # Simple keyword extraction (in production, use jieba or similar)
        # Remove common stop words
        stop_words = {'的', '了', '吗', '呢', '啊', '呀', '吧', '是', '在', '有', '和', '与'}

        # Split and filter
        words = re.findall(r'[\w]+', query)
        keywords = [w for w in words if w not in stop_words and len(w) > 1]

        return keywords[:10]  # Limit to top 10

    def _build_filters(self, analysis: QueryAnalysis) -> dict[str, Any]:
        """Build filter conditions based on query analysis."""
        filters: dict[str, Any] = {}

        # Add intent-based filters
        if analysis.intent == "shipping_inquiry":
            # Prefer documents about shipping
            filters["prefer_tags"] = ["物流", "配送", "发货"]

        if analysis.intent == "refund_inquiry":
            filters["prefer_tags"] = ["退货", "退款"]

        # Add entity-based filters
        if "time" in analysis.entities:
            filters["has_time_info"] = True

        return filters
