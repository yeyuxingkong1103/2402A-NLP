# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-LightRAG优化
tests/test_lightrag_v12.py —— LightRAG 模块单元测试
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lightrag_v12.rule_based_llm import (
    extract_entities, extract_relations,
    rule_based_entity_extraction, rule_based_keyword_extraction,
)


class TestRuleBasedEntityExtraction:
    """工单十二：规则实体抽取测试"""

    def test_extract_company(self):
        text = "武汉力源信息技术股份有限公司注册资本3000万元。"
        result = extract_entities(text)
        assert "武汉力源信息技术股份有限公司" in result
        assert "Organization" in result

    def test_extract_monetary_value(self):
        text = "公司注册资本3000万元，营业收入10亿元。"
        result = extract_entities(text)
        assert "3000万元" in result
        assert "10亿元" in result
        assert "MonetaryValue" in result

    def test_extract_percentage(self):
        text = "持股比例为35%。"
        result = extract_entities(text)
        assert "35%" in result
        assert "Percentage" in result

    def test_extract_relation_registered_capital(self):
        text = "武汉力源信息技术股份有限公司注册资本3000万元。"
        result = extract_relations(text)
        assert "注册资本" in result
        assert "武汉力源信息技术股份有限公司" in result
        assert "3000万元" in result

    def test_extract_relation_shareholding(self):
        text = "公司控股股东为武汉力源创业投资有限公司，持股比例为35%。"
        result = extract_relations(text)
        assert "持股比例" in result
        assert "35%" in result

    def test_full_entity_extraction_format(self):
        text = "武汉力源信息技术股份有限公司注册资本3000万元。"
        result = rule_based_entity_extraction(text)
        # 工单十二：LightRAG 格式 entity<|#|>name<|#|>type<|#|>desc
        for line in result.split("\n"):
            if line.startswith("entity"):
                parts = line.split("<|#|>")
                assert len(parts) == 4
            elif line.startswith("relation"):
                parts = line.split("<|#|>")
                assert len(parts) == 5

    def test_keyword_extraction(self):
        text = "武汉力源信息技术股份有限公司注册资本"
        result = rule_based_keyword_extraction(text)
        assert isinstance(result, str)
        assert len(result) > 0


class TestLightRAGWrapper:
    """工单十二：LightRAG 封装测试"""

    def test_get_lightrag(self):
        from src.lightrag_v12 import get_lightrag
        rag = get_lightrag()
        assert rag is not None
        assert rag.working_dir is not None

    def test_get_graph_stats(self):
        from src.lightrag_v12.lightrag_wrapper import get_graph_stats
        stats = get_graph_stats()
        assert "entities" in stats
        assert "relations" in stats
        assert isinstance(stats["entities"], int)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
