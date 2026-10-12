# -*- coding: utf-8 -*-
"""
工单十四：单元测试
"""
import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestQueryRouter:
    """测试查询路由器"""

    def test_text_query_detection(self):
        from src.query_router import is_image_query
        assert is_image_query("该静电除尘器的发明人是？") is False
        assert is_image_query("以下哪个描述符合该静电除尘器的特征？") is False

    def test_image_query_detection(self):
        from src.query_router import is_image_query
        assert is_image_query("第7页的图片中，部件4相对于部件5的位置关系") is True
        assert is_image_query("尺寸X1，X2，X3分别代表什么部件的间隔距离") is True
        assert is_image_query("气流方向首先经过哪个部件") is True

    def test_page_detection(self):
        from src.query_router import detect_page
        assert detect_page("在文件中第7页的图片中") == 7
        assert detect_page("普通问题") is None

    def test_route_query(self):
        from src.query_router import route_query
        r = route_query("第7页图片中部件4的位置")
        assert r["query_type"] == "image"
        assert r["page"] == 7
        r = route_query("发明人是谁")
        assert r["query_type"] == "text"


class TestPdfParser:
    """测试PDF图片提取"""

    def test_extract_images(self):
        from src.pdf_parser import extract_images_from_pdf
        pdf_path = "./附件/CN100342976C.pdf"
        if not os.path.exists(pdf_path):
            pytest.skip("PDF文件不存在")
        result = extract_images_from_pdf(pdf_path, "./data/test_pages")
        assert result["total_pages"] == 8
        assert len(result["pages"]) == 8
        assert all(p.get("raw_image") or p.get("rendered_image") for p in result["pages"])


class TestChunker:
    """测试分块器"""

    def test_chunk_ocr_results(self):
        from src.chunker import chunk_ocr_results
        ocr_results = [
            {"page": 1, "text": "这是第一页的文本内容。" * 50},
            {"page": 2, "text": "这是第二页的文本内容。" * 30},
        ]
        result = chunk_ocr_results(ocr_results, "test_doc", "test.pdf")
        assert result["total_chunks"] > 0
        assert all("page" in c and "text" in c for c in result["chunks"])


class TestOcrResults:
    """测试OCR结果文件存在且包含关键信息"""

    def test_ocr_file_exists(self):
        ocr_path = "./data/ocr/all_pages_ocr_vlm.json"
        assert os.path.exists(ocr_path), "OCR结果文件不存在"

    def test_ocr_contains_inventor(self):
        ocr_path = "./data/ocr/all_pages_ocr_vlm.json"
        if not os.path.exists(ocr_path):
            pytest.skip("OCR结果不存在")
        with open(ocr_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_text = " ".join(p.get("text", "") for p in data)
        assert "吉特勒" in all_text, "OCR未识别到发明人姓名"

    def test_ocr_contains_feature(self):
        ocr_path = "./data/ocr/all_pages_ocr_vlm.json"
        if not os.path.exists(ocr_path):
            pytest.skip("OCR结果不存在")
        with open(ocr_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_text = " ".join(p.get("text", "") for p in data)
        assert "圆锥形" in all_text and "80" in all_text, "OCR未识别到特征描述"


class TestKBExists:
    """测试知识库构建"""

    def test_kb_mapping_exists(self):
        assert os.path.exists("./data/kb_mapping.json"), "知识库映射文件不存在"

    def test_milvus_has_data(self):
        from src.vector_store import VectorStore
        vs = VectorStore()
        stats = vs.get_stats()
        assert stats.get("exists"), "Milvus collection不存在"
        # 远程 Milvus num_entities 可能返回 None，改用 query 验证
        chunks = vs.get_all_chunks()
        assert len(chunks) > 0, "Milvus中没有数据"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
