# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
tests/test_api_v3.py —— 工单三 API 接口单测

覆盖：
  1. 健康检查
  2. /api/v3/ask RAG 模式
  3. /api/v3/ask 纯 LLM 模式
  4. /api/v3/tables/{doc_id}
  5. /api/v3/tables/404
"""
import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

# 工单三：项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

# 工单三：显存优化
os.environ.setdefault("RAG_EMBED_DEVICE", "cuda")
os.environ.setdefault("RAG_EMBED_BATCH_SIZE", "8")
os.environ.setdefault("RAG_EMBED_MAX_SEQ", "512")

from src.api_v3 import app

client = TestClient(app)
_HAS_LLM = bool(os.getenv("DEEPSEEK_API_KEY")
                 and not os.getenv("DEEPSEEK_API_KEY", "").startswith("your_key"))


# ================= 1. 健康检查 =================
def test_health():
    """工单三：健康检查"""
    r = client.get("/api/v3/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["version"] == "v3"
    assert data["table_collection"] == "rag_tables"


# ================= 2. RAG 模式 =================
@pytest.mark.skipif(not _HAS_LLM, reason="无 DEEPSEEK_API_KEY")
def test_ask_rag_mode():
    """工单三：RAG 模式（表格+文本融合检索）"""
    r = client.post("/api/v3/ask", json={
        "question": "武汉力源信息技术股份有限公司本次发行股数是多少？",
        "doc_id": "招股说明书2",
        "use_table": True,
        "use_rag": True,
        "top_k": 3,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "rag_v3"
    assert len(data["answer"]) > 0
    assert data["latency_ms"] > 0
    assert "route" in data
    # 表格类问题应有表格候选
    assert data["route"]["route"] in ("table_only", "hybrid")


# ================= 3. 纯 LLM 模式 =================
@pytest.mark.skipif(not _HAS_LLM, reason="无 DEEPSEEK_API_KEY")
def test_ask_pure_llm_mode():
    """工单三：纯 LLM 模式（不检索）"""
    r = client.post("/api/v3/ask", json={
        "question": "什么是招股说明书？",
        "use_rag": False,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "pure_llm"
    assert len(data["answer"]) > 0
    assert len(data["references"]) == 0


# ================= 4. 获取文档表格 =================
def test_get_tables():
    """工单三：获取招股说明书2 的表格"""
    r = client.get("/api/v3/tables/招股说明书2")
    assert r.status_code == 200
    data = r.json()
    assert data["doc_id"] == "招股说明书2"
    assert data["total"] > 0
    assert len(data["tables"]) > 0
    # 检查表格结构
    tbl = data["tables"][0]
    assert "table_id" in tbl
    assert "rows" in tbl


# ================= 5. 404 表格 =================
def test_get_tables_not_found():
    """工单三：不存在的文档返回 404"""
    r = client.get("/api/v3/tables/nonexistent_doc")
    assert r.status_code == 404


# ================= 6. OpenAPI 文档 =================
def test_openapi_docs():
    """工单三：OpenAPI 文档可访问"""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    assert spec["info"]["title"].startswith("PDF 表格解析")
    assert "/api/v3/ask" in spec["paths"]
    assert "/api/v3/tables/{doc_id}" in spec["paths"]
