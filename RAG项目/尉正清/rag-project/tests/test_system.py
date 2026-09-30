# tests/test_system.py
"""系统与角色接口测试。"""
from tests.conftest import BASE_URL


class TestSystem:

    def test_root(self, client):
        r = client.get(BASE_URL + "/", timeout=15)
        assert r.status_code == 200
        body = r.json()
        assert body["name"]
        assert "/docs" in body["docs"]
        assert isinstance(body["endpoints"], list) and body["endpoints"]

    def test_health_all_dependencies_up(self, client):
        r = client.get(BASE_URL + "/api/health", timeout=20)
        assert r.status_code == 200
        data = r.json()["data"]
        for dep in ("mysql", "redis", "milvus", "llm"):
            assert data[dep] is True, "依赖 %s 不可用" % dep
        assert data["llm_model"]

    def test_stats_shape(self, client):
        r = client.get(BASE_URL + "/api/stats", params={"refresh": "true"}, timeout=60)
        assert r.status_code == 200
        data = r.json()["data"]
        assert "rag_knowledge" in data["collections"]
        assert data["collections"]["rag_knowledge"] > 0
        # 三个角色都应有知识
        for role in ("lawyer", "psychologist", "financial_advisor"):
            assert data["knowledge_per_role"].get(role, 0) > 0, "%s 无知识" % role
        assert isinstance(data["ingested_sources"], dict)

    def test_stats_cache_hit(self, client):
        """第二次请求应命中 Redis 缓存。"""
        client.get(BASE_URL + "/api/stats", params={"refresh": "true"}, timeout=60)
        r = client.get(BASE_URL + "/api/stats", timeout=30)
        assert r.status_code == 200
        assert "cached" in r.json()["msg"]

    def test_openapi_documented(self, client):
        """OpenAPI 文档可访问且覆盖全部路由。"""
        r = client.get(BASE_URL + "/openapi.json", timeout=20)
        assert r.status_code == 200
        paths = r.json()["paths"]
        for p in ("/api/chat/ask", "/api/roles", "/api/sessions",
                  "/api/ingest/dataset", "/api/update/document"):
            assert p in paths, "OpenAPI 缺少 %s" % p


class TestRoles:

    def test_list_roles(self, client):
        r = client.get(BASE_URL + "/api/roles", timeout=15)
        assert r.status_code == 200
        roles = r.json()["data"]
        keys = {x["role_key"] for x in roles}
        assert {"lawyer", "psychologist", "financial_advisor"} <= keys
        for x in roles:
            assert x["name"] and x["greeting"]

    def test_role_detail_contains_persona(self, client):
        r = client.get(BASE_URL + "/api/roles/lawyer", timeout=15)
        assert r.status_code == 200
        data = r.json()["data"]
        # 人设与规则由 MySQL 驱动，详情里必须能拿到
        assert data["persona"]
        assert data["rules"]
        assert data["fallback"]
        assert data["disclaimer"]

    def test_role_not_found(self, client):
        r = client.get(BASE_URL + "/api/roles/not_exist_role", timeout=15)
        assert r.status_code == 404

    def test_hot_roles(self, client):
        r = client.get(BASE_URL + "/api/roles/hot", params={"k": 5}, timeout=15)
        assert r.status_code == 200
        assert isinstance(r.json()["data"], list)
