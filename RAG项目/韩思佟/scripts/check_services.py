"""验收真实 MySQL、Redis、Milvus；可选检查本地模型服务。"""
import argparse
import json
import sys
from urllib.request import Request, urlopen

from pymilvus import MilvusClient

from app import db
from app.config import load_env, setting
from app.memory import ConversationMemory


def check(require_llm=False):
    load_env()
    if db.backend() != "mysql":
        raise RuntimeError("当前不是 MySQL；请检查 .env.local")
    db.init_db()
    db.ping()
    if not any(role["name"] == "医生" for role in db.list_roles()):
        raise RuntimeError("MySQL 中缺少医生角色")

    memory = ConversationMemory()
    if memory.backend != "redis" or not memory.ping():
        raise RuntimeError("当前不是实际 Redis")
    health_key = "rag:health-check"
    memory.client.set(health_key, "ok", ex=30)
    if memory.client.get(health_key) != "ok":
        raise RuntimeError("Redis 读写验收失败")
    memory.client.delete(health_key)

    client = MilvusClient(setting("RAG_MILVUS_URI"))
    collection = "doctor_knowledge"
    if not client.has_collection(collection):
        raise RuntimeError("Milvus 中还没有 doctor_knowledge，请先导入知识库")
    rows = client.query(collection, filter="id >= 0", output_fields=["id"], limit=2000)
    count = len({row["id"] for row in rows})
    client.close()
    if count <= 0:
        raise RuntimeError("Milvus collection 为空")

    result = {"mysql": "ok", "redis": "ok", "milvus": "ok", "vectors": count}
    if require_llm:
        base = setting("LLM_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
        request = Request(base + "/models", headers={"Authorization": "Bearer " + setting("LLM_API_KEY", "EMPTY")})
        with urlopen(request, timeout=15) as response:
            models = json.load(response).get("data", [])
        expected = setting("LLM_MODEL")
        if not any(model.get("id") == expected for model in models):
            raise RuntimeError(f"模型服务已连接，但没有找到 {expected}")
        result["llm"] = expected
    print("SERVICES_OK: " + json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-llm", action="store_true")
    try:
        check(parser.parse_args().require_llm)
    except Exception as exc:
        print(f"SERVICES_FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
