"""阶段6 审核与发布 e2e 验收（对应验收项 a~e）。

前置：迁移已完成（11 版本 approved）；Redis/Milvus/MySQL 在线；Embedding/Reranker 走线上 API。
流程：
  d) 未登录 → 401；普通用户访问管理员接口 → 403
  a) 版本置回 pending_review（模拟新导入未审核，向量是历史遗留）→ 检索命中不了它
  b) 管理员 approve → 触发索引 → 同一问题能检索到
  c) 另一版本 reject → Milvus 向量数对比（删除后减少）
  e) 贴审核留痕（谁、何时、决定、note）
  收尾：c 步驳回的版本恢复 approved 并重新索引（恢复知识库原状）
"""
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

# 项目根（scripts/e2e → scripts → rag），供下面导入 scripts/_env 使用
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pymysql
import requests
from pymilvus import MilvusClient

from scripts._env import load_project_env, mysql_config

# 加载 .env（供 Milvus 统计与 CLI 索引子进程使用），顺带剔除 http(s)_proxy
load_project_env(PROJECT_ROOT)

SESSION = requests.Session()
SESSION.trust_env = False
BASE = "http://127.0.0.1:8123"
BACKEND_DIR = r"C:\Users\92842\Desktop\rag\backend"
PY = r"C:\Users\92842\anaconda3\python.exe"
RUN = uuid.uuid4().hex[:8]
ADMIN_EMAIL = f"e2e_admin_{RUN}@qq.com"
PLAIN_EMAIL = f"e2e_plain_{RUN}@qq.com"
PASSWORD = "E2ePass2026"
DOC_LAW = 9          # 中华人民共和国劳动合同法（version 153）
DOC_LAW_KEY = "doc-5124c49b45167a9b61b5868a"
DOC_YEER = 6         # 职工带薪年休假条例（version 159，29 chunks）
QUESTION = "试用期最长可以约定多久？"
# 数据库连接参数从 .env 读（MYSQL_HOST/PORT/USER/PASSWORD/DATABASE），源码零口令
DB = mysql_config()


def sql(query: str, args: tuple = ()): 
    conn = pymysql.connect(**DB)
    cur = conn.cursor()
    cur.execute(query, args)
    rows = cur.fetchall()
    conn.commit()
    conn.close()
    return rows


def start_backend():
    env = os.environ.copy()
    env.pop("http_proxy", None)
    env.pop("https_proxy", None)
    # 集成测试专用环境：ENVIRONMENT=test 才会启用固定验证码旁路
    # （仅用于集成测试，生产禁止启用；见 app/auth/service.py 旁路说明）
    env["ENVIRONMENT"] = "test"
    log = open(r"C:\Users\92842\Desktop\rag\e2e_review_backend.log", "ab", buffering=0)
    proc = subprocess.Popen(
        [PY, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8123"],
        cwd=BACKEND_DIR, env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    for _ in range(60):
        try:
            if SESSION.get(f"{BASE}/health/ready", timeout=3).status_code == 200:
                return proc
        except requests.RequestException:
            pass
        time.sleep(1)
    proc.kill()
    raise RuntimeError("backend not ready")


def register_and_login(email: str, promote_admin: bool = False) -> str:
    r = SESSION.post(f"{BASE}/api/v1/auth/register/code",
                     json={"email": email, "purpose": "register"}, timeout=30)
    # 集成测试旁路（ENVIRONMENT=test）：固定码放行，不再依赖验证码回显
    code = "000000"
    r = SESSION.post(f"{BASE}/api/v1/auth/register",
                     json={"email": email, "password": PASSWORD, "code": code}, timeout=30)
    assert r.status_code == 200, r.text
    if promote_admin:
        sql("UPDATE users SET is_admin=1 WHERE email=%s", (email,))
    r = SESSION.post(f"{BASE}/api/v1/auth/login",
                     json={"email": email, "password": PASSWORD}, timeout=30)
    return r.json()["data"]["access_token"]


def search(token: str) -> list[dict]:
    r = SESSION.post(f"{BASE}/api/v1/legal/search",
                     json={"query": QUESTION, "top_k": 10},
                     headers={"Authorization": f"Bearer {token}"}, timeout=120)
    assert r.status_code == 200, f"{r.status_code} {r.text[:300]}"
    return r.json()["data"]["results"]


def milvus_keys() -> set[str]:
    """拉取全量实体 key 计数（走查询路径，标记删除立即反映；count(*) 不反映删除）。"""
    client = MilvusClient(uri=f"http://{os.environ['MILVUS_HOST']}:{os.environ.get('MILVUS_PORT', '19530')}")
    rows = client.query(
        collection_name=os.environ.get("MILVUS_COLLECTION_NAME", "legal_documents"),
        filter="chunk_key != ''",
        output_fields=["chunk_key"],
        limit=16384,
    )
    return {row["chunk_key"] for row in rows}


def milvus_row_count() -> int:
    return len(milvus_keys())


def main() -> None:
    proc = start_backend()
    print("[OK] 后端启动（8123）")

    admin_token = register_and_login(ADMIN_EMAIL, promote_admin=True)
    plain_token = register_and_login(PLAIN_EMAIL)
    print(f"[OK] 管理员 {ADMIN_EMAIL}（is_admin=1）与普通用户 {PLAIN_EMAIL} 就绪")

    # ---------- d) 权限 ----------
    r = SESSION.get(f"{BASE}/api/v1/admin/documents?status=pending_review", timeout=30)
    assert r.status_code == 401, r.status_code
    print("[OK] d) 未登录 GET /admin/documents →", r.status_code)
    r = SESSION.post(f"{BASE}/api/v1/admin/documents/{DOC_LAW}/review",
                     json={"decision": "approve"}, timeout=30)
    assert r.status_code == 401
    print("[OK] d) 未登录 POST review →", r.status_code)
    r = SESSION.get(f"{BASE}/api/v1/admin/documents?status=pending_review",
                    headers={"Authorization": f"Bearer {plain_token}"}, timeout=30)
    assert r.status_code == 403, r.status_code
    print("[OK] d) 普通用户 GET /admin/documents →", r.status_code)
    r = SESSION.post(f"{BASE}/api/v1/admin/documents/{DOC_LAW}/review",
                     json={"decision": "approve"},
                     headers={"Authorization": f"Bearer {plain_token}"}, timeout=30)
    assert r.status_code == 403
    print("[OK] d) 普通用户 POST review →", r.status_code)

    # ---------- a) 未审核版本检索不到 ----------
    rows = sql("SELECT id, version_status FROM document_versions WHERE document_id=%s", (DOC_LAW,))
    print("[现状] 劳动合同法版本:", rows)
    sql("UPDATE document_versions SET version_status='pending_review' WHERE document_id=%s", (DOC_LAW,))
    print("[OK] a) 已把劳动合同法版本置回 pending_review（模拟新导入未审核）")
    results = search(plain_token)
    hit_docs = {item["document_id"] for item in results}
    print(f"[OK] a) 检索「{QUESTION}」返回 {len(results)} 条，涉及文档: {hit_docs}")
    assert DOC_LAW_KEY not in hit_docs, "未审核版本竟能被检索到！"
    print("[OK] a) 结果中无劳动合同法（doc-5124…）——未审核内容检索不到（兜底生效）")

    # ---------- b) approve 后能检索到 ----------
    r = SESSION.post(f"{BASE}/api/v1/admin/documents/{DOC_LAW}/review",
                     json={"decision": "approve", "review_note": "已核对官方来源、生效日期"},
                     headers={"Authorization": f"Bearer {admin_token}"}, timeout=300)
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    print(f"[OK] b) approve → {data}")
    assert data["index_verified"] is True
    results = search(plain_token)
    hit_docs = {item["document_id"] for item in results}
    print(f"[OK] b) 同问题检索返回 {len(results)} 条，涉及文档: {hit_docs}")
    assert DOC_LAW_KEY in hit_docs, "审核通过后仍检索不到！"
    print("[OK] b) 劳动合同法重新出现——发布生效")

    # ---------- c) reject 删向量 ----------
    # 该版本先置回 pending_review（模拟待审核提交；其向量为历史遗留，尚在库中）
    sql("UPDATE document_versions SET version_status='pending_review' WHERE document_id=%s", (DOC_YEER,))
    print("[OK] c) 职工带薪年休假条例版本已置回 pending_review")
    before = milvus_row_count()
    print(f"[OK] c) reject 前 Milvus row_count = {before}")
    r = SESSION.post(f"{BASE}/api/v1/admin/documents/{DOC_YEER}/review",
                     json={"decision": "reject", "review_note": "e2e 验收：验证驳回删向量"},
                     headers={"Authorization": f"Bearer {admin_token}"}, timeout=300)
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    print(f"[OK] c) reject → {data}")
    # Milvus 标记删除有秒级传播延迟：轮询等待计数稳定（最多 60 秒）
    target = before - data["deleted_vectors"]
    client = MilvusClient(uri=f"http://{os.environ['MILVUS_HOST']}:{os.environ.get('MILVUS_PORT', '19530')}")
    coll = os.environ.get("MILVUS_COLLECTION_NAME", "legal_documents")
    after = milvus_row_count()
    for _ in range(12):
        if after == target:
            break
        time.sleep(5)
        after = milvus_row_count()
    print(f"[OK] c) reject 后 Milvus 可见向量数 = {after}（删除前 {before}，删除 {data['deleted_vectors']} 条）")
    assert after == target

    # 被删 key 直接查证：该版本全部 chunk_key 在 Milvus 里已不可见
    version_keys = [r[0] for r in sql(
        "SELECT chunk_key FROM document_chunks WHERE document_version_id=(SELECT id FROM document_versions WHERE document_id=%s)",
        (DOC_YEER,))]
    remaining = client.query(
        collection_name=coll,
        filter="chunk_key in [" + ",".join(f"'{k}'" for k in version_keys) + "]",
        output_fields=["chunk_key"], limit=100,
    )
    print(f"[OK] c) 被删版本全部 {len(version_keys)} 个 chunk_key 抽查可见数 = {len(remaining)}（预期 0）")
    assert len(remaining) == 0
    rows = sql("SELECT version_status, reviewed_by, reviewed_at, review_note FROM document_versions WHERE document_id=%s", (DOC_YEER,))
    assert rows[0][0] == "rejected"

    # ---------- e) 审核留痕 ----------
    print("[OK] e) 审核记录（谁/何时/决定/note）:")
    for row in sql(
        "SELECT dv.version_key, dv.version_status, u.email, dv.reviewed_at, dv.review_note "
        "FROM document_versions dv LEFT JOIN users u ON dv.reviewed_by=u.user_key "
        "WHERE dv.document_id IN (%s,%s)", (DOC_LAW, DOC_YEER)
    ):
        print("   ", row)

    # ---------- 收尾：恢复 c 步驳回的版本 ----------
    sql("UPDATE document_versions SET version_status='approved', processing_status='awaiting_embedding', "
        "reviewed_by=NULL, reviewed_at=NULL, review_note=NULL WHERE document_id=%s", (DOC_YEER,))
    env = os.environ.copy()
    env.pop("http_proxy", None); env.pop("https_proxy", None)
    cli = subprocess.run(
        [PY, "-m", "app.cli.index_legal_documents", "--limit", "1000", "--embedding-batch-size", "32"],
        cwd=BACKEND_DIR, env=env, capture_output=True, text=True, timeout=600,
    )
    print("[收尾] 重新索引 CLI:", cli.stdout.strip()[:200], cli.stderr.strip()[:200] if cli.returncode else "")
    assert cli.returncode == 0
    final = milvus_row_count()
    dist = sql("SELECT version_status, COUNT(*) FROM document_versions GROUP BY version_status")
    print(f"[收尾] Milvus row_count 恢复为 {final}；版本状态分布: {dist}")

    proc.terminate(); proc.wait(timeout=15)
    print("\nE2E_REVIEW_ALL_PASS")


if __name__ == "__main__":
    main()
