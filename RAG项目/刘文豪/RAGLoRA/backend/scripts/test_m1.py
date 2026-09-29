# -*- coding: utf-8 -*-
"""M1 验收脚本：健康检查 + 注册/登录/鉴权 + 角色 CRUD + 多用户隔离。

用法（服务需已启动）：
    D:\\anaconda3\\envs\\rag_env\\python.exe scripts\\test_m1.py
"""
import secrets
import sys
import time
from pathlib import Path

import httpx

# 允许以 `python scripts/test_m1.py` 方式运行（把 backend/ 加入模块搜索路径）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = "http://127.0.0.1:8000/api"
ok_count = 0
fail_count = 0


def check(label: str, cond: bool, detail: str = ""):
    global ok_count, fail_count
    if cond:
        ok_count += 1
        print(f"  [PASS] {label}" + (f"  {detail}" if detail else ""))
    else:
        fail_count += 1
        print(f"  [FAIL] {label}  {detail}")


def main():
    c = httpx.Client(timeout=30)

    print("=" * 70)
    print("① 健康检查")
    print("=" * 70)
    r = c.get(f"{BASE}/health", params={"deep": True})
    h = r.json()
    for name, info in h["checks"].items():
        mark = "OK  " if info.get("ok") else "FAIL"
        extra = info.get("version") or info.get("device") or info.get("collections") or info.get("error") or ""
        print(f"  {mark} {name:<8} {str(extra)[:70]}")
    check("四项基础依赖全绿", all(h["checks"][k]["ok"] for k in ("mysql", "redis", "qdrant", "ollama")))
    if "models" in h["checks"]:
        check("本地模型文件就位", h["checks"]["models"]["ok"],
              f"sparse_linear={h['checks']['models']['embed']['sparse_linear']}")
        check("GPU 可用", h["checks"]["gpu"]["ok"],
              f"{h['checks']['gpu'].get('device','')} 空闲 {h['checks']['gpu'].get('vram_free_gb','?')}GB")
    print()

    print("=" * 70)
    print("② 鉴权：注册 / 登录 / Token 校验")
    print("=" * 70)
    uname = f"alice_{int(time.time())}"
    pw = secrets.token_hex(8)   # 临时用户密码每次随机生成，不写死
    r = c.post(f"{BASE}/auth/register", json={"username": uname, "password": pw, "display_name": "Alice"})
    check("注册成功", r.status_code == 200, f"HTTP {r.status_code}")
    token = r.json()["access_token"]
    uid = r.json()["user"]["id"]
    print(f"       用户 id={uid}  token={token[:38]}…")

    r = c.post(f"{BASE}/auth/register", json={"username": uname, "password": pw})
    check("重复用户名被拒 (409)", r.status_code == 409, f"HTTP {r.status_code}")

    r = c.post(f"{BASE}/auth/login", json={"username": uname, "password": "wrongpass"})
    check("错误密码被拒 (401)", r.status_code == 401, f"HTTP {r.status_code}")

    r = c.post(f"{BASE}/auth/login", json={"username": uname, "password": pw})
    check("正确密码登录成功", r.status_code == 200)
    token = r.json()["access_token"]

    r = c.get(f"{BASE}/auth/me")
    check("无 Token 访问被拒 (401)", r.status_code == 401, f"HTTP {r.status_code}")

    r = c.get(f"{BASE}/auth/me", headers={"Authorization": f"Bearer {token}"})
    check("带 Token 取到本人信息", r.status_code == 200 and r.json()["username"] == uname)

    r = c.get(f"{BASE}/auth/me", headers={"Authorization": "Bearer tampered.token.here"})
    check("伪造 Token 被拒 (401)", r.status_code == 401, f"HTTP {r.status_code}")
    print()

    H = {"Authorization": f"Bearer {token}"}

    print("=" * 70)
    print("③ 角色：列表 / 详情 / 新建 / 修改 / 删除保护")
    print("=" * 70)
    r = c.get(f"{BASE}/characters", headers=H)
    chars = r.json()
    check("角色列表可取", r.status_code == 200 and len(chars) >= 2, f"共 {len(chars)} 个")
    for ch in chars:
        print(f"       [{ch['id']}] {ch['avatar']} {ch['name']:<12} 知识库={ch['kb_collection']}  内置={ch['is_builtin']}")

    did = next(ch["id"] for ch in chars if ch["slug"] == "doctor")
    lid = next(ch["id"] for ch in chars if ch["slug"] == "lawyer")

    r = c.get(f"{BASE}/characters/{did}", headers=H)
    d = r.json()
    check("角色详情含人格三层", all(d.get(k) for k in ("identity_block", "style_json", "domain_constraints", "prompt_template")))
    check("医生绑定 kb_medical", d["kb_collection"] == "kb_medical")
    check("律师绑定 kb_legal", c.get(f"{BASE}/characters/{lid}", headers=H).json()["kb_collection"] == "kb_legal")

    r = c.delete(f"{BASE}/characters/{did}", headers=H)
    check("内置角色拒绝删除 (403)", r.status_code == 403, f"HTTP {r.status_code}")

    r = c.post(f"{BASE}/characters", headers=H, json={
        "slug": f"coach_{int(time.time())}", "name": "健身教练", "category": "健康",
        "identity_block": "你是一位持证健身教练。", "kb_collection": "kb_medical",
    })
    check("新建自定义角色", r.status_code == 201, f"HTTP {r.status_code}")
    new_id = r.json()["id"]
    check("未传模板时自动套用骨架", "{context}" in (r.json()["prompt_template"] or ""))

    r = c.put(f"{BASE}/characters/{new_id}", headers=H, json={"name": "高级健身教练", "temperature": 0.5})
    check("修改角色生效", r.json()["name"] == "高级健身教练" and r.json()["temperature"] == 0.5)

    r = c.delete(f"{BASE}/characters/{new_id}", headers=H)
    check("删除自定义角色", r.status_code == 200, f"HTTP {r.status_code}")

    r = c.get(f"{BASE}/characters/999999", headers=H)
    check("不存在的角色返回 404", r.status_code == 404, f"HTTP {r.status_code}")
    print()

    print("=" * 70)
    print("④ 多用户隔离")
    print("=" * 70)
    uname2 = f"bob_{int(time.time())}"
    pw2 = secrets.token_hex(8)
    r2 = c.post(f"{BASE}/auth/register", json={"username": uname2, "password": pw2})
    H2 = {"Authorization": f"Bearer {r2.json()['access_token']}"}

    r = c.get(f"{BASE}/auth/me", headers=H2)
    check("用户2 看到的是自己", r.json()["username"] == uname2 and r.json()["id"] != uid)
    check("角色列表是共享资源（两用户可见同一批）",
          len(c.get(f"{BASE}/characters", headers=H2).json()) == len(chars))

    import pymysql
    from app.core.config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB
    conn = pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
                           password=MYSQL_PASSWORD, database=MYSQL_DB, charset="utf8mb4")
    with conn.cursor() as cur:
        cur.execute("SHOW TABLES")
        tables = [t[0] for t in cur.fetchall()]
        cur.execute("SELECT COUNT(*) FROM users")
        n_users = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM characters")
        n_chars = cur.fetchone()[0]
    conn.close()
    check("6 张表全部建好", len(tables) == 6, f"{tables}")
    check("用户已落库", n_users >= 2, f"{n_users} 人")
    check("角色已落库", n_chars >= 2, f"{n_chars} 个")
    print()

    print("=" * 70)
    print(f"结果: {ok_count} 通过 / {fail_count} 失败")
    print("=" * 70)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
