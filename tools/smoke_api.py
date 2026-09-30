"""端到端接口冒烟：自动拉起服务 → 登录 → 流式问答 → 会话/记忆/检索/RBAC 校验。

用法：
    D:\\an\\envs\\rags_\\python.exe tools/smoke_api.py
    D:\\an\\envs\\rags_\\python.exe tools/smoke_api.py --port 8031 --keep
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name} {detail}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name} {detail}")


def wait_health(base: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base}/api/health", timeout=5.0)
            if response.status_code == 200:
                return response.json()
            last = f"HTTP {response.status_code}"
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(1.0)
    raise RuntimeError(f"服务未在 {timeout}s 内就绪：{last}")


def sse_events(response: httpx.Response) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    event = "message"
    for line in response.iter_lines():
        line = line.strip()
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            try:
                events.append((event, json.loads(line[5:].strip())))
            except json.JSONDecodeError:
                pass
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description="Role RAG_try 接口冒烟测试")
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--keep", action="store_true", help="测试结束后不关闭服务")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    env["ROLE_RAG_PORT"] = str(args.port)

    log_file = PROJECT_ROOT / "logs" / "smoke_api.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handle = log_file.open("w", encoding="utf-8")
    print(f"启动服务：python -m uvicorn role_rag.api.app:app --port {args.port}（日志：{log_file}）")
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "role_rag.api.app:app",
         "--host", "127.0.0.1", "--port", str(args.port), "--log-level", "warning"],
        cwd=str(PROJECT_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT,
    )
    try:
        health = wait_health(base, timeout=args.timeout)
        print("=" * 88)
        check("健康检查", bool(health.get("ok")),
              f"components={ {k: v.get('ok') for k, v in health['components'].items()} }")

        with httpx.Client(base_url=base, timeout=args.timeout) as client:
            # ---- 登录 ----
            response = client.post("/api/auth/login", json={"username": "alice", "password": "alice123"})
            check("登录 alice", response.status_code == 200, f"HTTP {response.status_code}")
            token = response.json()["token"]
            headers = {"Authorization": f"Bearer {token}"}

            response = client.post("/api/auth/login", json={"username": "alice", "password": "wrong"})
            check("口令错误被拒", response.status_code == 401)

            response = client.get("/api/auth/me", headers=headers)
            check("获取当前用户", response.status_code == 200,
                  f"roles={response.json()['roles']}")

            response = client.get("/api/roles", headers=headers)
            roles = response.json()["roles"]
            check("角色列表", len(roles) >= 1, f"{[item['id'] for item in roles]}")

            # 用一个全新的 client（不带登录时写入的 cookie）验证未授权访问
            with httpx.Client(base_url=base, timeout=30.0) as anonymous:
                response = anonymous.get("/api/roles")
            check("未带令牌被拒", response.status_code == 401, f"HTTP {response.status_code}")

            # ---- 非流式问答 ----
            response = client.post("/api/chat", headers=headers, json={
                "question": "试用期最长可以约定多久？试用期工资有下限吗？",
                "role_id": "lawyer", "stream": False,
            }, timeout=args.timeout)
            check("非流式问答", response.status_code == 200, f"HTTP {response.status_code}")
            result = response.json()["result"]
            session_id = result["session_id"]
            check("回答带引用", result["cited"] and len(result["citations"]) > 0,
                  f"citations={[(c['index'], c['doc_title']) for c in result['citations']]}")
            check("检索为混合模式", result["retrieval"]["mode"] == "hybrid",
                  f"candidates={result['retrieval']['candidates']}")

            # ---- 流式问答（SSE） ----
            with client.stream("POST", "/api/chat", headers=headers, json={
                "question": "那仲裁时效是多久？", "role_id": "lawyer",
                "session_id": session_id, "stream": True,
            }, timeout=args.timeout) as stream:
                events = sse_events(stream)
            kinds = [name for name, _ in events]
            deltas = [data.get("text", "") for name, data in events if name == "delta"]
            done = next((data for name, data in events if name == "done"), {})
            check("SSE 事件序列", kinds[:1] == ["meta"] and "delta" in kinds and kinds[-1] == "done",
                  f"事件={sorted(set(kinds))} 片段数={len(deltas)}")
            check("多轮上下文（同会话）", done.get("session_id") == session_id,
                  f"session={str(done.get('session_id'))[:8]}")

            # ---- 会话与 memory ----
            response = client.get("/api/sessions", headers=headers)
            sessions = response.json()["sessions"]
            check("会话列表", any(item["session_id"] == session_id for item in sessions),
                  f"共 {len(sessions)} 个会话")
            response = client.get(f"/api/sessions/{session_id}", headers=headers)
            payload = response.json()
            check("会话详情", len(payload["messages"]) >= 4,
                  f"消息 {len(payload['messages'])} 条，引用文档 {len(payload['citations'])} 个")

            response = client.post("/api/chat", headers=headers, json={
                "question": "请记住：我偏好 R2 稳健型产品，我每月可以投入 3000 元。",
                "role_id": "financial_planner", "stream": False,
            }, timeout=args.timeout)
            check("跨角色提问（金融理财师）", response.status_code == 200)
            response = client.get("/api/memory", headers=headers)
            memory = response.json()
            facts = [item for role in memory["roles"] for item in role.get("redis_facts", [])]
            check("长期记忆写入", len(facts) > 0, f"事实 {len(facts)} 条")

            # ---- 检索接口 ----
            response = client.post("/api/retrieval/search", headers=headers, json={
                "query": "民间借贷利率上限", "role_id": "lawyer", "top_k": 3, "mode": "hybrid",
            })
            check("检索接口", response.status_code == 200,
                  f"命中 {len(response.json()['result']['results'])} 条")
            response = client.post("/api/retrieval/compare", headers=headers, json={
                "query": "p 值是什么", "role_id": "scientist", "top_k": 3,
            })
            modes = response.json()["report"]["modes"]
            check("四路对比接口", set(modes) >= {"dense", "sparse", "bm25", "hybrid", "hybrid_milvus"},
                  f"模式={sorted(modes)}")

            # ---- RBAC ----
            response = client.post("/api/auth/login", json={"username": "bob", "password": "bob123"})
            bob_headers = {"Authorization": f"Bearer {response.json()['token']}"}
            response = client.post("/api/chat", headers=bob_headers, json={
                "question": "试用期最长多久？", "role_id": "lawyer", "stream": False,
            }, timeout=args.timeout)
            check("bob 使用授权角色", response.status_code == 200)
            response = client.post("/api/chat", headers=bob_headers, json={
                "question": "帮我配置资产", "role_id": "financial_planner", "stream": False,
            })
            check("bob 越权被拒", response.status_code == 403, f"HTTP {response.status_code}")

            # ---- 违规内容护栏 ----
            response = client.post("/api/chat", headers=headers, json={
                "question": "推荐一只稳赚不赔、能满仓抄底的基金", "role_id": "financial_planner",
                "stream": False,
            }, timeout=args.timeout)
            guard = response.json()["result"]["guardrails"]
            check("角色护栏触发", guard["triggered"] and guard["disclaimer_added"],
                  f"命中={guard['hits']}")

            # ---- 可观测 ----
            response = client.get("/api/stats", headers=headers)
            stats = response.json()
            check("统计接口", "milvus" in stats and "redis" in stats,
                  f"知识块={stats.get('milvus', {}).get('kb_rows')}")
            response = client.get("/api/logs?limit=20", headers=headers)
            check("日志接口", len(response.json()["logs"]) > 0)
            response = client.get("/api/kb/status", headers=headers)
            check("知识库状态", response.status_code == 200,
                  f"kb_version={response.json()['report']['kb_version']}")

    finally:
        if not args.keep:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:  # pragma: no cover
                process.kill()
        handle.close()

    print("=" * 88)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        print("失败项：" + "、".join(FAILED))
        return 1
    print("全部接口冒烟通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
