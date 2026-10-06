# -*- coding: utf-8 -*-
"""
容器内功能验证脚本 - 部署测试
工单编号: 人工智能 NLP-RAG-金融问答系统部署

用法: docker exec financial-rag-qa python /app/test_deploy.py
"""
import os, sys, json, urllib.request, time

BASE = os.environ.get("TEST_BASE", "http://localhost:5008")


def test(name, fn):
    try:
        result = fn()
        print(f"  [OK] {name}")
        return result
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return None


def http_get(path):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=5) as r:
        return json.loads(r.read())


def http_post(path, data):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def main():
    print("=" * 50)
    print("  容器功能验证")
    print("  BASE:", BASE)
    print("=" * 50)

    # 1. 健康检查
    print("\n[1] 基础健康检查")
    health = test("GET /api/health", lambda: http_get("/api/health"))
    if health:
        print(f"      状态: {health.get('status')}")
        print(f"      磁盘: {health.get('checks', {}).get('disk', {}).get('free_mb', '?')}MB 剩余")

    # 2. 系统状态
    print("\n[2] 系统目录")
    dirs = test("GET /api/data/dirs", lambda: http_get("/api/data/dirs"))
    if dirs:
        for name, info in dirs.items():
            print(f"      {name}: {info.get('files', 0)} 文件, {info.get('size_kb', 0)}KB")

    # 3. 问答功能
    print("\n[3] Graph RAG 问答")
    q = {"question": "武汉力源的控股股东是谁?"}
    ans = test("POST /api/ask", lambda: http_post("/api/ask", q))
    if ans:
        print(f"      图谱中心: {ans.get('graph_result', {}).get('center_nodes', [])}")
        print(f"      图谱分: {ans.get('graph_result', {}).get('graph_score')}")
        print(f"      响应时间: {ans.get('response_time')}s")
        print(f"      答案: {(ans.get('rag_answer') or '')[:60]}...")

    # 4. 数据共享
    print("\n[4] 容器间数据共享")
    shared = test("POST /api/data/shared",
                  lambda: http_post("/api/data/shared", {"test": "ok", "time": int(time.time())}))
    shared_get = test("GET /api/data/shared", lambda: http_get("/api/data/shared"))

    # 5. V8 vs V9 对比
    print("\n[5] V8 vs V9 对比")
    cmp_data = test("POST /api/compare", lambda: http_post("/api/compare", {}))
    if cmp_data:
        v9p = cmp_data.get("v9", {}).get("summary", {}).get("avg_context_precision", "?")
        v9r = cmp_data.get("v9", {}).get("summary", {}).get("avg_context_recall", "?")
        print(f"      V9 Context Precision: {v9p} (目标 ≥ 0.80)")
        print(f"      V9 Context Recall:    {v9r} (目标 ≥ 0.90)")

    print("\n" + "=" * 50)
    print("✅ 容器功能验证完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
