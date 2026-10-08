# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-金融问答系统部署
scripts/test_deployed_v10.py —— 工单十 容器化部署验收脚本

对 docker compose 部署的金融问答系统执行验收测试：
  一、部署后金融问答服务测试通过
      1) GET  /api/v6/health 健康检查
      2) POST /api/v6/ask    3 道金融题（年报数字题 + 招股书题），校验金标关键词
  二、容器管理
      1) 容器启动/端口/日志：`docker compose ps` + `docker logs` 检查无 ERROR/Traceback
      2) 卷持久化：向 rag-data 写入标记文件 → compose down → up → 文件仍在；
         同时验证 rag-ui 与 rag-api 挂载同一卷（容器间数据共享）
      3) 网络配置：rag-ui 容器内访问 http://rag-api:8006（容器互访）、
         rag-api 容器内访问宿主机 Milvus host.docker.internal:19530
结果落盘 docs/deploy_v10_test_results.json。

用法：python scripts/test_deployed_v10.py
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

WORK_ORDER = "人工智能NLP-RAG-金融问答系统部署"
BASE_URL = "http://localhost:8006"
UI_URL = "http://localhost:8506"
OUT = Path("docs/deploy_v10_test_results.json")

# 工单十：验收金融问答（金标关键词取自工单七 test_questions_v7.json，已核实）
QUESTIONS = [
    {"question": "邮储银行2019年实现的营业收入是多少亿元？同比增长多少？",
     "doc_id": "邮储银行2019年报", "keywords": ["2768.09", "6.06"]},
    {"question": "中信证券2020年全年实现的营业收入和归属于公司股东的净利润分别是多少亿元？",
     "doc_id": "中信证券2020年报", "keywords": ["543.83", "149.02"]},
    {"question": "本次发行前公司总股本是多少？",
     "doc_id": "招股说明书1", "keywords": []},  # 招股书题：仅要求非空回答+有引用
]

results = {"work_order": WORK_ORDER, "ts": datetime.now().isoformat(),
           "checks": [], "passed": True}


def record(name, ok, detail=""):
    results["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail[:120]}")
    if not ok:
        results["passed"] = False


def sh(cmd, timeout=60):
    """工单十：执行 shell 并返回 (rc, stdout+stderr)"""
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, (p.stdout + p.stderr).strip()


# ========== 一、部署后金融问答服务测试 ==========
print("=" * 20, "一、金融问答服务测试", "=" * 20)
# 1) 健康检查
try:
    r = requests.get(f"{BASE_URL}/api/v6/health", timeout=15)
    ok = r.status_code == 200
    record("health_check", ok, f"status={r.status_code} body={r.text[:200]}")
except Exception as e:
    record("health_check", False, str(e))
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    sys.exit(1)

# 2) 金融问答
qa_details = []
for q in QUESTIONS:
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{BASE_URL}/api/v6/ask",
                          json={"question": q["question"], "doc_id": q["doc_id"]},
                          timeout=120)
        ms = (time.perf_counter() - t0) * 1000
        body = r.json()
        answer = (body.get("answer") or "").replace(",", "")
        refs = body.get("references", [])
        hit = [k for k in q["keywords"] if k in answer]
        ok = r.status_code == 200 and answer.strip() and len(refs) > 0 \
            and len(hit) == len(q["keywords"])
        qa_details.append({"question": q["question"], "doc_id": q["doc_id"],
                           "latency_ms": round(ms, 1), "keywords": q["keywords"],
                           "keywords_hit": hit, "n_references": len(refs),
                           "answer_preview": answer[:200], "ok": ok})
        record(f"qa[{q['doc_id']}]", ok,
               f"{ms:.0f}ms 命中关键词 {hit}/{q['keywords']} 引用 {len(refs)} 条")
    except Exception as e:
        record(f"qa[{q['doc_id']}]", False, str(e))
results["qa"] = qa_details

# 3) UI 可访问
try:
    r = requests.get(UI_URL, timeout=15)
    record("ui_reachable", r.status_code == 200, f"status={r.status_code}")
except Exception as e:
    record("ui_reachable", False, str(e))

# ========== 二、容器管理 ==========
print("=" * 20, "二、容器管理验收", "=" * 20)
# 1) 容器运行状态 + 日志无异常
rc, ps = sh("docker compose ps --format json", timeout=30)
running = [l for l in ps.splitlines() if '"running"' in l.lower() or '"Running"' in l]
record("containers_running", rc == 0 and len(running) >= 2,
       f"{len(running)} 个容器运行中")

for cname in ("rag-v10-api", "rag-v10-ui"):
    rc, logs = sh(f"docker logs {cname} 2>&1 | tail -200", timeout=30)
    bad = [l for l in logs.splitlines()
           if ("ERROR" in l or "Traceback" in l)
           and "0 error" not in l.lower()]
    record(f"logs_clean[{cname}]", rc == 0 and not bad,
           f"异常行数={len(bad)}" + (f" 例: {bad[0][:100]}" if bad else ""))

# 2) 卷持久化 + 容器间数据共享
marker = f"v10-persist-{int(time.time())}"
rc, _ = sh(f"docker exec rag-v10-api sh -c 'echo {marker} > /app/data/.v10_persist_test'")
record("volume_write", rc == 0, f"写入标记 {marker}")

# 容器间数据共享：rag-ui 能读到 rag-api 写入的文件（同一 named volume）
rc, out = sh("docker exec rag-v10-ui cat /app/data/.v10_persist_test")
record("volume_shared_between_containers", rc == 0 and marker in out,
       f"rag-ui 读取到: {out.strip()}")

print("[v10] compose down → up，验证 named volume 数据不丢失...")
sh("docker compose down", timeout=120)
rc, _ = sh("docker compose up -d", timeout=180)
# 等待 api 恢复（模型已加载过一次，卷内缓存复用）
ready = False
for i in range(120):
    try:
        if requests.get(f"{BASE_URL}/api/v6/health", timeout=5).status_code == 200:
            ready = True
            break
    except Exception:
        pass
    time.sleep(5)
record("service_recover_after_recreate", ready, f"等待 {5 * (i + 1)}s 内恢复")

rc, out = sh("docker exec rag-v10-api cat /app/data/.v10_persist_test")
record("volume_persistent_after_recreate", rc == 0 and marker in out,
       f"重建后读取到: {out.strip()}")
sh("docker exec rag-v10-api rm -f /app/data/.v10_persist_test")  # 清理标记

# 3) 网络配置：容器互访 + 容器→宿主机 Milvus
rc, out = sh("docker exec rag-v10-ui python -c \""
             "import requests;r=requests.get('http://rag-api:8006/api/v6/health',timeout=10);"
             "print(r.status_code)\"")
record("network_ui_to_api", rc == 0 and "200" in out,
       f"rag-ui → rag-api:8006 status={out.strip()}")

rc, out = sh("docker exec rag-v10-api python -c \""
             "import socket;s=socket.create_connection(('host.docker.internal',19530),timeout=5);"
             "print('milvus-ok')\"")
record("network_api_to_host_milvus", rc == 0 and "milvus-ok" in out,
       f"rag-api → 宿主机 Milvus: {out.strip()}")

# ========== 落盘 ==========
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
n_ok = sum(1 for c in results["checks"] if c["ok"])
print(f"\n[v10] 验收完成: {n_ok}/{len(results['checks'])} 项通过 → {OUT}")
sys.exit(0 if results["passed"] else 1)
