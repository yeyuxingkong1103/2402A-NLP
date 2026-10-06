# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-金融问答系统部署任务
# 模块：run_deploy_check —— 部署验证：对运行中的服务（容器或本地）做接口联通性 + 检索问答验证
# 用法：python evaluation/run_deploy_check.py [http://127.0.0.1:8100]
import json
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8100"
RESULTS = []


def call(method, path, payload=None, timeout=180):
    url = BASE + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8", "replace")
    return r.status, body, int((time.time() - t0) * 1000)


def rec(name, ok, detail):
    RESULTS.append({"检查项": name, "结果": "✅" if ok else "❌", "说明": detail})
    print("[%s] %s — %s" % ("OK " if ok else "FAIL", name, detail))


def main():
    print("target:", BASE)
    try:
        st, body, ms = call("GET", "/api/health")
        rec("健康检查 /api/health", st == 200 and "ok" in body.lower() or st == 200, "HTTP %s (%d ms): %s" % (st, ms, body[:120]))
    except Exception as e:  # noqa: BLE001
        rec("健康检查 /api/health", False, repr(e))
    try:
        st, body, ms = call("GET", "/api/kb/overview")
        d = json.loads(body).get("data", {})
        rec("知识库概览 /api/kb/overview", st == 200 and bool(d),
            "HTTP %s；chunks=%s docs=%s (%d ms)" % (st, d.get("chunks") or d.get("total"), d.get("docs"), ms))
    except Exception as e:  # noqa: BLE001
        rec("知识库概览 /api/kb/overview", False, repr(e))
    try:
        st, body, ms = call("POST", "/api/graph/query",
                            {"query": "中国太保2021年实现营业收入和保险业务收入分别是多少？"})
        d = json.loads(body).get("data", {})
        tri = d.get("graph_triples") or []
        rec("知识图谱接口 /api/graph/query", st == 200 and bool(d),
            "HTTP %s；实体=%s；三元组=%d 条 (%d ms)" % (st, d.get("entities"), len(tri), ms))
    except Exception as e:  # noqa: BLE001
        rec("知识图谱接口 /api/graph/query", False, repr(e))
    try:
        q = "平安银行2019年实现营业收入和净利润分别是多少？"
        st, body, ms = call("POST", "/api/ask", {"query": q}, timeout=300)
        d = json.loads(body).get("data", {})
        ans = (d.get("answer") or "")[:180].replace("\n", " ")
        rec("Graph RAG 问答 /api/ask", st == 200 and bool(ans),
            "HTTP %s；%d ms；答案：%s" % (st, ms, ans))
    except Exception as e:  # noqa: BLE001
        rec("Graph RAG 问答 /api/ask", False, repr(e))
    try:
        st, body, ms = call("GET", "/")
        rec("首页 /", st == 200 and "html" in body[:400].lower(), "HTTP %s (%d ms)" % (st, ms))
    except Exception as e:  # noqa: BLE001
        rec("首页 /", False, repr(e))
    try:
        st, body, ms = call("GET", "/static/graph.html")
        rec("图谱可视化页 /static/graph.html", st == 200 and len(body) > 1000,
            "HTTP %s；%d 字节 (%d ms)" % (st, len(body), ms))
    except Exception as e:  # noqa: BLE001
        rec("图谱可视化页 /static/graph.html", False, repr(e))
    print(json.dumps(RESULTS, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
