# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG 项目-金融问答系统部署

测试模块 04：**健康检查与端口探活**。

对应验收标准 2.1：「使用 docker run 命令能够成功启动容器，并在指定端口提供服务。」

本机等价验证下，"启动容器"对应 ``部署/脚本/本地等价验证.ps1`` 起的两个进程；
断言的对象完全一样：**指定端口 8200/8201 上真的有服务在响应**，且三档探针语义正确。

三档探针语义（这是编排系统能不能正确摘流量的关键）：
    /healthz  存活：进程在 → 200，不碰索引（避免索引慢导致容器被误杀）
    /health   健康：始终 200，但每个子项如实给 ok/原因（便于一眼看降级）
    /readyz   就绪：索引不可用 → 503（否则编排会把流量打到没索引的实例上）
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
API_PORT = int(os.environ.get("RAG_TEST_API_PORT", "8200"))
UI_PORT = int(os.environ.get("RAG_TEST_UI_PORT", "8201"))


def test_健康检查三档探针(api_base: str, http: requests.Session, save_evidence) -> None:
    got = {}
    for ep in ("/healthz", "/health", "/readyz"):
        resp = http.get(f"{api_base}{ep}", timeout=30)
        got[ep] = {"status": resp.status_code, "body": resp.json() if resp.headers.get(
            "content-type", "").startswith("application/json") else resp.text[:200]}

    assert got["/healthz"]["status"] == 200, "/healthz 存活探针应恒为 200"
    assert got["/health"]["status"] == 200, "/health 健康检查应恒为 200（诚实报告子项）"
    assert got["/readyz"]["status"] == 200, f"/readyz 索引就绪时应为 200：{got['/readyz']}"

    alive = got["/healthz"]["body"]
    assert alive["status"] == "alive" and alive["pid"] > 0, f"/healthz 内容异常：{alive}"

    health = got["/health"]["body"]
    assert health["ready"] is True, f"服务未就绪：{json.dumps(health, ensure_ascii=False)[:500]}"
    assert health["port"] == API_PORT, f"服务自报端口 {health['port']} ≠ 期望 {API_PORT}"
    for key in ("volume_logs", "volume_sessions", "volume_index", "volume_uploads",
                "volume_runtime", "session_store", "engine"):
        assert key in health["checks"], f"健康检查缺少子项 {key}"
    assert health["checks"]["volume_index"]["ok"] is True, "索引卷不可读"
    assert health["checks"]["engine"]["ok"] is True, f"引擎异常：{health['checks']['engine']}"

    save_evidence("健康检查-三档探针.json", {
        "说明": "GET /healthz、/health、/readyz 的真实响应（工单10 验收项②）",
        "api_base": api_base,
        "期望端口": API_PORT,
        "实测": got,
    })


def test_索引元信息与基座一致(api_base: str, http: requests.Session, save_evidence) -> None:
    resp = http.get(f"{api_base}/api/v1/index", timeout=60)
    assert resp.status_code == 200, resp.text[:300]
    info = resp.json()
    assert info["chunks"] == 8847, f"索引块数 {info['chunks']} ≠ 基座 8847"
    assert info["vector_dim"] == 1024, f"向量维度 {info['vector_dim']} ≠ 1024"
    assert info["faiss"] is True, "FAISS 未启用（容器里装了 faiss-cpu，应启用）"
    save_evidence("健康检查-索引元信息.json", info)


def test_UI端口提供服务(ui_base: str, http: requests.Session) -> None:
    """rag-ui 容器（工单7 Web 界面）也要在指定端口 8201 提供服务。"""
    resp = http.get(f"{ui_base}/health", timeout=30)
    assert resp.status_code == 200, f"UI /health 返回 {resp.status_code}"
    body = resp.json()
    assert body.get("status") in ("ok", "degraded"), f"UI 健康检查内容异常：{body}"

    page = http.get(f"{ui_base}/", timeout=30)
    assert page.status_code == 200, "UI 首页不可访问"
    assert "text/html" in page.headers.get("content-type", ""), "UI 首页不是 HTML"
    assert len(page.text) > 500, "UI 首页内容过短，可能没渲染"
    assert "问答" in page.text or "RAG" in page.text, "UI 首页未包含预期中文界面内容"


def test_响应头含请求追踪与服务耗时(api_base: str, http: requests.Session) -> None:
    resp = http.get(f"{api_base}/healthz", timeout=30)
    assert "x-request-id" in {k.lower() for k in resp.headers}, "缺少 x-request-id 请求追踪头"
    assert "x-process-time-ms" in {k.lower() for k in resp.headers}, "缺少 x-process-time-ms 耗时头"
    assert float(resp.headers["x-process-time-ms"]) >= 0


def test_指标端点为Prometheus文本格式(api_base: str, http: requests.Session, save_evidence) -> None:
    resp = http.get(f"{api_base}/metrics", timeout=30)
    assert resp.status_code == 200, resp.text[:200]
    text = resp.text
    for metric in ("rag_api_requests_total", "rag_api_ask_total", "rag_api_uptime_seconds"):
        assert metric in text, f"指标缺少 {metric}"
    assert "# TYPE" in text and "# HELP" in text, "不是合法 Prometheus exposition 格式"
    save_evidence("健康检查-指标.txt", text)


def test_服务信息与容器端口声明一致(api_base: str, http: requests.Session, save_evidence) -> None:
    resp = http.get(f"{api_base}/", timeout=30)
    assert resp.status_code == 200
    info = resp.json()
    assert info["ports"]["api"] == API_PORT and info["ports"]["ui"] == UI_PORT, \
        f"服务自报端口与 compose 映射不一致：{info['ports']}"
    assert "POST /api/v1/ask" in info["endpoints"]
    save_evidence("健康检查-服务信息.json", info)
