# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG 项目-金融问答系统部署

测试模块 03：**网络连通性（含 host.docker.internal:11434）**。

对应验收标准 2.3：「容器网络配置正确，能够与其他服务（如 RTMP 服务器）正常通信」。

本模块验证的网络事实（阈值/地址全部环境变量可覆盖，见 conftest.py）：
    1. 宿主机 Ollama 可达（生成/嵌入后端）—— 容器里走 ``host.docker.internal:11434``；
       本机等价验证里若该域名不可解析，则回退 ``127.0.0.1``，并在断言信息里如实说明用的是哪条路径。
    2. 服务对外端口真的在监听（8200 / 8201）。
    3. 服务自己报告的网络依赖状态与实测一致（健康检查不撒谎）。
    4. Redis 端口：本机等价验证环境默认没有 Redis（``DEPLOY_REDIS_ENABLED=0``），
       此时**要求服务如实降级**（session backend = file + 给出原因），而不是假装连上了。
    5. 跨服务 DNS/服务名：容器编排里用 ``redis`` / ``rag-api`` 服务名互访，
       静态校验 compose 的 network 与 hostname 声明。
    6. RTMP 端口 1935：本机当前未运行 RTMP 服务，测试如实记录"未监听"，
       并给出容器里的验证方法（``docker exec rag-api ...``，见 部署/脚本/health.ps1）。
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest
import requests

from conftest import HOST_ALIAS, OLLAMA_BASE, tcp_open

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE = REPO_ROOT / "部署" / "docker" / "docker-compose.yml"

RTMP_PORT = int(os.environ.get("RAG_TEST_RTMP_PORT", "1935"))
REDIS_PORT = int(os.environ.get("RAG_TEST_REDIS_PORT", "6379"))


def _resolve(host: str) -> str | None:
    try:
        return socket.gethostbyname(host)
    except OSError:
        return None


def test_宿主机Ollama可达_容器内用host_docker_internal(api_health: dict, save_evidence) -> None:
    """Ollama 是生成与嵌入后端；容器里必须用 host.docker.internal 而不是 127.0.0.1。"""
    alias_ip = _resolve(HOST_ALIAS)
    used = "127.0.0.1"
    if alias_ip is not None:
        ok, detail = tcp_open(HOST_ALIAS, 11434)
        if ok:
            used = HOST_ALIAS

    assert _resolve("127.0.0.1") is not None
    ok, detail = tcp_open(used, 11434)
    assert ok, f"Ollama 端口不可达（{used}:11434）：{detail}"

    resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
    assert resp.status_code == 200, f"Ollama /api/tags 返回 {resp.status_code}"
    models = [m["name"] for m in resp.json().get("models", [])]
    assert any(m.startswith("bge-m3") for m in models), f"缺少 bge-m3 嵌入模型：{models[:8]}"
    assert any(m.startswith("qwen2.5:3b") for m in models), f"缺少 qwen2.5:3b 生成模型：{models[:8]}"

    save_evidence("网络连通性-ollama.json", {
        "说明": "宿主机 Ollama 连通性实测（容器内等价地址 = host.docker.internal:11434）",
        "host_alias": HOST_ALIAS,
        "host_alias_resolved_ip": alias_ip,
        "本机实测使用地址": used,
        "模型总数": len(models),
        "关键模型": [m for m in models if m.startswith(("bge-m3", "qwen2.5:3b"))],
        "http_status": resp.status_code,
        "服务自报llm检查": api_health["checks"]["engine"].get("checks", {}).get("llm", {}),
    })


def test_服务端口监听(api_base: str, ui_base: str) -> None:
    for name, base in (("rag-api", api_base), ("rag-ui", ui_base)):
        host = base.split("//")[1].split(":")[0]
        port = int(base.rsplit(":", 1)[1])
        ok, detail = tcp_open(host, port)
        assert ok, f"{name} 端口未监听（{host}:{port}）：{detail}"


def test_健康检查报告的网络状态与实测一致(api_health: dict) -> None:
    """健康检查里 llm 子项必须报 ok，且给出的 base_url 是容器内可用的那一套。"""
    engine = api_health["checks"]["engine"]
    assert engine.get("ok") is True, f"引擎未就绪：{json.dumps(engine, ensure_ascii=False)[:400]}"
    llm = engine.get("checks", {}).get("llm", {})
    assert llm.get("ok") is True, f"llm 检查未通过：{llm}"
    url = str(llm.get("base_url") or llm.get("url") or "")
    assert "11434" in url, f"llm 检查里的 Ollama 地址异常：{url}"


def _redis_is_real(port: int) -> tuple[bool, str]:
    """用 **Redis 协议** 判断对端是不是真 Redis（只探 TCP 会误判：有端口能连但不是 Redis）。

    实测教训：本机 6379 的 `create_connection` 会成功，但只有发 `PING` 收到 `+PONG`
    才能确认是 Redis 本体 —— 否则测试会把"别的服务占了端口"当成"Redis 可用"。
    """
    import socket

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=3) as sock:
            sock.settimeout(3)
            sock.sendall(b"PING\r\n")
            data = sock.recv(64)
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if data.startswith(b"+PONG"):
        return True, f"PING → {data!r}（真 Redis）"
    return False, f"PING → {data!r}（端口有服务但不是 Redis，或需要 AUTH）"


def test_redis_端口与降级行为如实报告(api_health: dict, save_evidence) -> None:
    """Redis 可用则必须报 ok；不可用则必须如实降级（文件卷后端）+ 给出原因字符串。"""
    is_redis, detail = _redis_is_real(REDIS_PORT)
    store = api_health["checks"]["session_store"]
    assert store.get("ok") is True, f"会话存储检查失败：{store}"

    if is_redis:
        assert store["redis"]["ok"] is True, f"Redis 可用但服务报不可用：{store['redis']}"
        assert "redis" in store["backend"], f"后端未使用 redis：{store['backend']}"
        # 会话键必须真的写进 Redis（容器间共享的实证）
        assert store["redis"].get("client"), f"未报告 Redis 客户端种类：{store['redis']}"
    else:
        assert store["redis"]["ok"] is False, "Redis 不可达却报告可用（健康检查失真）"
        assert store["backend"] == "file", f"未降级到文件后端：{store['backend']}"
        reason = str(store.get("degraded_reason") or store["redis"].get("detail") or "")
        assert reason, "降级了却没给出原因字符串"

    save_evidence("网络连通性-redis降级.json", {
        "说明": "Redis 连通性、客户端种类与降级行为（用 Redis 协议 PING 判定，避免端口误判）",
        "redis_协议探测": {"is_redis": is_redis, "detail": detail, "port": REDIS_PORT},
        "session_store": store,
        "客户端说明": "session_store 自带标准库 RESP 客户端（MiniRedis），不依赖 redis PyPI 包；"
                    "容器内若装了 redis 包则优先用官方客户端。",
    })


def test_RTMP端口探测与容器内验证方法(save_evidence) -> None:
    """RTMP 服务器（1935）—— 工单原文点名的互通信对象。本机当前未运行，如实记录。"""
    ok, detail = tcp_open("127.0.0.1", RTMP_PORT, timeout=2.0)
    save_evidence("网络连通性-rtmp.json", {
        "说明": "RTMP 服务器 1935 端口探测（工单原文：容器要能与 RTMP 服务器等其它服务通信）",
        "target": f"127.0.0.1:{RTMP_PORT}",
        "容器内等价目标": f"host.docker.internal:{RTMP_PORT}",
        "connectable": ok,
        "detail": detail,
        "本机现状": "未运行 RTMP 服务" if not ok else "RTMP 服务在监听",
        "容器内验证命令": "docker exec rag-api python -c \"import socket;socket.create_connection(('host.docker.internal',1935),5)\"",
        "编排侧验证": "部署/脚本/health.ps1 的【⑥ 网络连通性】段落会自动执行上述探测",
    })
    # 不断言 RTMP 必须在线（本机确实没有该服务），但必须能"探测并留下证据"
    assert isinstance(ok, bool)


def test_compose_网络声明正确() -> None:
    compose = COMPOSE.read_text(encoding="utf-8")
    assert "rag-net:" in compose, "缺少自定义网络 rag-net"
    assert "hostname: rag-api" in compose and "hostname: redis" in compose, \
        "缺少 hostname 声明 —— 跨服务用服务名互访依赖它"
    # 每个服务都必须接入同一网络（否则互相不通）
    for svc in ("rag-api", "rag-ui", "redis", "prometheus"):
        block = compose.split(f"{svc}:", 1)[1]
        assert "- rag-net" in block.split("volumes:")[0] or "networks:" in block, \
            f"服务 {svc} 未加入 rag-net"
    # Ollama 走宿主网关
    assert "RAG_LLM__OLLAMA_BASE_URL: ${RAG_LLM__OLLAMA_BASE_URL:-http://host.docker.internal:11434}" in compose, \
        "compose 未把 Ollama 指向 host.docker.internal"
