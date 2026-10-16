# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG 项目-金融问答系统部署

测试模块 01：**部署配置校验**（``测试/测试脚本/test_01_部署配置.py``）。

对应验收：四项验收的公共前提 —— 配置文件必须自洽，端口/卷/阈值都要"看得见、可覆盖"。

不依赖运行中的服务（纯静态校验），因此**总是会跑**：
    1. docker-compose.yml 能被 PyYAML/文本解析出关键字段（端口 8200/8201、network、healthcheck）；
    2. Dockerfile 具备多阶段 / 非 root / HEALTHCHECK / PYTHONUNBUFFERED / 时区字体；
    3. .env.example 覆盖了「端口 / 卷 / 阈值 / 会话」四类键；
    4. 部署脚本齐备（build/up/down/logs/health）；
    5. 阈值配置化：部署层代码里不出现硬编码端口（除 dataclass 默认值）。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE = REPO_ROOT / "部署" / "docker" / "docker-compose.yml"
DOCKERFILE = REPO_ROOT / "部署" / "docker" / "Dockerfile"
ENV_EXAMPLE = REPO_ROOT / "部署" / "配置" / ".env.example"
SCRIPTS = REPO_ROOT / "部署" / "脚本"
SERVICE = REPO_ROOT / "研发" / "service"

#: compose 里必须出现的服务（工单要求：rag-api / rag-ui / redis / 可选 prometheus）
REQUIRED_SERVICES = ("rag-api", "rag-ui", "redis", "prometheus")
#: 必须存在的部署脚本
REQUIRED_SCRIPTS = ("build.ps1", "up.ps1", "down.ps1", "logs.ps1", "health.ps1")


@pytest.fixture(scope="module")
def compose_text() -> str:
    assert COMPOSE.is_file(), f"缺少 {COMPOSE}"
    return COMPOSE.read_text(encoding="utf-8")


def test_dockerfile_存在且为多阶段构建() -> None:
    assert DOCKERFILE.is_file(), f"缺少 {DOCKERFILE}"
    text = DOCKERFILE.read_text(encoding="utf-8")
    stages = re.findall(r"^FROM\s+\S+(?:\s+AS\s+(\S+))?", text, flags=re.MULTILINE | re.IGNORECASE)
    assert len(stages) >= 2, f"应为多阶段构建，实测 FROM 数量 = {len(stages)}"
    assert "AS builder" in text and "AS runtime" in text, "缺少 builder/runtime 两个阶段命名"
    assert "--from=builder" in text, "runtime 阶段没有从 builder 拷贝产物（多阶段就白做了）"


def test_dockerfile_非root用户与健康检查与编码与环境() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    checks = {
        "非 root 用户": r"^\s*USER\s+rag\b",
        "非 root 用户创建": r"useradd\s+-u\s+10001",
        "HEALTHCHECK 指令": r"^\s*HEALTHCHECK\b",
        "PYTHONUNBUFFERED": r"PYTHONUNBUFFERED=1",
        "时区 TZ": r"TZ=Asia/Shanghai",
        "中文字体": r"fonts-wqy-zenhei",
        "时区数据包": r"tzdata",
        "工作目录": r"WORKDIR\s+/app",
        "端口声明": r"EXPOSE\s+8200",
    }
    missing = [name for name, pat in checks.items() if not re.search(pat, text, flags=re.MULTILINE)]
    assert not missing, f"Dockerfile 缺少：{missing}"


def test_compose_服务与端口与网络(compose_text: str) -> None:
    for svc in REQUIRED_SERVICES:
        assert re.search(rf"^\s{{2}}{re.escape(svc)}:", compose_text, flags=re.MULTILINE), \
            f"compose 缺少服务 {svc}"
    # 指定端口：宿主 8200 → 容器 8200；宿主 8201 → 容器 8201
    assert re.search(r'"\$\{API_PORT:-8200\}:8200"', compose_text), "rag-api 未映射 8200"
    assert re.search(r'"\$\{UI_PORT:-8201\}:8201"', compose_text), "rag-ui 未映射 8201"
    assert "name: ${NETWORK_NAME:-rag-net}" in compose_text, "缺少自定义网络 rag-net"
    assert "driver: bridge" in compose_text, "网络驱动不是 bridge"
    assert "host-gateway" in compose_text, "缺少 host.docker.internal 的 host-gateway 映射（Linux 必需）"


def test_compose_卷持久化与容器间共享(compose_text: str) -> None:
    # 关键数据卷：索引 / 日志 / 会话 / 模型 / 上传 / 运行期
    for vol in ("rag-index", "rag-log", "rag-session", "rag-model", "rag-upload", "rag-runtime"):
        assert re.search(rf"^\s{{2}}{vol}:", compose_text, flags=re.MULTILINE), f"缺少卷声明 {vol}"
    # 容器间数据共享：rag-session 必须同时出现在 rag-api 与 rag-ui 的 volumes 里
    assert compose_text.count("rag-session:/app/sessions") >= 2, \
        "rag-session 未被多个服务挂载 —— 无法证明容器间数据共享"
    # 绑定挂载（工单要求"具名卷 + 绑定挂载"）
    assert "type: bind" in compose_text or "../.." in compose_text, "缺少绑定挂载"
    # 索引卷只读（防止运行期写坏索引）
    assert re.search(r"rag-index:/app/data/index:ro", compose_text), "索引卷应为只读挂载"


def test_compose_重启策略_依赖_资源限制_日志轮转(compose_text: str) -> None:
    # restart / networks / logging 用 YAML 锚点 x-common 声明一次、四个服务 <<: *common 复用，
    # 因此**不能**数源文件里的出现次数（那会误判）；这里断言"锚点存在 + 每个服务都引用了它"。
    assert "restart: unless-stopped" in compose_text, "缺少 restart: unless-stopped"
    assert re.search(r"^x-common:\s*&common", compose_text, flags=re.MULTILINE), \
        "缺少 x-common 锚点（restart/networks/logging 的公共片段）"

    # 用「服务定义行」切块：只认顶格两空格 + 名字 + 冒号 + 行尾，且不是 YAML 锚点
    blocks: dict[str, str] = {}
    current: str | None = None
    for line in compose_text.splitlines():
        m = re.match(r"^  ([a-z][\w-]*):\s*$", line)
        if m:
            current = m.group(1)
            blocks[current] = ""
        elif current is not None:
            blocks[current] += line + "\n"
    services = [s for s in ("rag-api", "rag-ui", "redis", "prometheus") if s in blocks]
    assert len(services) == 4, f"compose 里未找到 4 个服务定义：{sorted(blocks)}"
    for svc in services:
        assert "<<: *common" in blocks[svc], \
            f"服务 {svc} 未引用 x-common 锚点（缺 restart/日志轮转/网络）"

    assert "depends_on:" in compose_text and "condition: service_healthy" in compose_text, \
        "缺少 depends_on 健康依赖"
    # 资源限制（工单17 要用）
    assert "deploy:" in compose_text and "limits:" in compose_text, "缺少资源限制"
    assert "cpus:" in compose_text and "memory:" in compose_text, "资源限制必须同时限制 CPU 与内存"
    # 日志轮转：验收要求"运行过程无异常日志"，日志必须可控
    assert 'max-size' in compose_text and 'max-file' in compose_text, "缺少日志轮转配置"
    # 顶层 version 字段已废弃，会打印 obsolete 警告 → 必须不写
    assert not re.search(r"^version:", compose_text, flags=re.MULTILINE), \
        "顶层 version: 已被 Compose V2 废弃，会产生警告日志"


def test_env_example_覆盖四类配置() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    groups = {
        "端口": ["API_PORT", "UI_PORT"],
        "卷/路径": ["VOL_PREFIX", "HOST_INDEX_DIR"],
        "阈值": ["RAG_RETRIEVAL__TOP_K", "RAG_ANSWER__MIN_VECTOR_SIMILARITY", "DEPLOY_MAX_CONCURRENCY"],
        "会话共享": ["DEPLOY_SESSION_TTL_S", "DEPLOY_REDIS_URL", "DEPLOY_REDIS_ENABLED"],
        "网络": ["NETWORK_NAME", "NETWORK_SUBNET", "RAG_LLM__OLLAMA_BASE_URL"],
    }
    missing = {g: [k for k in keys if k not in text] for g, keys in groups.items()}
    missing = {g: v for g, v in missing.items() if v}
    assert not missing, f".env.example 缺少配置键：{missing}"


def test_部署脚本齐备() -> None:
    missing = [s for s in REQUIRED_SCRIPTS if not (SCRIPTS / s).is_file()]
    assert not missing, f"缺少部署脚本：{missing}"
    for extra in ("本地等价验证.ps1", "本地停止.ps1"):
        assert (SCRIPTS / extra).is_file(), f"缺少 Docker 不可用时的等价验证脚本：{extra}"


def test_部署层代码不硬编码端口与阈值() -> None:
    """交付纪律：阈值/端口配置化（除 dataclass 默认值外，业务代码不出现魔法数字）。"""
    import ast

    text = (SERVICE / "api.py").read_text(encoding="utf-8")
    # 用 AST 剔除 docstring/注释后再查数字字面量 —— 文档里写 "8200" 是说明，不是硬编码
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if isinstance(node, ast.Expr) and node.value is not None:
                continue
            # 把字符串字面量替换成占位，避免 docstring 干扰
    numeric_literals = [n.value for n in ast.walk(tree)
                        if isinstance(n, ast.Constant) and isinstance(n.value, int)]
    forbidden = {p for p in (8200, 8201) if p in numeric_literals}
    assert not forbidden, f"api.py 出现硬编码端口字面量：{sorted(forbidden)}（应走 cfg.api_port）"

    # 超时也不得写死（除探针用的极小超时属实现细节）
    assert "timeout=60" not in text and "timeout=120" not in text, "api.py 出现硬编码超时"

    cfg = (SERVICE / "config.py").read_text(encoding="utf-8")
    assert "DEPLOY_API_PORT" in cfg and "DEPLOY_MAX_CONCURRENCY" in cfg, "配置项未收敛到环境变量"
    # 默认值必须集中在 dataclass 默认值里，且能被环境变量覆盖
    assert "api_port: int = 8200" in cfg, "端口默认值应集中在 DeployConfig 默认值"
    assert 'environ, "DEPLOY_API_PORT"' in cfg, "环境变量未接线到配置字段"
