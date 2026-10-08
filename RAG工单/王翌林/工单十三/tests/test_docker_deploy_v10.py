# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-金融问答系统部署
tests/test_docker_deploy_v10.py —— 工单十 Docker 部署产物静态单测（不依赖 docker daemon）

校验 Dockerfile / docker-compose.yml / .dockerignore / 启停脚本 / 验收脚本齐备且
包含验收要求的关键配置：端口 8006/8506、named volume 持久化、容器间数据共享挂载、
自定义网络、host-gateway 访问宿主机 Milvus、.env 不入镜像、工单编号注释齐全。
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK_ORDER = "人工智能NLP-RAG-金融问答系统部署"


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


# ---------- 文件齐备 ----------
def test_deploy_files_exist():
    for rel in ("Dockerfile", "docker-compose.yml", ".dockerignore",
                "requirements-docker.txt",
                "scripts/start_docker_v10.sh", "scripts/stop_docker_v10.sh",
                "scripts/test_deployed_v10.py"):
        assert (ROOT / rel).is_file(), f"缺少 {rel}"


# ---------- 工单编号注释 ----------
def test_work_order_comment_present():
    for rel in ("Dockerfile", "docker-compose.yml", ".dockerignore",
                "requirements-docker.txt",
                "scripts/start_docker_v10.sh", "scripts/stop_docker_v10.sh",
                "scripts/test_deployed_v10.py"):
        assert WORK_ORDER in read(rel), f"{rel} 缺少工单编号注释"


# ---------- Dockerfile ----------
def test_dockerfile_base_and_port():
    df = read("Dockerfile")
    assert "FROM python:3.10-slim" in df
    assert re.search(r"EXPOSE\s+8006\s+8506", df), "需暴露 8006/8506"
    assert "uvicorn" in df and "src.api_v6:app" in df


def test_dockerfile_cpu_torch_index():
    df = read("Dockerfile")
    # 工单十：CPU 版 torch，避免 CUDA wheel 数 GB 体积
    assert "download.pytorch.org/whl/cpu" in df


def _requirement_lines(rel="requirements-docker.txt"):
    """工单十：仅取有效依赖行（去注释/空行），避免注释中的关键词干扰断言"""
    return [l.split("#")[0].strip() for l in read(rel).splitlines()
            if l.split("#")[0].strip()]


def test_dockerfile_no_cuda_requirements():
    lines = _requirement_lines()
    assert not any("+cu" in l for l in lines), "docker 依赖不得含 CUDA 版 wheel"
    assert not any("paddleocr" in l.lower() for l in lines), "paddleocr 为可选依赖不入镜像"
    assert any(l.startswith("torch==") for l in lines)


# ---------- compose：端口 ----------
def test_compose_ports():
    yml = read("docker-compose.yml")
    assert '"8006:8006"' in yml, "rag-api 端口映射缺失"
    assert '"8506:8506"' in yml, "rag-ui 端口映射缺失"


# ---------- compose：卷持久化 + 容器间共享 ----------
def test_compose_named_volume_persistence():
    yml = read("docker-compose.yml")
    assert re.search(r"volumes:\s*\n\s*rag-data:", yml), "需声明 named volume rag-data"
    # api 与 ui 均挂载同一卷 → 容器间数据共享
    assert yml.count("rag-data:/app/data") == 2


def test_compose_models_bind_mount_readonly():
    yml = read("docker-compose.yml")
    mounts = re.findall(r"/home/dabaie/models:/home/dabaie/models:ro", yml)
    assert len(mounts) == 2, "api/ui 均需只读挂载模型目录"


# ---------- compose：网络 ----------
def test_compose_network():
    yml = read("docker-compose.yml")
    assert "driver: bridge" in yml, "需自定义 bridge 网络"
    assert "host.docker.internal:host-gateway" in yml, "需 host-gateway 访问宿主机 Milvus"
    assert "MILVUS_HOST: host.docker.internal" in yml


# ---------- 密钥不入镜像 ----------
def test_env_not_in_image():
    assert any(l.rstrip("/") == ".env" for l in _requirement_lines(".dockerignore")), \
        ".env 必须被 .dockerignore 排除"
    df = read("Dockerfile")
    assert not re.search(r"^\s*COPY\s+\.env", df, re.M), "Dockerfile 不得 COPY .env"


# ---------- 启停脚本 ----------
def test_scripts_executable_and_safe():
    start = read("scripts/start_docker_v10.sh")
    assert "docker compose build" in start and "docker compose up -d" in start
    assert "api/v6/health" in start, "启动脚本需含健康检查"
    stop = read("scripts/stop_docker_v10.sh")
    # 默认 down 不带 -v，保留数据卷（验收②数据不丢失）
    assert "docker compose down\n" in stop or "docker compose down\r" in stop \
        or re.search(r"docker compose down$", stop, re.M)
    assert "down -v" in stop and "--prune" in stop, "删卷需显式 --prune"


# ---------- 验收脚本覆盖点 ----------
def test_acceptance_script_covers_requirements():
    s = read("scripts/test_deployed_v10.py")
    for kw in ("health_check", "qa[", "logs_clean", "volume_persistent_after_recreate",
               "volume_shared_between_containers", "network_ui_to_api",
               "network_api_to_host_milvus", "docs/deploy_v10_test_results.json"):
        assert kw in s, f"验收脚本缺少 {kw}"
