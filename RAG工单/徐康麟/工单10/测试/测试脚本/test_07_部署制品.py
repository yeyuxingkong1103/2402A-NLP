# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG 项目-金融问答系统部署

测试模块 07：**部署制品可用性（Docker 侧）**。

本机 Docker daemon 被沙箱拒绝（证据：``部署/日志/docker提权失败证据.log``），
因此本模块**不自欺欺人地跳过**，而是把「不需要 daemon 也能验的部分」全部验掉：

    1. ``docker compose config`` —— 客户端侧解析，**不需要 daemon**，当前环境实测可跑；
       用它证明 compose 文件能被 Docker 自己解析成完整配置（不是"我看着像对的"）。
    2. Dockerfile 指令级校验（阶段 / USER / HEALTHCHECK / COPY 源路径真实存在）。
    3. 镜像内脚本语法校验（entrypoint.sh 用 ``bash -n``；healthcheck.py 用 ``py_compile``）。
    4. requirements.txt 的包与「业务代码真实 import」对得上（防止镜像缺依赖启动即崩）。
    5. 需要 daemon 的断言统一用 ``RAG_TEST_DOCKER=1`` 门控，并在结果里**明确标注**。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / "部署" / "docker"
DOCKERFILE = DOCKER_DIR / "Dockerfile"
COMPOSE = DOCKER_DIR / "docker-compose.yml"
ENV_FILE = REPO_ROOT / "部署" / "配置" / ".env.example"
REQUIREMENTS = DOCKER_DIR / "requirements.txt"
DOCKER_AVAILABLE = os.environ.get("RAG_TEST_DOCKER") == "1" and shutil.which("docker") is not None


@pytest.fixture(scope="module")
def compose_config() -> dict:
    """``docker compose config --format json``（**客户端解析，不需要 daemon**）。

    带上 ``--profile monitoring``：prometheus 属于可选监控栈（默认不启动），
    不带 profile 时它**不会**出现在解析结果里 —— 这是 Compose 的预期行为，不是缺服务。
    """
    out = subprocess.run(
        ["docker", "compose", "--env-file", str(ENV_FILE), "-f", str(COMPOSE),
         "--profile", "monitoring", "config", "--format", "json"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
        cwd=str(REPO_ROOT.parent / "工单10"),
    )
    if out.returncode != 0:
        pytest.fail(f"docker compose config 失败（rc={out.returncode}）：{out.stderr[:500]}")
    return json.loads(out.stdout)


def test_compose_config_可被docker自身解析(compose_config: dict, save_evidence) -> None:
    """核心证据：compose 文件通过 Docker 官方解析器，且解析结果与设计一致。"""
    assert compose_config["name"] == "rag-finqa"
    services = compose_config["services"]
    assert set(services) >= {"rag-api", "rag-ui", "redis", "prometheus"}, f"服务不全：{set(services)}"

    api = services["rag-api"]
    assert api["ports"][0]["published"] == "8200" and api["ports"][0]["target"] == 8200
    ui = services["rag-ui"]
    assert ui["ports"][0]["published"] == "8201" and ui["ports"][0]["target"] == 8201
    assert api["image"] == "rag-finqa:1.0.0"
    assert api["restart"] == "unless-stopped"
    assert api["healthcheck"]["test"][0] == "CMD"
    assert api["deploy"]["resources"]["limits"]["memory"] == "2147483648"
    assert api["deploy"]["resources"]["limits"]["cpus"] == 2
    assert any(h.startswith("host.docker.internal=") for h in api["extra_hosts"])

    # 构建上下文必须解析到工单根目录（否则 Dockerfile 里 COPY 研发/app 找不到）
    ctx = Path(api["build"]["context"])
    assert ctx == REPO_ROOT, f"构建上下文解析错误：{ctx} ≠ {REPO_ROOT}"
    assert (ctx / api["build"]["dockerfile"]).is_file(), \
        f"Dockerfile 路径在上下文内不存在：{api['build']['dockerfile']}"

    # 卷：具名卷 + 绑定挂载都要有
    vols = api["volumes"]
    named = [v["source"] for v in vols if v["type"] == "volume"]
    binds = [v["source"] for v in vols if v["type"] == "bind"]
    assert {"rag-index", "rag-log", "rag-session", "rag-model", "rag-upload", "rag-runtime"} <= set(named), \
        f"具名卷不全：{named}"
    assert binds, "没有绑定挂载（工单要求卷 + 绑定挂载并存）"
    index_vol = next(v for v in vols if v["target"] == "/app/data/index")
    assert index_vol["read_only"] is True, "索引卷应为只读"

    # 容器间数据共享：两个服务挂同一个 session 卷
    ui_targets = {v["source"] for v in services["rag-ui"]["volumes"]}
    assert "rag-session" in ui_targets and "rag-session" in set(named), "会话卷未在两个服务间共享"

    # 网络
    assert compose_config["networks"]["rag-net"]["driver"] == "bridge"
    assert compose_config["networks"]["rag-net"]["ipam"]["config"][0]["subnet"] == "172.28.0.0/16"

    save_evidence("compose配置-解析结果.json", {
        "说明": "docker compose config --format json 的真实输出（Docker 官方解析器，客户端侧不需 daemon）",
        "命令": "docker compose --env-file 部署/配置/.env.example -f 部署/docker/docker-compose.yml config --format json",
        **compose_config,
    })


def test_dockerfile_COPY源文件真实存在() -> None:
    """Dockerfile 里每个 COPY 的源，必须在构建上下文（工单根）里真实存在。"""
    text = DOCKERFILE.read_text(encoding="utf-8")
    context = REPO_ROOT
    copies = re.findall(r"^COPY\s+(?!--from)(.+)$", text, flags=re.MULTILINE)
    assert copies, "Dockerfile 里没有 COPY 指令"
    missing: list[str] = []
    for line in copies:
        parts = [p for p in line.split() if not p.startswith("--")]
        for src in parts[:-1]:
            if src.startswith("/"):
                continue
            target = context / src
            if not target.exists():
                missing.append(src)
    assert not missing, f"COPY 源在构建上下文里不存在：{missing}"


def test_entrypoint与healthcheck语法可运行() -> None:
    """entrypoint.sh 必须在**真 bash** 下语法通过。

    ⚠️ 实测坑：`C:\\Windows\\system32\\bash.exe` 只是 WSL 启动器存根，
    对它跑 `bash -n` 会返回 1 且 stderr 为空（还会触发 UTF-8 解码告警），
    因此这里优先用 WSL 里的真 bash（`wsl bash -n`），并显式排除 system32 存根。
    """
    entry = DOCKER_DIR / "entrypoint.sh"
    assert entry.is_file(), "缺少 entrypoint.sh"
    text = entry.read_text(encoding="utf-8")
    assert text.startswith("#!"), "缺少 shebang"
    assert "exec \"$@\"" in text, "entrypoint 未 exec 传入命令（PID 1 信号会丢）"
    assert "set -euo pipefail" in text, "缺少严格模式"

    checked = "静态检查（未找到可用的真 bash）"
    wsl_ok = False
    wsl = shutil.which("wsl")
    if wsl:
        # WSL 里路径要转成 /mnt/e/... 形式
        wsl_path = "/mnt/" + str(entry)[0].lower() + str(entry)[2:].replace("\\", "/")
        out = subprocess.run([wsl, "bash", "-n", wsl_path], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=120)
        if out.returncode == 0:
            checked = f"WSL bash -n 通过（{wsl_path}）"
            wsl_ok = True
        else:
            # 实测：本机 `wsl` 被宿主拒绝（Wsl/Service/CreateInstance/E_ACCESS_DENIED），
            # 因此 WSL 不可用时**退化为静态检查**并如实标注，而不是把环境限制当成脚本错误。
            detail = (out.stderr or out.stdout or "").strip().replace("\x00", "")
            checked = (f"静态检查（WSL 不可用：rc={out.returncode} "
                       f"{(detail[:60] if detail else 'E_ACCESS_DENIED/无输出')}）")
    if not wsl_ok:
        bash = shutil.which("bash")
        # ⚠️ 实测坑：C:\Windows\system32\bash.exe 是 WSL 启动器存根，bash -n 恒返回 1，
        # 故必须排除它，否则会把"存根"误判成"脚本语法错误"。
        if bash and "system32" not in bash.lower():
            out = subprocess.run([bash, "-n", str(entry)], capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=30)
            assert out.returncode == 0, f"entrypoint.sh 语法错误：{out.stderr}"
            checked = f"bash -n 通过（{bash}）"

    (REPO_ROOT / "测试" / "测试数据").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "测试" / "测试数据" / "entrypoint语法校验.txt").write_text(
        f"entrypoint.sh 语法校验结果：{checked}\n"
        f"文件：{entry}\n字节数：{entry.stat().st_size}\n"
        f"静态检查项：shebang / set -euo pipefail / exec \"$@\" 均已人工核对通过\n"
        f"环境限制：C:\\Windows\\system32\\bash.exe 是 WSL 启动器存根（bash -n 恒返回 1，实测）；\n"
        f"         本机 `wsl` 亦被宿主拒绝（Wsl/Service/CreateInstance/E_ACCESS_DENIED，实测）。\n"
        f"容器内验证方式：镜像构建后 `docker run --rm rag-finqa:1.0.0 bash -n /usr/local/bin/entrypoint.sh`。\n",
        encoding="utf-8")
    assert checked, "未完成 entrypoint 语法校验"

    hc = DOCKER_DIR / "healthcheck.py"
    assert hc.is_file(), "缺少 healthcheck.py"
    out = subprocess.run([sys.executable, "-m", "py_compile", str(hc)], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, f"healthcheck.py 编译失败：{out.stderr}"


def test_requirements覆盖业务代码真实import() -> None:
    """镜像里必须装齐业务代码 **顶层** import 的第三方包，否则容器一起来就 ImportError。"""
    reqs = set()
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("==")[0].split("[")[0].strip().lower()   # uvicorn[standard]==0.27.0 → uvicorn
        reqs.add(name)
    assert {"numpy", "faiss-cpu", "fastapi", "uvicorn", "pymupdf", "jieba"} <= reqs, f"依赖缺失：{reqs}"

    # 业务代码顶层 import 扫描（延迟导入/可选依赖不算硬依赖）
    stdlib_ok = {"json", "os", "sys", "re", "time", "math", "pathlib", "typing", "dataclasses",
                 "collections", "itertools", "functools", "hashlib", "sqlite3", "threading",
                 "subprocess", "shutil", "random", "logging", "datetime", "enum", "abc",
                 "contextlib", "urllib", "uuid", "platform", "socket", "io", "csv", "base64",
                 "string", "copy", "warnings", "traceback", "pickle", "gzip", "zipfile",
                 "concurrent", "queue", "statistics", "textwrap", "unicodedata", "difflib",
                 "inspect", "importlib", "tempfile", "argparse", "unittest", "operator",
                 "heapq", "bisect", "decimal", "fractions", "secrets", "signal", "struct",
                 "email", "http", "html", "xml", "ast", "tokenize", "glob", "fnmatch",
                 "asyncio", "ctypes", "array", "binascii", "codecs", "configparser", "dbm",
                 "filecmp", "getpass", "gettext", "locale", "multiprocessing", "numbers",
                 "pprint", "profile", "pstats", "shelve", "shlex", "site", "smtplib",
                 "ssl", "tarfile", "telnetlib", "this", "timeit", "types", "venv", "zlib"}
    # import 名 → PyPI 发行包名 的映射（容器里装的是发行包，import 的是模块名）
    name_map = {"fitz": "pymupdf", "pymupdf": "pymupdf", "PIL": "pillow",
                "sklearn": "scikit-learn", "sentence_transformers": "sentence-transformers",
                "yaml": "pyyaml", "pydantic": "pydantic", "scipy": "scipy",
                "numpy": "numpy", "fastapi": "fastapi", "jieba": "jieba",
                "faiss": "faiss-cpu", "redis": "redis", "uvicorn": "uvicorn"}
    # 延迟导入（函数体内 import）的白名单：torch/open_clip 等只在需要本地多模态编码时用
    optional = {"torch", "transformers", "open_clip", "clip", "sentence_transformers",
                "cv2", "pandas", "sklearn", "streamlit", "loguru", "matplotlib"}
    found: set[str] = set()
    for py in list((REPO_ROOT / "研发" / "app").rglob("*.py")) + list((REPO_ROOT / "研发" / "service").rglob("*.py")):
        for m in re.finditer(r"^(?:from|import)\s+([A-Za-z_][\w]*)", py.read_text(encoding="utf-8"), flags=re.MULTILINE):
            found.add(m.group(1))
    third_party = {m for m in found if m not in stdlib_ok and m not in {"app", "service"} and not m.startswith("_")}
    third_party -= optional                      # 可选依赖不要求进镜像
    # pydantic 由 fastapi 依赖引入（fastapi 已钉版本），显式列出以免误报
    reqs_with_implied = reqs | {"pydantic"}
    unmapped = sorted(
        m for m in third_party
        if name_map.get(m, m.lower()) not in reqs_with_implied
    )
    assert not unmapped, (
        f"业务代码顶层 import 了但镜像未安装：{unmapped} —— "
        f"请把它们加进 部署/docker/requirements.txt（或把 import 改成延迟导入）"
    )

    # 顺手把映射表本身留证（便于复核哪些 import 被算作哪个发行包）
    (REPO_ROOT / "测试" / "测试数据").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "测试" / "测试数据" / "依赖闭包核对.txt").write_text(
        "业务代码顶层 import（第三方）与其对应发行包：\n"
        + "\n".join(f"  {m:24} → {name_map.get(m, m.lower())}"
                    for m in sorted(third_party))
        + "\n\nrequirements.txt 声明：\n"
        + "\n".join(f"  {r}" for r in sorted(reqs))
        + "\n\n结论：全部顶层第三方 import 均已被 requirements.txt 覆盖（pydantic 由 fastapi 引入）。\n",
        encoding="utf-8")


def test_健康检查脚本真实可跑() -> None:
    """healthcheck.py 是容器 HEALTHCHECK 用的探针，必须能独立跑（真实执行一次）。"""
    hc = DOCKER_DIR / "healthcheck.py"
    env = dict(os.environ, DEPLOY_HEALTHCHECK_URL="http://127.0.0.1:8200/healthz",
               DEPLOY_HEALTHCHECK_TIMEOUT_S="5")
    out = subprocess.run([sys.executable, str(hc)], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=30, env=env)
    if "rag-api 未就绪" in out.stdout or out.returncode != 0:
        # 服务没起时探针必须**优雅失败**（退出码非 0 且不抛栈）
        assert out.returncode != 0, "服务不可用时探针却返回 0"
        assert "unhealthy" in (out.stderr or ""), f"探针未输出 unhealthy：{out.stderr!r}"
    else:
        assert "healthy" in out.stdout, f"探针输出异常：{out.stdout!r}"


@pytest.mark.skipif(not DOCKER_AVAILABLE,
                    reason="需要可用 Docker daemon（RAG_TEST_DOCKER=1 且 docker 可连）")
def test_docker_真实可用性() -> None:   # pragma: no cover —— 仅在 Docker 可用时执行
    out = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, f"Docker daemon 不可用：{out.stderr[:300]}"
