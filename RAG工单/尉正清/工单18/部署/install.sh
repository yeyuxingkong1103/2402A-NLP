#!/usr/bin/env bash
# 工单编号：人工智能NLP-RAG-文档质量评估 Skill 与 RAGFlow 集成
#
# 部署前置检查 + 容器侧依赖安装。做四件事：
#
#   1. 检查 conda 环境 rag_gd —— **只检查，缺包只提示，绝不安装**。
#      那个环境是已交付的工单 01-10 在用的，被上游硬锁，往里装新包会砸掉它们。
#   2. 检查 docker / docker compose 可用、容器在跑。
#   3. 往 RAGFlow 容器的 venv 里装 pymupdf（import fitz）。
#      镜像自带的 /ragflow/.venv 里没有它，而质检要解析 PDF 才能分文字型/扫描型。
#   4. 检查 docker-compose.yml 引用的宿主路径是否都存在。
#      路径不存在时 docker 不报错，它会**建一个空目录**顶上去，
#      结果就是容器里 Skill 代码是空的、端点一调就 500 —— 所以必须提前手动查。
#
# 用法：
#   bash install.sh          # 全量检查 + 装 pymupdf
#   bash install.sh --check  # 只检查，什么都不装，不改任何东西
#
# 重要：容器一旦重建（docker compose down 之后再 up、或 up --force-recreate），
#       venv 可写层里的 pymupdf 会跟着丢，必须重跑本脚本。见 DEPLOYMENT.md 常见问题。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-${SCRIPT_DIR}/../研发/ragflow/docker/docker-compose.yml}"
CONTAINER="${CONTAINER:-docker-ragflow-cpu-1}"
CONTAINER_PY="/ragflow/.venv/bin/python"

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

FAIL=0
ok()   { echo "  [OK]   $*"; }
warn() { echo "  [注意] $*"; }
bad()  { echo "  [错误] $*"; FAIL=1; }

# Git Bash 会把 /ragflow/... 这类参数改写成 C:/Program Files/Git/ragflow/...，报
# "no such file or directory"。设了 MSYS_NO_PATHCONV 它就不多管闲事了（别的 shell 无副作用）。
dexec() { MSYS_NO_PATHCONV=1 docker exec "$@"; }

echo "=============================================================="
echo " 工单18 部署检查：${CHECK_ONLY:+=只检查模式}"
echo " 脚本目录：${SCRIPT_DIR}"
echo "=============================================================="

# ---------------------------------------------------------------- 1/4 conda 环境
echo "==> 1/4 检查 conda 环境 rag_gd（本地跑 Skill 用）"

RAG_GD_PY="${RAG_GD_PYTHON:-}"
if [ -z "${RAG_GD_PY}" ]; then
    for cand in \
        "/d/Anaconda/envs/rag_gd/python.exe" \
        "/mnt/d/Anaconda/envs/rag_gd/python.exe" \
        "D:/Anaconda/envs/rag_gd/python.exe" \
        "${HOME}/anaconda3/envs/rag_gd/bin/python" \
        "${HOME}/miniconda3/envs/rag_gd/bin/python"
    do
        if [ -x "${cand}" ]; then RAG_GD_PY="${cand}"; break; fi
    done
fi

if [ -z "${RAG_GD_PY}" ]; then
    bad "没找到 rag_gd 环境的解释器。用 RAG_GD_PYTHON=/路径/python 显式指定，"
    echo "        或先确认 conda 装在哪：D:\\Anaconda\\envs\\rag_gd\\python.exe"
else
    ok "rag_gd 解释器：${RAG_GD_PY}"
    # 只查这三个包在不在，缺了也只是提示。环境是冻结的（工单 01-10 在用）
    for mod in fitz yaml docx; do
        if "${RAG_GD_PY}" -c "import ${mod}" >/dev/null 2>&1; then
            ok "  依赖 ${mod} 可导入"
        else
            warn "  依赖 ${mod} 导不进来。**不要 pip install 补** ——"
            echo "        rag_gd 是工单 01-10 的交付环境，被上游硬锁；"
            echo "        容器里跑质检不走这个环境（走 /ragflow/.venv），所以不影响端点。"
        fi
    done
fi

# ---------------------------------------------------------------- 2/4 docker
echo "==> 2/4 检查 docker / docker compose"

DC=()
if command -v docker >/dev/null 2>&1; then
    ok "docker 可用：$(docker --version 2>/dev/null | head -1)"
else
    bad "找不到 docker 命令"
fi

if docker compose version >/dev/null 2>&1; then
    DC=(docker compose)
    ok "docker compose（插件版）可用"
elif command -v docker-compose >/dev/null 2>&1; then
    DC=(docker-compose)
    ok "docker-compose（独立版）可用"
else
    bad "docker compose 不可用，起不了 RAGFlow"
fi

if [ -f "${COMPOSE_FILE}" ]; then
    ok "compose 文件：${COMPOSE_FILE}"
else
    bad "compose 文件不存在：${COMPOSE_FILE}"
fi

if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "${CONTAINER}"; then
    ok "容器 ${CONTAINER} 正在运行"
    RUNNING=1
elif docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "${CONTAINER}"; then
    warn "容器 ${CONTAINER} 存在但没在跑，先执行：bash start.sh"
    RUNNING=0
else
    warn "容器 ${CONTAINER} 还不存在（没起过），先执行：bash start.sh"
    RUNNING=0
fi

# ---------------------------------------------------------------- 3/4 pymupdf
echo "==> 3/4 容器侧依赖：pymupdf（fitz）"

if [ "${RUNNING}" != "1" ]; then
    warn "容器没在跑，跳过。起来之后重跑本脚本。"
elif dexec "${CONTAINER}" "${CONTAINER_PY}" -c 'import fitz' >/dev/null 2>&1; then
    ok "pymupdf 已装（import fitz 正常）"
elif [ "${CHECK_ONLY}" = "1" ]; then
    bad "pymupdf 没装。去掉 --check 重跑本脚本，或手动执行："
    echo "        docker exec ${CONTAINER} /usr/local/bin/uv pip install \\"
    echo "            --python ${CONTAINER_PY} pymupdf"
else
    warn "pymupdf 没装，开始安装（约 20MB，走宿主网络）"
    # 用 uv，不用 pip —— 实测镜像里 /ragflow/.venv 是 uv 建的，
    # **没有 pip 可执行文件，也没有 pip 模块**（python -m pip 报 No module named pip）。
    # 直接 pip install 会 "not found"。
    set +e
    dexec "${CONTAINER}" /usr/local/bin/uv pip install --python "${CONTAINER_PY}" pymupdf 2>&1 | tail -3
    rc=${PIPESTATUS[0]}
    if [ "${rc}" != "0" ]; then
        warn "uv 方式失败（exit=${rc}），退回 python -m pip 重试"
        dexec "${CONTAINER}" "${CONTAINER_PY}" -m pip install pymupdf 2>&1 | tail -3
        rc=${PIPESTATUS[0]}
    fi
    set -e
    if [ "${rc}" = "0" ] && dexec "${CONTAINER}" "${CONTAINER_PY}" -c 'import fitz' >/dev/null 2>&1; then
        ok "pymupdf 装好了，import fitz 通过"
    else
        bad "pymupdf 安装失败（exit=${rc}）。手动重试："
        echo "        docker exec ${CONTAINER} /usr/local/bin/uv pip install \\"
        echo "            --python ${CONTAINER_PY} pymupdf"
    fi
fi

# ---------------------------------------------------------------- 4/4 挂载路径
echo "==> 4/4 检查 compose 引用的宿主路径"

if [ -f "${COMPOSE_FILE}" ]; then
    COMPOSE_DIR="$(cd "$(dirname "${COMPOSE_FILE}")" && pwd)"
    MISSING=0
    # 从 compose 里抠出所有以 ./ 或 ../ 开头的 bind mount 源路径
    while IFS= read -r src; do
        [ -n "${src}" ] || continue
        if [ -e "${COMPOSE_DIR}/${src}" ]; then
            ok "  ${src}"
        else
            bad "  缺：${src}"
            MISSING=$((MISSING + 1))
        fi
    done < <(grep -oE '^[[:space:]]*-[[:space:]]*\.\.?/[^:[:space:]]+' "${COMPOSE_FILE}" \
             | sed -E 's/^[[:space:]]*-[[:space:]]*//' | sort -u)

    if [ "${MISSING}" != "0" ]; then
        echo "        路径不存在时 docker 会拿一个空目录顶上，容器里看到的 Skill 是空的，"
        echo "        端点会报「质检失败」。把缺的目录/文件补回源码树再起容器。"
    fi
fi

echo "=============================================================="
if [ "${FAIL}" = "0" ]; then
    echo " 检查通过。下一步：bash quality_api.sh（需要先 export API_KEY）"
else
    echo " 有项目未通过（见上面的 [错误]）。"
fi
echo "=============================================================="
exit "${FAIL}"
