#!/usr/bin/env bash
# 工单编号：人工智能NLP-RAG-文档质量评估 Skill 与 RAGFlow 集成
#
# 起 RAGFlow，并等它真的能响应 HTTP。
#
# 用的 compose 是 研发/ragflow/docker/docker-compose.yml，里面已经包含工单18 的两条挂载：
#   ~/研发/patches/api/apps/document_app.py        -> /ragflow/api/apps/document_app.py
#   ~/研发/document_quality_assessment             -> /ragflow/document_quality_assessment
# 也就是说：**不用改 compose**，本脚本起来的就是带质检端点的版本。
#
# 用法：
#   bash start.sh              # 起容器 + 等健康检查（默认最多等 300 秒）
#   WAIT=600 bash start.sh     # 机器慢就多等会儿
#
# 起来之后如果提示容器是新建的，记得再跑一次 bash install.sh 装 pymupdf。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-${SCRIPT_DIR}/../研发/ragflow/docker/docker-compose.yml}"
CONTAINER="${CONTAINER:-docker-ragflow-cpu-1}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:9380/}"
WAIT="${WAIT:-300}"
CONTAINER_PY="/ragflow/.venv/bin/python"

dexec() { MSYS_NO_PATHCONV=1 docker exec "$@"; }

echo "==> 检查 compose 文件：${COMPOSE_FILE}"
[ -f "${COMPOSE_FILE}" ] || { echo "[错误] 找不到 compose 文件" >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "[错误] docker compose 不可用" >&2; exit 1; }

# .env 不在版本库里（里面有本机密钥，不该提交），所以克隆下来是没有的。
# compose 用 env_file: .env 引用它，缺了会直接报错起不来 —— 这里自动补一份。
ENV_FILE="$(dirname "${COMPOSE_FILE}")/.env"
ENV_EXAMPLE="$(dirname "${COMPOSE_FILE}")/.env.example"
if [ ! -f "${ENV_FILE}" ]; then
    if [ -f "${ENV_EXAMPLE}" ]; then
        cp "${ENV_EXAMPLE}" "${ENV_FILE}"
        echo "==> .env 不存在，已从 .env.example 复制一份"
        echo "    ⚠ 若需要调用大模型，请把 DEEPSEEK_API_KEY 填进去，"
        echo "      或直接 export DEEPSEEK_API_KEY=...（环境变量优先）"
    else
        echo "[错误] 既没有 .env 也没有 .env.example：${ENV_FILE}" >&2
        exit 1
    fi
fi

# 记下起之前的容器 ID：前后不一样就说明容器是新建的，venv 里的东西全没了
BEFORE_ID="$(docker inspect -f '{{.Id}}' "${CONTAINER}" 2>/dev/null || true)"

echo "==> docker compose up -d"
docker compose -f "${COMPOSE_FILE}" up -d

echo "==> 等容器进入 running（最多 60 秒）"
for i in $(seq 1 12); do
    if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
        echo "    容器已运行（第 ${i} 次探测）"
        break
    fi
    [ "${i}" = "12" ] && { echo "[错误] 容器 60 秒还没起来，看日志：docker logs ${CONTAINER} --tail 50" >&2; exit 1; }
    sleep 5
done

echo "==> 等 HTTP 服务就绪：${HEALTH_URL}（最多 ${WAIT} 秒）"
# RAGFlow 启动要 1~2 分钟，先起 nginx/API，再起后台任务。
# 判据放宽：只要 curl 不是 000（连不上），就算 HTTP 层活了；401/404 也算活。
ELAPSED=0
while [ "${ELAPSED}" -lt "${WAIT}" ]; do
    CODE="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "${HEALTH_URL}" 2>/dev/null || echo 000)"
    if [ "${CODE}" != "000" ]; then
        echo "    HTTP 已响应（状态码 ${CODE}，用时约 ${ELAPSED}s）"
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [ "${ELAPSED}" -ge "${WAIT}" ]; then
    echo "[错误] ${WAIT} 秒内 HTTP 没起来。排查：" >&2
    echo "        docker logs ${CONTAINER} --tail 100" >&2
    echo "        docker exec ${CONTAINER} ps aux | grep ragflow_server" >&2
    exit 1
fi

# ------------------------------------------------------------------ 收尾提醒
AFTER_ID="$(docker inspect -f '{{.Id}}' "${CONTAINER}" 2>/dev/null || true)"
if [ "${BEFORE_ID}" != "${AFTER_ID}" ]; then
    echo "==> 容器是新建/重建的（ID 变了）"
    if dexec "${CONTAINER}" "${CONTAINER_PY}" -c 'import fitz' >/dev/null 2>&1; then
        echo "    pymupdf 在，不用管"
    else
        echo "    [注意] pymupdf 不在容器里了 —— 质检端点到 PDF 那步会报 No module named 'fitz'"
        echo "           现在补：bash install.sh"
    fi
fi

echo "==> 起好了。验收步骤："
echo "    export API_KEY=ragflow-xxxxxxxx   # 获取方法见 DEPLOYMENT.md 第 4 步"
echo "    bash quality_api.sh"
