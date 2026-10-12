#!/usr/bin/env bash
# 工单编号：人工智能NLP-RAG-文档质量评估 Skill 与 RAGFlow 集成
#
# 停 RAGFlow。
#
# 用法：
#   bash stop.sh             # docker compose down：停容器并删除容器（数据在 volume 里，安全）
#   bash stop.sh --volumes   # 连 volume 一起删 —— 知识库数据、上传的文档全没，慎用
#
# 注意：compose down 会**删掉容器**，容器可写层里手动装的 pymupdf 也随之丢失。
#       下次 bash start.sh 之后必须重跑 bash install.sh。
#       只想暂停不丢东西的话，用 docker compose stop（见 DEPLOYMENT.md）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-${SCRIPT_DIR}/../研发/ragflow/docker/docker-compose.yml}"
CONTAINER="${CONTAINER:-docker-ragflow-cpu-1}"

DOWN_ARGS=()
if [ "${1:-}" = "--volumes" ]; then
    echo "[注意] 带 --volumes：volume 会被删除，知识库数据和已上传文档不可恢复。"
    printf '确认继续？输入 yes 回车：'
    read -r ANSWER || true
    [ "${ANSWER}" = "yes" ] || { echo "已取消。"; exit 0; }
    DOWN_ARGS+=(--volumes)
fi

# 空数组在 set -u 下的老坑：直接 "${DOWN_ARGS[@]}" 在 bash 4.3 及更早会报未绑定变量
compose_down() {
    if [ "${#DOWN_ARGS[@]}" = "0" ]; then
        docker compose -f "${COMPOSE_FILE}" down
    else
        docker compose -f "${COMPOSE_FILE}" down "${DOWN_ARGS[@]}"
    fi
}

echo "==> 检查 compose 文件：${COMPOSE_FILE}"
[ -f "${COMPOSE_FILE}" ] || { echo "[错误] 找不到 compose 文件" >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "[错误] docker compose 不可用" >&2; exit 1; }

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
    echo "==> docker compose down ${DOWN_ARGS[*]:-}"
    compose_down
else
    echo "==> 容器 ${CONTAINER} 本来就不存在，还是跑一遍 down 收尾"
    compose_down
fi

echo "==> 已停。残余检查："
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
    echo "    [注意] 容器还在：docker rm -f ${CONTAINER}"
else
    echo "    容器已删除"
fi
echo "    下次 start.sh 起来后记得：bash install.sh（补 pymupdf）"
