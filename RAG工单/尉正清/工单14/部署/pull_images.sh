#!/bin/bash
# 工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失
# 拉取 RAGFlow 依赖镜像（带自动重试）
#
# ⚠️ 本机 Docker Hub 直连被墙（实测 18 秒超时），daemon.json 里配了三个国内加速器，
#    但实测只有 docker.m.daocloud.io 可达（HTTP 401 = 正常）。该加速器**偶发 TLS 超时**，
#    所以这里每个镜像最多重试 5 次 —— 单次失败不代表源不可用。
set -u
IMAGES=(
  "infiniflow/ragflow:v0.23.1"
  "elasticsearch:8.11.3"
  "mysql:8.0.39"
  "valkey/valkey:8"
  "quay.io/minio/minio:RELEASE.2025-06-13T11-33-47Z"
)
MAX_TRY=5
for IMG in "${IMAGES[@]}"; do
    if docker image inspect "$IMG" >/dev/null 2>&1; then
        echo "[跳过] $IMG 已存在"; continue
    fi
    ok=0
    for n in $(seq 1 $MAX_TRY); do
        echo "[$IMG] 第 $n/$MAX_TRY 次尝试 ..."
        if docker pull "$IMG" 2>&1 | tail -3 | grep -qE "Pull complete|Image is up to date|Status: Downloaded"; then
            echo "[成功] $IMG"; ok=1; break
        fi
        sleep 5
    done
    [ "$ok" = 0 ] && echo "[失败] $IMG 重试 $MAX_TRY 次仍未成功"
done
echo; echo "=== 镜像清单 ==="
docker images --format "  {{.Repository}}:{{.Tag}} | {{.Size}}"
