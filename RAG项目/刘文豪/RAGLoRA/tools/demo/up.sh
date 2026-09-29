#!/usr/bin/env bash
# 按需拉起 Docker 组件。用法: bash tools/demo/up.sh {milvus|neo4j|attu|all}
set -euo pipefail
ROOT_DIR="D:/桌面/RAGLoRA"
TARGET="${1:-}"

case "$TARGET" in
  milvus|neo4j|attu) ;;
  all)
    # 内存预算护栏：本机同时运行着 Dify，提交量（可用物理内存）仅约 2.2G。
    # 而 Milvus standalone 约需 2.5G（含嵌入式 etcd/minio），Neo4j 约需 1.5G，
    # 两者合计约 4G，明显超出预算。同时拉起会引发换页抖动甚至 OOM-kill，
    # 进而波及正在运行的 Dify 容器，因此默认拒绝，需显式强制。
    if [ "${RAGLORA_FORCE_ALL:-0}" != "1" ]; then
      cat >&2 <<'EOF'
[已中止] `all` 会同时拉起 Milvus 与 Neo4j，超出本机内存预算。

  内存估算：Milvus standalone ≈ 2.5G（含嵌入式 etcd/minio）
            Neo4j            ≈ 1.5G
            合计             ≈ 4.0G
  本机提交量（可用物理内存）仅 ≈ 2.2G（Dify 常驻占用之外所剩）。
  两者并存会触发换页抖动甚至 OOM-kill，可能连带杀掉本机正在运行的 Dify 容器。

  建议逐个启动、用完即停，复用内存：
    bash tools/demo/up.sh milvus    # 用完 -> bash tools/demo/down.sh milvus
    bash tools/demo/up.sh neo4j     # 用完 -> bash tools/demo/down.sh neo4j
  （先 down 再 up 另一个，任何时刻只驻留一个组件，峰值内存 ≈ 2.5G 仍在预算内。）

  若你已确认内存充足、确需同时启动，请显式强制：
    RAGLORA_FORCE_ALL=1 bash tools/demo/up.sh all
EOF
      exit 3
    fi
    ;;
  *) echo "用法: bash tools/demo/up.sh {milvus|neo4j|attu|all}" >&2; exit 1 ;;
esac

up_one() {
  local svc="$1"
  echo "==> 启动 $svc（compose 项目 raglora）"
  docker compose -f "$ROOT_DIR/docker/docker-compose.$svc.yml" up -d || {
    local rc=$?
    echo "[$svc] docker compose up 失败（退出码 $rc），不再等待就绪。" >&2
    return "$rc"
  }
}

if [ "$TARGET" = "all" ]; then
  up_one milvus
  up_one neo4j
  # Attu 是 Milvus 的 Web 界面，几乎不占内存（约 100MB 级），
  # 但它**依赖 Milvus**，所以只在 milvus/neo4j 都起之后才拉。
  up_one attu
else
  up_one "$TARGET"
fi

echo "等待服务就绪（20s）..."
sleep 20
docker ps --filter "name=raglora-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
