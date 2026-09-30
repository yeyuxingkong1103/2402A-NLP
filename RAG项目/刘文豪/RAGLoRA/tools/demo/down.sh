#!/usr/bin/env bash
# 停止 Docker 组件并释放内存。用法: bash tools/demo/down.sh {milvus|neo4j|attu|all}
set -euo pipefail
ROOT_DIR="D:/桌面/RAGLoRA"
TARGET="${1:-all}"

case "$TARGET" in
  milvus|neo4j|attu|all) ;;
  *)
    echo "用法: bash tools/demo/down.sh {milvus|neo4j|attu|all}" >&2
    exit 1
    ;;
esac

# 安全护栏：本机同时运行着 Dify（11 个 docker-* 容器，其 compose 项目名为 `docker`）。
# 真正把 raglora 与 Dify 隔开的是 compose 文件里声明的项目名 `name: raglora` ——
# 它保证 compose 只操作 raglora 项目的容器/网络/卷，绝不会碰到 docker-* 那一堆。
# 因此护栏校验的就是这个性质：目标 compose 文件必须仍声明 `name: raglora`。
# （注意：护栏不校验容器名。容器叫 milvus/neo4j 与 Dify 无关，按容器名拦截既拦不住
#  Dify 的 docker-*/raglora-* 命名，又会误伤用户自己 docker run --name neo4j 起的容器。）
# 若将来有人在 docker/ 下新增一个漏写 name: 的 compose 文件，其项目名会退化为目录名
# `docker`，从而与 Dify 串台 —— 这里直接中止。
down_one() {
  local svc="$1"
  grep -q '^name: raglora$' "$ROOT_DIR/docker/docker-compose.$svc.yml" || {
    echo "[$svc] 项目名护栏失效（compose 未声明 name: raglora），已中止以免误伤其它 compose 项目" >&2
    return 2
  }
  echo "==> 停止 $svc（compose 项目 raglora）"
  docker compose -f "$ROOT_DIR/docker/docker-compose.$svc.yml" down || {
    local rc=$?
    echo "[$svc] docker compose down 失败（退出码 $rc）" >&2
    return "$rc"
  }
}

rc=0
for svc in milvus neo4j; do
  if [ "$TARGET" = "all" ] || [ "$TARGET" = "$svc" ]; then
    down_one "$svc" || rc=$?
  fi
done

echo "已停止。当前 raglora 容器："
docker ps --filter "name=raglora-" --format "table {{.Names}}\t{{.Status}}"

exit "$rc"
