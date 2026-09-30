#!/usr/bin/env bash
# 把 Milvus standalone（etcd + minio + milvus）用 Docker 起来 —— 目标是让 Milvus
# 不再依赖那台 VMware VM（即"从 VMware 换成 WSL"的终点）。
#
# 用法（在 WSL / 任何能访问 docker 的 Linux 里）：
#   bash scripts/milvus_docker_up.sh              # 起（幂等：已有同名容器则优先复用）
#   bash scripts/milvus_docker_up.sh --recreate   # 先 down 再 up（删容器，数据卷保留）
#   bash scripts/milvus_docker_up.sh --status     # 只看状态
#   bash scripts/milvus_docker_up.sh --down       # 停并删除容器（数据卷保留在磁盘上）
#
# 从 Windows 侧（PowerShell，零引号）：
#   wsl -d Ubuntu -e bash /mnt/e/deepseekharness/LLaMA-Factory/scripts/milvus_docker_up.sh
#
# ⚠️ 执行前必须先在【管理员】PowerShell 里删掉指向 VM 的端口转发，否则端口冲突：
#     netsh interface portproxy delete v4tov4 listenaddress=127.0.0.1 listenport=19530
#   （本脚本会检测 19530 是否已被占用并给出提示，但删转发需要管理员权限，脚本无权代做）
#
# 关键点（都是踩过的坑）：
#   * **已有同名容器时优先复用**：Docker Desktop 里可能本来就有一套
#     milvus-etcd / milvus-minio / milvus-standalone（例如 5 天前建的、停在 Exited）。
#     这时直接 docker start（etcd → minio → standalone），**零下载、不动数据卷**；
#     不跑 compose up -d —— 否则必然报容器名冲突：
#       Error response from daemon: Conflict. The container name "/milvus-minio" is already in use
#   * 数据卷放 $HOME/milvus（WSL 的 ext4）。**不要放 /mnt/e** —— drvfs 上跑数据库很慢。
#   * 优先拉官方 compose；国内拉 GitHub 常超时，拉不到就用脚本内置的等价 compose。
#   * 镜像来自 Docker Hub / quay.io，国内可能很慢甚至拉不动；失败会原样打印错误。
#   * 末尾的"就绪"只到**端口级**：端口通了不等于 Milvus 能执行 DDL。起服务前请按
#     docs/WSL.md 第 8.3 节用 pymilvus 反复 list_collections() 确认。

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$HERE/.." && pwd)"
MILVUS_DIR="${MILVUS_DIR:-$HOME/milvus}"
LOG="$WORKSPACE/.tmp/milvus_docker_up.log"
MILVUS_VERSION="${MILVUS_VERSION:-v2.4.15}"
MILVUS_PORT=19530
#: 这套 compose 的三个容器名（依赖顺序：etcd → minio → standalone）
CONTAINERS="milvus-etcd milvus-minio milvus-standalone"

mkdir -p "$MILVUS_DIR" "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1

ACTION="up"
RECREATE=0
case "${1:-}" in
    --down)     ACTION="down" ;;
    --status)   ACTION="status" ;;
    --recreate) ACTION="up"; RECREATE=1 ;;
    ""|--up)    ACTION="up" ;;
    -h|--help)
        sed -n '2,28p' "${BASH_SOURCE[0]}"
        exit 0 ;;
    *) echo "未知参数：$1（可用：--up / --recreate / --down / --status / --help）" >&2; exit 2 ;;
esac

echo "=== Milvus in Docker：$ACTION  $(date -Is) ==="
echo "目录      = $MILVUS_DIR"
echo "版本      = $MILVUS_VERSION（仅 compose 会用到；复用已有容器时不重新拉镜像）"
echo "端口      = $MILVUS_PORT"
echo "日志      = $LOG"
echo

if ! command -v docker >/dev/null 2>&1; then
    echo "!!! 找不到 docker。修复建议："
    echo "    1) Windows 侧确认 Docker Desktop 已启动"
    echo "    2) Docker Desktop → Settings → Resources → WSL Integration 勾上 Ubuntu"
    echo "    3) 本发行版里验证：docker --version"
    exit 2
fi
docker --version
if ! docker info >/dev/null 2>&1; then
    echo "!!! docker 装上了但连不上引擎（Docker Desktop 没启动？）"
    docker info 2>&1 | head -5
    exit 2
fi
echo

port_busy() {
    python3 - "$MILVUS_PORT" <<'PY'
import socket, sys
port = int(sys.argv[1])
try:
    socket.create_connection(("127.0.0.1", port), timeout=2).close()
    print("busy")
except Exception:
    print("free")
PY
}

# 容器的 State.Status；不存在时返回 missing（docker inspect 失败即视为不存在）
container_state() {
    local st
    st="$(docker inspect -f '{{.State.Status}}' "$1" 2>/dev/null)" || st="missing"
    [ -n "$st" ] || st="missing"
    printf '%s' "$st"
}

if [ "$ACTION" = "status" ]; then
    echo "--- 19530 状态 ---"
    echo "  127.0.0.1:$MILVUS_PORT = $(port_busy)"
    echo "--- 容器 ---"
    docker ps -a --filter "name=milvus" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}' 2>&1
    exit 0
fi

if [ "$ACTION" = "down" ]; then
    echo "--- 停止并删除 milvus 相关容器（数据卷保留在 $MILVUS_DIR/volumes）---"
    if [ -f "$MILVUS_DIR/docker-compose.yml" ]; then
        ( cd "$MILVUS_DIR" && docker compose down ) 2>&1 | tail -12
    else
        for c in $CONTAINERS; do
            docker rm -f "$c" >/dev/null 2>&1 && echo "  已删除 $c"
        done
    fi
    echo "完成。"
    exit 0
fi

# ---------------- up ----------------
if [ "$RECREATE" -eq 1 ]; then
    echo "--- [0/5] --recreate：先停并删除现有容器（数据卷保留在 $MILVUS_DIR/volumes）---"
    if [ -f "$MILVUS_DIR/docker-compose.yml" ]; then
        ( cd "$MILVUS_DIR" && docker compose down ) 2>&1 | tail -12
    else
        for c in $CONTAINERS; do
            docker rm -f "$c" >/dev/null 2>&1 && echo "  已删除 $c"
        done
    fi
    echo "  容器已删除，$MILVUS_DIR/volumes 未动 —— 下面按正常流程 compose up -d 重建。"
    echo
fi

echo "--- [1/5] 检查 $MILVUS_PORT 是否已被占用 ---"
state="$(port_busy)"
echo "  127.0.0.1:$MILVUS_PORT = $state"
if [ "$state" = "busy" ]; then
    echo
    echo "  ⚠️ 端口已被占用。若那是你之前加的、指向 VM 的 portproxy，必须先删掉，否则"
    echo "     Docker 发布端口会失败。请在【管理员】PowerShell 执行："
    echo
    echo "       netsh interface portproxy delete v4tov4 listenaddress=127.0.0.1 listenport=19530"
    echo "       netsh interface portproxy show all"
    echo
    echo "     删完再重跑本脚本。（若占用者是别的程序，自己判断是否要停它。）"
    echo
    echo "  已继续进行，但很可能起不来 —— 下一节的容器状态会暴露问题。"
fi
echo

echo "--- [2/5] 探测是否已有同名容器（docker ps -a --filter name=milvus）---"
docker ps -a --filter "name=milvus" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' 2>&1
echo
existing=0; exited=0; running=0
for c in $CONTAINERS; do
    st="$(container_state "$c")"
    printf '  %-20s %s\n' "$c" "$st"
    [ "$st" != "missing" ] && existing=$((existing + 1))
    [ "$st" = "exited" ] && exited=$((exited + 1))
    [ "$st" = "running" ] && running=$((running + 1))
done
if [ "$existing" -eq 0 ]; then
    MODE="compose"
    echo "  未发现同名容器 → 走 docker compose up -d（首次要拉三个镜像，国内可能很慢）"
elif [ "$existing" -eq 3 ] && [ "$exited" -eq 3 ]; then
    MODE="reuse"
    echo "  ✓ 三个容器都在且都停在 Exited → 【复用已有容器】（零下载、不动数据卷）"
    echo "    复用不会重新拉镜像，所以容器里的 Milvus 版本可能与本脚本 compose 默认版本不同，"
    echo "    以下面的【容器状态】为准。"
elif [ "$existing" -eq 3 ] && [ "$running" -eq 3 ]; then
    MODE="running"
    echo "  ✓ 三个容器都已存在且都在运行 → 跳过 compose up -d（避免容器名冲突），直接等就绪"
else
    MODE="abnormal"
    echo
    echo "  ⚠️ 已有容器处于【异常 / 不完整】状态（existing=$existing exited=$exited running=$running）"
    echo "     可能原因：上次只起了一半、手工删过其中一个、或某个容器卡在 restarting/paused。"
    echo "     本脚本在这种状态下【不删容器】，请二选一："
    echo "       1) 先看现场：docker ps -a --filter name=milvus"
    echo "                    docker logs --tail 40 milvus-standalone"
    echo "       2) 确认可以重建后跑：bash scripts/milvus_docker_up.sh --recreate"
    echo "          它执行 docker compose down && docker compose up -d ——"
    echo "          ⚠️ down 会【删掉这三个容器】，但 $MILVUS_DIR/volumes 里的数据保留。"
    echo
    echo "  → 本脚本到此为止（退出码 4），没有改动任何容器。"
    exit 4
fi
echo

NEED_COMPOSE=0
[ "$MODE" = "compose" ] && NEED_COMPOSE=1

echo "--- [3/5] 准备 compose 文件 ---"
if [ "$NEED_COMPOSE" -eq 0 ]; then
    echo "  复用/已在运行模式：不需要 compose 文件（跳过）"
else
    COMPOSE="$MILVUS_DIR/docker-compose.yml"
    if [ -f "$COMPOSE" ]; then
        echo "  已存在，复用：$COMPOSE"
    else
        ok=0
        for url in \
            "https://github.com/milvus-io/milvus/releases/download/${MILVUS_VERSION}/milvus-standalone-docker-compose.yml" \
            "https://raw.githubusercontent.com/milvus-io/milvus/${MILVUS_VERSION}/deployments/docker/standalone/docker-compose.yml"
        do
            echo "  尝试下载：$url"
            if curl -fsSL --max-time 25 "$url" -o "$COMPOSE.tmp" 2>&1; then
                if [ -s "$COMPOSE.tmp" ]; then
                    mv "$COMPOSE.tmp" "$COMPOSE"; echo "  ✓ 下载成功"; ok=1; break
                fi
            fi
            echo "  ✗ 失败或超时"
        done
        rm -f "$COMPOSE.tmp"
        if [ "$ok" -ne 1 ]; then
            echo
            echo "  官方 compose 拉不到（国内对 GitHub 常超时）→ 使用脚本内置的等价 compose。"
            echo "  ⚠️ 内置版按 Milvus ${MILVUS_VERSION} 的常见部署写成；若你用别的版本，请自行核对。"
            cat > "$COMPOSE" <<EOF
services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.16
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ./volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://etcd:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2024-05-28T17-19-04Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - ./volumes/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:${MILVUS_VERSION}
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ./volumes/milvus:/var/lib/milvus
    ports:
      - "${MILVUS_PORT}:19530"
      - "9091:9091"
    depends_on:
      - etcd
      - minio
EOF
            echo "  已写入 $COMPOSE"
        fi
    fi
fi
echo

echo "--- [4/5] 起容器 ---"
if [ "$MODE" = "reuse" ]; then
    echo "  复用已有容器（零下载、不动数据卷）：按依赖顺序 docker start（etcd → minio → standalone）"
    for c in $CONTAINERS; do
        printf '  %-20s ' "$c"
        if docker start "$c" >/dev/null 2>&1; then
            echo "已启动"
        else
            echo "启动失败（下面会重新列出状态与日志）"
        fi
        sleep 2   # 先让 etcd/minio 起来，standalone 之后才能连上
    done
elif [ "$MODE" = "running" ]; then
    echo "  三个容器已在运行 → 不执行 compose up -d（同名容器会冲突）"
else
    echo "  docker compose up -d（首次要拉三个镜像，国内可能很慢）"
    ( cd "$MILVUS_DIR" && docker compose up -d ) 2>&1 | tail -40
    echo "  compose 退出码 = ${PIPESTATUS[0]}"
    echo "  提示：若报容器名冲突（Conflict. The container name ... is already in use），"
    echo "        说明已有同名容器 —— 重跑不加参数的版本会优先复用；要重建则用 --recreate。"
fi
echo

echo "--- [5/5] 等 $MILVUS_PORT 端口就绪（最多 180s）---"
python3 - <<'PY'
import socket, time, sys
port = 19530
t0 = time.time()
while time.time() - t0 < 180:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=3).close()
        print("  ✓ 127.0.0.1:%d 已就绪，用时 %.1fs" % (port, time.time() - t0))
        sys.exit(0)
    except Exception:
        time.sleep(4)
print("  ✗ 180s 内未就绪 —— 看下面的容器状态与日志")
sys.exit(1)
PY
ready=$?
echo

echo "--- 容器状态 ---"
docker ps -a --filter "name=milvus" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' 2>&1
echo

if [ "$ready" -ne 0 ]; then
    echo "--- standalone 日志尾部 40 行（排查用；注意时间戳，别把几天前的旧日志当成刚才的）---"
    docker logs --tail 40 milvus-standalone 2>&1
    echo
    echo "常见原因："
    echo "  1) 19530 被 portproxy 占用 → 先删转发（见上面命令）再重跑"
    echo "  2) 镜像拉取失败/超时（国内网络）→ 给 Docker Desktop 配置镜像加速后再重跑"
    echo "  3) etcd/minio 没起来 → docker logs milvus-etcd / milvus-minio"
    echo "  4) 现有容器状态异常 → bash scripts/milvus_docker_up.sh --recreate"
    echo "     （down 删容器但保留 $MILVUS_DIR/volumes 里的数据）"
    exit 3
fi

echo "=== 成功：Milvus 现在跑在本地 Docker 里 ==="
echo "⚠️ 注意：上面只确认了【端口级】就绪；端口通 ≠ Milvus 能执行 DDL。"
echo "   起服务前请先确认它真能干活（pymilvus 反复 list_collections() 直到成功）："
echo "   docs/WSL.md 第 8.3 节有可直接粘贴的命令。"
echo "下一步："
echo "  1) 用本地 Milvus 起服务（同一台机器上的 127.0.0.1:19530）："
echo "     wsl -d Ubuntu -e bash /mnt/e/deepseekharness/LLaMA-Factory/.tmp/wsl_run_local.sh"
echo "  2) 首次启动时交付 collection 是空的，应用会自动 ingest knowledge/（3 篇 / 16 个 chunk）"
echo "  3) 确认无误后可以不再开那台 VMware VM"
