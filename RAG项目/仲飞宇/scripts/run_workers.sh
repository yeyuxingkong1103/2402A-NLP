#!/usr/bin/env bash
# 多 worker 启动：为 nginx 负载均衡提供后端（deploy/nginx.conf 的 upstream 就是 8000/8001/8002）。
#
# 与 run.sh 的关系：run.sh 起**一个** uvicorn 并写死 logs/uvicorn.pid，第二次调用会被
# 单实例守卫挡掉。本脚本起多个实例，每个实例有独立的 pid 文件与日志，互不干扰。
#
# 用法：
#   bash scripts/run_workers.sh                 # 起 8000/8001/8002
#   PORTS="8000 8001" bash scripts/run_workers.sh
#   bash scripts/run_workers.sh --status        # 看各 worker 状态 + /health
#   bash scripts/run_workers.sh --stop          # 停掉全部 worker
#   # 压测「系统吞吐上限」（把 LLM 换成回显，隔离出接口+检索的能力）：
#   LLM_PROVIDER=dummy bash scripts/run_workers.sh
#
# ⚠️ 两个必须知道的约束（踩过，别再重新推导）：
#
# 1. **Milvus Lite 不能多进程共享。** Lite 用 `./data/milvus.db` 时会独占数据目录的文件锁，
#    第二个 worker 连不上向量库（DataDirLockedError / ping 失败）。所以多 worker 必须把
#    MILVUS_DB_URI 指向独立的 Milvus 服务（本脚本默认已指向 127.0.0.1:19530）。
#    起独立服务的定义见 deploy/docker-compose.milvus.yml。
#
# 2. **APP_ENV 会反压这里设的环境变量。** app/core/config.py:16 的 load_dotenv 默认
#    override=False（真实环境变量优先），但 :21 在 APP_ENV 非空时会用 .env.{APP_ENV}
#    以 override=True 再加载一次。所以若 shell 里带着 APP_ENV，本脚本导出的
#    MILVUS_DB_URI 会被 .env.prod 等文件覆盖掉，切库静默失效——这里直接拦下来报错。
#
# 另：每个 worker 的应用日志按端口分文件（logs/app-<port>.log，靠下面注入的
# WORKER_PORT）。三个进程共写一个 app.log 不行：按天轮转是「改名 + 新建同名文件」，
# 各进程持自己的 fd，跨天时输的那个进程日志直接丢。详见 app/core/logging_config.py。
#
# 另：worker 之间共享短期记忆靠 MEMORY_BACKEND=redis（RedisMemoryStore 的 key 是
# memory:{session}:{role}，天然跨进程）；BM25 索引则是**每进程**缓存
# （app/core/retrieve/hybrid_retriever.py:81），/knowledge/upload 只在处理它的那个
# worker 里失效——多 worker 下新上传的内容在其他 worker 关键词召不回。压测是只读，
# 不受影响，但线上多实例部署要注意这点。

set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/lib.sh

[[ -d ".venv" ]] && source .venv/bin/activate

HOST="${HOST:-0.0.0.0}"
# 端口刻意跳过 8001：那是 BGE 重排服务（RERANK_BASE_URL=http://127.0.0.1:8001/v1）。
# deploy/nginx.conf 原有的 upstream 写的是 8000/8001/8002，因为那份配置从没真跑过，
# 一直没暴露这个撞车——起 worker 前必须先把 upstream 改成本脚本的端口。
PORTS="${PORTS:-8000 8002 8003}"
# 多 worker 必须是共享的向量库；保持与 deploy/docker-compose.milvus.yml 的端口一致
export MILVUS_DB_URI="${MILVUS_DB_URI:-http://127.0.0.1:19530}"
# 跨 worker 共享短期记忆
export MEMORY_BACKEND="${MEMORY_BACKEND:-redis}"

pidfile_of()  { echo "logs/uvicorn-$1.pid"; }
logout_of()   { echo "logs/uvicorn-$1.out"; }
running()     { is_running "$(pidfile_of "$1")"; }

mkdir -p logs

# ---------- --stop / --status ----------
MODE="${1:-start}"
case "$MODE" in
--stop)
    for p in $PORTS; do
        stop_by_pidfile "worker :$p" "$(pidfile_of "$p")"
    done
    exit 0
    ;;
--status)
    for p in $PORTS; do
        if running "$p"; then
            h="$(curl -sf -m 5 "http://127.0.0.1:$p/health" || echo '{"status":"无响应"}')"
            printf '  :%s pid=%-6s %s\n' "$p" "$(pid_of "$(pidfile_of "$p")")" "$h"
        else
            printf '  :%s 未运行\n' "$p"
        fi
    done
    exit 0
    ;;
esac

# ---------- 前置检查 ----------
if [[ -n "${APP_ENV:-}" ]]; then
    echo "❌ shell 里带着 APP_ENV=${APP_ENV}：app/core/config.py:21 会用 .env.${APP_ENV}"
    echo "   以 override=True 覆盖本脚本导出的 MILVUS_DB_URI，切库会静默失效。"
    echo "   请 unset APP_ENV 后重试（或改用 APP_ENV 那套配置的完整设置）。"
    exit 2
fi

# Lite uri 是本地路径；多 worker 共享它必然失败，提前拦下而不是等第二个 worker 报错
if [[ "$MILVUS_DB_URI" != http* ]]; then
    echo "❌ MILVUS_DB_URI=$MILVUS_DB_URI 是本地文件路径（Milvus Lite）。"
    echo "   Lite 独占数据目录文件锁，多个 worker 只有一个能连上向量库。"
    echo "   先起独立服务再重试："
    echo "     docker compose -f deploy/docker-compose.milvus.yml up -d"
    echo "     MILVUS_DB_URI=http://127.0.0.1:19530 .venv/bin/python scripts/seed.py"
    exit 2
fi

if ! timeout 5 bash -c "cat < /dev/null > /dev/tcp/127.0.0.1/$(echo "$MILVUS_DB_URI" | grep -oE '[0-9]+$')" 2>/dev/null; then
    echo "⚠️  ${MILVUS_DB_URI} 端口探测不通，worker 起来后 /health 里 milvus 会是 unavailable"
fi

# 单实例 run.sh 占着 8000 时先提示，否则起 worker 会端口冲突
if is_running logs/uvicorn.pid; then
    echo "⚠️  logs/uvicorn.pid 里的单实例还在跑（pid $(pid_of logs/uvicorn.pid)）。"
    echo "   它会占用同一个端口，先停掉：bash scripts/shutdown.sh"
fi

# ---------- 启动 ----------
echo "启动 worker（共享向量库 $MILVUS_DB_URI，记忆后端 $MEMORY_BACKEND）"
for p in $PORTS; do
    if running "$p"; then
        echo "  :$p 已在运行（pid $(cat "$(pidfile_of "$p")")），跳过"
        continue
    fi
    # WORKER_PORT 让每个 worker 的**应用日志**也按端口分文件（logs/app-<port>.log）。
    # 不注入的话三个 worker 共写一个 app.log：日志轮转是「改名 + 新建」，多进程各持自己的
    # fd，跨天时输的那个进程日志直接丢（实测见 app/core/logging_config.py 注释）。
    # 与上面 uvicorn-<port>.out 是同一套"带端口"的规矩。
    WORKER_PORT="$p" nohup uvicorn app.main:app --host "$HOST" --port "$p" > "$(logout_of "$p")" 2>&1 &
    echo $! > "$(pidfile_of "$p")"
    echo "  :$p 已启动（pid $(pid_of "$(pidfile_of "$p")")，日志 $(logout_of "$p") + logs/app-$p.log）"
done

# ---------- 等就绪 ----------
echo "等待各 worker /health 就绪…"
for p in $PORTS; do
    if wait_health "http://127.0.0.1:$p/health" 30; then
        echo "  ✅ :$p $(curl -s -m 5 http://127.0.0.1:$p/health | head -c 120)"
    else
        echo "  ❌ :$p 健康检查超时，看 $(logout_of "$p") 末尾"
        tail -5 "$(logout_of "$p")" 2>/dev/null | sed 's/^/     /' || true
    fi
done

echo
echo "nginx 上游（deploy/nginx.conf 的 upstream rag_backend）应对准这些端口：$PORTS"
echo "停全部：bash scripts/run_workers.sh --stop"
