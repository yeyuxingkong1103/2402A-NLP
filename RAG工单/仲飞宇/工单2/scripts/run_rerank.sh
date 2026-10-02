#!/usr/bin/env bash
# 启动本地 bge-reranker 重排服务（rerank_service/.venv，端口默认 8001）
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/lib.sh

HOST="${RERANK_HOST:-127.0.0.1}"
PORT="${RERANK_PORT:-8001}"
# pid 与日志都带端口：以前写死 logs/rerank.pid，换个端口再起一个会把它覆盖掉，
# 先起的那个实例就成了没人管、也没法按 pid 停的孤儿——而每个孤儿常驻 ~2.2GB。
PIDFILE="logs/rerank-${PORT}.pid"
LOGFILE="logs/rerank-${PORT}.out"
MODEL_ID="${RERANK_MODEL:-BAAI/bge-reranker-v2-m3}"

# 优先复用本地已缓存的模型目录，避免重复下载
find_local_model() {
    # 主路径：scripts/download_rerank_model.sh 下到 D 盘的整目录。
    # 备路径：早期从 HF 缓存下的残留（snapshots/<rev>/），有时也可能在 Windows 侧家目录。
    local direct="${RERANK_MODEL_DIR:-/mnt/d/models/bge-reranker-v2-m3}"
    [[ -f "$direct/model.safetensors" ]] && { echo "$direct"; return 0; }

    local cache_dirs=(
        "/mnt/c/Users/15277/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots"
        "$HOME/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots"
    )
    local d snap
    for d in "${cache_dirs[@]}"; do
        [[ -d "$d" ]] || continue
        snap="$(find "$d" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)"
        [[ -n "$snap" && -f "$snap/model.safetensors" ]] && { echo "$snap"; return 0; }
    done
    return 1
}

if [[ ! -d "rerank_service/.venv" ]]; then
    echo "❌ 未找到 rerank_service/.venv。请先安装依赖："
    echo "   python3 -m venv rerank_service/.venv"
    echo "   rerank_service/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu"
    echo "   rerank_service/.venv/bin/pip install -r rerank_service/requirements.txt"
    exit 1
fi

if is_running "$PIDFILE"; then
    echo "重排服务已在运行（端口 $PORT，pid $(pid_of "$PIDFILE")）"
    exit 0
fi

# 其它端口上是否已经有实例？app 只认 .env 里的 RERANK_BASE_URL 一个地址，
# 多出来的实例没有任何一边会用到，却各自常驻一份 ~2.2GB 的模型权重——
# WSL 内存上限只有几 GB（C:\Users\15277\.wslconfig），多塞两个就够触发 OOM，
# 而 WSL 一 OOM，PyCharm 那个 wsl.exe+ConPTY 的终端就跟着掉。
# 这里只警告不动手：不阻断 start_all.sh 的调用；收摊交给 scripts/shutdown.sh。
strays="$(ps -eo pid,args 2>/dev/null | grep "[u]vicorn rerank_service.server:app" | grep -vE -- "--port[= ]${PORT}\$" || true)"
if [[ -n "$strays" ]]; then
    echo "⚠️  其它端口上已有重排实例在跑："
    echo "$strays" | sed 's/^/      /'
    echo "      每个实例独立加载一份 ~2.2GB 的 bge-reranker 权重，"
    echo "      但 RERANK_BASE_URL 只指向一个地址，其余纯占内存。"
    echo "      确认用不到就先收掉：bash scripts/shutdown.sh"
    echo
fi

MODEL_PATH="$(find_local_model || true)"
[[ -n "$MODEL_PATH" ]] || MODEL_PATH="$MODEL_ID"

# 国内直连 huggingface.co 通常不通（实测 HTTP 000），模型未缓存时走 hf-mirror 镜像；
# 已显式设置 HF_ENDPOINT 时尊重用户选择。
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p logs
echo "启动 bge-reranker 重排服务: http://$HOST:$PORT"
echo "模型: $MODEL_PATH"
# PYTHONUNBUFFERED：stdout 一旦重定向到文件，python 就从行缓冲切成块缓冲（~8KB），
# "Loading weights" 和 uvicorn 的启动日志会全卡在缓冲区里，日志看起来一片空白——
# 排查时极易被误判成「进程崩了」。冷启动本来就要 1–2 分钟，这段时间日志有输出很重要。
nohup env PYTHONUNBUFFERED=1 RERANK_MODEL="$MODEL_PATH" rerank_service/.venv/bin/uvicorn rerank_service.server:app --host "$HOST" --port "$PORT" > "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
echo "已启动，pid=$(cat "$PIDFILE")，日志: $LOGFILE"
