#!/usr/bin/env bash
# ============================================================================
# deploy/start.sh —— **运行脚本**：一条命令把服务起起来
#
# 顺序（顺序有讲究，别乱）：
#   [1/5] 读配置（.env）与解析参数
#   [2/5] 依赖服务：Redis（装了才起；没装会自动降级为内存记忆）
#   [3/5] 大模型服务：vLLM（可选，--with-vllm；本机没卡时不要开）
#   [4/5] 应用：scripts/run_api.py（nohup 后台 + pid 文件）
#   [5/5] 等健康检查（**用 /livez**）并打印 PID / 日志 / 地址
#
# 用法：
#   bash deploy/start.sh                       # 用 .env 的配置起服务
#   bash deploy/start.sh --offline             # 离线模式（内存库+零依赖嵌入+mock 大模型）
#   bash deploy/start.sh --with-vllm --vllm-model /models/qwen27b
#   bash deploy/start.sh --restart             # 已在跑就先停再起
#   bash deploy/start.sh --dry-run
#
# 退出码：0 成功；1 参数错；30 依赖未就绪；31 Redis/依赖启动失败（含"要求严格"时）；
#         32 vLLM 未在超时内就绪；33 API 启动失败；34 API 健康检查超时。
#
# 幂等：已在跑且没给 --restart 时**不重复拉起**，直接提示并返回 0。
# 日志：脚本自身写 logs/start-YYYYmmdd.log；API 的 stdout/stderr 写 logs/api.log。
#
# 为什么健康检查用 /livez 而不是 /health：
#   实测 /health 每次都现采依赖探针，依赖不可达时单次 2.2s 且占住工作线程
#   （8 线程 ⇒ 吞吐被钉在 ~3.6 QPS）；/livez 742.8 QPS / P50 38ms。
#   见 docs/LOAD-TEST.md §3。
# ============================================================================
set -euo pipefail

LR_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=deploy/lib.sh
source "$LR_LIB_DIR/lib.sh"
lr_init "start"

OFFLINE=0
WITH_VLLM=0
NO_REDIS=0
RESTART=0
WAIT_SECONDS=90
VLLM_MODEL=""
VLLM_PORT=""
VLLM_GPU_UTIL="0.80"
API_HOST=""
API_PORT=""
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
运行脚本（Linux）：起 Redis →（可选）vLLM → API，并等健康检查。

用法：bash deploy/start.sh [选项] [-- 额外传给 run_api.py 的参数]

选项：
  --offline             离线模式（内存向量库 + 零依赖向量化 + mock 大模型；无卡无密钥可跑通）
  --with-vllm           同时拉起 vLLM（需要 GPU；参数见下面三项）
  --vllm-model PATH     vLLM 模型目录（必填当用 --with-vllm；可用 .env 的 VLLM_MODEL）
  --vllm-port PORT      vLLM 端口（默认 30000，与 .env 的 OPENAI_COMPAT_BASE_URL 对齐）
  --vllm-gpu-util F     显存占比（默认 0.80；同一张卡还要跑嵌入/重排时别调太高）
  --api-host HOST       覆盖 API_HOST（默认取 .env；对外服务常要 0.0.0.0）
  --api-port PORT       覆盖 API_PORT（默认取 .env，再默认 8000）
  --no-redis            不起 Redis（连不上会自动降级为内存记忆）
  --restart             已在跑时先停再起
  --wait SECONDS        健康检查最长等待（默认 90）
  --dry-run             只打印将执行的操作
  -h, --help            显示本帮助

退出码：0 成功；1 参数错；30 依赖未就绪；32 vLLM 未就绪；33 API 启动失败；34 健康检查超时。
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --offline) OFFLINE=1 ;;
    --with-vllm) WITH_VLLM=1 ;;
    --vllm-model) shift; VLLM_MODEL="${1:?--vllm-model 需要值}" ;;
    --vllm-port) shift; VLLM_PORT="${1:?--vllm-port 需要值}" ;;
    --vllm-gpu-util) shift; VLLM_GPU_UTIL="${1:?--vllm-gpu-util 需要值}" ;;
    --api-host) shift; API_HOST="${1:?--api-host 需要值}" ;;
    --api-port) shift; API_PORT="${1:?--api-port 需要值}" ;;
    --no-redis) NO_REDIS=1 ;;
    --restart) RESTART=1 ;;
    --wait) shift; WAIT_SECONDS="${1:?--wait 需要值}" ;;
    --dry-run) LR_DRY_RUN=1 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) usage; lr_error "未知参数：$1"; exit 1 ;;
  esac
  shift
done
export LR_DRY_RUN

cd "$LR_ROOT"
RUN_DIR="$LR_ROOT/run"
LOG_DIR="$LR_ROOT/logs"
mkdir -p "$RUN_DIR" "$LOG_DIR"
API_PID_FILE="$RUN_DIR/api.pid"
VLLM_PID_FILE="$RUN_DIR/vllm.pid"
API_LOG="$LOG_DIR/api.log"
VLLM_LOG="$LOG_DIR/vllm.log"
VENV_PY="$LR_ROOT/.venv/bin/python"

lr_section "启动服务（项目根：$LR_ROOT）"

# ---------------------------------------------------------------------------
# [1/5] 读配置
#   注意：`scripts/run_api.py` 自己也会读 .env（它有 load_dotenv），
#   但这里**显式 source 一遍**更稳：本脚本自己要用的 API_HOST/API_PORT 也得在这儿拿到，
#   而且"环境变量已存在则优先"的语义在两边一致。
#   ⚠️ `set -a` 让 source 进来的每个变量都自动 export。
# ---------------------------------------------------------------------------
lr_section "[1/5] 读取配置"
if [[ -f "$LR_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$LR_ROOT/.env"
  set +a
  lr_ok "已加载 $LR_ROOT/.env"
else
  lr_warn "没有 .env —— 将用代码内默认值（先跑 bash deploy/deploy.sh 生成一个更省心）"
fi
: "${API_HOST:=127.0.0.1}"
: "${API_PORT:=8000}"
: "${VLLM_PORT:=30000}"
: "${VLLM_MODEL:=}"
export API_HOST API_PORT

if [[ ! -x "$VENV_PY" ]]; then
  # ⚠️ 2026-09-29 真机自检抓到的缺陷：`--dry-run` 在新机器上（还没跑 install.sh）
  #    必然缺 .venv，若在这里 exit 30，干跑就永远不可能成功 —— 而干跑正是
  #    "起服务前想知道会发生什么"的手段。⇒ 干跑降级为 WARN 并继续，
  #    **正式跑依旧是硬失败**（退出码 30，并给出补救命令）。
  if [[ "$LR_DRY_RUN" == "1" ]]; then
    lr_warn "[dry-run] 虚拟环境尚不存在：$VENV_PY"
    lr_info "[dry-run] 正式跑前需要先执行：bash deploy/install.sh"
    lr_info "[dry-run] 后续步骤按「虚拟环境已就绪」继续打印"
  else
    lr_error "没有虚拟环境：$VENV_PY —— 先跑 bash deploy/install.sh"
    exit 30
  fi
fi
if [[ -x "$VENV_PY" ]]; then
  lr_info "Python：$("$VENV_PY" -V 2>&1)"
else
  lr_info "[dry-run] 虚拟环境未创建，跳过 python 版本回读"
fi
lr_info "API：http://$API_HOST:$API_PORT"
[[ "$OFFLINE" == "1" ]] && lr_info "模式：**离线**（内存库 + 零依赖嵌入 + mock 大模型）"

# 幂等：已经在跑就不再拉一个（两个进程抢同一份 Milvus Lite 文件锁会静默降级！）
EXISTING_PID="$(lr_read_pid "$API_PID_FILE")"
if lr_pid_alive "$EXISTING_PID"; then
  if [[ "$RESTART" == "1" ]]; then
    lr_warn "检测到已在运行（pid=$EXISTING_PID），--restart：先停再起"
    lr_run bash "$LR_LIB_DIR/stop.sh" --keep-vllm --keep-redis || true
  else
    lr_ok "服务已在运行（pid=$EXISTING_PID）。要重启请加 --restart"
    lr_info "  健康检查：curl -s http://127.0.0.1:$API_PORT/livez"
    lr_info "  看日志：  tail -f $API_LOG"
    exit 0
  fi
else
  # pid 文件存在但进程没了：这是"上次异常退出"，明确说清而不是静默忽略
  if [[ -n "$EXISTING_PID" ]]; then
    lr_warn "pid 文件里有 $EXISTING_PID，但该进程已不存在（上次异常退出？）—— 继续启动"
  fi
fi

# ---------------------------------------------------------------------------
# [2/5] Redis
# ---------------------------------------------------------------------------
lr_section "[2/5] Redis（短期记忆）"
if [[ "$NO_REDIS" == "1" ]]; then
  lr_info "按参数跳过（--no-redis）：短期记忆会降级为内存实现，**重启后窗口会丢**"
elif ! lr_have redis-server; then
  lr_warn "没装 redis-server：降级为内存记忆（会话窗口不跨重启）。"
  lr_warn "  需要的话：bash deploy/install.sh --with-redis"
else
  if redis-cli ping >/dev/null 2>&1; then
    lr_ok "Redis 已在运行（$(redis-cli ping 2>/dev/null)）"
  else
    lr_info "启动 Redis（daemonize）"
    lr_run redis-server --daemonize yes || lr_warn "Redis 启动失败，代码会自动降级"
    sleep 1
    lr_info "Redis 状态：$(redis-cli ping 2>/dev/null || echo '未就绪（将降级）')"
  fi
fi

# ---------------------------------------------------------------------------
# [3/5] vLLM（可选）
#
# ⚠️ **vLLM 启动四要素**（缺一个就起不来，见 handoff/CLOUD-ENV-FACTS.md §7.1）：
#   ① LD_LIBRARY_PATH 要含 `nvidia/cu13/lib`（vllm 的 _C 按 CUDA13 编译，
#      而 torch 自带的 cudart 是 12 ⇒ 不加就是 libcudart.so.13 not found）；
#   ② PATH 里要有 `ninja`（FlashInfer/inductor 的 JIT 会 FileNotFoundError）；
#   ③ 必须用 `vllm serve` **子命令**（0.29 的入口变了，旧的 api_server 已废弃）；
#   ④ 4090 没有原生 FP8 内核 ⇒ FP8 权重走 **Marlin（weight-only）**，属预期行为。
# ---------------------------------------------------------------------------
lr_section "[3/5] vLLM（可选）"
if [[ "$WITH_VLLM" != "1" ]]; then
  lr_info "跳过（未指定 --with-vllm）。服务会用 .env 里配好的 OPENAI_COMPAT_BASE_URL 找模型。"
else
  if ! lr_have nvidia-smi; then
    lr_error "--with-vllm 需要 GPU，但本机没有 nvidia-smi"
    exit 30
  fi
  [[ -n "$VLLM_MODEL" ]] || lr_die "--with-vllm 需要 --vllm-model PATH（或用 .env 的 VLLM_MODEL）"
  if [[ ! -d "$VLLM_MODEL" ]]; then
    lr_error "模型目录不存在：$VLLM_MODEL"
    exit 30
  fi
  if lr_port_listening "$VLLM_PORT"; then
    lr_ok "端口 $VLLM_PORT 已有服务在监听 —— 复用（不重复起）"
  else
    VLLM_BIN="$LR_ROOT/.venv/bin/vllm"
    [[ -x "$VLLM_BIN" ]] || VLLM_BIN="$(command -v vllm || true)"
    [[ -n "$VLLM_BIN" ]] || lr_die "找不到 vllm 可执行文件（先 bash deploy/install.sh --full）"
    # 四要素之一：把 CUDA13 运行库塞进 LD_LIBRARY_PATH
    CU13="$LR_ROOT/.venv/lib/python3.10/site-packages/nvidia/cu13/lib"
    export LD_LIBRARY_PATH="$CU13:${LD_LIBRARY_PATH:-}"
    [[ -d "$CU13" ]] || lr_warn "找不到 $CU13 —— 若 import vllm 报 libcudart.so.13，就是这里的问题"
    # 四要素之二：PATH 里要有 ninja
    lr_have ninja || lr_warn "PATH 里没有 ninja（JIT 可能失败）：pip install ninja 或装进 .venv/bin"
    lr_info "启动 vLLM：model=$VLLM_MODEL port=$VLLM_PORT gpu-util=$VLLM_GPU_UTIL"
    lr_info "  （FP8 权重在 4090 上走 Marlin weight-only，属预期）"
    # served-model-name 必须与 `.env` 的 LLM_MODEL 一致（否则请求 404 model_missing）。
    # 优先级：环境变量 VLLM_SERVED_NAME > .env 的 LLM_MODEL > 目录名。
    SERVED="${VLLM_SERVED_NAME:-${LLM_MODEL:-$(basename "$VLLM_MODEL")}}"
    lr_info "served-model-name=$SERVED（要与 .env 的 LLM_MODEL 一致）"
    if [[ "$LR_DRY_RUN" == "1" ]]; then
      lr_info "[dry-run] 将后台启动 vLLM"
    else
      # setsid + nohup：脱离当前会话，脚本退出后不被带走
      setsid nohup "$VLLM_BIN" serve \
        --model "$VLLM_MODEL" \
        --served-model-name "$SERVED" \
        --host 127.0.0.1 --port "$VLLM_PORT" \
        --dtype auto --gpu-memory-utilization "$VLLM_GPU_UTIL" \
        --trust-remote-code >>"$VLLM_LOG" 2>&1 </dev/null &
      echo $! >"$VLLM_PID_FILE"
      lr_info "vLLM 已启动（pid=$(cat "$VLLM_PID_FILE")，日志 $VLLM_LOG），等它就绪…"
      if ! lr_wait_http "http://127.0.0.1:$VLLM_PORT/health" 300 2; then
        lr_error "vLLM 未在超时内就绪。看日志：tail -n 50 $VLLM_LOG"
        lr_error "  常见：显存不够（--vllm-gpu-util 调小）、LD_LIBRARY_PATH 缺 cu13、缺 ninja"
        exit 32
      fi
    fi
  fi
fi

# ---------------------------------------------------------------------------
# [4/5] 启动 API
# ---------------------------------------------------------------------------
lr_section "[4/5] 启动 API"
API_ARGS=(scripts/run_api.py --host "$API_HOST" --port "$API_PORT")
[[ "$OFFLINE" == "1" ]] && API_ARGS+=(--offline)
if (( ${#EXTRA_ARGS[@]} )); then API_ARGS+=("${EXTRA_ARGS[@]}"); fi

lr_info "命令：$VENV_PY ${API_ARGS[*]}"
if [[ "$LR_DRY_RUN" == "1" ]]; then
  lr_info "[dry-run] 将后台启动 API"
else
  # < /dev/null 不能省：历史上 `systemctl start mysql` 之类会抢 stdin 把流程吃掉
  # （VM-DEPLOY.md §7 发现 1）。
  setsid nohup "$VENV_PY" "${API_ARGS[@]}" >>"$API_LOG" 2>&1 </dev/null &
  echo $! >"$API_PID_FILE"
  sleep 3
  API_PID="$(cat "$API_PID_FILE")"
  if ! lr_pid_alive "$API_PID"; then
    lr_error "API 进程起来就退了（pid=$API_PID）。日志尾部："
    tail -n 30 "$API_LOG" >&2 || true
    exit 33
  fi
  lr_ok "API 已启动：pid=$API_PID"
fi

# ---------------------------------------------------------------------------
# [5/5] 健康检查
#   ⚠️ 用 /livez：它**不探测依赖**（O(1)）。/health 是深度体检，依赖挂掉时很贵。
#      但要注意 /livez 只证明"进程活着"；**就绪**还要看依赖 —— 所以这里额外
#      打一次 /health 的摘要进日志（只看，不拿它做门禁）。
# ---------------------------------------------------------------------------
lr_section "[5/5] 健康检查"
if [[ "$LR_DRY_RUN" == "1" ]]; then
  lr_info "[dry-run] 跳过健康检查"
else
  if ! lr_wait_http "http://127.0.0.1:$API_PORT/livez" "$WAIT_SECONDS" 2; then
    lr_error "API 健康检查超时。日志尾部："
    tail -n 30 "$API_LOG" >&2 || true
    exit 34
  fi
  # 就绪信息（不阻塞）：/health 里能看到 vector_store / llm provider / 降级原因
  HEALTH="$(curl -s -m 10 "http://127.0.0.1:$API_PORT/health" 2>/dev/null || true)"
  if [[ -n "$HEALTH" ]]; then
    # `|| true` 同理：head 读满就退出，printf 可能吃到 SIGPIPE；这只是打印摘要，不该影响启动
    lr_info "健康摘要（/health）：$(printf '%s' "$HEALTH" | head -c 400 || true)"
  fi
fi

lr_snapshot_runtime "启动后"
if [[ "$LR_DRY_RUN" == "1" ]]; then
  # 干跑**没有**启动任何东西，标题不能说"已就绪"（那是假读数）
  lr_section "干跑结束（未启动任何服务）"
else
  lr_section "服务已就绪"
fi
lr_info "  API：      http://$API_HOST:$API_PORT"
lr_info "  存活探针： curl -s http://127.0.0.1:$API_PORT/livez"
lr_info "  网页控制台：http://$API_HOST:$API_PORT/ui"
lr_info "  API 日志： $API_LOG（tail -f 跟踪）"
lr_info "  脚本日志： $LR_LOG_FILE"
lr_info "  停止服务： bash deploy/stop.sh"
