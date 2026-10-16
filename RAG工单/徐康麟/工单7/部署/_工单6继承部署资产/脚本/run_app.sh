#!/usr/bin/env bash
# =============================================================================
# 工单：人工智能NLP-RAG-图像内容解析及检索优化
# 文件：部署/脚本/run_app.sh —— 一键启动 / 自检入口（Linux / macOS / WSL 用；Windows 主用 run_app.ps1）
#
# 与 部署/脚本/run_app.ps1 行为对齐，差别只在「解释器怎么找」：
#   · Windows 走仓库统一入口 run_py.ps1（固定工单1 venv）；
#   · 本脚本直接用 python3（或环境变量 RAG_SCHEDULER_PYTHON 指定的解释器），
#     并显式设置 PYTHONPATH=<仓库根>/研发、PYTHONUTF8=1。
#
# 用法：
#   bash 部署/脚本/run_app.sh --mode api --port 8600              # 前台起 FastAPI
#   bash 部署/脚本/run_app.sh --mode web --port 8123              # 前台起纯标准库中文 Web
#   bash 部署/脚本/run_app.sh --mode cli --question "……"           # CLI 单次问答
#   bash 部署/脚本/run_app.sh --health-check-only                 # 【验收自检】退出码 0 = 通过
#   bash 部署/脚本/run_app.sh --health-check-only --smoke-all     # 自检 + 全部端点冒烟
#
# 退出码：0 全部检查通过 / 1 存在失败检查 / 3 索引缺失 / 127 找不到解释器
#
# ⚠️ 本机（Windows 沙箱）实测无法执行本脚本：bash.exe 只是 WSL 转发器，沙箱下
#    `wsl -l -q` 返回 E_ACCESSDENIED —— 故本文件在交付机上是**静态审查**（未实跑），
#    实跑证据由 run_app.ps1 提供。详见 部署/交付物核验.md「未完成项」。
# =============================================================================
set -u

MODE="api"
BIND_HOST="127.0.0.1"
PORT=""
QUESTION=""
HEALTH_CHECK_ONLY=0
SMOKE_ALL=0
NO_WARMUP=0
TIMEOUT_SEC=180

while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --bind) BIND_HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --question) QUESTION="$2"; shift 2 ;;
    --health-check-only) HEALTH_CHECK_ONLY=1; shift ;;
    --smoke-all) SMOKE_ALL=1; shift ;;
    --no-warmup) NO_WARMUP=1; shift ;;
    --timeout) TIMEOUT_SEC="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "[参数错误] 未知参数：$1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT" || exit 1

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$ROOT/研发${PYTHONPATH:+:$PYTHONPATH}"

PY="${RAG_SCHEDULER_PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[失败] 找不到 Python 解释器：$PY（可用环境变量 RAG_SCHEDULER_PYTHON 指定）" >&2
  exit 127
fi

if [ -z "$PORT" ]; then
  if [ "$MODE" = "web" ]; then PORT=8123; else PORT=8600; fi
fi
BASE_URL="http://$BIND_HOST:$PORT"
LOG_DIR="$ROOT/部署/日志"
mkdir -p "$LOG_DIR"
EVIDENCE="$LOG_DIR/自检结果.json"

FAILED=0
CHECKS=""
STARTED_BY_US=0
SERVER_PID=""

add_check() { # $1=name $2=ok(0/1) $3=detail
  if [ "$2" -eq 1 ]; then mark="PASS"; else mark="FAIL"; FAILED=$((FAILED + 1)); fi
  echo "  [$mark] $1 —— $3"
  CHECKS="$CHECKS{\"name\":\"$1\",\"ok\":$([ "$2" -eq 1 ] && echo true || echo false),\"detail\":\"$3\"},"
}

http_get() { curl -s -o /tmp/gd4_body.$$ -w '%{http_code}' --max-time "${2:-10}" "$1" 2>/dev/null || echo 000; }
http_post() { curl -s -o /tmp/gd4_body.$$ -w '%{http_code}' --max-time "${3:-180}" -H 'Content-Type: application/json; charset=utf-8' -d "$2" "$1" 2>/dev/null || echo 000; }

ready() {
  code="$(http_get "$BASE_URL/health" 5)"
  [ "$code" = "200" ] || [ "$code" = "503" ]
}

stop_app() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}

# ---------------------------------------------------------------- CLI 模式
if [ "$MODE" = "cli" ]; then
  [ -n "$QUESTION" ] || QUESTION="武汉兴图新科电子股份有限公司注册资本是多少？"
  echo "[CLI] $PY 研发/app/main.py --ask \"$QUESTION\""
  "$PY" 研发/app/main.py --ask "$QUESTION"
  exit $?
fi

echo "=============================================================================="
echo "工单4 部署自检：模式=$MODE 地址=$BASE_URL 预热=$([ "$NO_WARMUP" -eq 0 ] && echo True || echo False)"
echo "=============================================================================="

if ready; then
  echo "[复用] $BASE_URL/health 已有服务在响应（本次不新起、结束时也不停它）"
else
  if [ "$MODE" = "web" ]; then
    CMD=("$PY" 研发/app/main.py --serve --host "$BIND_HOST" --port "$PORT")
  else
    CMD=("$PY" 部署/脚本/serve_api.py --host "$BIND_HOST" --port "$PORT")
  fi
  [ "$NO_WARMUP" -eq 0 ] || CMD+=("--no-warmup")
  echo "[启动] ${CMD[*]}"
  "${CMD[@]}" > "$LOG_DIR/serve_${MODE}_${PORT}.out" 2>&1 &
  SERVER_PID=$!
  STARTED_BY_US=1
  ok=0
  deadline=$(( $(date +%s) + TIMEOUT_SEC ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    sleep 1
    if ready; then ok=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
  done
  if [ "$ok" -ne 1 ]; then
    echo "[失败] 服务 ${TIMEOUT_SEC}s 内未就绪——如实标记失败（不伪造）"
    echo "[诊断] 子进程输出末 20 行："
    tail -n 20 "$LOG_DIR/serve_${MODE}_${PORT}.out" 2>/dev/null | sed 's/^/    /'
    add_check "服务就绪" 0 "${TIMEOUT_SEC}s 内 $BASE_URL/health 无响应"
    [ "$STARTED_BY_US" -eq 1 ] && stop_app
    exit 1
  fi
  echo "[就绪] 服务已响应 $BASE_URL/health"
fi

# 1) /health
CODE="$(http_get "$BASE_URL/health" 60)"
BODY="$(cat /tmp/gd4_body.$$ 2>/dev/null || echo '')"
STATUS="$(printf '%s' "$BODY" | "$PY" -c 'import json,sys;print((json.load(sys.stdin) or {}).get("status"))' 2>/dev/null || echo unknown)"
add_check "/health HTTP 200" "$([ "$CODE" = "200" ] && echo 1 || echo 0)" "HTTP $CODE"
add_check "/health status=ok" "$([ "$STATUS" = "ok" ] && echo 1 || echo 0)" "status=$STATUS"

CHUNKS="$(printf '%s' "$BODY" | "$PY" -c 'import json,sys;d=json.load(sys.stdin) or {};print(((d.get("checks") or {}).get("index") or {}).get("chunks"))' 2>/dev/null || echo None)"
IMGS="$(printf '%s' "$BODY" | "$PY" -c 'import json,sys;d=json.load(sys.stdin) or {};print(((d.get("checks") or {}).get("index") or {}).get("image_chunks"))' 2>/dev/null || echo None)"
case "$CHUNKS" in ''|None) add_check "索引已加载" 0 "chunks=$CHUNKS" ;; *) add_check "索引已加载" 1 "chunks=$CHUNKS image_chunks=$IMGS" ;; esac

# 2) 端点冒烟
if [ "$SMOKE_ALL" -eq 1 ]; then
  C="$(http_post "$BASE_URL/ask" '{"question":"武汉兴图新科电子股份有限公司注册资本是多少？","lang":"zh","use_cache":true}' 180)"
  A="$(cat /tmp/gd4_body.$$)"
  add_check "POST /ask" "$([ "$C" = "200" ] && printf '%s' "$A" | grep -q '5,520' && echo 1 || echo 0)" "HTTP $C"
  C="$(http_post "$BASE_URL/retrieve" '{"question":"变更募投项目的程序","top_k":5}' 180)"
  A="$(cat /tmp/gd4_body.$$)"
  add_check "POST /retrieve" "$([ "$C" = "200" ] && printf '%s' "$A" | grep -q '"hit_count"' && echo 1 || echo 0)" "HTTP $C"
  if [ "$MODE" = "api" ]; then
    C="$(http_post "$BASE_URL/ask/stream" '{"question":"公司来自军用领域的收入占比分别是多少？","lang":"zh","use_cache":true}' 180)"
    A="$(cat /tmp/gd4_body.$$)"
    add_check "POST /ask/stream" "$([ "$C" = "200" ] && printf '%s' "$A" | grep -q 'data:' && echo 1 || echo 0)" "HTTP $C（SSE）"
    C="$(http_post "$BASE_URL/eval" '{"limit":3,"top_k":5,"use_cache":true,"include_details":false}' 600)"
    A="$(cat /tmp/gd4_body.$$)"
    add_check "POST /eval（前 3 题）" "$([ "$C" = "200" ] && printf '%s' "$A" | grep -q '"accuracy"' && echo 1 || echo 0)" "HTTP $C"
  else
    echo "  [SKIP] POST /ask/stream —— web 模式无该端点（只有 FastAPI 服务提供）"
  fi
  C="$(http_get "$BASE_URL/questions" 60)"; add_check "GET /questions" "$([ "$C" = "200" ] && echo 1 || echo 0)" "HTTP $C"
  C="$(http_get "$BASE_URL/" 60)"; add_check "GET /" "$([ "$C" = "200" ] && echo 1 || echo 0)" "HTTP $C"
fi

# 3) 留痕
{
  printf '{\n  "generated_by": "部署/脚本/run_app.sh",\n'
  printf '  "finished_at": "%s",\n' "$(date -Iseconds)"
  printf '  "mode": "%s",\n  "base_url": "%s",\n' "$MODE" "$BASE_URL"
  printf '  "health_http": %s,\n  "health_status": "%s",\n' "${CODE:-0}" "${STATUS:-unknown}"
  printf '  "index_chunks": %s,\n  "image_chunks": %s,\n' "${CHUNKS:-null}" "${IMGS:-null}"
  printf '  "smoke_all": %s,\n  "exit_code": %s,\n' "$([ "$SMOKE_ALL" -eq 1 ] && echo true || echo false)" "$FAILED"
  printf '  "checks": [%s],\n' "${CHECKS%,}"
  printf '  "note": "本文件由部署自检实跑写出；数字来自本次运行，未预填。"\n}\n'
} > "$EVIDENCE"
echo "[OK] 已写自检留痕 $EVIDENCE"

if [ "$HEALTH_CHECK_ONLY" -eq 0 ]; then
  echo ""
  echo "服务仍在运行（未加 --health-check-only）：$BASE_URL  —— Ctrl+C 停止"
  wait "$SERVER_PID" 2>/dev/null || true
  exit "$FAILED"
fi

if [ "$STARTED_BY_US" -eq 1 ]; then
  echo "[清理] 关闭本次自检拉起的服务进程…"
  stop_app
fi

echo ""
if [ "$FAILED" -gt 0 ]; then
  echo "结论：$FAILED 项检查失败，退出码 1"
  printf '%s' "$BODY" | grep -q 'E_INDEX_MISSING' && { echo "原因：索引缺失——先跑 研发/scripts/build_index.py --incremental --include-images"; exit 3; }
  exit 1
fi
echo "结论：自检全部通过（模式=$MODE，$BASE_URL）"
exit 0
