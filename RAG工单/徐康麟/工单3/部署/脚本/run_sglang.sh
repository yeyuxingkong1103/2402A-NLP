#!/usr/bin/env bash
# =============================================================================
# 工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 文件：部署/脚本/run_sglang.sh —— 算力云 GPU 上启动 SGLang 的 OpenAI 兼容服务
#
# 与 run_vllm.sh 同构：二选一即可，产品侧只认 RAG_LLM__OPENAI_BASE_URL（标准库 urllib 调用
# /v1/chat/completions，**不需要 openai 包**），因此 vLLM / SGLang 可互换。
#
# ⚠️ 诚实声明（不得改写）：本机**无 GPU 且断网无法安装 sglang**，
#   本脚本**未在本机执行过**（沙箱禁止创建命名管道，bash 无法运行）。请在算力云上自验：
#       bash -n 部署/脚本/run_sglang.sh
#       bash 部署/脚本/run_sglang.sh --check
#
# 用法（仓库根目录 = 工单3）：
#   bash 部署/脚本/run_sglang.sh --check
#   bash 部署/脚本/run_sglang.sh --background
#   MODEL=Qwen/Qwen2.5-7B-Instruct PORT=30000 bash 部署/脚本/run_sglang.sh
#   bash 部署/脚本/run_sglang.sh --stop
#
# 退出码：0=成功；1=运行期失败；2=预检失败；3=端口占用；4=健康检查失败。
# =============================================================================
set -uo pipefail

WORK_ORDER='人工智能NLP-RAG-PDF文档的表格解析及检索优化'
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/部署/日志"
DEPLOY_LOG="${LOG_DIR}/deploy.log"
mkdir -p "${LOG_DIR}"
export PYTHONIOENCODING='utf-8' PYTHONUTF8='1'

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-30000}"                 # SGLang 默认端口
HOST="${HOST:-0.0.0.0}"
TP="${TP:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"
PID_FILE="${LOG_DIR}/sglang_${PORT}.pid.json"
OUT_LOG="${LOG_DIR}/sglang_${PORT}.out.log"
ERR_LOG="${LOG_DIR}/sglang_${PORT}.err.log"

CHECK_ONLY=0
BACKGROUND=0
STOP=0
while [ $# -gt 0 ]; do
    case "$1" in
        --check) CHECK_ONLY=1; shift ;;
        --background|-b) BACKGROUND=1; shift ;;
        --stop) STOP=1; shift ;;
        -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "❌ 未知参数：$1"; exit 2 ;;
    esac
done

log_event() { # $1=event $2=level $3=func $4=inputs $5=outputs $6=elapsed_ms $7=error
    local event="$1" level="${2:-INFO}" func="${3:-}" inputs="${4:-}" outputs="${5:-}" elapsed="${6:-}" err="${7:-}"
    local ts
    ts="$(date -Iseconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S%z)"
    printf '{"ts": "%s", "level": "%s", "event": "%s", "func": "%s", "work_order": "%s", "module": "部署/脚本/run_sglang.sh", "pid": %s, "inputs": "%s", "outputs": "%s", "elapsed_ms": %s, "error": "%s"}\n' \
        "${ts}" "${level}" "${event}" "${func}" "${WORK_ORDER}" "$$" \
        "${inputs//\"/\\\"}" "${outputs//\"/\\\"}" \
        "$(if [ -n "${elapsed}" ]; then printf '%s' "${elapsed}"; else printf 'null'; fi)" \
        "${err//\"/\\\"}" >> "${DEPLOY_LOG}" 2>/dev/null || echo "[log 降级] 无法写 ${DEPLOY_LOG}" >&2
    if [ "${level}" = 'ERROR' ] || [ "${level}" = 'CRITICAL' ] || [ "${level}" = 'WARNING' ]; then
        echo "[${level}] ${event} :: ${func} :: ${err}"
    fi
}

PY="${RAG_SCHEDULER_PYTHON:-python3}"
command -v "${PY}" >/dev/null 2>&1 || { log_event 'func.error' 'ERROR' 'resolve_python' "PY=${PY}" '' '' '找不到解释器'; echo "❌ 找不到 ${PY}"; exit 127; }

if [ "${STOP}" -eq 1 ]; then
    [ -f "${PID_FILE}" ] || { echo "❌ 未找到 PID 文件：${PID_FILE}"; exit 1; }
    pid="$("${PY}" -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))['pid'])" "${PID_FILE}")"
    kill -TERM "${pid}" 2>/dev/null; sleep 2; kill -0 "${pid}" 2>/dev/null && kill -KILL "${pid}" 2>/dev/null
    rm -f "${PID_FILE}"
    log_event 'func.exit' 'INFO' 'stop_sglang' "pid=${pid}" 'stopped=1' '0'
    echo "✅ 已停止 SGLang（PID ${pid}）"; exit 0
fi

# ---- 预检：真实 import sglang + CUDA ----
log_event 'func.enter' 'INFO' 'check_sglang' "port=${PORT} model=${MODEL} tp=${TP}"
CHECK_OUT="$("${PY}" - <<'PY'
import json
info = {"sglang": None, "torch": None, "cuda": None, "device_count": None, "gpu_names": [], "error": None}
try:
    import sglang  # noqa: F401
    info["sglang"] = getattr(sglang, "__version__", "unknown")
except Exception as exc:  # noqa: BLE001
    info["error"] = f"sglang 不可用：{type(exc).__name__}: {exc}"
try:
    import torch
    info["torch"] = torch.__version__
    info["cuda"] = bool(torch.cuda.is_available())
    if info["cuda"]:
        info["device_count"] = int(torch.cuda.device_count())
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(info["device_count"])]
except Exception as exc:  # noqa: BLE001
    info["torch"] = info["torch"] or f"不可用：{type(exc).__name__}: {exc}"
print(json.dumps(info, ensure_ascii=False))
PY
)"
echo "   SGLang 预检：${CHECK_OUT}"
CUDA_OK="$("${PY}" -c "import json,sys;d=json.loads(sys.argv[1]);print('1' if d.get('cuda') else '0')" "${CHECK_OUT}" 2>/dev/null || echo 0)"
SGLANG_OK="$("${PY}" -c "import json,sys;d=json.loads(sys.argv[1]);print('1' if d.get('sglang') else '0')" "${CHECK_OUT}" 2>/dev/null || echo 0)"
log_event 'func.exit' 'INFO' 'check_sglang' "port=${PORT}" "cuda=${CUDA_OK} sglang=${SGLANG_OK}" '0'
if [ "${SGLANG_OK}" != '1' ]; then
    log_event 'func.error' 'ERROR' 'check_sglang' "port=${PORT}" '' '' 'sglang 不可用'
    echo "❌ sglang 不可用（本机断网装不上属于预期）。云端安装："
    echo "     pip install -r 部署/配置/requirements-cloud.txt"
    exit 2
fi
if [ "${CUDA_OK}" != '1' ]; then
    log_event 'run_sglang.no_gpu' 'WARNING' 'check_sglang' "port=${PORT}" '' '' '未检测到可用 CUDA'
    echo "⚠️ 未检测到可用 CUDA —— SGLang GPU 后端无法工作（本沙箱无法确认 GPU 型号，禁止编造）"
    exit 2
fi

# ---- 端口检查 ----
if ! "${PY}" - "${PORT}" <<'PY'
import socket, sys
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
except OSError as exc:
    print(f"BUSY {exc}")
    sys.exit(1)
finally:
    sock.close()
print("FREE")
PY
then
    log_event 'func.error' 'ERROR' 'port_free' "port=${PORT}" '' '' '端口被占用'
    echo "❌ 端口 ${PORT} 已被占用"; exit 3
fi

SGLANG_ARGS=(--model-path "${MODEL}" --host "${HOST}" --port "${PORT}" --tp "${TP}"
             --mem-fraction-static "${MEM_FRACTION_STATIC}" --context-length "${CONTEXT_LENGTH}")
echo "🚀 启动 SGLang：${PY} -m sglang.launch_server ${SGLANG_ARGS[*]}"
echo "   模型：${MODEL}　端口：${PORT}　TP：${TP}　（本地未实测：本机无 GPU）"

if [ "${CHECK_ONLY}" -eq 1 ]; then
    echo "ℹ️ --check：仅预检，不启动"
    echo "   启动后给产品的环境变量："
    echo "     RAG_LLM__BACKEND=openai"
    echo "     RAG_LLM__OPENAI_BASE_URL=http://127.0.0.1:${PORT}/v1"
    echo "     RAG_LLM__OPENAI_MODEL=${MODEL}"
    exit 0
fi

if [ "${BACKGROUND}" -eq 0 ]; then
    log_event 'run_sglang.start_foreground' 'INFO' 'run_sglang' "model=${MODEL} port=${PORT} tp=${TP}"
    "${PY}" -m sglang.launch_server "${SGLANG_ARGS[@]}"
    exit $?
fi

if command -v setsid >/dev/null 2>&1; then
    setsid nohup "${PY}" -m sglang.launch_server "${SGLANG_ARGS[@]}" >"${OUT_LOG}" 2>"${ERR_LOG}" < /dev/null &
else
    nohup "${PY}" -m sglang.launch_server "${SGLANG_ARGS[@]}" >"${OUT_LOG}" 2>"${ERR_LOG}" < /dev/null &
fi
child_pid=$!
"${PY}" - "${PID_FILE}" "${child_pid}" "${MODEL}" "${PORT}" "${OUT_LOG}" "${ERR_LOG}" <<'PY'
import json, sys, time
pid_file, pid, model, port, out_log, err_log = sys.argv[1:7]
open(pid_file, "w", encoding="utf-8").write(json.dumps(
    {"pid": int(pid), "model": model, "port": int(port), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
     "out_log": out_log, "err_log": err_log}, ensure_ascii=False, indent=2))
PY
log_event 'run_sglang.start_background' 'INFO' 'run_sglang' "model=${MODEL} port=${PORT}" "pid=${child_pid}" '0'

echo "⏳ 等待 SGLang 就绪（/health → /v1/models，最多 ${HEALTH_TIMEOUT} s）…"
if "${PY}" - "http://127.0.0.1:${PORT}" "${HEALTH_TIMEOUT}" <<'PY'
import sys, time, urllib.request
base, timeout = sys.argv[1], float(sys.argv[2])
start, last = time.time(), ""
while time.time() - start < timeout:
    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(base + path, timeout=5) as resp:
                if resp.status == 200:
                    print(f"{path} -> 200")
                    sys.exit(0)
        except Exception as exc:  # noqa: BLE001
            last = f"{path}: {type(exc).__name__}: {exc}"
    time.sleep(2)
print("FAIL " + last)
sys.exit(1)
PY
then
    log_event 'func.exit' 'INFO' 'wait_sglang' "port=${PORT}" "pid=${child_pid}" '0'
    echo "✅ SGLang 就绪（PID ${child_pid}）"
    cat <<EOF

--- 运维信息 ---
   停止：bash 部署/脚本/run_sglang.sh --stop
   输出日志：${OUT_LOG}
   错误日志：${ERR_LOG}
   给产品的环境变量：
     RAG_LLM__BACKEND=openai
     RAG_LLM__OPENAI_BASE_URL=http://127.0.0.1:${PORT}/v1
     RAG_LLM__OPENAI_MODEL=${MODEL}
EOF
    exit 0
else
    log_event 'func.error' 'ERROR' 'wait_sglang' "port=${PORT}" '' '' '健康检查未通过'
    echo "❌ SGLang 未就绪；输出尾部："
    tail -n 30 "${OUT_LOG}" 2>/dev/null
    tail -n 30 "${ERR_LOG}" 2>/dev/null
    exit 4
fi
