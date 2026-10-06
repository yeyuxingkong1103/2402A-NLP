#!/usr/bin/env bash
# =============================================================================
# 工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 文件：部署/脚本/run_vllm.sh —— 算力云 GPU 上启动 vLLM 的 OpenAI 兼容服务
#
# 为什么需要它：产品侧的 LLM 后端是可插拔的 ollama → **openai 兼容 vLLM/SGLang** → extractive，
#   云端把 vLLM 拉起来后，只要设置 RAG_LLM__OPENAI_BASE_URL，问答链路即自动改走该后端
#   （研发/app/core/llm_client.py 用标准库 urllib 调 /v1/chat/completions，**不需要 openai 包**）。
#
# ⚠️ 诚实声明（不得改写）：本机**无 GPU 且断网无法安装 vllm**，
#   本脚本**未在本机执行过**（沙箱也禁止创建命名管道，bash 起不来）。请在算力云上按下方命令自验：
#       bash -n 部署/脚本/run_vllm.sh          # 语法检查
#       bash 部署/脚本/run_vllm.sh --check     # 环境预检（不启动）
#
# 用法（仓库根目录 = 工单3）：
#   bash 部署/脚本/run_vllm.sh --check
#   bash 部署/脚本/run_vllm.sh                                  # 前台启动（默认 Qwen2.5-7B-Instruct / 8000）
#   MODEL=Qwen/Qwen2.5-14B-Instruct PORT=8000 bash 部署/脚本/run_vllm.sh --background
#   bash 部署/脚本/run_vllm.sh --stop
#
# 退出码：0=成功；1=运行期失败；2=预检失败（未装 vllm / 无 GPU）；3=端口占用；4=健康检查失败。
# =============================================================================
set -uo pipefail

WORK_ORDER='人工智能NLP-RAG-PDF文档的表格解析及检索优化'
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/部署/日志"
DEPLOY_LOG="${LOG_DIR}/deploy.log"
mkdir -p "${LOG_DIR}"
export PYTHONIOENCODING='utf-8' PYTHONUTF8='1'

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"     # 云端模型名（未在本机验证过可用性）
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TP="${TP:-1}"                                   # tensor parallel = GPU 张数
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DTYPE="${DTYPE:-auto}"
API_KEY="${VLLM_API_KEY:-}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"         # 首次加载权重可能数分钟
PID_FILE="${LOG_DIR}/vllm_${PORT}.pid.json"
OUT_LOG="${LOG_DIR}/vllm_${PORT}.out.log"
ERR_LOG="${LOG_DIR}/vllm_${PORT}.err.log"

CHECK_ONLY=0
BACKGROUND=0
STOP=0
while [ $# -gt 0 ]; do
    case "$1" in
        --check) CHECK_ONLY=1; shift ;;
        --background|-b) BACKGROUND=1; shift ;;
        --stop) STOP=1; shift ;;
        -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "❌ 未知参数：$1"; exit 2 ;;
    esac
done

log_event() { # $1=event $2=level $3=func $4=inputs $5=outputs $6=elapsed_ms $7=error
    local event="$1" level="${2:-INFO}" func="${3:-}" inputs="${4:-}" outputs="${5:-}" elapsed="${6:-}" err="${7:-}"
    local ts
    ts="$(date -Iseconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S%z)"
    printf '{"ts": "%s", "level": "%s", "event": "%s", "func": "%s", "work_order": "%s", "module": "部署/脚本/run_vllm.sh", "pid": %s, "inputs": "%s", "outputs": "%s", "elapsed_ms": %s, "error": "%s"}\n' \
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

# ---- 停止 ----
if [ "${STOP}" -eq 1 ]; then
    if [ ! -f "${PID_FILE}" ]; then echo "❌ 未找到 PID 文件：${PID_FILE}"; exit 1; fi
    pid="$("${PY}" -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))['pid'])" "${PID_FILE}")"
    kill -TERM "${pid}" 2>/dev/null; sleep 2; kill -0 "${pid}" 2>/dev/null && kill -KILL "${pid}" 2>/dev/null
    rm -f "${PID_FILE}"
    log_event 'func.exit' 'INFO' 'stop_vllm' "pid=${pid} port=${PORT}" 'stopped=1' '0'
    echo "✅ 已停止 vLLM（PID ${pid}）"; exit 0
fi

# ---- 预检：真实 import vllm + CUDA 可用性（不用 find_spec 冒充可用）----
log_event 'func.enter' 'INFO' 'check_vllm' "port=${PORT} model=${MODEL} tp=${TP}"
CHECK_OUT="$("${PY}" - <<'PY'
import json
info = {"vllm": None, "torch": None, "cuda": None, "device_count": None, "gpu_names": [], "error": None}
try:
    import vllm  # noqa: F401
    info["vllm"] = getattr(vllm, "__version__", "unknown")
except Exception as exc:  # noqa: BLE001 —— 显式报告，不静默
    info["error"] = f"vllm 不可用：{type(exc).__name__}: {exc}"
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
echo "   vLLM 预检：${CHECK_OUT}"
CUDA_OK="$("${PY}" -c "import json,sys;d=json.loads(sys.argv[1]);print('1' if d.get('cuda') else '0')" "${CHECK_OUT}" 2>/dev/null || echo 0)"
VLLM_OK="$("${PY}" -c "import json,sys;d=json.loads(sys.argv[1]);print('1' if d.get('vllm') else '0')" "${CHECK_OUT}" 2>/dev/null || echo 0)"
log_event 'func.exit' 'INFO' 'check_vllm' "port=${PORT}" "cuda=${CUDA_OK} vllm=${VLLM_OK}" '0'
if [ "${VLLM_OK}" != '1' ]; then
    log_event 'func.error' 'ERROR' 'check_vllm' "port=${PORT}" '' '' 'vllm 不可用'
    echo "❌ vllm 不可用（本机断网装不上属于预期）。云端安装："
    echo "     pip install -r 部署/配置/requirements-cloud.txt"
    echo "     pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124"
    exit 2
fi
if [ "${CUDA_OK}" != '1' ]; then
    log_event 'run_vllm.no_gpu' 'WARNING' 'check_vllm' "port=${PORT}" '' '' '未检测到可用 CUDA（本沙箱无法确认 GPU 型号，禁止编造）'
    echo "⚠️ 未检测到可用 CUDA —— GPU 版 vLLM 无法工作；请确认算力云实例已挂载 4090 且驱动正常"
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

VLLM_ARGS=(--model "${MODEL}" --host "${HOST}" --port "${PORT}" --tensor-parallel-size "${TP}"
           --max-model-len "${MAX_MODEL_LEN}" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" --dtype "${DTYPE}")
[ -n "${API_KEY}" ] && VLLM_ARGS+=(--api-key "${API_KEY}")

echo "🚀 启动 vLLM：${PY} -m vllm.entrypoints.openai.api_server ${VLLM_ARGS[*]}"
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
    log_event 'run_vllm.start_foreground' 'INFO' 'run_vllm' "model=${MODEL} port=${PORT} tp=${TP}"
    "${PY}" -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}"
    exit $?
fi

if command -v setsid >/dev/null 2>&1; then
    setsid nohup "${PY}" -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" >"${OUT_LOG}" 2>"${ERR_LOG}" < /dev/null &
else
    nohup "${PY}" -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" >"${OUT_LOG}" 2>"${ERR_LOG}" < /dev/null &
fi
child_pid=$!
"${PY}" - "${PID_FILE}" "${child_pid}" "${MODEL}" "${PORT}" "${OUT_LOG}" "${ERR_LOG}" <<'PY'
import json, sys, time
pid_file, pid, model, port, out_log, err_log = sys.argv[1:7]
open(pid_file, "w", encoding="utf-8").write(json.dumps(
    {"pid": int(pid), "model": model, "port": int(port), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
     "out_log": out_log, "err_log": err_log}, ensure_ascii=False, indent=2))
PY
log_event 'run_vllm.start_background' 'INFO' 'run_vllm' "model=${MODEL} port=${PORT}" "pid=${child_pid}" '0'

echo "⏳ 等待 /v1/models 就绪（最多 ${HEALTH_TIMEOUT} s；首次加载权重较慢）…"
if "${PY}" - "http://127.0.0.1:${PORT}/v1/models" "${HEALTH_TIMEOUT}" <<'PY'
import sys, time, urllib.request
url, timeout = sys.argv[1], float(sys.argv[2])
start = time.time()
while time.time() - start < timeout:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            print(resp.read().decode("utf-8", errors="replace")[:300])
            sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        last = f"{type(exc).__name__}: {exc}"
    time.sleep(2)
print("FAIL " + last)
sys.exit(1)
PY
then
    log_event 'func.exit' 'INFO' 'wait_vllm' "url=/v1/models" "pid=${child_pid}" '0'
    echo "✅ vLLM 就绪（PID ${child_pid}）"
    cat <<EOF

--- 运维信息 ---
   停止：bash 部署/脚本/run_vllm.sh --stop
   输出日志：${OUT_LOG}
   错误日志：${ERR_LOG}
   给产品的环境变量：
     RAG_LLM__BACKEND=openai
     RAG_LLM__OPENAI_BASE_URL=http://127.0.0.1:${PORT}/v1
     RAG_LLM__OPENAI_MODEL=${MODEL}
   （Embedding 仍可用本机 Ollama bge-m3，或另起一个嵌入服务）
EOF
    exit 0
else
    log_event 'func.error' 'ERROR' 'wait_vllm' "url=/v1/models" '' '' '健康检查未通过'
    echo "❌ vLLM 未就绪；输出尾部："
    tail -n 30 "${OUT_LOG}" 2>/dev/null
    tail -n 30 "${ERR_LOG}" 2>/dev/null
    exit 4
fi
