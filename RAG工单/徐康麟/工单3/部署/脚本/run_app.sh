#!/usr/bin/env bash
# =============================================================================
# 工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 文件：部署/脚本/run_app.sh —— 统一启动入口（Linux / 算力云 / 容器内；与 run_app.ps1 行为对齐）
#
# 用途：环境与索引预检 → 端口检查 → 启动服务 → 健康检查 → 打印日志与停止方式；
#       每步写结构化 JSON Lines 到 部署/日志/deploy.log（ts/level/event/func/inputs/outputs/elapsed_ms/error），
#       禁止静默失败：任何失败都以非 0 退出码结束并打印原因。
#
# 模式：
#   fallback  纯标准库界面（http.server）—— 无需 streamlit，任何环境可跑
#   api       FastAPI 服务（uvicorn app.main:app，**必须 --app-dir 研发**）
#   streamlit 真 Streamlit 应用（需 pip install streamlit；本机不含该依赖，见 --allow-degraded）
#   check     只做预检，不启动
#
# 用法（仓库根目录 = 工单3）：
#   bash 部署/脚本/run_app.sh --mode check
#   bash 部署/脚本/run_app.sh --mode api --port 8600
#   bash 部署/脚本/run_app.sh --mode api --port 8600 --background
#   bash 部署/脚本/run_app.sh --stop --port 8600
#   bash 部署/脚本/run_app.sh --mode streamlit --env-file 部署/配置/config.example.env --allow-degraded
#
# 退出码：0=成功；1=运行期失败；2=预检失败；3=端口占用；4=模式依赖缺失；127=解释器缺失。
# =============================================================================
set -uo pipefail

WORK_ORDER='人工智能NLP-RAG-PDF文档的表格解析及检索优化'
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}" || { echo "❌ 无法进入仓库根目录：${REPO_ROOT}"; exit 1; }

LOG_DIR="${REPO_ROOT}/部署/日志"
DEPLOY_LOG="${LOG_DIR}/deploy.log"
mkdir -p "${LOG_DIR}"
export PYTHONIOENCODING='utf-8'
export PYTHONUTF8='1'
export PYTHONDONTWRITEBYTECODE='1'   # 禁止向可能只读的共享 venv 写 .pyc

MODE='fallback'
PORT=0
SERVER_HOST=''
ENV_FILE=''
PYTHON_BIN=''
HEALTH_TIMEOUT=90
BACKGROUND=0
STOP=0
ALLOW_DEGRADED=0
NO_WARMUP=0

# -----------------------------------------------------------------------------
# 结构化日志：单行 JSON（不依赖 loguru / jq；python3 只用来做 JSON 转义）
# -----------------------------------------------------------------------------
json_escape() {
    python3 - "$1" <<'PY' 2>/dev/null || printf '"%s"' "${1//\"/\\\"}"
import json, sys
print(json.dumps(sys.argv[1], ensure_ascii=False))
PY
}

log_event() { # $1=event $2=level $3=func $4=inputs $5=outputs $6=elapsed_ms $7=error
    local event="$1" level="${2:-INFO}" func="${3:-}" inputs="${4:-}" outputs="${5:-}" elapsed="${6:-}" err="${7:-}"
    local ts
    ts="$(date -Iseconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S%z)"
    {
        printf '{"ts": %s, "level": %s, "event": %s, "func": %s, "work_order": %s, "module": "部署/脚本/run_app.sh", "pid": %s, "inputs": %s, "outputs": %s, "elapsed_ms": %s, "error": %s}\n' \
            "$(json_escape "${ts}")" "$(json_escape "${level}")" "$(json_escape "${event}")" "$(json_escape "${func}")" \
            "$(json_escape "${WORK_ORDER}")" "$$" \
            "$(if [ -n "${inputs}" ]; then json_escape "${inputs}"; else printf 'null'; fi)" \
            "$(if [ -n "${outputs}" ]; then json_escape "${outputs}"; else printf 'null'; fi)" \
            "$(if [ -n "${elapsed}" ]; then printf '%s' "${elapsed}"; else printf 'null'; fi)" \
            "$(if [ -n "${err}" ]; then json_escape "${err}"; else printf 'null'; fi)"
    } >> "${DEPLOY_LOG}" 2>/dev/null || echo "[log 降级] 无法写 ${DEPLOY_LOG}" >&2
    if [ "${level}" = 'ERROR' ] || [ "${level}" = 'CRITICAL' ] || [ "${level}" = 'WARNING' ]; then
        echo "[${level}] ${event} :: ${func} :: ${err}"
    fi
}

# 函数入口/出口包装：成功写 func.exit（含耗时与输出摘要），失败写 func.error（含原因）后返回该退出码
run_step() { # $1=func $2=inputs $3...=command
    local func="$1" inputs="$2"; shift 2
    local start end code
    start="$(date +%s%3N 2>/dev/null || echo 0)"
    log_event 'func.enter' 'INFO' "${func}" "${inputs}"
    "$@"
    code=$?
    end="$(date +%s%3N 2>/dev/null || echo 0)"
    if [ "${code}" -eq 0 ]; then
        log_event 'func.exit' 'INFO' "${func}" "${inputs}" "exit_code=0" "$((end - start))"
    else
        log_event 'func.error' 'ERROR' "${func}" "${inputs}" "" "$((end - start))" "退出码 ${code}"
    fi
    return "${code}"
}

usage() {
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --mode) MODE="${2:-}"; shift 2 ;;
        --port) PORT="${2:-0}"; shift 2 ;;
        --host) SERVER_HOST="${2:-}"; shift 2 ;;
        --env-file) ENV_FILE="${2:-}"; shift 2 ;;
        --python) PYTHON_BIN="${2:-}"; shift 2 ;;
        --health-timeout) HEALTH_TIMEOUT="${2:-90}"; shift 2 ;;
        --background|-b) BACKGROUND=1; shift ;;
        --stop) STOP=1; shift ;;
        --allow-degraded) ALLOW_DEGRADED=1; shift ;;
        --no-warmup) NO_WARMUP=1; shift ;;
        -h|--help) usage ;;
        *) echo "❌ 未知参数：$1"; usage ;;
    esac
done

case "${MODE}" in fallback|api|streamlit|check) ;; *) echo "❌ --mode 只接受 fallback|api|streamlit|check"; exit 2 ;; esac

# -----------------------------------------------------------------------------
# 1. env 文件（KEY=VALUE，空值=不设置）
# -----------------------------------------------------------------------------
import_env_file() {
    local path="$1" line key value count=0
    [ -z "${path}" ] && return 0
    if [ ! -f "${path}" ]; then log_event 'func.error' 'ERROR' 'import_env_file' "env_file=${path}" '' '' '文件不存在'; return 2; fi
    while IFS= read -r line || [ -n "${line}" ]; do
        line="${line#"${line%%[![:space:]]*}"}"
        case "${line}" in ''|'#'*) continue ;; esac
        key="${line%%=*}"; value="${line#*=}"
        [ "${key}" = "${line}" ] && continue
        key="$(echo "${key}" | tr -d '[:space:]')"
        value="$(echo "${value}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
        [ -z "${value}" ] && continue
        export "${key}=${value}"
        count=$((count + 1))
    done < "${path}"
    log_event 'func.exit' 'INFO' 'import_env_file' "env_file=${path}" "applied_keys=${count}" '0'
    return 0
}

# -----------------------------------------------------------------------------
# 2. 解释器
# -----------------------------------------------------------------------------
resolve_python() {
    if [ -n "${PYTHON_BIN}" ]; then echo "${PYTHON_BIN}"; return; fi
    if [ -n "${RAG_SCHEDULER_PYTHON:-}" ]; then echo "${RAG_SCHEDULER_PYTHON}"; return; fi
    if command -v python3 >/dev/null 2>&1; then echo python3; return; fi
    if command -v python >/dev/null 2>&1; then echo python; return; fi
    echo ''
}

# -----------------------------------------------------------------------------
# 3. 预检（真实 import；索引/语料/Ollama/OpenAI 兼容后端）
# -----------------------------------------------------------------------------
preflight() {
    local py="$1" mode="$2" deps_json='[]'
    case "${mode}" in
        api) deps_json='["fastapi","uvicorn"]' ;;
        streamlit) deps_json='["streamlit"]' ;;
    esac
    export RAG_PREFLIGHT_MODE_DEPS="${deps_json}"
    local tmp_json
    tmp_json="$(mktemp)"
    PYTHONPATH="${REPO_ROOT}/研发:${PYTHONPATH:-}" "${py}" - "${tmp_json}" <<'PY'
import json, os, sys, urllib.request, pathlib
out_path = sys.argv[1]
sys.path.insert(0, str(pathlib.Path("研发").resolve()))
out = {"python": sys.version.split()[0], "executable": sys.executable, "cwd": os.getcwd()}
missing = []
for mod in ("numpy", "pymupdf", "jieba"):
    try:
        __import__(mod)
    except Exception as exc:  # noqa: BLE001 —— 显式收集，不静默
        missing.append({"module": mod, "error": f"{type(exc).__name__}: {exc}"})
out["missing_core"] = missing
mode_missing = []
for mod in json.loads(os.environ.get("RAG_PREFLIGHT_MODE_DEPS", "[]")):
    try:
        __import__(mod)
    except Exception as exc:  # noqa: BLE001
        mode_missing.append({"module": mod, "error": f"{type(exc).__name__}: {exc}"})
out["missing_mode_deps"] = mode_missing
index_dir = pathlib.Path("研发/data/index")
manifests = sorted(index_dir.glob("*/index_manifest.json"))
out["index_manifests"] = [str(p) for p in manifests]
out["chunks"] = out["vectors"] = out["dim"] = out["bm25_vocab"] = out["embed_model"] = None
out["index_dir_used"] = None
out["index_files"] = []
if manifests:
    manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
    embedding = manifest.get("embedding") or {}
    bm25 = manifest.get("bm25") or {}
    out["chunks"] = manifest.get("total_chunks") or embedding.get("count") or bm25.get("count")
    out["vectors"] = embedding.get("count")
    out["dim"] = embedding.get("dim")
    out["bm25_vocab"] = bm25.get("vocab_size")
    out["embed_model"] = embedding.get("model")
    out["index_dir_used"] = str(manifests[-1].parent)
    out["index_files"] = [f.get("file_name") for f in (manifest.get("files") or [])]
raw = pathlib.Path("研发/data/raw")
out["raw_pdfs"] = sorted(p.name for p in raw.glob("*.pdf")) if raw.is_dir() else []
base = os.environ.get("RAG_LLM__OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
try:
    with urllib.request.urlopen(base + "/api/tags", timeout=0.5) as resp:
        names = [r.get("name") for r in json.loads(resp.read().decode("utf-8") or "{}").get("models", [])]
    out["ollama"] = {"available": True, "models": names[:8]}
except Exception as exc:  # noqa: BLE001
    out["ollama"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
oa = os.environ.get("RAG_LLM__OPENAI_BASE_URL", "")
if oa:
    try:
        with urllib.request.urlopen(oa.rstrip("/") + "/models", timeout=0.5) as resp:
            out["openai_compat"] = {"available": True, "probe": "HTTP 200"}
    except Exception as exc:  # noqa: BLE001
        out["openai_compat"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
else:
    out["openai_compat"] = {"available": False, "error": "未配置 RAG_LLM__OPENAI_BASE_URL"}
pathlib.Path(out_path).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False))
PY
    local code=$?
    unset RAG_PREFLIGHT_MODE_DEPS
    if [ ${code} -ne 0 ]; then log_event 'func.error' 'ERROR' 'preflight' "mode=${mode}" '' '' "预检脚本失败 exit=${code}"; rm -f "${tmp_json}"; return 2; fi
    PREFLIGHT_JSON="${tmp_json}"
    return 0
}

# 用 python 解析预检 JSON（不依赖 jq），输出「问题清单」，并回填若干 shell 变量
read_preflight() {
    local json_file="$1" mode="$2" allow_degraded="$3"
    local parse_out
    parse_out="$("${PY}" - "${json_file}" "${mode}" "${allow_degraded}" <<'PY'
import json, sys
data = json.loads(open(sys.argv[1], encoding="utf-8").read())
mode, allow_degraded = sys.argv[2], sys.argv[3] == "1"
problems = []
core_missing = data.get("missing_core") or []
if core_missing:
    problems.append("核心依赖缺失：" + "；".join(f"{m['module']}（{m['error']}）" for m in core_missing))
mode_missing = data.get("missing_mode_deps") or []
degraded = False
if mode_missing:
    detail = "；".join(f"{m['module']}（{m['error']}）" for m in mode_missing)
    if mode == "streamlit" and allow_degraded:
        degraded = True
        print("DEGRADED " + detail)
    elif mode == "streamlit":
        problems.append(f"模式 streamlit 依赖缺失：{detail} —— 本机/本容器无该依赖；请用 --mode fallback（纯标准库）")
    else:
        problems.append(f"模式 {mode} 依赖缺失：{detail}")
if not data.get("index_manifests"):
    problems.append("索引不存在（未找到 研发/data/index/*/index_manifest.json）：先跑 研发/scripts/parse_corpus.py 再跑 研发/scripts/build_index.py")
if not data.get("raw_pdfs"):
    problems.append("语料为空：研发/data/raw 下没有 *.pdf（按目录自动发现，禁止硬编码文件名）")
print("SUMMARY chunks=%s vectors=%s dim=%s bm25_vocab=%s embed=%s pdfs=%s ollama=%s" % (
    data.get("chunks"), data.get("vectors"), data.get("dim"), data.get("bm25_vocab"), data.get("embed_model"),
    ",".join(data.get("raw_pdfs") or []),
    "可用" if (data.get("ollama") or {}).get("available") else "不可用→生成按设计降级 extractive"))
print("DEGRADED" if degraded else "PROBLEMS %d" % len(problems))
for p in problems:
    print("PROBLEM " + p)
PY
)"
    PREFLIGHT_CHUNKS="$(echo "${parse_out}" | sed -n 's/^SUMMARY chunks=\([^ ]*\).*/\1/p')"
    PREFLIGHT_SUMMARY="$(echo "${parse_out}" | sed -n 's/^SUMMARY //p')"
    PREFLIGHT_PROBLEMS="$(echo "${parse_out}" | sed -n 's/^PROBLEM //p')"
    if echo "${parse_out}" | grep -q '^DEGRADED'; then
        MODE='fallback'
        log_event 'run_app.mode_degraded' 'WARNING' 'read_preflight' "mode=streamlit" 'fallback' '' "$(echo "${parse_out}" | grep '^DEGRADED' | head -1)"
        echo "⚠️ streamlit 不可用 → 按 --allow-degraded 显式降级为 fallback 模式"
    fi
    if [ -n "${PREFLIGHT_PROBLEMS}" ]; then
        echo "❌ 预检未通过："
        echo "${PREFLIGHT_PROBLEMS}" | while IFS= read -r p; do echo "   - ${p}"; done
        log_event 'run_app.preflight_failed' 'ERROR' 'read_preflight' "mode=${mode}" "${PREFLIGHT_PROBLEMS}" '' '预检失败'
        [ "${mode}" = 'streamlit' ] && return 4
        return 2
    fi
    echo "✅ 预检通过：${PREFLIGHT_SUMMARY}"
    return 0
}

# -----------------------------------------------------------------------------
# 4. 端口检查（python 监听一次，跨平台一致）
# -----------------------------------------------------------------------------
port_free() {
    "${PY}" - "$1" <<'PY'
import socket, sys
port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", port))
    print("FREE")
except OSError as exc:
    print(f"BUSY {exc}")
finally:
    sock.close()
PY
}

# -----------------------------------------------------------------------------
# 5. 健康检查（python 轮询，无 curl 依赖）
# -----------------------------------------------------------------------------
wait_health() {
    "${PY}" - "$1" "$2" <<'PY'
import json, sys, time, urllib.request
url, timeout = sys.argv[1], float(sys.argv[2])
start = time.time()
last = ""
while time.time() - start < timeout:
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if resp.status == 200:
                print("OK %d %.0f %s" % (resp.status, (time.time() - start) * 1000, body[:600]))
                sys.exit(0)
            last = f"HTTP {resp.status}"
    except Exception as exc:  # noqa: BLE001
        last = f"{type(exc).__name__}: {exc}"
    time.sleep(0.7)
print("FAIL %.0f %s" % ((time.time() - start) * 1000, last))
sys.exit(1)
PY
}

# =============================================================================
# 主流程
# =============================================================================
PY="$(resolve_python)"
if [ -z "${PY}" ] || ! command -v "${PY}" >/dev/null 2>&1; then
    log_event 'func.error' 'ERROR' 'resolve_python' "RAG_SCHEDULER_PYTHON=${RAG_SCHEDULER_PYTHON:-}" '' '' '找不到解释器'
    echo "❌ 找不到 Python 解释器（可用 --python 或 RAG_SCHEDULER_PYTHON 指定）"
    exit 127
fi
log_event 'func.exit' 'INFO' 'resolve_python' '' "python=$(${PY} -c 'import sys;print(sys.version.split()[0])') path=${PY}" '0'

if [ "${STOP}" -eq 1 ]; then
    target_port="${PORT:-8600}"
    pid_file="${LOG_DIR}/run_app_${target_port}.pid.json"
    if [ ! -f "${pid_file}" ]; then
        log_event 'func.error' 'ERROR' 'stop_app' "pid_file=${pid_file}" '' '' 'PID 文件不存在'
        echo "❌ 未找到 PID 文件：${pid_file}（该端口未经本脚本后台启动）"
        exit 1
    fi
    pid="$("${PY}" -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))['pid'])" "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then kill -TERM "${pid}" 2>/dev/null; sleep 1; kill -0 "${pid}" 2>/dev/null && kill -KILL "${pid}" 2>/dev/null; fi
    rm -f "${pid_file}"
    log_event 'func.exit' 'INFO' 'stop_app' "pid=${pid} port=${target_port}" 'stopped=1' '0'
    echo "✅ 已停止（PID ${pid}，端口 ${target_port}）"
    exit 0
fi

if [ -n "${ENV_FILE}" ]; then
    import_env_file "${ENV_FILE}" || { echo "❌ env 文件处理失败"; exit 2; }
fi
[ -n "${SERVER_HOST}" ] && export RAG_SERVER__HOST="${SERVER_HOST}"
[ "${PORT}" -gt 0 ] && export RAG_SERVER__PORT="${PORT}"

if ! run_step 'preflight' "mode=${MODE}" preflight "${PY}" "${MODE}"; then
    echo "❌ 预检执行失败"; exit 2
fi
read_preflight "${PREFLIGHT_JSON}" "${MODE}" "${ALLOW_DEGRADED}"
code=$?
rm -f "${PREFLIGHT_JSON}"
if [ ${code} -ne 0 ]; then exit ${code}; fi

if [ "${MODE}" = 'check' ]; then
    echo "ℹ️ --mode check：仅预检，不启动服务"
    exit 0
fi

port_to_use="${PORT:-0}"
[ "${port_to_use}" -eq 0 ] && { [ "${MODE}" = 'streamlit' ] && port_to_use=8501 || port_to_use=8600; }
host_to_use="${SERVER_HOST:-${RAG_SERVER__HOST:-127.0.0.1}}"
health_host="${host_to_use}"; case "${health_host}" in 0.0.0.0|::) health_host=127.0.0.1 ;; esac
[ "${MODE}" = 'streamlit' ] && health_path='/_stcore/health' || health_path='/api/health'
health_url="http://${health_host}:${port_to_use}${health_path}"

if [ "$(port_free "${port_to_use}")" != 'FREE' ]; then
    log_event 'func.error' 'ERROR' 'port_free' "port=${port_to_use}" '' '' '端口被占用'
    echo "❌ 端口 ${port_to_use} 已被占用：换端口（--port 8601）或先停止（--stop --port ${port_to_use}）"
    exit 3
fi

case "${MODE}" in
    fallback)
        APP_ARGS=("研发/app/ui/serve_fallback.py" "--host" "${host_to_use}" "--port" "${port_to_use}")
        [ "${NO_WARMUP}" -eq 1 ] && APP_ARGS+=("--no-warmup")
        ;;
    api)
        APP_ARGS=("-m" "uvicorn" "app.main:app" "--app-dir" "研发" "--host" "${host_to_use}" "--port" "${port_to_use}")
        ;;
    streamlit)
        APP_ARGS=("-m" "streamlit" "run" "研发/app/ui/streamlit_app.py" "--server.address" "${host_to_use}" \
                  "--server.port" "${port_to_use}" "--server.headless" "true")
        ;;
esac

echo "🚀 启动模式 ${MODE}：${PY} ${APP_ARGS[*]}"
echo "   访问地址：http://${host_to_use}:${port_to_use}/　健康检查：${health_url}"

if [ "${BACKGROUND}" -eq 0 ]; then
    log_event 'run_app.start_foreground' 'INFO' 'run_app' "mode=${MODE} port=${port_to_use} args=${APP_ARGS[*]}"
    "${PY}" "${APP_ARGS[@]}"
    rc=$?
    log_event "$([ ${rc} -eq 0 ] && echo func.exit || echo run_app.exit_foreground)" "$([ ${rc} -eq 0 ] && echo INFO || echo ERROR)" 'run_app' \
        "mode=${MODE} port=${port_to_use}" "exit_code=${rc}" '0'
    exit ${rc}
fi

out_log="${LOG_DIR}/run_app_${port_to_use}.out.log"
err_log="${LOG_DIR}/run_app_${port_to_use}.err.log"
pid_file="${LOG_DIR}/run_app_${port_to_use}.pid.json"
# setsid 在部分精简容器里不存在 → 显式回退到 nohup（两者都保证父进程退出后子进程继续）
if command -v setsid >/dev/null 2>&1; then
    setsid nohup "${PY}" "${APP_ARGS[@]}" >"${out_log}" 2>"${err_log}" < /dev/null &
else
    log_event 'run_app.setsid_missing' 'WARNING' 'run_app' 'mode='${MODE} '' '' 'setsid 不存在，改用 nohup'
    nohup "${PY}" "${APP_ARGS[@]}" >"${out_log}" 2>"${err_log}" < /dev/null &
fi
child_pid=$!
"${PY}" - "${pid_file}" "${child_pid}" "${MODE}" "${port_to_use}" "${host_to_use}" "${PY}" "${out_log}" "${err_log}" <<'PY'
import json, os, sys, time
pid_file, pid, mode, port, host, python, out_log, err_log = sys.argv[1:9]
meta = {"pid": int(pid), "mode": mode, "port": int(port), "host": host, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": python, "out_log": out_log, "err_log": err_log, "run_id": os.environ.get("RAG_RUN_ID")}
open(pid_file, "w", encoding="utf-8").write(json.dumps(meta, ensure_ascii=False, indent=2))
PY
log_event 'run_app.start_background' 'INFO' 'run_app' "mode=${MODE} port=${port_to_use} pid=${child_pid}" "pid_file=${pid_file}" '0'

echo "⏳ 等待健康检查（最多 ${HEALTH_TIMEOUT} s）…"
if ! run_step 'wait_health' "url=${health_url}" wait_health "${health_url}" "${HEALTH_TIMEOUT}"; then
    log_event 'run_app.health_failed' 'ERROR' 'wait_health' "url=${health_url}" '' '' '健康检查未通过'
    echo "❌ 健康检查未通过：${health_url}"
    echo "   进程输出尾部（${out_log}）："; tail -n 25 "${out_log}" 2>/dev/null
    echo "   错误输出尾部（${err_log}）："; tail -n 25 "${err_log}" 2>/dev/null
    echo "   清理：bash 部署/脚本/run_app.sh --stop --port ${port_to_use}"
    exit 1
fi

log_event 'run_app.health_ok' 'INFO' 'run_app' "url=${health_url}" "pid=${child_pid}" '0'
echo "✅ 健康检查通过（PID ${child_pid}）"
cat <<EOF

--- 运维信息 ---
   停止：bash 部署/脚本/run_app.sh --stop --port ${port_to_use}
   PID 文件：${pid_file}
   进程输出：${out_log}
   应用日志：${LOG_DIR}/app.log
   错误日志：${LOG_DIR}/error.log
   链路日志：${LOG_DIR}/rag_trace.jsonl
   部署日志：${DEPLOY_LOG}
   日志自检：python3 部署/脚本/verify_logs.py
EOF
exit 0
