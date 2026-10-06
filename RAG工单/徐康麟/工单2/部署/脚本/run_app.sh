#!/usr/bin/env bash
# =============================================================================
# 工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 阶段：部署 / 算力云 Linux —— 应用启动（Streamlit 界面 + OpenAI 兼容后端）
# =============================================================================
# 用途
#   在算力云（Linux）上启动 RAG 问答应用：
#     界面：优先 Streamlit（app/ui/streamlit_app.py），不可用时回退纯标准库
#           http.server（app/ui/serve_fallback.py），两者共用同一套 app/core。
#     LLM ：默认走 vLLM/SGLang 的 OpenAI 兼容接口（RAG_LLM__BACKEND=openai），
#           也可用 --backend ollama 指向 Ollama。
#   本机 Windows 请改用同目录 run_app.ps1（Ollama + serve_fallback）。
#
# 用法
#   ./run_app.sh --help
#   ./run_app.sh                                  # 默认 streamlit + openai:8000
#   ./run_app.sh --ui fallback --port 8600
#   ./run_app.sh --backend ollama --llm-model qwen2.5:3b
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

UI="${RAG_UI:-auto}"                       # auto | streamlit | fallback
HOST="${RAG_HOST:-0.0.0.0}"
PORT="${RAG_PORT:-8600}"
LLM_BACKEND="${RAG_LLM_BACKEND_UI:-openai}" # openai | ollama | extractive
LLM_BASE_URL="${RAG_LLM__BASE_URL:-http://127.0.0.1:8000/v1}"
LLM_MODEL="${RAG_LLM__MODEL:-qwen2.5-7b-instruct}"
LLM_API_KEY="${RAG_LLM__API_KEY:-EMPTY}"
OLLAMA_BASE_URL="${RAG_LLM__OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
OLLAMA_MODEL="${RAG_LLM__MODEL_OLLAMA:-qwen2.5:3b}"
EMBED_BACKEND="${RAG_EMBEDDING__BACKEND:-ollama}"
EMBED_MODEL="${RAG_EMBEDDING__OLLAMA_MODEL:-bge-m3:latest}"
INDEX_DIR="${RAG_INDEX_DIR:-${REPO_ROOT}/研发/data/index/bge-m3-1024}"
SKIP_PREFLIGHT=0

usage() {
  cat <<'EOF'
用法: run_app.sh [选项]

选项:
      --ui MODE          auto(默认)|streamlit|fallback —— 界面实现
      --host HOST        监听地址（默认 0.0.0.0）
  -p, --port PORT        监听端口（默认 8600）
      --backend B        openai(默认，vLLM/SGLang) | ollama | extractive
      --base-url URL     OpenAI 兼容地址（默认 http://127.0.0.1:8000/v1）
      --llm-model NAME   生成模型名（默认 qwen2.5-7b-instruct）
      --api-key KEY      后端鉴权 key（默认 EMPTY）
      --ollama-url URL   Ollama 地址（--backend ollama 时用）
      --embed-model NAME 嵌入模型（默认 bge-m3:latest；必须与索引维度一致）
      --index-dir DIR    索引目录（默认 研发/data/index/bge-m3-1024）
      --skip-preflight   跳过前置检查
  -h, --help             显示本帮助

环境变量: RAG_UI RAG_HOST RAG_PORT RAG_LLM_BACKEND_UI RAG_LLM__BASE_URL RAG_LLM__MODEL
          RAG_LLM__API_KEY RAG_LLM__OLLAMA_BASE_URL RAG_EMBEDDING__BACKEND
          RAG_EMBEDDING__OLLAMA_MODEL RAG_INDEX_DIR PYTHON_BIN（默认 python3）

退出码: 0 正常退出 | 2 前置检查失败 | 3 索引/入口文件缺失 | 5 后端不可达 | 6 端口占用
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ui) UI="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    -p|--port) PORT="$2"; shift 2 ;;
    --backend) LLM_BACKEND="$2"; shift 2 ;;
    --base-url) LLM_BASE_URL="$2"; shift 2 ;;
    --llm-model) LLM_MODEL="$2"; shift 2 ;;
    --api-key) LLM_API_KEY="$2"; shift 2 ;;
    --ollama-url) OLLAMA_BASE_URL="$2"; shift 2 ;;
    --embed-model) EMBED_MODEL="$2"; shift 2 ;;
    --index-dir) INDEX_DIR="$2"; shift 2 ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[错误] 未知参数: $1" >&2; usage; exit 2 ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
log()  { echo "[$(date '+%F %T')] $*"; }
fail() { echo "[失败] $*" >&2; exit "${2:-2}"; }

STREAMLIT_PY="${REPO_ROOT}/研发/app/ui/streamlit_app.py"
FALLBACK_PY="${REPO_ROOT}/研发/app/ui/serve_fallback.py"
for f in "$STREAMLIT_PY" "$FALLBACK_PY"; do
  [[ -f "$f" ]] || fail "缺少界面入口文件: $f" 3
done

# ---------------- 前置检查 ----------------
if [[ "$SKIP_PREFLIGHT" -eq 0 ]]; then
  log "=== 前置检查 ==="
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "找不到 $PYTHON_BIN"
  log "python      : $(command -v "$PYTHON_BIN") $("$PYTHON_BIN" -V 2>&1)"

  [[ -f "${INDEX_DIR}/vectors.npy" ]] || fail "索引未就绪：${INDEX_DIR}/vectors.npy 不存在。请先构建：${PYTHON_BIN} ${REPO_ROOT}/研发/scripts/build_index.py --rebuild" 3
  log "索引目录    : ${INDEX_DIR}"

  if [[ "$LLM_BACKEND" == "openai" ]]; then
    if curl -sf --max-time 5 "${LLM_BASE_URL}/models" >/dev/null 2>&1; then
      log "LLM 后端    : ${LLM_BASE_URL} 可达"
    else
      log "LLM 后端    : ${LLM_BASE_URL} **不可达**（请先运行 run_vllm.sh / run_sglang.sh）"
      fail "OpenAI 兼容后端不可达：${LLM_BASE_URL}" 5
    fi
  elif [[ "$LLM_BACKEND" == "ollama" ]]; then
    curl -sf --max-time 5 "${OLLAMA_BASE_URL}/api/tags" >/dev/null 2>&1 \
      && log "Ollama      : ${OLLAMA_BASE_URL} 可达" \
      || fail "Ollama 不可达：${OLLAMA_BASE_URL}" 5
  else
    log "LLM 后端    : extractive（不调用外部服务）"
  fi

  if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
    fail "端口 ${PORT} 已被占用" 6
  fi
fi

# ---------------- 选择界面 ----------------
HAVE_STREAMLIT=0
if "$PYTHON_BIN" -c "import streamlit" >/dev/null 2>&1; then HAVE_STREAMLIT=1; fi

case "$UI" in
  auto)      [[ "$HAVE_STREAMLIT" -eq 1 ]] && UI_MODE="streamlit" || UI_MODE="fallback" ;;
  streamlit) UI_MODE="streamlit" ;;
  fallback)  UI_MODE="fallback" ;;
  *) fail "未知 --ui 取值: $UI（auto|streamlit|fallback）" ;;
esac

if [[ "$UI_MODE" == "streamlit" && "$HAVE_STREAMLIT" -eq 0 ]]; then
  fail "未安装 streamlit，无法使用 --ui streamlit。云端安装：pip install 'streamlit>=1.30.0'；或改用 --ui fallback" 2
fi

# ---------------- 导出环境变量并启动 ----------------
log "=== 应用配置 ==="
export RAG_LLM__BACKEND="$LLM_BACKEND"
export RAG_LLM__MODEL="$LLM_MODEL"
export RAG_LLM__BASE_URL="$LLM_BASE_URL"
export RAG_LLM__OLLAMA_BASE_URL="$OLLAMA_BASE_URL"
export RAG_LLM__API_KEY="$LLM_API_KEY"
export RAG_EMBEDDING__BACKEND="$EMBED_BACKEND"
export RAG_EMBEDDING__OLLAMA_MODEL="$EMBED_MODEL"
log "界面        : ${UI_MODE}"
log "RAG_LLM__BACKEND=${RAG_LLM__BACKEND}  RAG_LLM__MODEL=${RAG_LLM__MODEL}"
log "RAG_LLM__BASE_URL=${RAG_LLM__BASE_URL}"
log "RAG_EMBEDDING__BACKEND=${RAG_EMBEDDING__BACKEND}  RAG_EMBEDDING__OLLAMA_MODEL=${RAG_EMBEDDING__OLLAMA_MODEL}"
log "监听        : http://${HOST}:${PORT}"

cd "$REPO_ROOT"
if [[ "$UI_MODE" == "streamlit" ]]; then
  [[ "$LLM_BACKEND" == "ollama" ]] && export RAG_LLM__MODEL="$OLLAMA_MODEL"
  log "命令: ${PYTHON_BIN} -m streamlit run ${STREAMLIT_PY} --server.address ${HOST} --server.port ${PORT} --server.headless true"
  exec "$PYTHON_BIN" -m streamlit run "$STREAMLIT_PY" \
      --server.address "$HOST" --server.port "$PORT" --server.headless true
else
  log "命令: ${PYTHON_BIN} ${FALLBACK_PY} --host ${HOST} --port ${PORT}"
  exec "$PYTHON_BIN" "$FALLBACK_PY" --host "$HOST" --port "$PORT"
fi
