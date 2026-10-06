#!/usr/bin/env bash
# =============================================================================
# 工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 文件：部署/docker/entrypoint.sh —— 容器入口（模式分发 + 可选等待 LLM 就绪）
#
# 用法（容器 CMD 或 docker run 末位参数；默认 api）：
#   api        启动 FastAPI（uvicorn app.main:app，--app-dir 研发）
#   fallback   启动纯标准库界面（http.server；镜像内无需额外依赖）
#   streamlit  启动 Streamlit（需镜像内已装 streamlit，见 requirements-cloud.txt）
#   check      只做环境预检（依赖/索引/语料/LLM 探测）后退出，用于 CI 与部署自检
#
# 环境变量：
#   RAG_APP_MODE        默认 api（等价于首个位置参数）
#   RAG_WAIT_FOR_LLM    1 = 启动前等待 LLM 端点就绪（默认 0；等待上限 RAG_WAIT_TIMEOUT，默认 300 s）
#   RAG_SERVER__HOST/PORT、RAG_DATA__*、RAG_LOG__DIR、RAG_LLM__* 见 部署/配置/config.example.env
#
# 说明：预热由 `app.main` 的 startup 事件（build_engine(warmup=True)）完成，
#       以及 fallback/streamlit 入口自身调用 engine.warmup()，本脚本不再重复预热。
# =============================================================================
set -euo pipefail

WORK_ORDER='人工智能NLP-RAG-PDF文档的表格解析及检索优化'
MODE="${1:-${RAG_APP_MODE:-api}}"
echo "[entrypoint] 工单：${WORK_ORDER}"
echo "[entrypoint] 模式=${MODE} 解释器=$(python --version 2>&1)"

cd /app

wait_for() {
    python - "$1" "${RAG_WAIT_TIMEOUT:-300}" <<'PY'
import sys, time, urllib.request
url, timeout = sys.argv[1], float(sys.argv[2])
start = time.time()
while time.time() - start < timeout:
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            if resp.status == 200:
                print(f"[entrypoint] LLM 就绪：{url}")
                sys.exit(0)
    except Exception as exc:  # noqa: BLE001 —— 等待期允许失败，超时才报错
        last = f"{type(exc).__name__}: {exc}"
    time.sleep(2)
print(f"[entrypoint] ❌ LLM 未就绪（{timeout}s 超时）：{url}：{last}")
sys.exit(1)
PY
}

if [ "${RAG_WAIT_FOR_LLM:-0}" = "1" ]; then
    case "${RAG_LLM__BACKEND:-auto}" in
        openai)
            wait_for "${RAG_LLM__OPENAI_BASE_URL:-http://vllm:8000/v1}/models"
            ;;
        *)
            wait_for "${RAG_LLM__OLLAMA_BASE_URL:-http://ollama:11434}/api/tags"
            ;;
    esac
fi

case "${MODE}" in
    api)
        echo "[entrypoint] 启动 FastAPI：127.0.0.1 内部 → ${RAG_SERVER__HOST}:${RAG_SERVER__PORT}"
        exec python -m uvicorn app.main:app --app-dir 研发 \
             --host "${RAG_SERVER__HOST:-0.0.0.0}" --port "${RAG_SERVER__PORT:-8600}"
        ;;
    fallback)
        exec python 研发/app/ui/serve_fallback.py \
             --host "${RAG_SERVER__HOST:-0.0.0.0}" --port "${RAG_SERVER__PORT:-8600}"
        ;;
    streamlit)
        echo "[entrypoint] 启动 Streamlit（需镜像内已安装 streamlit，否则会立即报 ModuleNotFoundError）"
        exec python -m streamlit run 研发/app/ui/streamlit_app.py \
             --server.address "${RAG_SERVER__HOST:-0.0.0.0}" --server.port "${RAG_SERVER__PORT:-8600}" \
             --server.headless true
        ;;
    check)
        echo "[entrypoint] 仅预检：依赖 + 索引 + 语料 + LLM 探测"
        python - <<'PY'
import importlib, json, pathlib, urllib.request, os, sys
work_order = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"
print("工单：", work_order)
missing = []
for mod in ("numpy", "pymupdf", "jieba", "fastapi", "uvicorn"):
    try:
        importlib.import_module(mod)
    except Exception as exc:  # noqa: BLE001 —— 显式列出，不静默
        missing.append(f"{mod}: {type(exc).__name__}: {exc}")
manifests = sorted(pathlib.Path("研发/data/index").glob("*/index_manifest.json"))
pdfs = sorted(p.name for p in pathlib.Path("研发/data/raw").glob("*.pdf"))
base = os.environ.get("RAG_LLM__OLLAMA_BASE_URL", "http://127.0.0.1:11434")
try:
    with urllib.request.urlopen(base + "/api/tags", timeout=0.5) as resp:
        ollama = f"可用（{len(json.loads(resp.read().decode('utf-8') or '{}').get('models', []))} 个模型）"
except Exception as exc:  # noqa: BLE001
    ollama = f"不可用（{type(exc).__name__}）→ 生成按设计降级 extractive"
print("缺失依赖：", missing or "无")
print("索引 manifest：", [str(p) for p in manifests] or "无（请先跑 parse_corpus.py + build_index.py）")
print("语料 PDF：", pdfs or "无")
print("Ollama：", ollama)
sys.exit(1 if (missing or not manifests or not pdfs) else 0)
PY
        ;;
    *)
        echo "[entrypoint] ❌ 未知模式：${MODE}（可选 api|fallback|streamlit|check）" >&2
        exit 2
        ;;
esac
