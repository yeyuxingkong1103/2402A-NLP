#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${RAG_PARSER_VENV:-/home/lenovo/.venvs/rag-parsers}"
CORE_VENV="${RAG_VENV:-/home/lenovo/.venvs/rag-roleplay}"
cd "$ROOT"
[[ -x "$VENV/bin/python" ]] || {
  echo "请先运行 bash deploy/local/setup_parsers.sh" >&2
  exit 1
}
export PATH="$VENV/bin:$PATH"
"$VENV/bin/python" -m app.offline_pipeline prepare "$@"
echo "DOCUMENTS_READY: 解析、清洗和分块完成。"
echo "下一步：$CORE_VENV/bin/python -m app.offline_pipeline embed"
echo "再执行：$CORE_VENV/bin/python -m app.offline_pipeline sync --dry-run"
echo "确认预览后正式执行：$CORE_VENV/bin/python -m app.offline_pipeline sync"
echo "最后执行 bash deploy/local/start.sh，重启在线服务并刷新内存BM25。"
