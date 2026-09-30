#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing: $1" >&2
    exit 1
  fi
}

require_cmd python
require_cmd node
require_cmd npm

if [ ! -f .env ]; then
  echo "missing: .env (copy .env.example and fill local values)" >&2
  exit 1
fi

PYTHONPATH=backend python - <<'PY'
from app.core.config import get_settings
settings = get_settings()
print('app:', settings.app_name)
print('mysql:', f'{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}')
print('redis:', f'{settings.redis_host}:{settings.redis_port}/{settings.redis_db}')
print('milvus:', f'{settings.milvus_host}:{settings.milvus_port}/{settings.milvus_collection_knowledge}')
print('deepseek_configured:', bool(settings.deepseek_api_key and not settings.deepseek_api_key.startswith('your-')))
PY

cd frontend
npm --version >/dev/null
node --version >/dev/null
echo "environment check passed"
