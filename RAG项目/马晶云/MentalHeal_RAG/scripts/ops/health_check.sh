#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
python - <<'PY'
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

url = os.environ.get('BASE_URL', 'http://127.0.0.1:8000').rstrip('/') + '/health'
try:
    with urlopen(url, timeout=5) as response:
        payload = json.loads(response.read().decode('utf-8'))
        code = response.status
except HTTPError as exc:
    payload = json.loads(exc.read().decode('utf-8'))
    code = exc.code
except (URLError, TimeoutError) as exc:
    print(f'health_check_failed: {type(exc).__name__}: {exc}', file=sys.stderr)
    sys.exit(1)
print(json.dumps(payload, ensure_ascii=False, indent=2))
sys.exit(0 if code == 200 and payload.get('status') == 'ok' else 1)
PY
