#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
[[ -f .env.local ]] || { echo "缺少 .env.local，请先运行 setup.sh" >&2; exit 1; }
set -a
source .env.local
set +a
[[ -n "${AUTODL_SSH_HOST:-}" && "$AUTODL_SSH_HOST" != region-instance.autodl.com ]] || {
  echo "请把 AutoDL SSH 登录信息填入 .env.local 的 AUTODL_SSH_*。" >&2; exit 1;
}
echo "隧道运行期间保持此终端打开：本地 8001 -> AutoDL 127.0.0.1:8001"
exec ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -NT -L 127.0.0.1:8001:127.0.0.1:8001 \
  -p "${AUTODL_SSH_PORT}" "${AUTODL_SSH_USER}@${AUTODL_SSH_HOST}"
