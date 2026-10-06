#!/bin/sh
set -eu

# The .env file is also used for local development, so translate host-local
# database URLs only inside the container.
rewrite_host() {
  printf '%s' "$1" | sed \
    -e 's#://127\\.0\\.0\\.1:#://host.docker.internal:#g' \
    -e 's#://localhost:#://host.docker.internal:#g'
}

export MYSQL_URL="$(rewrite_host "${MYSQL_URL:-}")"
export REDIS_URL="$(rewrite_host "${REDIS_URL:-}")"
export MILVUS_URI="$(rewrite_host "${MILVUS_URI:-}")"

exec "$@"
