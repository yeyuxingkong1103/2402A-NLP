#!/usr/bin/env bash
set -euo pipefail
command -v docker >/dev/null || { echo "请先安装 Docker Engine"; exit 1; }
docker compose version >/dev/null || { echo "请安装 Docker Compose 插件"; exit 1; }
command -v python3 >/dev/null || { echo "请先安装 Python 3.11+"; exit 1; }
[ -f .env ] || cp .env.example .env
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
docker compose pull
echo "安装完成。请修改 .env 后执行 scripts/run.sh"

