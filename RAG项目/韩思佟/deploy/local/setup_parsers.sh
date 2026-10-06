#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${RAG_PARSER_VENV:-/home/lenovo/.venvs/rag-parsers}"
cd "$ROOT"
python3 -m venv "$VENV"
export PATH="$VENV/bin:$PATH"
"$VENV/bin/python" -m pip install --upgrade pip
# MinerU pipeline 在无显卡的 Ubuntu 上使用 CPU 版 PyTorch。
if ! "$VENV/bin/python" -c "import torch, torchvision" >/dev/null 2>&1; then
  "$VENV/bin/python" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
"$VENV/bin/python" -m pip install -r requirements-parsers.txt
# 主项目虚拟环境已有 PyTorch 时，也允许解析环境复用它，避免重复占用数 GB。
CORE_SITE="${RAG_CORE_SITE_PACKAGES:-$HOME/.venvs/rag-roleplay/lib/python3.12/site-packages}"
if ! "$VENV/bin/python" -c "import torch" >/dev/null 2>&1 && [[ -d "$CORE_SITE/torch" ]]; then
  printf '%s\n' "$CORE_SITE" > "$VENV/lib/python3.12/site-packages/rag-core.pth"
fi
"$VENV/bin/python" -m app.offline_pipeline check --require-all
echo "PARSERS_READY: PaddleOCR 和 MinerU 已安装在 $VENV"
