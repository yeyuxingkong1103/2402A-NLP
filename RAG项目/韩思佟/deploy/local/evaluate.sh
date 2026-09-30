#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP_VENV="${RAG_VENV:-/home/lenovo/.venvs/rag-roleplay}"
EVAL_VENV="${RAG_EVAL_VENV:-/home/lenovo/.venvs/rag-roleplay-eval}"
cd "$ROOT"
[[ -x "$APP_VENV/bin/python" ]] || { echo "先运行 bash deploy/local/setup.sh 创建项目环境。" >&2; exit 1; }
if [[ "${1:-}" == "--setup" ]]; then
  # 新环境拥有独立依赖；只将原环境的第三方库路径作为补充，复用体积较大的Torch。
  "$APP_VENV/bin/python" -m venv "$EVAL_VENV"
  APP_PACKAGES="$("$APP_VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  EVAL_PACKAGES="$("$EVAL_VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  printf '%s\n' "$APP_PACKAGES" > "$EVAL_PACKAGES/zz_project_packages.pth"
  "$EVAL_VENV/bin/python" -m pip install --ignore-installed -r requirements-eval.txt
  "$EVAL_VENV/bin/python" -c 'import ragas, openai; from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall; print("评测环境:", ragas.__version__, "OpenAI:", openai.__version__)'
  echo "安装完成。先执行 bash deploy/local/evaluate.sh --limit 8 校验，再执行 --run 开始付费评测。"
  exit 0
fi
if [[ ! -x "$EVAL_VENV/bin/python" ]]; then
  echo "请先执行 bash deploy/local/evaluate.sh --setup" >&2
  exit 1
fi
# --run 是调用在线裁判的明确开关；无此参数只校验，不生成假分数。
export RAG_ENV_FILE="$ROOT/.env.local"
export RAGAS_DO_NOT_TRACK=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$EVAL_VENV/bin/python" -m app.evaluate ragas "$@"
