#!/usr/bin/env bash
# 完成本地精排及真实评测；每一步失败即停止，不能用旧报告冒充通过。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${RAG_VENV:-/home/lenovo/.venvs/rag-roleplay}"
RUN_PAID=false
case "${1:-}" in
  --run-paid) RUN_PAID=true; shift ;;
  "") ;;
  *) echo "用法：bash deploy/local/finish_rag.sh [--run-paid]" >&2; exit 2 ;;
esac
cd "$ROOT"
[[ -x "$VENV/bin/python" ]] || { echo "请先执行 bash deploy/local/setup.sh" >&2; exit 1; }
export RAG_ENV_FILE="$ROOT/.env.local"
export RAG_RERANK_ENABLED=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$VENV/bin/python" -m scripts.prepare_reranker --check || "$VENV/bin/python" -m scripts.prepare_reranker
"$VENV/bin/python" -m unittest discover -s tests -p 'test_single_core.py' -v
"$VENV/bin/python" -m unittest discover -s tests -p 'test_retrieval_metrics.py' -v
bash deploy/local/start.sh
"$VENV/bin/python" -m app.evaluate retrieval --compare-rerank
bash deploy/local/evaluate.sh --setup
bash deploy/local/evaluate.sh --limit 8 --output outputs/ragas/input_validation
if [[ "$RUN_PAID" == true ]]; then
  # 只有显式 --run-paid 才执行两题、两种排序的真实在线付费评测。
  bash deploy/local/evaluate.sh --run --limit 2 --compare-rerank
else
  echo "已跳过付费RAGAS。需要时显式执行：bash deploy/local/finish_rag.sh --run-paid"
fi
echo "VERIFIED: 查看 outputs/retrieval_report.md 与 outputs/ragas/generation_report.md"
echo "问答页面 http://127.0.0.1:8000/chat；逐行学习 http://127.0.0.1:8000/learn"
