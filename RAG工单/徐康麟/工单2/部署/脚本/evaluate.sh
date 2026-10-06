#!/usr/bin/env bash
# =============================================================================
# 工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 阶段：部署 / 算力云 Linux —— 优化前后对比评估
# =============================================================================
# 用途
#   在算力云（或任意 Linux）上跑**优化前后对比评估**：
#     默认调用 优化/脚本/compare_optimization.py（产出 CSV + Markdown 对比报告）；
#     缺失时可加 --allow-fallback 回落到 研发/scripts/evaluate.py（只评优化后系统）。
#   与同目录 evaluate.ps1 等价（本机 Windows 用 .ps1）。
#
# CLI 契约（compare_optimization.py 必须接受；T7 按此实现）
#   --mode {rag,extractive}  --limit N  --out DIR  --golden PATH
#
# 用法
#   ./evaluate.sh --help
#   ./evaluate.sh                                  # rag 模式全量
#   ./evaluate.sh --mode extractive --limit 10
#   ./evaluate.sh --allow-fallback --out /tmp/eval
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODE="${RAG_EVAL_MODE:-rag}"
LIMIT="${RAG_EVAL_LIMIT:-0}"
OUT_DIR="${RAG_EVAL_OUT:-}"
GOLDEN="${RAG_EVAL_GOLDEN:-}"
FILENAME="${RAG_EVAL_FILENAME:-}"
COMPARE_SCRIPT="${RAG_COMPARE_SCRIPT:-${REPO_ROOT}/优化/脚本/compare_optimization.py}"
ALLOW_FALLBACK=0
DRY_RUN=0

usage() {
  cat <<'EOF'
用法: evaluate.sh [选项]

选项:
      --mode MODE       rag(默认，最终交付路径) | extractive
      --limit N         仅评估前 N 题（0=全部，默认 0）
      --out DIR         输出目录（默认各脚本自身默认，通常 优化/评估结果）
      --golden PATH     判分基准 golden_qa.jsonl（默认 测试/测试数据/golden_qa.jsonl）
      --filename NAME   输出文件名前缀（仅回落 evaluate.py 时使用）
      --compare-script PATH  对比脚本路径（默认 优化/脚本/compare_optimization.py）
      --allow-fallback  对比脚本缺失时回落 研发/scripts/evaluate.py
      --dry-run         只打印将执行的命令
  -h, --help            显示本帮助

环境变量: RAG_EVAL_MODE RAG_EVAL_LIMIT RAG_EVAL_OUT RAG_EVAL_GOLDEN
          RAG_EVAL_FILENAME RAG_COMPARE_SCRIPT PYTHON_BIN（默认 python3）

退出码: 0 成功 | 2 解释器问题 | 3 缺少脚本 | 其他 = 被调用脚本退出码
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --golden) GOLDEN="$2"; shift 2 ;;
    --filename) FILENAME="$2"; shift 2 ;;
    --compare-script) COMPARE_SCRIPT="$2"; shift 2 ;;
    --allow-fallback) ALLOW_FALLBACK=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[错误] 未知参数: $1" >&2; usage; exit 2 ;;
  esac
done

case "$MODE" in rag|extractive) ;; *) echo "[错误] --mode 只能是 rag 或 extractive" >&2; exit 2 ;; esac

PYTHON_BIN="${PYTHON_BIN:-python3}"
log()  { echo "[$(date '+%F %T')] $*"; }
fail() { echo "[失败] $*" >&2; exit "${2:-2}"; }

FALLBACK_SCRIPT="${REPO_ROOT}/研发/scripts/evaluate.py"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "找不到 $PYTHON_BIN（可用 PYTHON_BIN 覆盖）"
log "=== 优化前后对比评估 ==="
log "仓库根目录  : ${REPO_ROOT}"
log "模式 / 题数 : ${MODE} / $([[ "$LIMIT" -eq 0 ]] && echo '全部' || echo "前 ${LIMIT} 题")"
log "python      : $(command -v "$PYTHON_BIN") $("$PYTHON_BIN" -V 2>&1)"

# ---------------- 选择脚本 ----------------
SELECTED=""
KIND=""
if [[ -f "$COMPARE_SCRIPT" ]]; then
  SELECTED="$COMPARE_SCRIPT"; KIND="compare"
  log "对比脚本    : ${COMPARE_SCRIPT}"
elif [[ "$ALLOW_FALLBACK" -eq 1 ]]; then
  [[ -f "$FALLBACK_SCRIPT" ]] || fail "回落脚本也不存在: $FALLBACK_SCRIPT" 3
  SELECTED="$FALLBACK_SCRIPT"; KIND="fallback"
  log "对比脚本缺失: ${COMPARE_SCRIPT}"
  log "已按 --allow-fallback 回落到 ${FALLBACK_SCRIPT}（仅评估优化后系统，不做基线对比）"
else
  log "对比脚本缺失: ${COMPARE_SCRIPT}"
  log "  ① 等 T7 产出后重跑本脚本；"
  log "  ② 或加 --allow-fallback 先跑 研发/scripts/evaluate.py。"
  fail "缺少对比脚本" 3
fi

# ---------------- 组装命令 ----------------
ARGS=( "$SELECTED" --mode "$MODE" )
[[ "$LIMIT" -gt 0 ]] && ARGS+=( --limit "$LIMIT" )
[[ -n "$OUT_DIR" ]] && ARGS+=( --out "$OUT_DIR" )
[[ -n "$GOLDEN" ]] && ARGS+=( --golden "$GOLDEN" )
if [[ "$KIND" == "fallback" && -n "$FILENAME" ]]; then ARGS+=( --filename "$FILENAME" ); fi

log "将要执行: ${PYTHON_BIN} ${ARGS[*]}"
if [[ "$DRY_RUN" -eq 1 ]]; then
  log "[dry-run] 仅打印，不执行。"
  exit 0
fi

cd "$REPO_ROOT"
set +e
"$PYTHON_BIN" "${ARGS[@]}"
RC=$?
set -e
log "评估结束，退出码 ${RC}"
exit "$RC"
