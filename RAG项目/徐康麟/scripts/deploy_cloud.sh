#!/usr/bin/env bash
# ============================================================================
# deploy_cloud.sh —— AutoDL 云上「一键（幂等）部署」的**薄入口**
#
# 这个文件故意只做三件事：定位自己 -> 找到 python -> 把参数原样转给
# `scripts/deploy_cloud.py`。所有逻辑（环境/依赖/引擎/flashinfer 处置/模型解析与
# 校验/配置注入/启动/自检）都在那个 Python 文件里，便于单测与 dry-run。
#
# 用法（在实例上、项目根目录下）：
#   bash scripts/deploy_cloud.sh --dry-run            # 先看计划，不改系统
#   bash scripts/deploy_cloud.sh --engine sglang --yes
#   bash scripts/deploy_cloud.sh --engine vllm --skip-model-download
#   bash scripts/deploy_cloud.sh --selfcheck          # 健康检查 + bench_serving 三档
#   bash scripts/deploy_cloud.sh --diagnose           # 失败时的诊断清单
#
# 环境变量：
#   PYTHON   指定解释器（默认 python3；conda 环境建好后可指到
#            /root/autodl-tmp/conda_envs/rag/bin/python）
#   DEPLOY_ARGS  追加参数（便于 systemd/定时任务里复用同一条命令）
#
# 语法自检（本地/实例都行）：
#   bash -n scripts/deploy_cloud.sh        # 只看语法
#   shellcheck scripts/deploy_cloud.sh     # 有 shellcheck 更好
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[FAIL] 找不到解释器：$PYTHON_BIN（可用 PYTHON=/path/to/python 指定）" >&2
    exit 10
fi

# shellcheck disable=SC2086  # DEPLOY_ARGS 有意按空格拆开（文档里就是这么用的）
exec "$PYTHON_BIN" "$HERE/deploy_cloud.py" "$@" ${DEPLOY_ARGS:-}
