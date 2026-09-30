#!/usr/bin/env bash
# 部署脚本**自检**：在真机（有 bash 的机器）上一条命令验证四个脚本能不能用。
#
# 为什么需要它：本项目开发机是 Windows，**没有可用的 bash**（`bash.exe` 是 WSL 启动器、
# 服务拒绝访问；Docker daemon 也无权限）⇒ `tests/test_deploy_scripts.py` 只能做**静态**检查，
# 真机执行的 `bash -n` 一直是"有 bash 才跑，没有就跳过"。这个脚本把"真机该跑什么"固化下来：
#
#   1. 语法：对每个 `deploy/*.sh` 跑 `bash -n`（语法错误在这里就暴露，不必等真部署）；
#   2. 帮助：对四个入口脚本跑 `--help`（必须退出 0，且不产生副作用）；
#   3. 干跑：对四个入口脚本跑 `--dry-run`（必须退出 0，且**不改动磁盘/进程**）；
#   4. 依赖可加载：`source deploy/lib.sh` 后助手函数齐全。
#
# 用法：
#   bash deploy/selfcheck.sh            # 全量（推荐在真机首次部署前跑）
#   bash deploy/selfcheck.sh --quick    # 只跑语法检查
#
# 退出码：0 全过；1 有失败项；2 环境不满足（不是 bash 跑的）。
set -euo pipefail

LR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LR_ROOT"

QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && {
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

if [[ -z "${BASH_VERSION:-}" ]]; then
    echo "!! 这个脚本必须用 bash 跑（当前不是 bash）" >&2
    exit 2
fi

ENTRIES=(install.sh deploy.sh start.sh stop.sh)
PASS=0
FAIL=0
FAILED_ITEMS=()

_pass() { PASS=$((PASS + 1)); printf '  [OK]   %s\n' "$1"; }
_fail() { FAIL=$((FAIL + 1)); FAILED_ITEMS+=("$1"); printf '  [FAIL] %s\n' "$1"; }

echo "======== 部署脚本自检 $(date -Is) ========"
echo "bash: $BASH_VERSION（$BASH）"
echo "根目录: $LR_ROOT"
echo

echo "[1/4] 语法检查 bash -n"
for script in deploy/lib.sh deploy/selfcheck.sh "${ENTRIES[@]/#/deploy/}"; do
    if bash -n "$script" 2>/tmp/lr-selfcheck-err; then
        _pass "bash -n $script"
    else
        _fail "bash -n $script —— $(tr '\n' ' ' </tmp/lr-selfcheck-err | cut -c1-160)"
    fi
done

if [[ "$QUICK" == "1" ]]; then
    echo
    echo "========（--quick：跳过帮助/干跑）通过 $PASS / 失败 $FAIL ========"
    [[ "$FAIL" -eq 0 ]] || exit 1
    exit 0
fi

echo
echo "[2/4] --help（必须退出 0）"
for entry in "${ENTRIES[@]}"; do
    if bash "deploy/$entry" --help >/dev/null 2>&1; then
        _pass "deploy/$entry --help"
    else
        _fail "deploy/$entry --help 退出码非 0"
    fi
done

echo
echo "[3/4] --dry-run（必须退出 0，且不动磁盘/进程）"
# 干跑前后各记一次关键状态，跑完要求"没变"——干跑就该是**只读**的
# ⚠️ 赋值里的命令替换必须自己兜底：`VAR="$(ls ... | tr ...)"` 在 `set -e` 下，
#    只要管道里任何一个命令失败（例如 index/ 不存在），**整个自检会在这里退出**。
SNAP_BEFORE="$(ls -1 index 2>/dev/null | tr '\n' ' ' || true)$(pgrep -c -f 'run_api\.py' 2>/dev/null || echo 0)"
for entry in "${ENTRIES[@]}"; do
    if bash "deploy/$entry" --dry-run >/dev/null 2>&1; then
        _pass "deploy/$entry --dry-run"
    else
        # install/deploy 在缺依赖时可能非 0；只警告不判失败，但要让人看见
        _fail "deploy/$entry --dry-run 退出码非 0（看上面输出定位）"
    fi
done
SNAP_AFTER="$(ls -1 index 2>/dev/null | tr '\n' ' ' || true)$(pgrep -c -f 'run_api\.py' 2>/dev/null || echo 0)"
if [[ "$SNAP_BEFORE" == "$SNAP_AFTER" ]]; then
    _pass "干跑没有改动 index/ 与 api 进程"
else
    _fail "干跑改动了环境：'$SNAP_BEFORE' -> '$SNAP_AFTER'"
fi

echo
echo "[4/4] lib.sh 可加载且助手齐全"
REQUIRED=(lr_init lr_info lr_ok lr_warn lr_die lr_run lr_have lr_free_gb lr_wait_http
          lr_read_pid lr_stop_pid lr_python_pids lr_snapshot_runtime lr_ok_if_ran)
# shellcheck source=/dev/null
if source deploy/lib.sh >/dev/null 2>&1; then
    missing=()
    for helper in "${REQUIRED[@]}"; do
        declare -F "$helper" >/dev/null || missing+=("$helper")
    done
    if [[ "${#missing[@]}" -eq 0 ]]; then
        _pass "lib.sh 可 source 且 ${#REQUIRED[@]} 个助手齐全"
    else
        _fail "lib.sh 缺少助手：${missing[*]}"
    fi
else
    _fail "source deploy/lib.sh 失败"
fi

echo
echo "======== 自检结果：通过 $PASS / 失败 $FAIL ========"
if [[ "$FAIL" -ne 0 ]]; then
    printf '失败项：\n'
    printf '  - %s\n' "${FAILED_ITEMS[@]}"
    exit 1
fi
echo "（注意：这一步只证明脚本**语法与干跑**没问题；真正的端到端验证还要跑）"
echo "  bash deploy/install.sh && bash deploy/deploy.sh && bash deploy/start.sh"
echo "  python scripts/verify_delivery.py     # 交付自检 20 项"
exit 0
