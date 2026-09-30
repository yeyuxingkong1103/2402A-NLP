#!/usr/bin/env bash
# 把工作区代码同步到 WSL 自己的文件系统（默认 ~/legal-rag），并可选用部署自检。
#
# 用法（在 WSL 内）：
#   bash scripts/wsl_sync.sh                 # 只同步 + 校验
#   bash scripts/wsl_sync.sh --deploy        # 同步后跑 scripts/deploy.sh
#   bash scripts/wsl_sync.sh --test          # 同步后跑 pytest（含 --deploy 的效果）
#   bash scripts/wsl_sync.sh --dst ~/x       # 自定义目标目录
#
# 从 Windows 侧一条命令调用：
#   powershell -ExecutionPolicy Bypass -File scripts\sync_to_wsl.ps1 [-Deploy] [-Test]
#
# 设计要点（每一条都是踩过的坑）：
#   * 必须拷到 WSL 自己的文件系统（ext4），**不要**在 /mnt/e 上直接跑 —— drvfs 的
#     小文件 I/O 慢近 10 倍（同一套测试 Windows 侧 180s、Linux 侧 20s）。
#   * 排除表必须严：运行时产物 + 缓存 + IDE 目录，尤其 **.ssh**（里面有工作区私钥，
#     不该在 WSL 里多存一份）。
#   * 用 --delete 让目标等于工作区；排除项用**数组**传，避免 unquoted 变量被二次
#     分词 / 通配展开。
#   * 同步后做只读校验（关键文件是否存在、有没有误带 .ssh），不只看 rsync 的退出码。
#   * 本脚本自身位于工作区内，靠 $BASH_SOURCE 自定位，不写死任何绝对路径。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$SCRIPT_DIR/.." && pwd)"
DST="$HOME/legal-rag"

# 同时打到控制台与日志文件：让 Windows 侧能**事后**读取（尤其 --test 的结果，
# 之前只打控制台，队长读不到，只能让人再跑一遍）。
LOG="${LOG:-$SRC/.tmp/wsl_sync.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1
DO_DEPLOY=0
DO_TEST=0

usage() {
    cat <<'EOF'
用法：bash scripts/wsl_sync.sh [选项]

  --dst <目录>   目标目录（默认：$HOME/legal-rag）
  --deploy       同步后运行 scripts/deploy.sh（建/刷 .venv + 依赖 + 导入自检）
  --test         同步后运行 pytest tests -q（隐含 --deploy）
  -h, --help     显示本帮助

退出码：
  0  成功
  1  工作区结构不对（$SRC 下没有 legal_rag/）
  2  参数错误
  3  同步失败（rsync 非 0）
  4  关键文件校验失败
  5  --deploy 失败
  6  --test 失败
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dst)    DST="${2:?--dst 需要参数}"; shift 2 ;;
        --deploy) DO_DEPLOY=1; shift ;;
        --test)   DO_TEST=1; DO_DEPLOY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ ! -d "$SRC/legal_rag" ]; then
    echo "!!! 工作区结构不对：$SRC 下没有 legal_rag/" >&2
    exit 1
fi

echo "=== 同步工作区到 WSL $(date -Is) ==="
echo "源   = $SRC"
echo "目标 = $DST"
echo "用户 = $(whoami)    python3 = $(python3 -V 2>&1)"
echo

# --- 清理历史遗留的"不该在 WSL 里"的目录 -------------------------------------
# 早期版本的排除表漏了这些；.ssh 尤其要删（工作区私钥不该在此多存一份）。
echo "--- [1/4] 清理历史遗留目录 ---"
for d in .ssh .idea .npm-cache .pip-cache .testtmp; do
    if [ -e "$DST/$d" ]; then rm -rf "$DST/$d"; echo "  已删除 $d"; fi
done
echo "  完成"
echo

# --- 同步 -------------------------------------------------------------------
echo "--- [2/4] rsync 同步（--delete，目标 == 工作区）---"
EXC=(
    --exclude=data --exclude=.venv --exclude=__pycache__ --exclude='*.pyc'
    --exclude=.tmp --exclude=logs --exclude=run --exclude=index --exclude=uploads
    --exclude=.pytest_cache --exclude=.pytmp --exclude=.pytest-tmp --exclude=.testtmp
    --exclude=.agent-teams --exclude=.idea --exclude=.npm-cache --exclude=.pip-cache
    --exclude=.ssh
)
mkdir -p "$DST"
if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "${EXC[@]}" "$SRC/" "$DST/"
else
    echo "  本机没有 rsync；改用 tar 管道（注意：这种写法不会删除目标里多余的文件）"
    ( cd "$SRC" && tar \
        --exclude='./data' --exclude='./.venv' --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='./.tmp' --exclude='./logs' --exclude='./run' --exclude='./index' \
        --exclude='./uploads' --exclude='./.pytest_cache' --exclude='./.pytmp' \
        --exclude='./.pytest-tmp' --exclude='./.testtmp' --exclude='./.agent-teams' \
        --exclude='./.idea' --exclude='./.npm-cache' --exclude='./.pip-cache' --exclude='./.ssh' \
        -cf - . ) | ( cd "$DST" && tar -xf - )
fi
rc=$?
if [ "$rc" -ne 0 ]; then echo "!!! 同步失败，退出码 $rc" >&2; exit 3; fi
echo "  rsync 退出码 = 0"
echo

# --- 只读校验 ---------------------------------------------------------------
echo "--- [3/4] 校验 ---"
n=$(cd "$DST" && find . -type f -not -path './.venv/*' -not -path '*/__pycache__/*' | wc -l)
echo "  目标文件数（不含 .venv/__pycache__）= $n"
miss=0
for f in legal_rag/config.py legal_rag/api/app.py legal_rag/generate/openai_compat.py \
         scripts/deploy.sh scripts/run_api.py run.sh shutdown.sh requirements.txt \
         tests/test_openai_compat.py docs/WSL.md docs/CLOUD.md; do
    if [ -f "$DST/$f" ]; then
        echo "  OK    $f"
    else
        echo "  缺失  $f"; miss=$((miss + 1))
    fi
done
if [ -e "$DST/.ssh" ]; then echo "  !!! .ssh 仍在（含工作区私钥，必须删掉）"; miss=$((miss + 1)); else echo "  .ssh 未带入 ✓"; fi
if [ -d "$DST/data" ]; then echo "  !!! data/ 被带入（不该同步）"; miss=$((miss + 1)); else echo "  data/ 未带入 ✓"; fi
if [ "$miss" -ne 0 ]; then echo "!!! 校验失败（$miss 项）" >&2; exit 4; fi
echo

# --- 可选：部署自检 / 跑测试 -------------------------------------------------
if [ "$DO_DEPLOY" -eq 1 ]; then
    echo "--- [4/4] 部署自检（scripts/deploy.sh）---"
    ( cd "$DST" && bash scripts/deploy.sh </dev/null ) || { echo "!!! deploy.sh 失败" >&2; exit 5; }
    echo
else
    echo "--- [4/4] 已跳过部署自检（未传 --deploy）---"
    echo "  如需：bash scripts/wsl_sync.sh --deploy"
    echo
fi

if [ "$DO_TEST" -eq 1 ]; then
    echo "--- 附加：pytest tests -q ---"
    ( cd "$DST" && .venv/bin/python -m pytest tests -q ) || { echo "!!! pytest 失败" >&2; exit 6; }
    echo
fi

echo "=== 完成：$DST 已与工作区一致 $(date -Is) ==="
