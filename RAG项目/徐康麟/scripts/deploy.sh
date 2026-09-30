#!/usr/bin/env bash
# ============================================================================
# deploy.sh —— 法律 RAG「一键部署 + 自检」脚本（在目标 Ubuntu / WSL 主机上运行）
#
# 它把 VM-DEPLOY.md「§2 一条命令完成『同步 + 刷依赖 + 自检』」里那条已被实测复跑过的
# 命令固化成脚本，并补上：版本门槛、可定位的报错、幂等、可选起服务。
# 目标机上的路径按 VM-DEPLOY.md §1 的口径是 ~/legal-rag（VM 用户 dshagent）。
#
# 依赖的事实来源（不要凭空改）：
#   * VM-DEPLOY.md §2.1  远端解字节码 + pip install -r requirements.txt + 五项导入自检
#   * VM-DEPLOY.md §3    刷新依赖命令：.venv/bin/python -m pip install -r requirements.txt
#   * VM-DEPLOY.md §4    五项导入自检：legal_rag.metrics / .observability / .logging_setup
#                        / .config / .system_metrics
#   * VM-DEPLOY.md §7 发现 1（实测坑）：远端 `systemctl start mysql` 会抢走 stdin 吃掉流程，
#                        所以起服务必须写 `bash run.sh < /dev/null`（< /dev/null 不能省）
#   * VM-DEPLOY.md §1    VM 上是 Python 3.10.12 + 项目自带 .venv
#
# 用法：
#   bash scripts/deploy.sh                    # 只做 5 步自检，不起服务
#   bash scripts/deploy.sh --start            # 自检完顺手起服务（内部 bash run.sh </dev/null）
#   bash scripts/deploy.sh --start --offline  # 起服务时走离线模式（无 key / 无 GPU）
#   bash scripts/deploy.sh --skip-deps        # 跳过刷依赖（虚拟环境已就绪、赶时间时用）
#   bash scripts/deploy.sh --help             # 看用法
#
# 在目标机上执行（VM-DEPLOY.md §1 里的 VM）：
#   ssh -i .ssh/dsh_ubuntu_ed25519 dshagent@<VM_IP>
#   cd ~/legal-rag && bash scripts/deploy.sh --start
#
# 退出码：0 全通过；1 项目结构不对（缺 legal_rag/）；2 缺 python3；3 python3 < 3.10；
#         4 建虚拟环境失败；5 刷依赖失败；6 清理字节码失败；7 导入自检未全过；8 起服务失败
#
# 幂等：可重复执行；已存在的 .venv 不会重建；pip 已是「已满足则不动」；再跑一遍结果一致。
# ============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 基础：定位脚本自身的绝对路径（用 BASH_SOURCE，避免被 cd 影响）
#
# 注意：严格模式（set -euo pipefail）已在**文件顶部**执行 —— 不支持 -u / -o pipefail
# 的老 bash 会在那里直接报错退出，**没有**「先打印一句人话再退出」的机会
# （早期注释曾如此声称，与实际执行顺序不符，已更正）。
# ---------------------------------------------------------------------------
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

VENV_DIR="$APP_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"
REQ_FILE="$APP_DIR/requirements.txt"

START=0
OFFLINE=0
SKIP_DEPS=0

usage() {
  cat <<'USAGE'
法律 RAG 一键部署 + 自检（在目标 Ubuntu / WSL 主机上运行）

用法：
  bash scripts/deploy.sh [选项]

选项：
  --start        自检全部通过后，用 `bash run.sh < /dev/null` 起服务
  --offline      起服务时附加 --offline（无密钥 / 无 GPU 时先跑通链路）
  --skip-deps    跳过第 [2/5] 步的依赖刷新（虚拟环境已装好、只想快速自检时用）
  -h, --help     显示本帮助

五步：
  [1/5] 检查 python3（要求 3.10+，打印实际版本）
  [2/5] 虚拟环境：没有 .venv 就创建；有则复用（用 .venv/bin/python -m pip 刷 requirements.txt）
  [3/5] 清理 __pycache__ / *.pyc（不碰 .venv，沿用 VM-DEPLOY.md §2.1 的 find 写法）
  [4/5] 五项导入自检：legal_rag.metrics / .observability / .logging_setup / .config / .system_metrics
  [5/5] 收尾（默认只打印后续命令；带 --start 则真正起服务）

示例：
  bash scripts/deploy.sh                 # 纯自检
  bash scripts/deploy.sh --start         # 自检 + 起服务
  bash scripts/deploy.sh --skip-deps     # 不刷依赖，快速自检

退出码：
  0=全通过 1=项目结构不对（缺 legal_rag/） 2=缺 python3 3=python3 版本过低
  4=建虚拟环境失败 5=刷依赖失败 6=清理字节码失败 7=导入自检未全过 8=起服务失败
USAGE
}

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --start)     START=1 ;;
    --offline)   OFFLINE=1 ;;
    --skip-deps) SKIP_DEPS=1 ;;
    -h|--help)   usage; exit 0 ;;
    *)
      echo "未知参数：$1" >&2
      echo "用 bash scripts/deploy.sh --help 查看用法。" >&2
      exit 2
      ;;
  esac
  shift
done

# 严格模式已在文件顶部启用（见文件头说明），此处不再重复 set：重复虽无害，但会让读者
# 误以为「老 bash 能在参数解析后再退出」—— 实际顺序是顶部先执行 set。

echo "=================================================================="
echo " 法律 RAG 部署自检"
echo " 项目目录：$APP_DIR"
echo " 开始时间：$(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

# ---------------------------------------------------------------------------
# [1/5] 检查 python3
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] 检查 python3（要求 3.10+）"
if ! command -v python3 >/dev/null 2>&1; then
  echo "[1/5] 失败：没有找到 python3。" >&2
  cat >&2 <<'HINT'
     修复建议（Ubuntu / WSL，二选一）：
       sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
       # 或者装 3.10/3.11/3.12 后确认 python3 -V 输出 >= 3.10
     若 python3 已装但不在 PATH：export PATH="/usr/bin:$PATH" 后再跑本脚本。
HINT
  exit 2
fi

PY3_BIN="$(command -v python3)"

if ! PY3_VERSION="$("$PY3_BIN" -c 'import platform; print(platform.python_version())' 2>/dev/null)"; then
  echo "[1/5] 失败：python3 存在但连版本都取不到（可能是损坏的解释器）。" >&2
  echo "      实际路径：$PY3_BIN" >&2
  echo "      手动确认：$PY3_BIN -V" >&2
  exit 3
fi

if ! "$PY3_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "[1/5] 失败：需要 Python 3.10+，当前是 $PY3_VERSION（路径 $PY3_BIN）。" >&2
  cat >&2 <<'HINT'
     修复建议（Ubuntu 22.04+ 自带 3.10，20.04 需要自行安装）：
       sudo apt-get install -y python3.10 python3.10-venv python3.10-pip
       # 或把 3.10+ 的解释器放到 PATH 最前面，并保证 `python3 -V` 是新版本
HINT
  exit 3
fi
echo "[1/5] OK：$PY3_BIN 版本 $PY3_VERSION"

if [[ ! -d "$APP_DIR/legal_rag" ]]; then
  echo "[1/5] 失败：项目目录下没有 legal_rag/，确认脚本放在 <项目根>/scripts/ 里。" >&2
  echo "      当前项目目录被识别为：$APP_DIR" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# [2/5] 虚拟环境 + 依赖
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] 虚拟环境与依赖"

install_deps() {
  local pip_py="$1"
  local out
  echo "      刷新依赖：$pip_py -m pip install -r requirements.txt"
  if ! out="$("$pip_py" -m pip install -r "$REQ_FILE" 2>&1)"; then
    echo "[2/5] 失败：pip 安装依赖未成功，下面是 pip 的原样输出（末 25 行）。" >&2
    printf '%s\n' "$out" | tail -n 25 >&2
    cat >&2 <<'HINT'
     修复建议（按出现顺序试）：
       1) 先升级 pip 再重试：
            <venv>/bin/python -m pip install -U pip setuptools wheel
       2) 若是离线环境（VM-DEPLOY.md §1 记录 VM 无外网依赖问题），加国内源：
            <venv>/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
       3) 若报「No module named venv / ensurepip」，说明系统缺 venv 包：
            sudo apt-get install -y python3-venv
       4) 磁盘满也会报错：df -h . 看一眼可用空间。
HINT
    return 1
  fi
  printf '%s\n' "$out" | tail -n 3
  return 0
}

if [[ -x "$VENV_PY" ]]; then
  echo "[2/5] 复用已有虚拟环境：$VENV_DIR"
  echo "      Python 版本：$("$VENV_PY" -V 2>&1)"
  if [[ "$SKIP_DEPS" -eq 1 ]]; then
    echo "      --skip-deps：跳过依赖刷新"
  else
    install_deps "$VENV_PY" || exit 5
  fi
else
  echo "[2/5] 未发现虚拟环境，正在创建：$VENV_DIR"
  if ! "$PY3_BIN" -m venv "$VENV_DIR"; then
    echo "[2/5] 失败：创建虚拟环境失败。" >&2
    cat >&2 <<'HINT'
     修复建议：
       sudo apt-get install -y python3-venv python3-pip
       <python3> -m venv .venv          # 手动复现看完整报错
       若 .venv 是「残留的不完整目录」，先删掉再重跑：
         rm -rf .venv && bash scripts/deploy.sh
HINT
    exit 4
  fi
  if [[ ! -x "$VENV_PY" ]]; then
    echo "[2/5] 失败：虚拟环境建好了但没有 $VENV_PY。" >&2
    echo "      修复建议：rm -rf .venv 后重跑；若仍失败，先 sudo apt-get install -y python3-venv" >&2
    exit 4
  fi
  echo "      创建完成：$("$VENV_PY" -V 2>&1)"
  install_deps "$VENV_PY" || exit 5
fi

# 依赖齐全的自证：五项自检里 config / system_metrics 会 import pydantic 等，
# 但为了把「依赖没装好」与「代码有问题」分开报，这里先单独确认 pydantic 可导入。
if ! "$VENV_PY" -c 'import pydantic' >/dev/null 2>&1; then
  echo "[2/5] 失败：虚拟环境里 import pydantic 就失败了 —— 这是依赖问题，不是代码问题。" >&2
  echo "      修复建议：$VENV_PY -m pip install -r $REQ_FILE" >&2
  echo "                （或加 --skip-deps 前后对比，确认是不是这一步没跑）" >&2
  exit 5
fi
echo "[2/5] OK：依赖可导入（pydantic 自证通过）"

# ---------------------------------------------------------------------------
# [3/5] 清理字节码
# 沿用 VM-DEPLOY.md §2.1 的写法：先把 .venv 剪掉，再删 __pycache__
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] 清理 __pycache__ / *.pyc（不动 .venv）"
if ! find "$APP_DIR" -path "$VENV_DIR" -prune -o -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null; then
  echo "[3/5] 失败：清理 __pycache__ 命令返回非零。" >&2
  cat >&2 <<'HINT'
     修复建议：
       1) 看是不是权限问题：ls -ld <项目根>/legal_rag   （owner 必须是当前用户）
       2) 手动重试：find . -path ./.venv -prune -o -name __pycache__ -type d -print
       3) 只读挂载也会这样：mount | grep " $(pwd) "  看有没有 ro
       注意：这一步失败不致命，修好后重跑本脚本即可（幂等）。
HINT
  exit 6
fi
# 顺带清掉散落的 .pyc（VM-DEPLOY.md §2.1 的排除项里含 *.pyc）
find "$APP_DIR" -path "$VENV_DIR" -prune -o -name '*.pyc' -type f -delete 2>/dev/null || true
echo "[3/5] OK：字节码已清理（.venv 保持原样）"

# ---------------------------------------------------------------------------
# [4/5] 五项导入自检（VM-DEPLOY.md §4）
# 用一个 python 进程逐个 import，任何一项失败都打印模块文件路径 + 异常类型，便于定位
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] 五项导入自检"

IMPORT_CHECK='
import importlib, sys, traceback
mods = [
    "legal_rag.metrics",
    "legal_rag.observability",
    "legal_rag.logging_setup",
    "legal_rag.config",
    "legal_rag.system_metrics",
]
ok = 0
failed = []
for name in mods:
    try:
        m = importlib.import_module(name)
    except BaseException as exc:
        failed.append(name)
        print("FAIL import %-28s -> %s: %s" % (name, type(exc).__name__, exc))
        traceback.print_exc()
    else:
        print("OK   import %-28s from %s" % (name, getattr(m, "__file__", "<built-in>")))
        ok += 1
print("IMPORT_SUMMARY ok=%d fail=%d %s" % (ok, len(failed), failed))
sys.exit(0 if not failed else 7)
'

IMPORT_OUT=""
IMPORT_RC=0
# 在项目根目录下执行：这样 `legal_rag` 包就在 sys.path[0]（即当前目录）里，能直接 import。
IMPORT_OUT="$("$VENV_PY" -c "$IMPORT_CHECK" 2>&1)" || IMPORT_RC=$?
printf '%s\n' "$IMPORT_OUT"

if [[ "$IMPORT_RC" -ne 0 ]]; then
  echo "[4/5] 失败：导入自检没有全过（python 退出码 $IMPORT_RC）。" >&2
  echo "      上面每条 FAIL 行里的「模块名 -> 异常类型: 异常信息」就是定位点；" >&2
  echo "      traceback 指向的就是缺文件/缺依赖的那一行。" >&2
  cat >&2 <<'HINT'
     修复建议：
       1) 若是 ModuleNotFoundError: No module named 'legal_rag.xxx'
          → 该文件根本没同步上来。在 Windows 侧重跑 scripts/sync_to_remote.ps1，
            再用「逐字节自检」确认 vm 缺少数 = 0。
       2) 若是 ModuleNotFoundError: No module named 'pydantic' / 'redis' / 'pymilvus'
          → 是依赖问题：<venv>/bin/python -m pip install -r requirements.txt
            （本脚本默认会做，若你加了 --skip-deps 就漏了这一步）
       3) 若是 SyntaxError
          → 同步过程把文件写坏了（例如 CRLF 或传输截断）：
            <venv>/bin/python -m py_compile <报错文件>
            并在 Windows 侧重跑同步（脚本已按 LF/UTF-8 处理，正常不会出现）。
       4) 若是 psutil / 其它可选依赖
          → 系统指标会降级但不应报错；把完整 traceback 贴给归属人。
HINT
  exit 7
fi
echo "[4/5] OK：五项全部导入成功"

# ---------------------------------------------------------------------------
# [5/5] 收尾：可选起服务
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] 收尾"

if [[ "$START" -eq 1 ]]; then
  echo "      起服务：bash run.sh < /dev/null   （离线=$OFFLINE）"
  echo "      注意：run.sh 里的 sudo systemctl start mysql 会抢 stdin，"
  echo "            所以这里的 < /dev/null 是实测必需的，不能省（VM-DEPLOY.md §7 发现 1）。"
  RUN_ARGS=()
  if [[ "$OFFLINE" -eq 1 ]]; then
    RUN_ARGS+=(--offline)
  fi
  if ! bash "$APP_DIR/run.sh" "${RUN_ARGS[@]}" </dev/null; then
    echo "[5/5] 失败：run.sh 未能把服务拉起来。" >&2
    cat >&2 <<'HINT'
     修复建议：
       1) 看日志尾部： tail -n 40 run/api.log
       2) 若提示「服务已在运行」，先停： bash shutdown.sh   然后重跑
       3) 若端口被占： ss -ltnp | grep ':8000 '   （VM-DEPLOY.md §1：8080 被 Open WebUI 占用，
          本项目默认 8000；可在 .env 里改 API_PORT）
       4) 想先排除依赖/网络因素： bash run.sh --offline < /dev/null
HINT
    exit 8
  fi
  PORT="${API_PORT:-8000}"
  echo "      健康检查：curl -s http://127.0.0.1:${PORT}/health"
else
  echo "      已跳过起服务（未传 --start）。"
  echo "      起服务命令（注意 < /dev/null）：bash run.sh < /dev/null"
fi

echo ""
echo "=================================================================="
echo " 部署自检完成：5/5 步通过（$(date '+%Y-%m-%d %H:%M:%S')）"
if [[ "$START" -eq 1 ]]; then
  echo " 服务已尝试启动，日志：$APP_DIR/run/api.log"
else
  echo " 下一步（可选）：bash scripts/deploy.sh --start"
fi
echo "=================================================================="
exit 0
