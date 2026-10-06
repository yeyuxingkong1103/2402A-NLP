#!/usr/bin/env bash
# ============================================================================
# deploy/install.sh —— **一键安装运行环境**（Linux / Ubuntu 为主）
#
# 它做什么（按顺序，每步都有日志）：
#   [1/6] 环境体检：Linux、python3 ≥ 3.10、磁盘余量与 pip 缓存目录
#   [2/6] pip 缓存与 TMPDIR 归位（**刻意放到不含模型的那个盘**）
#   [3/6] 建/复用虚拟环境 .venv
#   [4/6] 安装依赖（requirements.txt；--full 时再装 requirements-full.txt）
#   [5/6] 可选依赖：Redis / Ollama（--with-redis / --with-ollama；失败不致命但必须明说）
#   [6/6] 自检：五项导入 + pip check
#
# 用法：
#   bash deploy/install.sh                    # 装最小可跑集合（离线/mock 也能跑通）
#   bash deploy/install.sh --full             # 额外装 requirements-full（torch 等大件）
#   bash deploy/install.sh --with-redis --with-ollama
#   bash deploy/install.sh --dry-run          # 只打印要做什么，不动系统
#   bash deploy/install.sh --cache-dir /root/.cache/pip --tmp-dir /root/tmp
#   bash deploy/install.sh --help
#
# 退出码：0 成功；1 参数错；10 平台不对；11 缺 python3；12 python 版本过低；
#         13 磁盘空间不足；14 建虚拟环境失败；15 装依赖失败；16 自检未通过。
#
# 幂等：已存在的 .venv 不重建；pip 是"已满足则不动"；重复跑结果一致。
# 日志：logs/install-YYYYmmdd.log（目录可用 LEGAL_RAG_LOG_DIR 覆盖）。
#
# 事实来源（不要凭空改）：
#   * requirements.txt 是**最小可跑集合**（README 第 1 步）；
#   * requirements-full.txt 里是大件（torch / FlagEmbedding / chromadb / pymilvus 等）；
#   * 真机教训 N.6：pip cache 与 TMPDIR 若与 29G 模型同盘，安装峰值会写满数据盘，
#     报 `No space left on device`，而收尾清理后又"看着还有空间" ⇒ 必须预检 + 换盘。
# ============================================================================
set -euo pipefail

LR_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=deploy/lib.sh
source "$LR_LIB_DIR/lib.sh"
lr_init "install"

# ---------------------------------------------------------------------------
# 默认值（可被命令行覆盖）
# ---------------------------------------------------------------------------
WITH_FULL=0
WITH_REDIS=0
WITH_OLLAMA=0
PIP_INDEX_URL_DEFAULT="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
CACHE_DIR=""
TMP_DIR=""

usage() {
  cat <<'USAGE'
一键安装运行环境（Linux）

用法：bash deploy/install.sh [选项]

选项：
  --full                额外安装 requirements-full.txt（torch / FlagEmbedding 等大件）
  --with-redis          尝试安装并启动 Redis（缺了就自动降级为内存记忆，非必须）
  --with-ollama         尝试安装 Ollama（本地大模型/嵌入；没有也能用 OpenAI 兼容后端）
  --pip-index URL       pip 镜像地址（默认清华；国外机器可给官方源）
  --cache-dir DIR       pip 缓存目录（默认 ~/.cache/pip，**别放模型盘**）
  --tmp-dir DIR         临时目录 TMPDIR（默认 /tmp，**别放模型盘**）
  --dry-run             只打印将执行的操作，不改动系统
  -h, --help            显示本帮助

退出码：0 成功；1 参数错；10 平台不对；11 缺 python3；12 版本过低；
        13 空间不足；14 建虚拟环境失败；15 装依赖失败；16 自检未通过。
USAGE
}

# ---------------------------------------------------------------------------
# 解析参数：手写而不是 getopt —— getopt 在 macOS/BSD 与 GNU 上行为不同，
# 手写最可控，也能给出中文报错。
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full) WITH_FULL=1 ;;
    --with-redis) WITH_REDIS=1 ;;
    --with-ollama) WITH_OLLAMA=1 ;;
    --pip-index) shift; PIP_INDEX_URL_DEFAULT="${1:?--pip-index 需要值}" ;;
    --cache-dir) shift; CACHE_DIR="${1:?--cache-dir 需要值}" ;;
    --tmp-dir) shift; TMP_DIR="${1:?--tmp-dir 需要值}" ;;
    --dry-run) LR_DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; lr_error "未知参数：$1"; exit 1 ;;
  esac
  shift
done
export LR_DRY_RUN

cd "$LR_ROOT"
VENV_DIR="$LR_ROOT/.venv"
VENV_PY="$VENV_DIR/bin/python"

lr_section "法律 RAG 环境安装开始（项目根：$LR_ROOT）"
lr_snapshot_runtime "安装前"
[[ "$LR_DRY_RUN" == "1" ]] && lr_warn "当前是 --dry-run：只打印，不改动系统"

# ---------------------------------------------------------------------------
# [1/6] 环境体检
# ---------------------------------------------------------------------------
lr_section "[1/6] 环境体检"

# 平台：本脚本面向 Linux（/proc、apt、systemd 都按 Linux 写）。
# 在 macOS 上跑会缺 apt，明确报错比"跑到一半莫名失败"好。
case "$(uname -s 2>/dev/null || echo unknown)" in
  Linux) lr_ok "平台检查通过：Linux" ;;
  *) lr_warn "当前不是 Linux（$(uname -s 2>/dev/null || echo 未知)）："
     lr_warn "  本脚本按 Linux 写（apt / /proc / systemd）。生产环境请在 Linux 上跑；"
     lr_warn "  若只是本地试跑，请改用 Windows 侧的 .ps1 流程或 WSL。" ;;
esac

if ! lr_have python3; then
  lr_error "没有 python3。Ubuntu 安装：sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip"
  exit 11
fi
PY_VERSION="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PY_MAJOR="$(python3 -c 'import sys; print(sys.version_info[0])')"
PY_MINOR="$(python3 -c 'import sys; print(sys.version_info[1])')"
lr_info "python3 版本：$PY_VERSION（$(command -v python3)）"
if (( PY_MAJOR < 3 || (PY_MAJOR == 3 && PY_MINOR < 10) )); then
  lr_error "需要 Python 3.10+，当前 $PY_VERSION"
  exit 12
fi

# 磁盘：先把"数据盘/模型盘"和"pip 缓存盘"分开看，再决定装不装得下。
# 阈值取经验值：最小集合 ~1G；--full 里有 torch 等 ~6-8G。
NEED_GB=2
[[ "$WITH_FULL" == "1" ]] && NEED_GB=12
lr_need_gb "$LR_ROOT" "$NEED_GB" "项目所在盘（venv+依赖）" || exit 13

# ---------------------------------------------------------------------------
# [2/6] pip 缓存与 TMPDIR 归位
#   真机教训 N.6：这两处若与模型同盘，安装峰值写满盘 ⇒ 装到一半失败、
#   收尾后"看着还有空间"，非常难查。所以这里**显式指定并打印**。
# ---------------------------------------------------------------------------
lr_section "[2/6] pip 缓存与临时目录"

: "${CACHE_DIR:=$HOME/.cache/pip}"
: "${TMP_DIR:=/tmp}"
mkdir -p "$CACHE_DIR" "$TMP_DIR" 2>/dev/null || true
export PIP_CACHE_DIR="$CACHE_DIR"
export TMPDIR="$TMP_DIR"
lr_info "PIP_CACHE_DIR=$PIP_CACHE_DIR（可用 $(lr_free_gb "$PIP_CACHE_DIR" 2>/dev/null || echo '?')G）"
lr_info "TMPDIR=$TMPDIR（可用 $(lr_free_gb "$TMPDIR" 2>/dev/null || echo '?')G）"
if [[ "${PIP_CACHE_DIR%%/*}" == "${LR_ROOT%%/*}" ]]; then
  lr_warn "pip 缓存与项目在同一个挂载点；若项目盘同时放模型，安装峰值可能写满它"
fi

# ---------------------------------------------------------------------------
# [3/6] 虚拟环境
# ---------------------------------------------------------------------------
lr_section "[3/6] 虚拟环境"
if [[ -x "$VENV_PY" ]]; then
  lr_ok "复用已存在的虚拟环境：$VENV_DIR"
else
  lr_info "创建虚拟环境：$VENV_DIR"
  # 某些发行版的 venv 需要单独的 python3-venv 包，失败时给出可执行的补救命令。
  if ! lr_run python3 -m venv "$VENV_DIR"; then
    lr_error "创建虚拟环境失败。Ubuntu 上常见原因是缺 python3-venv："
    lr_error "  sudo apt-get install -y python3-venv"
    exit 14
  fi
  lr_ok_if_ran "虚拟环境创建完成"
fi
if [[ ! -x "$VENV_PY" ]]; then
  # ⚠️ 2026-09-29 真机自检抓到的缺陷：`--dry-run` **不会**真的建 venv，
  #    于是这一行在**新机器**上必然失败 ⇒ 整个干跑退出 14，起不到"部署前预检"的作用。
  #    所以干跑时把"产物还不存在"降级为 WARN 并继续（后续用 venv python 的步骤本来
  #    也只是打印）；**正式跑依旧是硬失败**（不许静默通过）。
  if [[ "$LR_DRY_RUN" == "1" ]]; then
    lr_warn "[dry-run] 虚拟环境尚不存在（正式跑会在上一步创建）：$VENV_PY"
    lr_info "[dry-run] 后续步骤按「虚拟环境已就绪」继续打印，不会真的执行"
  else
    lr_error "虚拟环境里没有可执行的 python：$VENV_PY"
    exit 14
  fi
fi

# ---------------------------------------------------------------------------
# [4/6] 安装依赖
# ---------------------------------------------------------------------------
lr_section "[4/6] 安装依赖"
PIP_ARGS=(--disable-pip-version-check --index-url "$PIP_INDEX_URL_DEFAULT")
lr_info "pip 源：$PIP_INDEX_URL_DEFAULT"

lr_run "$VENV_PY" -m pip install --upgrade pip "${PIP_ARGS[@]}" \
  || lr_warn "升级 pip 失败（不致命，继续装依赖）"

if ! lr_run "$VENV_PY" -m pip install -r "$LR_ROOT/requirements.txt" "${PIP_ARGS[@]}"; then
  lr_error "安装 requirements.txt 失败。常见原因与处理："
  lr_error "  1) 网络/镜像不可达 ⇒ 换 --pip-index，或用官方源"
  lr_error "  2) 空间不足 ⇒ 见上面第 [1] 步的读数，清缓存或换盘"
  lr_error "  3) 单个包编译失败 ⇒ 日志里有具体包名，先单独装它看完整报错"
  exit 15
fi
lr_ok_if_ran "requirements.txt 安装完成"

if [[ "$WITH_FULL" == "1" ]]; then
  lr_info "额外安装 requirements-full.txt（torch 等大件，可能需要十几分钟）"
  if ! lr_run "$VENV_PY" -m pip install -r "$LR_ROOT/requirements-full.txt" "${PIP_ARGS[@]}"; then
    lr_error "安装 requirements-full.txt 失败。注意：国内装 torch 建议用镜像的 torch 源，"
    lr_error "  或先只装需要的子集（真机实测 torch 是最容易卡住的那个）。"
    exit 15
  fi
  lr_ok_if_ran "requirements-full.txt 安装完成"
fi

# ---------------------------------------------------------------------------
# [5/6] 可选依赖（失败不致命，但**必须明说**"没装上、会走什么降级路径"）
# ---------------------------------------------------------------------------
lr_section "[5/6] 可选依赖"

if [[ "$WITH_REDIS" == "1" ]]; then
  if lr_have redis-server; then
    lr_ok "Redis 已安装：$(redis-server --version 2>/dev/null | head -c 60)"
  elif lr_have apt-get; then
    lr_run sudo apt-get update -y >/dev/null 2>&1 || lr_warn "apt-get update 失败，继续试装"
    lr_run sudo apt-get install -y redis-server || lr_warn "Redis 安装失败"
  else
    lr_warn "没有 apt-get，无法自动装 Redis；可跳过（代码会降级为内存记忆）"
  fi
  if lr_have redis-server; then
    # 起 Redis 只是为了"装完可用"；用 daemonize 不占前台。
    lr_run redis-cli ping >/dev/null 2>&1 || lr_run redis-server --daemonize yes || true
    lr_info "Redis 状态：$(redis-cli ping 2>/dev/null || echo '未就绪')"
  fi
else
  lr_info "跳过 Redis（未指定 --with-redis）。**没装也没关系**：短期记忆会降级为内存实现，"
  lr_info "  但**重启后会话窗口会丢**，生产建议装上。"
fi

if [[ "$WITH_OLLAMA" == "1" ]]; then
  if lr_have ollama; then
    lr_ok "Ollama 已安装：$(ollama --version 2>/dev/null | head -c 60)"
  else
    lr_info "安装 Ollama（官方脚本，需要能访问 ollama.com）"
    if ! lr_run bash -c 'curl -fsSL https://ollama.com/install.sh | sh'; then
      lr_warn "Ollama 安装失败（网络/权限）。可跳过："
      lr_warn "  * 用 OpenAI 兼容后端（OPENAI_COMPAT_BASE_URL 指向 vLLM/云 API）"
      lr_warn "  * 或离线模式（EMBEDDING_PROVIDER=offline + LLM_PROVIDER=mock）"
    fi
  fi
  lr_have ollama && lr_info "Ollama 模型列表：$(ollama list 2>/dev/null | head -3 | tr '\n' ' ')"
else
  lr_info "跳过 Ollama（未指定 --with-ollama）"
fi

# ---------------------------------------------------------------------------
# [6/6] 自检
#   五项导入取自 VM-DEPLOY.md §4（历史验收用过的组合），再补一个 pip check。
# ---------------------------------------------------------------------------
lr_section "[6/6] 自检"
if [[ "$LR_DRY_RUN" == "1" ]]; then
  # ⚠️ `lr_run` 在干跑时只打印并返回 0 ⇒ 若照原样走下去会打印"自检通过"，
  #    而**根本没有 import 过任何模块**（虚报通过比不报更糟，本项目明令禁止）。
  lr_info "[dry-run] 跳过五项导入自检与 pip check（正式跑会真跑它们）"
  lr_info "[dry-run] 注意：干跑只证明「流程与参数」没问题，**不**证明依赖装好了"
else
  if lr_run "$VENV_PY" - <<'PYCHECK'
# 导入自检：能 import 通说明依赖没缺关键件（比"pip 说装好了"更可信）
import importlib
mods = ["legal_rag.metrics", "legal_rag.observability", "legal_rag.logging_setup",
        "legal_rag.config", "legal_rag.system_metrics"]
for name in mods:
    importlib.import_module(name)
    print(f"  [OK] import {name}")
PYCHECK
  then
    lr_ok "五项导入自检通过"
  else
    lr_error "导入自检失败：依赖可能缺了关键件（上面有具体模块名）"
    exit 16
  fi

  if lr_run "$VENV_PY" -m pip check; then
    lr_ok "pip check 通过（没有版本冲突）"
  else
    lr_warn "pip check 报出依赖冲突：不一定致命，但请看清是哪个包"
  fi
fi

lr_run "$VENV_PY" --version >/dev/null 2>&1 || true
if [[ -x "$VENV_PY" ]]; then
  lr_info "Python 版本：$("$VENV_PY" --version 2>&1)"
else
  # 干跑且没建 venv：不要去执行一个不存在的解释器（否则日志里会出现一行莫名的
  # "No such file or directory"，看起来像安装失败）。
  lr_info "[dry-run] 虚拟环境未创建，跳过 python 版本回读"
fi

# ---------------------------------------------------------------------------
# 收尾
# ---------------------------------------------------------------------------
lr_section "安装完成"
if [[ -x "$VENV_PY" ]]; then
  lr_ok "环境就绪：$VENV_PY"
else
  # 只有干跑会走到这里（正式跑在 [3/6] 没建出 venv 就已经退出 14 了）
  lr_warn "[dry-run] 干跑结束：环境**尚未**就绪（正式跑会建 .venv 并装依赖）"
fi
lr_info "下一步："
lr_info "  1) 生成配置：bash deploy/deploy.sh            # 会从 .env.example 生成 .env"
lr_info "  2) 起服务：  bash deploy/start.sh             # 默认离线/mock 也能起"
lr_info "  3) 停服务：  bash deploy/stop.sh"
lr_info "日志：$LR_LOG_FILE"
