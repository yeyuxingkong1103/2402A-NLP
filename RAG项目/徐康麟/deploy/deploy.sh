#!/usr/bin/env bash
# ============================================================================
# deploy/deploy.sh —— **一键部署**：把项目"放到位并初始化"（**不启动服务**）
#
# 与 deploy/install.sh 的分工：
#   * install.sh 负责"机器上有没有 Python 环境/依赖"；
#   * deploy.sh  负责"项目文件、目录、配置、索引"。
#   两者都幂等，可以先跑 deploy.sh（它会提示你先跑 install.sh），也可以反过来。
#
# 它做什么：
#   [1/6] 结构与版本校验（legal_rag/ 在不在、python3 版本）
#   [2/6] 运行目录准备（index/ logs/ run/ uploads/ data/）
#   [3/6] 配置：从 .env.example 生成 .env（**绝不覆盖已有**，除非 --force-env）
#   [4/6] 依赖就绪检查（有 .venv 就复用；没有就提示跑 install.sh）
#   [5/6] 建知识库索引（可选：--index；--rebuild 强制重建）
#   [6/6] 部署后自检 + 打印下一步
#
# 用法：
#   bash deploy/deploy.sh                      # 只做结构与配置准备
#   bash deploy/deploy.sh --index              # 顺便建索引（离线/mock 也能建）
#   bash deploy/deploy.sh --index --rebuild    # 强制重建索引
#   bash deploy/deploy.sh --archive pkg.tgz --dir /opt/legal-rag   # 从压缩包部署到指定目录
#   bash deploy/deploy.sh --dry-run
#
# 退出码：0 成功；1 参数错；20 项目结构不对；21 缺 python3；22 生成配置失败；
#         23 依赖未就绪（请先跑 install.sh）；24 建索引失败。
#
# 日志：logs/deploy-YYYYmmdd.log（目录可用 LEGAL_RAG_LOG_DIR 覆盖）。
# ============================================================================
set -euo pipefail

LR_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=deploy/lib.sh
source "$LR_LIB_DIR/lib.sh"
lr_init "deploy"

TARGET_DIR=""
ARCHIVE=""
DO_INDEX=0
REBUILD=0
ROLE_ID=""
FORCE_ENV=0
SKIP_DEPS=1          # 默认**不**自动装依赖：安装是会改系统的动作，应该显式要求
SOURCE_DIR="$LR_ROOT"

usage() {
  cat <<'USAGE'
一键部署（Linux）：准备目录、生成配置、可选建索引。**不启动服务**（用 deploy/start.sh）。

用法：bash deploy/deploy.sh [选项]

选项：
  --dir DIR             部署到的目录（默认=本仓库所在目录；配合 --archive 时是新目标）
  --archive FILE        从一个 .tar.gz 部署（解包到 --dir）
  --index               建知识库索引（默认不建）
  --rebuild             建索引时强制重建（与 --index 同用）
  --role ROLE           索引归属的角色 role_id（默认用 .env 里的 DEFAULT_ROLE_ID）
  --env-file FILE       用指定的配置文件当模板（默认 .env.example）
  --force-env           **覆盖**已存在的 .env（默认绝不覆盖；覆盖前会备份成 .env.bak.<时间>）
  --install-deps        部署前顺手跑 deploy/install.sh（会改动系统，默认不做）
  --dry-run             只打印将执行的操作
  -h, --help            显示本帮助

退出码：0 成功；1 参数错；20 项目结构不对；21 缺 python3；22 生成配置失败；
        23 依赖未就绪；24 建索引失败。
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir) shift; TARGET_DIR="${1:?--dir 需要值}" ;;
    --archive) shift; ARCHIVE="${1:?--archive 需要值}" ;;
    --index) DO_INDEX=1 ;;
    --rebuild) REBUILD=1; DO_INDEX=1 ;;
    --role) shift; ROLE_ID="${1:?--role 需要值}" ;;
    --env-file) shift; ENV_TEMPLATE="${1:?--env-file 需要值}" ;;
    --force-env) FORCE_ENV=1 ;;
    --install-deps) SKIP_DEPS=0 ;;
    --dry-run) LR_DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; lr_error "未知参数：$1"; exit 1 ;;
  esac
  shift
done
export LR_DRY_RUN

: "${ENV_TEMPLATE:=$SOURCE_DIR/.env.example}"
: "${TARGET_DIR:=$SOURCE_DIR}"

# 安全闸门：`--archive` + 默认目标目录 = 把压缩包**解到源码树里**（会覆盖工作副本）。
# 这是很容易手滑的组合，所以直接拦下来，要求显式给 --dir。
if [[ -n "$ARCHIVE" && "$TARGET_DIR" == "$SOURCE_DIR" ]]; then
  lr_error "--archive 与默认目标目录相同：会把压缩包解到**当前源码树**里（覆盖工作副本）。"
  lr_error "  请显式指定落地目录，例如：--dir /opt/legal-rag"
  exit 1
fi

lr_section "一键部署开始"
lr_info "源目录：$SOURCE_DIR"
lr_info "目标目录：$TARGET_DIR"
[[ -n "$ARCHIVE" ]] && lr_info "压缩包：$ARCHIVE"
[[ "$LR_DRY_RUN" == "1" ]] && lr_warn "当前是 --dry-run：只打印，不改动磁盘"

# ---------------------------------------------------------------------------
# [1/6] 结构与版本校验
# ---------------------------------------------------------------------------
lr_section "[1/6] 结构与版本校验"
if [[ ! -d "$SOURCE_DIR/legal_rag" ]]; then
  lr_error "在 $SOURCE_DIR 下找不到 legal_rag/ —— 这不是项目根目录。"
  lr_error "  请把脚本放在仓库根的 deploy/ 下运行，或用 --archive 指定压缩包。"
  exit 20
fi
lr_ok "结构检查通过：$SOURCE_DIR/legal_rag 存在"

if ! lr_have python3; then
  lr_error "没有 python3：先跑 bash deploy/install.sh（或 apt-get install python3 python3-venv）"
  exit 21
fi
lr_info "python3：$(python3 -V 2>&1)（$(command -v python3)）"

# ---------------------------------------------------------------------------
# [2/6] 目录准备（幂等：已存在就复用）
#   这几个目录都是**运行时会写**的，提前建好并说明用途，避免部署后第一次跑才发现没权限。
# ---------------------------------------------------------------------------
lr_section "[2/6] 运行目录"
for sub in index logs run uploads data; do
  if [[ -d "$TARGET_DIR/$sub" ]]; then
    lr_info "  已存在：$sub/"
  else
    lr_run mkdir -p "$TARGET_DIR/$sub"
    lr_info "  已创建：$sub/"
  fi
done
lr_need_gb "$TARGET_DIR" 1 "索引/日志/上传落盘" || true

# 如果用的是压缩包部署：解包到目标目录（--strip-components=1 去掉顶层目录名）
if [[ -n "$ARCHIVE" ]]; then
  lr_section "[2b/6] 从压缩包解包"
  [[ -f "$ARCHIVE" ]] || lr_die "压缩包不存在：$ARCHIVE"
  lr_run mkdir -p "$TARGET_DIR"
  # 先看清单再解包：让日志里留下"这次到底部署了哪些文件"的证据。
  lr_info "压缩包内容（前 10 条）："
  # ⚠️ `tar | head` 在 `set -o pipefail` 下会被 **SIGPIPE 判成失败**（head 读满 10 行就退出，
  #    tar 下一次写拿到 EPIPE、退出码 141）⇒ 再被 `set -e` 一拦，`--archive` 模式**中途退出**。
  #    这里显式吞掉这个"预期内"的失败：它只影响日志展示，下一行的解包照常执行。
  tar -tzf "$ARCHIVE" 2>/dev/null | head -10 | while read -r line; do lr_info "    $line"; done || true
  lr_run tar -xzf "$ARCHIVE" -C "$TARGET_DIR" --strip-components=1
  # ⚠️ **权限归一化**（2026-09-29 真机踩到）：在 Windows 上打的包（`tar.exe`）里
  #    每个文件的 mode 是 **0666**（world-writable），解到 Linux 上就是
  #    `-rw-rw-rw-` —— 任何人可改部署脚本与代码，而 `tests/test_recall_ab_driver.py`
  #    里那条"脚本不该 world-writable"守卫会直接变红。
  #    这里解包后立刻收回 group/other 的写位，并把该可执行的恢复成 755。
  if [[ "$LR_DRY_RUN" != "1" ]]; then
    chmod -R go-w "$TARGET_DIR" 2>/dev/null || lr_warn "chmod go-w 失败（检查部署目录权限）"
    # shellcheck disable=SC2038
    find "$TARGET_DIR/scripts" "$TARGET_DIR/deploy" -maxdepth 1 -name '*.sh' \
      -exec chmod 755 {} + 2>/dev/null || true
    for extra in run.sh shutdown.sh; do
      [[ -f "$TARGET_DIR/$extra" ]] && chmod 755 "$TARGET_DIR/$extra"
    done
    lr_info "已归一化权限：world-writable 位已收回，*.sh 恢复 755"
  fi
  lr_ok_if_ran "解包完成：$TARGET_DIR"
fi

# ---------------------------------------------------------------------------
# [3/6] 配置：.env
#   原则：**绝不覆盖已有配置**（那会悄悄改掉线上行为）。要覆盖必须显式 --force-env，
#   且先备份。生成后把"必须人工确认的项"列出来 —— 尤其 AUTH_REQUIRED（对外发布的闸门）。
# ---------------------------------------------------------------------------
lr_section "[3/6] 配置（.env）"
ENV_FILE="$TARGET_DIR/.env"
if [[ -f "$ENV_FILE" && "$FORCE_ENV" != "1" ]]; then
  lr_ok "已存在 $ENV_FILE —— **保持不动**（要覆盖用 --force-env，会先备份）"
else
  if [[ ! -f "$ENV_TEMPLATE" ]]; then
    lr_error "找不到配置模板：$ENV_TEMPLATE"
    exit 22
  fi
  if [[ -f "$ENV_FILE" ]]; then
    BACKUP="$ENV_FILE.bak.$(date +%Y%m%d-%H%M%S)"
    lr_warn "--force-env：先把现有配置备份到 $BACKUP"
    lr_run cp -p "$ENV_FILE" "$BACKUP"
  fi
  lr_run cp -p "$ENV_TEMPLATE" "$ENV_FILE"
  lr_ok "已从 $ENV_TEMPLATE 生成 $ENV_FILE"
fi
lr_warn "请人工确认这几项（它们决定「能不能对外」与「用哪个模型」）："
lr_info "  AUTH_REQUIRED=true        —— 交付/对外必须打开（代码默认 false）"
lr_info "  LLM_PROVIDER / OPENAI_COMPAT_BASE_URL / LLM_MODEL  —— 指向你的大模型服务"
lr_info "  VECTOR_STORE / EMBEDDING_PROVIDER / RERANK_PROVIDER —— 生产建议 milvus + ollama(bge-m3)"
lr_info "  API_HOST / API_PORT       —— 对外服务时 API_HOST 常要设 0.0.0.0"
lr_info "  （配置文件：$ENV_FILE）"

# ---------------------------------------------------------------------------
# [4/6] 依赖就绪
# ---------------------------------------------------------------------------
lr_section "[4/6] 依赖就绪"
VENV_PY="$TARGET_DIR/.venv/bin/python"
if [[ "$SKIP_DEPS" == "0" ]]; then
  lr_info "--install-deps：调用 deploy/install.sh"
  lr_run bash "$LR_LIB_DIR/install.sh" || lr_die "install.sh 失败"
fi
if [[ -x "$VENV_PY" ]]; then
  lr_ok "虚拟环境就绪：$VENV_PY（$("$VENV_PY" -V 2>&1)）"
else
  # ⚠️ 2026-09-29 真机自检抓到的缺陷：`--dry-run` 是**部署前预检**，而依赖这一步
  #    正是"预检"要报告的东西之一。新机器上（还没跑 install.sh）干跑必然缺 .venv，
  #    若在这里直接 exit 23，干跑永远不可能成功。⇒ 干跑降级为 WARN 并继续，
  #    **正式跑依旧是硬失败**（提示里给出可执行的补救命令）。
  if [[ "$LR_DRY_RUN" == "1" ]]; then
    lr_warn "[dry-run] 虚拟环境尚不存在：$VENV_PY"
    lr_info "[dry-run] 正式跑前需要先执行：bash deploy/install.sh（或给本脚本加 --install-deps）"
    lr_info "[dry-run] 后续步骤按「依赖已就绪」继续打印"
  else
    lr_error "没有可用的虚拟环境：$VENV_PY"
    lr_error "  先跑：bash deploy/install.sh        （或加 --install-deps 让本脚本代跑）"
    exit 23
  fi
fi

# ---------------------------------------------------------------------------
# [5/6] 建索引（可选）
#   ⚠️ 不建索引也能起服务，但**检索会命中 0 条** —— 所以这里给出明确提示，
#      而不是让用户以为"部署完了就能问"。
# ---------------------------------------------------------------------------
lr_section "[5/6] 知识库索引"
if [[ "$DO_INDEX" != "1" ]]; then
  lr_warn "跳过建索引（未指定 --index）。**跳过不影响起服务，但检索会 0 命中。**"
  lr_info "  需要时执行：bash deploy/deploy.sh --index --rebuild"
else
  IDX_ARGS=(scripts/build_index.py --source knowledge/)
  [[ "$REBUILD" == "1" ]] && IDX_ARGS+=(--rebuild)
  [[ -n "$ROLE_ID" ]] && IDX_ARGS+=(--role-id "$ROLE_ID")
  # 离线模式：内存向量库 + 零依赖向量化 ⇒ 没有 GPU / 没有 Ollama 也能把索引建起来。
  # 生产上真实链路请用 --provider milvus --embedding ollama（见 .env 的配置项）。
  IDX_ARGS+=(--offline)
  lr_info "建索引命令：$VENV_PY ${IDX_ARGS[*]}"
  if ! lr_run "$VENV_PY" "${IDX_ARGS[@]}"; then
    lr_error "建索引失败（上面有具体报错）。常见原因："
    lr_error "  1) knowledge/ 为空 ⇒ 先把语料放进去"
    lr_error "  2) 维度冲突 ⇒ 换 --collection 或清理旧库（**别乱删别人的表**）"
    lr_error "  3) 嵌入后端不可达 ⇒ 先 --offline 跑通链路"
    exit 24
  fi
  lr_ok_if_ran "索引建立完成"
fi

# ---------------------------------------------------------------------------
# [6/6] 部署后自检
# ---------------------------------------------------------------------------
lr_section "[6/6] 部署后自检"
if [[ "$LR_DRY_RUN" == "1" ]]; then
  # 干跑不许打印"自检通过"——它根本没跑（虚报通过比不报更糟）
  lr_info "[dry-run] 跳过部署后自检（正式跑会 import legal_rag.config / legal_rag.api.app）"
else
  if lr_run "$VENV_PY" - <<'PYCHECK'
import importlib, sys
for name in ("legal_rag.config", "legal_rag.api.app"):
    importlib.import_module(name)
print("  [OK] 关键模块可导入")
PYCHECK
  then
    lr_ok "自检通过"
  else
    lr_error "自检失败：项目文件可能不完整"
    exit 20
  fi
fi

lr_snapshot_runtime "部署后"
if [[ "$LR_DRY_RUN" == "1" ]]; then
  # 干跑只报告"会做什么"与"缺什么"，不冒充"已部署完成"
  lr_section "干跑结束（未改动磁盘）"
  lr_info "上面若出现 [dry-run] WARN（例如缺 .venv），请先补齐再正式跑本脚本。"
else
  lr_section "部署完成"
fi
lr_info "下一步："
lr_info "  1) 编辑配置：${EDITOR:-vi} $ENV_FILE"
lr_info "  2) 起服务：  bash deploy/start.sh                 # 离线试跑"
lr_info "               bash deploy/start.sh --with-vllm ...  # 同时拉起 vLLM"
lr_info "  3) 探活：    curl -s http://127.0.0.1:\${API_PORT:-8000}/livez"
lr_info "日志：$LR_LOG_FILE"
