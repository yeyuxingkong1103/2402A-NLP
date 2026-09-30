#!/usr/bin/env bash
# ============================================================
# install.sh —— 环境安装脚本（第 11 步实现）
# 作用：在全新 Linux 机器上，把运行本项目所需的依赖装齐
# 支持系统：Ubuntu / Debian（apt 系）、CentOS / RHEL（yum 系）
# 用法：
#   bash scripts/install.sh              # 真正安装
#   bash scripts/install.sh --dry-run    # 只打印将要执行的命令，不实际安装
# ============================================================

set -euo pipefail                 # 任一命令失败即退出；未定义变量报错；管道任一段失败即失败

# ---------- 全局变量 ----------
DRY_RUN=0                         # 干跑开关：1 只打印命令，0 真正执行
ENV_NAME="power_rag"              # conda 环境名
PY_VERSION="3.10"                 # conda 环境的 Python 版本
MINICONDA_DIR="$HOME/miniconda3"  # Miniconda 安装目录
# 发行版信息文件路径；允许用环境变量覆盖，便于在非目标系统（如本机 Git Bash）上演示检测流程
OS_RELEASE_FILE="${OS_RELEASE_FILE:-/etc/os-release}"
PKG_MGR=""                        # 包管理器，探测后填 apt 或 yum
OS_ID=""                          # 系统标识，从 /etc/os-release 读取
OS_LIKE=""                        # 系统兼容标识，Debian 系常写 debian

cd "$(dirname "$0")/.."           # 切到项目根目录，后面统一用相对路径
PROJECT_DIR="$(pwd)"              # 项目根目录的绝对路径

# ---------- 参数解析 ----------
for arg in "$@"; do               # 逐个处理命令行参数
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;       # 打开干跑模式
    -h|--help)                    # 打印用法后退出
      echo "用法：bash scripts/install.sh [--dry-run]"
      exit 0 ;;
    *) echo "未知参数：$arg（可用 --dry-run / --help）" >&2; exit 2 ;;   # 不认识的参数直接报错
  esac
done

# ---------- 输出与执行辅助 ----------
info() { echo -e "\033[32m[信息]\033[0m $*"; }        # 绿色：正常进度
warn() { echo -e "\033[33m[警告]\033[0m $*"; }        # 黄色：只警告不中断
err()  { echo -e "\033[31m[错误]\033[0m $*" >&2; }    # 红色：错误信息，输出到 stderr

run() {                                               # 执行命令：干跑模式只打印
  if [ "$DRY_RUN" -eq 1 ]; then                       # 干跑模式
    printf '  [dry-run] '                             # 前缀
    printf '%q ' "$@"                                 # %q 会把含空格的参数转义，打印出来可直接复制执行
    echo                                              # 换行收尾
  else                                                # 正常模式
    "$@"                                              # 真正执行，参数按原样传递
  fi
}

has_cmd() { command -v "$1" >/dev/null 2>&1; }        # 命令是否存在：存在返回 0

step() { echo; echo "===== $* ====="; }               # 打印带分隔的步骤标题

# ============================================================
# 第 1 步：识别操作系统
# ============================================================
step "1/10 识别操作系统"

if [ ! -f "$OS_RELEASE_FILE" ]; then                  # 没有该文件说明不是主流 Linux 发行版
  err "找不到 $OS_RELEASE_FILE，无法识别系统；本脚本只支持 Ubuntu / Debian / CentOS / RHEL"
  exit 1
fi

# shellcheck disable=SC1091
. "$OS_RELEASE_FILE"                                  # 读入发行版信息，得到 ID / ID_LIKE / PRETTY_NAME
OS_ID="${ID:-}"                                       # 发行版标识，如 ubuntu / debian / centos / rhel
OS_LIKE="${ID_LIKE:-}"                                # 兼容标识，如 debian / rhel fedora
info "检测到系统：${PRETTY_NAME:-$OS_ID}"              # 打印人类可读的系统名

case "$OS_ID" in                                      # 按发行版选包管理器
  ubuntu|debian|linuxmint|pop) PKG_MGR="apt" ;;       # Debian 系用 apt
  centos|rhel|rocky|almalinux|fedora) PKG_MGR="yum" ;; # RHEL 系用 yum
  *)
    case "$OS_LIKE" in                                # ID 认不出时再看 ID_LIKE
      *debian*) PKG_MGR="apt" ;;                      # 兼容 Debian 的都用 apt
      *rhel*|*fedora*) PKG_MGR="yum" ;;               # 兼容 RHEL 的都用 yum
      *)
        err "不支持的系统：$OS_ID（本脚本只支持 Ubuntu / Debian / CentOS / RHEL）"
        exit 1 ;;
    esac ;;
esac
info "包管理器：$PKG_MGR"                              # 打印选定结果

# ============================================================
# 第 2 步：检查 root 或 sudo 权限
# ============================================================
step "2/10 检查管理员权限"

SUDO=""                                               # 需要提权时存放 "sudo"，否则为空
if [ "$(id -u)" -eq 0 ]; then                         # 当前就是 root
  info "当前用户是 root，无需 sudo"
elif has_cmd sudo; then                               # 普通用户但装了 sudo
  SUDO="sudo"                                         # 后续系统级命令统一加 sudo
  info "当前用户不是 root，将使用 sudo 提权"
else                                                  # 既不是 root 也没有 sudo
  err "既不是 root 也没有 sudo，无法安装系统依赖；请用 root 或安装 sudo 后重试"
  exit 1
fi

# ============================================================
# 第 3 步：安装系统基础依赖
# ============================================================
step "3/10 安装系统基础依赖"

if [ "$PKG_MGR" = "apt" ]; then                       # Debian 系
  run $SUDO apt-get update -y                         # 先更新软件源索引
  run $SUDO apt-get install -y curl wget git build-essential   # 装编译工具链与下载工具
else                                                  # RHEL 系
  run $SUDO yum install -y curl wget git gcc gcc-c++  # 装编译工具链与下载工具
fi
info "系统基础依赖处理完成"

# ============================================================
# 第 4 步：检查并安装 Miniconda
# ============================================================
step "4/10 检查并安装 Miniconda"

if [ -x "$MINICONDA_DIR/bin/conda" ]; then            # 已有可执行的 conda
  info "已安装 Miniconda：$MINICONDA_DIR，跳过安装"
else                                                  # 没装则下载安装
  info "未检测到 Miniconda，准备安装到 $MINICONDA_DIR"
  # 用 Miniconda 官方安装脚本静默安装；-b 批处理、-p 指定目录
  run bash -c "curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && bash /tmp/miniconda.sh -b -p '$MINICONDA_DIR'"
fi

# 把 conda 初始化进当前 shell，后面才能直接用 conda 命令
if [ "$DRY_RUN" -eq 0 ]; then                         # 干跑模式不改变当前环境
  # shellcheck disable=SC1091
  eval "$("$MINICONDA_DIR/bin/conda" shell.bash hook)" # 加载 conda 的 shell 钩子
fi
if [ "$DRY_RUN" -eq 1 ]; then                         # 干跑模式并没有真的装
  info "conda 将就绪（dry-run 未实际安装）"            # 如实说明
else                                                  # 正常模式
  info "conda 已就绪"                                 # 已完成
fi

# ============================================================
# 第 5 步：创建 conda 环境
# ============================================================
step "5/10 创建 conda 环境 $ENV_NAME（Python $PY_VERSION）"

if [ "$DRY_RUN" -eq 0 ] && conda env list | grep -qE "^${ENV_NAME}\s"; then   # 环境已存在
  info "conda 环境 $ENV_NAME 已存在，跳过创建"
else                                                  # 不存在则创建
  run conda create -y -n "$ENV_NAME" "python=$PY_VERSION"     # 建环境并指定 Python 版本
fi

# ============================================================
# 第 6 步：安装 Python 依赖
# ============================================================
step "6/10 安装 Python 依赖（requirements.txt）"

if [ "$DRY_RUN" -eq 1 ]; then                         # 干跑模式：只打印将要执行的命令
  echo "  [dry-run] conda activate $ENV_NAME"
  echo "  [dry-run] pip install -r requirements.txt"
else                                                  # 正常模式
  # shellcheck disable=SC1091
  eval "$("$MINICONDA_DIR/bin/conda" shell.bash hook)" # 再次确保 conda 可用
  conda activate "$ENV_NAME"                          # 激活项目环境
  info "已激活环境：$CONDA_DEFAULT_ENV"
  pip install -r "$PROJECT_DIR/requirements.txt"      # 安装项目依赖
fi
info "Python 依赖处理完成"

# ============================================================
# 第 7 步：检查并安装 Redis
# ============================================================
step "7/10 检查并安装 Redis"

if has_cmd redis-server || has_cmd redis-cli; then    # 已装 Redis
  info "Redis 已安装，跳过安装"
else                                                  # 未装则安装
  if [ "$PKG_MGR" = "apt" ]; then                     # Debian 系包名
    run $SUDO apt-get install -y redis-server
  else                                                # RHEL 系包名
    run $SUDO yum install -y redis
  fi
fi

# 开机自启并立刻启动；CentOS 的单元名是 redis，Ubuntu 是 redis-server，两个都试
run bash -c "$SUDO systemctl enable redis-server 2>/dev/null || $SUDO systemctl enable redis 2>/dev/null || true"
run bash -c "$SUDO systemctl start  redis-server 2>/dev/null || $SUDO systemctl start  redis 2>/dev/null || true"
info "Redis 已设置开机自启并启动（默认端口 6379）"

# ============================================================
# 第 8 步：检查并安装 MySQL
# ============================================================
step "8/10 检查并安装 MySQL"

if has_cmd mysql; then                                # 已装 MySQL 客户端
  info "MySQL 已安装，跳过安装"
else                                                  # 未装则安装
  if [ "$PKG_MGR" = "apt" ]; then                     # Debian 系包名
    run $SUDO apt-get install -y mysql-server
  else                                                # RHEL 系包名
    run $SUDO yum install -y mysql-server
  fi
fi

run bash -c "$SUDO systemctl enable mysqld 2>/dev/null || $SUDO systemctl enable mysql 2>/dev/null || true"
run bash -c "$SUDO systemctl start  mysqld 2>/dev/null || $SUDO systemctl start  mysql 2>/dev/null || true"
info "MySQL 已设置开机自启并启动（默认端口 3306）"

# 建库需要密码，脚本不替用户决定，这里只把命令打出来让用户自己执行
echo
warn "MySQL 需要手动创建数据库，请执行下面这条命令（会提示输入 MySQL 密码）："
echo "    mysql -u root -p -e \"CREATE DATABASE rag CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;\""
echo "  建完库后，把 .env 里的 MYSQL_USER / MYSQL_PASSWORD / MYSQL_DB 填成对应值。"
echo

# ============================================================
# 第 9 步：提示 Milvus 安装方式（不自动装 Docker）
# ============================================================
step "9/10 Milvus 向量库（需 Docker，本脚本不自动安装 Docker）"

warn "本脚本不自动安装 Docker；请先自行装好 Docker，再执行下面这条命令启动 Milvus："
echo
echo "    docker run -d --name milvus-standalone \\"
echo "      -p 19530:19530 -p 9091:9091 \\"
echo "      milvusdb/milvus:v2.6.2 standalone"
echo
info "说明：镜像名 milvusdb/milvus，容器名 milvus-standalone，端口 19530（SDK 连接）+ 9091（健康检查）"
info "容器已存在时用 docker start milvus-standalone 即可"

# ============================================================
# 第 10 步：打印安装结果与下一步
# ============================================================
step "10/10 安装结果汇总"

# 逐项探测并在终端打出状态；干跑模式一律标成"未验证"
check_status() {                                      # 入参：显示名、探测命令名
  local label="$1" cmd="$2"                           # 显示名与命令名
  if [ "$DRY_RUN" -eq 1 ]; then                       # 干跑模式不探测
    printf "  %-12s %s\n" "$label" "未验证（dry-run）"
  elif has_cmd "$cmd"; then                           # 命令存在即视为已安装
    printf "  %-12s %s\n" "$label" "已安装（$(command -v "$cmd")）"
  else                                                # 没找到命令
    printf "  %-12s %s\n" "$label" "未安装"
  fi
}
check_status "conda"  conda                           # conda
check_status "redis"  redis-server                    # Redis
check_status "mysql"  mysql                           # MySQL
check_status "docker" docker                          # Docker（Milvus 依赖它）
check_status "node"   node                            # Node（前端依赖它）

echo
info "下一步："
echo "    1) 复制配置模板并填写真实值："
echo "         cp .env.example .env && vi .env"
echo "    2) 启动前后端服务："
echo "         bash scripts/run.sh"
echo
info "提醒：Milvus 必须先启动（第 9 步的命令），否则后端启动时向量库探测会失败。"
info "首次使用还需要灌数据：python -m ingest && python -m vector_store --rebuild"
