#!/usr/bin/env bash
# ============================================================
# reindex.sh —— 一键重建知识库（第 13 步实现）
# 作用：按 .env 的 PDF_FILES 依次跑「解析 → 增强 → 入库」三步，
#       把 data/parsed_chunks.json、data/enriched_chunks.json 与 Milvus 集合全部重刷一遍。
#       改完 PDF_FILES 后跑这个脚本即可让清单生效。
# 用法：bash scripts/reindex.sh
# ============================================================

set -euo pipefail                 # 失败即退出；未定义变量报错；管道任一段失败即失败

# ---------- 全局变量 ----------
ENV_NAME="power_rag"              # conda 环境名，与 install.sh / run.sh 保持一致
MINICONDA_DIR="$HOME/miniconda3"  # Miniconda 安装目录
LOG_FILE="logs/reindex.log"       # 重建过程日志，最后从里面 grep 结果

cd "$(dirname "$0")/.."           # 切到项目根目录，后面统一用相对路径
PROJECT_DIR="$(pwd)"              # 项目根目录绝对路径
mkdir -p logs                     # 日志目录不存在就创建

# ---------- 输出辅助 ----------
info() { echo -e "\033[32m[信息]\033[0m $*"; }        # 绿色：正常进度
warn() { echo -e "\033[33m[警告]\033[0m $*"; }        # 黄色：可以容忍的异常
err()  { echo -e "\033[31m[错误]\033[0m $*" >&2; }    # 红色：错误信息

# ============================================================
# 第 1 步：检查 conda 环境
# ============================================================
echo "===== 1/4 检查 conda 环境 ====="

if [ ! -x "$MINICONDA_DIR/bin/conda" ]; then           # conda 不在预期位置
  err "找不到 conda：$MINICONDA_DIR/bin/conda；请先执行 bash scripts/install.sh"
  exit 1
fi

# shellcheck disable=SC1091
eval "$("$MINICONDA_DIR/bin/conda" shell.bash hook)"   # 加载 conda 的 shell 钩子

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then  # 环境列表里没有目标环境
  err "conda 环境 $ENV_NAME 不存在；请先执行 bash scripts/install.sh"
  exit 1
fi

conda activate "$ENV_NAME"                             # 激活环境
info "已激活 conda 环境：$CONDA_DEFAULT_ENV"

# ============================================================
# 第 2 步：检查 .env 与 PDF_FILES
# ============================================================
echo "===== 2/4 检查配置 ====="

if [ ! -f .env ]; then                                 # 没有 .env 就不知道解析什么、连哪里
  err "配置文件 .env 不存在；请执行 cp .env.example .env 并填写后重试"
  exit 1
fi

# 取 .env 里的 PDF_FILES 值（只取第一个匹配行，等号右边整段）
PDF_FILES_RAW="$(grep -E "^PDF_FILES=" .env | head -1 | cut -d= -f2- || true)"
if [ -z "$PDF_FILES_RAW" ]; then                       # 空值＝扫全目录
  warn "PDF_FILES 为空，本次将扫描 PDF_DIR 下的全部 PDF"
else                                                   # 非空：数一下配了几个
  COUNT="$(printf '%s' "$PDF_FILES_RAW" | awk -F, '{print NF}')"
  info "PDF_FILES 配了 $COUNT 个条目（逗号分隔）"
fi

# ============================================================
# 第 3 步：执行一键重建
# ============================================================
echo "===== 3/4 执行一键重建（解析 → 增强 → 入库）====="

# 重建会覆盖 parsed_chunks.json / enriched_chunks.json 并重建 Milvus 集合，先把话说清楚
info "注意：本操作会覆盖 data/ 下的解析与增强结果，并重建 Milvus 集合"

# 用 tee 同时写日志与屏幕；pipefail 保证 python 失败时整条管道判失败
if ! python -m ingest --reindex 2>&1 | tee "$LOG_FILE"; then
  err "一键重建失败，详情见 $LOG_FILE"
  exit 1                                               # 返回非 0，供上层脚本判断
fi

# ============================================================
# 第 4 步：从日志里汇总每步耗时与最终 row_count
# ============================================================
echo "===== 4/4 结果汇总 ====="

for label in "解析 PDF" "离线增强" "向量入库"; do        # 逐步骤抓日志里的输出行
  line="$(grep -F "$label" "$LOG_FILE" | tail -1 || true)"   # 取该步最后一条匹配
  [ -n "$line" ] && echo "  $line"                       # 抓到就打印
done

# 从日志里 grep 最终 row_count（程序在结尾会打印）
ROW="$(grep -oE "row_count = [0-9]+" "$LOG_FILE" | tail -1 | grep -oE "[0-9]+" || true)"
if [ -n "$ROW" ]; then                                 # 抓到了就报出来
  info "Milvus 集合最终 row_count = $ROW"
else                                                   # 没抓到说明查询失败
  warn "日志里没抓到 row_count，请手动执行 curl http://localhost:8000/kb/status 核对"
fi

echo
info "一键重建完成。完整日志：$LOG_FILE"
info "如需查看接口侧状态：curl http://localhost:8000/kb/status"
