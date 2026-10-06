#!/usr/bin/env bash
# ============================================================
# 一键完整运行流程（基于 RAG 的心理医生多角色陪伴系统）
#
# 流程：0/5 读取配置 → 1/5 环境自检 → 2/5 初始化数据库与角色
#       → 3/5 构建知识库 → 4/5 启动服务 → 5/5 冒烟验证
#
# 用法：
#   bash scripts/run_all.sh                  # 完整流程
#   bash scripts/run_all.sh --skip-ingest    # 跳过知识库构建（已入库时最快跑通）
#   bash scripts/run_all.sh --drop-existing  # 重建索引（先清空该角色旧向量与元数据）
#   bash scripts/run_all.sh --port 8001      # 指定服务端口
#   bash scripts/run_all.sh --no-smoke       # 跳过冒烟验证
#   bash scripts/run_all.sh --only-check     # 只做环境自检，不做任何变更
#   bash scripts/run_all.sh --help           # 查看帮助
#
# 前置条件：MySQL(3307) / Redis(6379) / Milvus(19530) 已启动
# ============================================================
set -euo pipefail

# 项目根目录 = 本脚本所在目录（scripts/）的上一级
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ---------- 参数解析 ----------
SKIP_INGEST=0      # 是否跳过知识库构建
DROP_EXISTING=""   # 是否重建索引（--drop-existing 参数透传）
API_PORT_ARG=""    # 命令行指定的端口
DO_SMOKE=1         # 是否做冒烟验证
ONLY_CHECK=0       # 是否仅自检

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-ingest)   SKIP_INGEST=1 ;;
    --drop-existing) DROP_EXISTING="--drop-existing" ;;
    --port)          shift; API_PORT_ARG="${1:-}" ;;
    --no-smoke)      DO_SMOKE=0 ;;
    --only-check)    ONLY_CHECK=1 ;;
    -h|--help)       sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "[!] 未知参数：$1（用 --help 查看用法）"; exit 2 ;;
  esac
  shift
done

# ---------- 输出辅助函数 ----------
step() { echo; echo "────────────────────────────────────────────"; echo "▶ $*"; echo "────────────────────────────────────────────"; }
ok()   { echo "[√] $*"; }
warn() { echo "[!] $*"; }
die()  { echo "[x] $*" >&2; exit 1; }

# ---------- 0/5 读取配置 ----------
step "0/5 读取配置"
if [ ! -f .env ]; then
  warn "未找到 .env，正在从 .env.example 复制"
  cp .env.example .env
  die "请先编辑 .env（填写 LLM_API_KEY / JWT_SECRET_KEY / ADMIN_PASSWORD）后重试"
fi
set -a; source .env; set +a   # set -a 自动导出变量；source 后关闭
ok ".env 已加载"

PYTHON_BIN="${PYTHON_BIN:-/home/dabaie/code/my_project/.venv/bin/python}"
API_PORT="${API_PORT_ARG:-${API_PORT:-8000}}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"

# ---------- 1/5 环境自检 ----------
step "1/5 环境自检"

# 虚拟环境必须存在且可执行，否则后续所有 Python 步骤都会失败
[ -x "$PYTHON_BIN" ] || die "虚拟环境不存在或不可执行：$PYTHON_BIN"
ok "Python：$("$PYTHON_BIN" -V 2>&1)"

# 两个本地模型目录必须存在（RAG 检索与重排的基础）
for model_path in "${EMBEDDING_MODEL_PATH:-}" "${RERANKER_MODEL_PATH:-}"; do
  [ -n "$model_path" ] || die "EMBEDDING_MODEL_PATH / RERANKER_MODEL_PATH 未配置"
  [ -d "$model_path" ] || die "本地模型缺失：$model_path（请先下载 BGE-M3 / BGE-Reranker-v2-M3）"
  ok "模型存在：$model_path"
done

# 端口探测：/dev/tcp 是 bash 内置能力，无需依赖 nc / telnet
check_port() {
  local host="$1" port="$2" name="$3"
  if timeout 3 bash -c "cat < /dev/null > /dev/tcp/$host/$port" 2>/dev/null; then
    ok "$name 可连通：$host:$port"
    return 0
  fi
  warn "$name 无法连接：$host:$port"
  return 1
}

INFRA_OK=1
check_port "${DB_HOST:-127.0.0.1}"     "${DB_PORT:-3307}"      "MySQL"  || INFRA_OK=0
check_port "${REDIS_HOST:-127.0.0.1}"  "${REDIS_PORT:-6379}"   "Redis"  || INFRA_OK=0
check_port "${MILVUS_HOST:-127.0.0.1}" "${MILVUS_PORT:-19530}" "Milvus" || INFRA_OK=0
# 三个外部依赖缺一不可：直接终止，避免后续步骤产生半成品数据
[ "$INFRA_OK" = 1 ] || die "基础设施未就绪：请先启动 MySQL / Redis / Milvus 后重试"

if [ "$ONLY_CHECK" = 1 ]; then
  step "仅自检模式：全部检查通过，未做任何变更"
  exit 0
fi

# ---------- 2/5 初始化数据库与角色 ----------
step "2/5 初始化 MySQL 表结构 / Milvus Collection / 角色 / 管理员"
# init_db 是幂等的：已存在的表/角色/管理员不会重复创建
"$PYTHON_BIN" scripts/init_db.py
ok "数据库初始化完成"

# ---------- 3/5 构建知识库 ----------
step "3/5 构建知识库（解析 → 分块 → 向量化 → 入库）"
if [ "$SKIP_INGEST" = 1 ]; then
  warn "已跳过知识库构建（--skip-ingest）"
else
  echo "[*] 说明：文档较多且含扫描件，首次构建耗时较长"
  echo "    可在另一个终端执行 bash scripts/check_ingest.sh 观察进度"
  # shellcheck disable=SC2086  # DROP_EXISTING 需按空/非空展开为独立参数
  "$PYTHON_BIN" scripts/ingest_knowledge.py --all ${DROP_EXISTING}
  ok "知识库构建完成"
fi

# ---------- 4/5 启动服务 ----------
step "4/5 启动后端服务"
bash scripts/run.sh --port "$API_PORT"

# 轮询健康检查：首次启动需加载 BGE-M3 / Reranker，最多等 180 秒
echo "[*] 等待服务就绪（首次需加载 BGE-M3 / Reranker，最多 180s）..."
READY=0
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 3
done
[ "$READY" = 1 ] || die "服务未在 180s 内就绪，请查看日志：$LOG_DIR/uvicorn.out"
ok "健康检查通过：http://127.0.0.1:$API_PORT/health"
ok "接口文档：http://127.0.0.1:$API_PORT/docs"

# ---------- 5/5 冒烟验证 ----------
step "5/5 冒烟验证"
if [ "$DO_SMOKE" = 0 ]; then
  warn "已跳过冒烟验证（--no-smoke）"
else
  BASE="http://127.0.0.1:$API_PORT/api/v1"
  DEMO_USER="demo$(date +%H%M%S)"   # 时间戳后缀避免用户名冲突
  DEMO_PASS="demo123456"

  # 1) 注册演示用户
  if curl -fsS -X POST "$BASE/auth/register" -H "Content-Type: application/json" \
       -d "{\"username\":\"$DEMO_USER\",\"password\":\"$DEMO_PASS\",\"nickname\":\"演示用户\"}" >/dev/null 2>&1; then
    ok "注册成功：$DEMO_USER"
  else
    warn "注册失败（用户名可能已存在），改用已有账号继续"
    DEMO_USER="demo001"
  fi

  # 2) 登录并提取 access_token（用项目 venv 的 python 解析 JSON，避免依赖 jq）
  TOKEN="$(curl -fsS -X POST "$BASE/auth/login" -H "Content-Type: application/json" \
            -d "{\"username\":\"$DEMO_USER\",\"password\":\"$DEMO_PASS\"}" \
          | "$PYTHON_BIN" -c "import sys,json;print(json.load(sys.stdin)['data']['access_token'])" 2>/dev/null || true)"
  [ -n "${TOKEN:-}" ] || die "登录失败，未取到 access_token（请检查账号密码）"
  ok "登录成功，已获取 access_token"

  # 3) 角色列表（正常应返回 3 个心理医生）
  PERSONA_COUNT="$(curl -fsS "$BASE/personas" \
    | "$PYTHON_BIN" -c "import sys,json;d=json.load(sys.stdin).get('data');d=d.get('items',[]) if isinstance(d,dict) else (d or []);print(len(d))" 2>/dev/null || echo '?')"
  ok "心理医生角色数：$PERSONA_COUNT"

  # 4) 非流式聊天：验证「检索 + 提示词 + LLM + 记忆」全链路
  echo "[*] 非流式聊天测试（persona_id=1，最多等待 60s）..."
  ANSWER="$(curl -fsS --max-time 60 -X POST "$BASE/chat" \
              -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
              -d '{"persona_id":1,"message":"我最近总是失眠，心里很慌。"}' \
            | "$PYTHON_BIN" -c "import sys,json;d=json.load(sys.stdin).get('data') or {};print(str(d.get('answer') or d.get('content') or '')[:120])" 2>/dev/null || true)"
  if [ -n "${ANSWER:-}" ]; then
    ok "聊天返回：$ANSWER"
  else
    warn "聊天未返回内容（请检查 LLM_API_KEY 是否有效、知识库是否已入库）"
  fi
fi

# ---------- 汇总 ----------
step "运行完成"
echo "  服务地址 : http://127.0.0.1:$API_PORT"
echo "  接口文档 : http://127.0.0.1:$API_PORT/docs"
echo "  运行日志 : $LOG_DIR/uvicorn.out"
echo "  PID 文件 : $PROJECT_ROOT/data/uvicorn.pid"
echo
echo "  停止服务 : bash scripts/shutdown.sh"
echo "  导入巡检 : bash scripts/check_ingest.sh"
echo "  角色刷新 : $PYTHON_BIN scripts/seed_personas.py"
echo "  质量评测 : $PYTHON_BIN scripts/eval_ragas.py --persona-id 1"
