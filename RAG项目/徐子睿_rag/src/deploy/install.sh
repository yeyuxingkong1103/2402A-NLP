#!/usr/bin/env bash
# ============================================================
# deploy/install.sh —— 全新 Linux 服务器的一键安装与常驻部署
#
# 用法（需要 root）：sudo bash deploy/install.sh
#
# 环境变量：
#   APP_DIR       安装目录（默认 /opt/rag-roleplay）
#   PYTHON_BIN    Python 解释器（默认 python3，需 3.11+）
#   PORT          服务端口（默认 8901）
#   SERVICE_USER 运行服务的用户（默认取 SUDO_USER，没有则 root）
#   REPO_URL      可选。给了就 git clone/pull；不给则用 APP_DIR 里已有的文件
#
# 本脚本做的事（顺序即依赖关系）：
#   1. 检查是否 root            —— 后面要装系统包、写 systemd unit
#   2. 按发行版装系统依赖       —— apt / dnf / yum 三种包管理器
#   3. 准备代码目录             —— 有 REPO_URL 就拉取，否则用已有文件
#   4. 建 venv 并装依赖
#   5. 生成 .env（从 .env.example 复制，不覆盖已有的）
#   6. 注册并启动 systemd 服务  —— 实现开机自启与崩溃自动重启
#   7. 探活
#
# 注意它启动的是 backend/server.py（老主线），WorkingDirectory 设为 ${APP_DIR}/backend，
# 所以 ExecStart 里是 `uvicorn server:app` 而非 `uvicorn backend.server:app`，
# 最后探的也是 backend 的 /api/health（不是新架构的 /health）。
# ============================================================
set -Eeuo pipefail
# set -Eeuo pipefail 说明：
#   -e 失败即退、-u 禁止未定义变量、-o pipefail 管道任一环失败即整体失败
#   -E 让 ERR 陷阱能作用于函数内部与子 shell（这里是更严格的错误传播）

# 全部支持环境变量覆盖，且都有合理默认值，做到"开箱即用又可按需调整"
APP_DIR="${APP_DIR:-/opt/rag-roleplay}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PORT="${PORT:-8901}"
# ${SUDO_USER:-root} 嵌套在 ${SERVICE_USER:-...} 里：
# 优先用调用 sudo 的那个用户（SUDO_USER），拿不到才回退 root。
# 这样以 sudo 方式安装时，服务会以普通用户身份运行，而不是以 root 跑应用。
SERVICE_USER="${SERVICE_USER:-${SUDO_USER:-root}}"

# EUID 是当前有效用户 id，0 表示 root。
# 必须提前检查：后面的 apt/dnf 装包和写 systemd 都要求 root，
# 跑到一半才失败会留下装了一半的系统。
if [[ "${EUID}" -ne 0 ]]; then
  echo "请使用 sudo 运行：sudo bash deploy/install.sh"
  exit 1
fi

# 按发行版选择包管理器 —— 三种都覆盖，找不到就明确报错退出，
# 而不是继续跑下去然后在一个奇怪的地方失败
if command -v apt-get >/dev/null 2>&1; then
  # DEBIAN_FRONTEND=noninteractive 抑制 apt 的交互式提问（比如时区选择）。
  # 不加的话脚本会卡在交互提示上，在无人值守的部署里表现为"挂住了"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  # build-essential 是必需的：FlagEmbedding 等包的部分依赖需要编译
  apt-get install -y python3 python3-venv python3-pip curl git build-essential
elif command -v dnf >/dev/null 2>&1; then
  dnf install -y python3 python3-pip gcc gcc-c++ make curl git
elif command -v yum >/dev/null 2>&1; then
  yum install -y python3 python3-pip gcc gcc-c++ make curl git
else
  echo "不支持的 Linux 包管理器；请先安装 Python 3.11+、pip、git、curl"
  exit 1
fi

mkdir -p "${APP_DIR}"
# 只有给了 REPO_URL 才从 git 取代码。
# 已存在 .git 时用 pull --ff-only（快进合并）：不做 merge commit，
# 冲突时直接失败而不是留下一个需要人工解决的中间状态 —— 部署脚本不该自作主张合并。
if [[ -n "${REPO_URL:-}" ]]; then
  if [[ -d "${APP_DIR}/.git" ]]; then
    git -C "${APP_DIR}" pull --ff-only
  else
    git clone "${REPO_URL}" "${APP_DIR}"
  fi
else
  echo "未设置 REPO_URL：使用 ${APP_DIR} 中已有项目文件。"
fi

# 校验关键文件是否就位。检查 backend/ 目录是因为下面的 systemd unit 要用它作为工作目录 ——
# 缺了它服务起来就会失败，提前拦住比让 systemd 反复重启好
if [[ ! -f "${APP_DIR}/requirements.txt" || ! -d "${APP_DIR}/backend" ]]; then
  echo "缺少项目文件：请设置 REPO_URL，或先将项目上传到 ${APP_DIR}。"
  exit 1
fi

cd "${APP_DIR}"
"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# cp -n 表示"目标已存在就不覆盖"（no-clobber）。
# 这一点很关键：重新部署时不该把线上已经调好的 .env 冲掉。
# 末尾 `|| true` 是因为目标已存在时 cp -n 返回非 0，
# 而"配置已存在"显然是正常情况，不该让部署中断。
cp -n .env.example .env 2>/dev/null || true

# 把项目目录交给服务运行用户：systemd 以 SERVICE_USER 身份启动服务，
# 若目录属主是 root，服务将无法写入 data/、logs/ 等目录
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${APP_DIR}"

# 写 systemd unit。EOF 不加引号 => 内部变量会被展开成实际值（这里正是想要的）
cat > /etc/systemd/system/rag-roleplay.service <<EOF
[Unit]
Description=RAG Roleplay FastAPI Service
After=network.target

[Service]
User=${SERVICE_USER}
WorkingDirectory=${APP_DIR}/backend
EnvironmentFile=-${APP_DIR}/.env
ExecStart=${APP_DIR}/.venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port ${PORT}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# daemon-reload：改了 unit 文件必须让 systemd 重新读取，否则下面 start 用的还是旧的
systemctl daemon-reload
# enable：设为开机自启
systemctl enable rag-roleplay.service
# restart 而不是 start：重新部署时服务可能已在运行，restart 两种情况都能正确处理
systemctl restart rag-roleplay.service
# 打印状态。--no-pager 防止输出被送进 less 而卡住脚本；
# 末尾 `|| true` 是因为服务启动失败时此命令返回非 0，
# 但我们希望继续执行到最后那行探活，把诊断信息都打出来再结束
systemctl --no-pager --full status rag-roleplay.service || true
# 最后探活一次，作为"部署是否成功"的最终结论。
# 探的是 backend 主线的 /api/health（与上面的 ExecStart 对应）
curl --fail --silent "http://127.0.0.1:${PORT}/api/health" || true
