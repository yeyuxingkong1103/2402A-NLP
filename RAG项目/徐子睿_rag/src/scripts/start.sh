#!/usr/bin/env bash
# ============================================================
# scripts/start.sh —— 启动新架构服务（src/api/main.py）
#
# 用法：bash scripts/start.sh
# 环境变量：HOST（默认 127.0.0.1）、PORT（默认 8902）
#     例：PORT=9000 bash scripts/start.sh
#
# 与 启动.bat 的区别：那个启动的是 backend.server:app（老主线），
# 本脚本启动的是 src.api.main:app（新架构：JWT + 多角色 + 离线/在线分层）。
# ============================================================
set -euo pipefail
# set -euo pipefail 三个开关：
#   -e  任一命令失败就立刻退出，不让错误被后续步骤掩盖
#   -u  引用未定义变量时报错，防止笔误变成静默的空字符串
#   -o pipefail  管道中任一环失败即视为整条失败（默认只看最后一个命令的退出码）

# 切到项目根目录：本脚本在 scripts/ 下，$0 的目录上一级就是根目录。
# 不切换的话，uvicorn 的模块搜索路径不对、找不到 src 包。
cd "$(dirname "$0")/.."

# ${VAR:-默认值} 语法：变量未设置或为空时用默认值 —— 让脚本开箱即用又能被覆盖。
#
# 端口默认 8902，与 backend 主线（启动.bat / deploy 用 8901）**分开**：
#   两条线如果共用 8901，后启动的那个会 bind 失败（Errno 10048），
#   而 uvicorn 会先打印 "Application startup complete" 再报错，看起来像启动成功了 ——
#   实际请求全打到先启动的那个应用上（表现为 /health 返回 404 之类），极难自查。
python -m uvicorn src.api.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8902}"
