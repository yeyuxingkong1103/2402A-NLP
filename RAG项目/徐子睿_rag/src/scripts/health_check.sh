#!/usr/bin/env bash
# ============================================================
# scripts/health_check.sh —— 健康检查（探活）
#
# 用法：bash scripts/health_check.sh
# 环境变量：BASE_URL（默认 http://127.0.0.1:8902 —— src 新架构的端口；
#     本脚本探的是 src 线的 /health，与 backend 主线的 /api/health 不是同一个）
#     例：BASE_URL=http://10.0.0.5:8902 bash scripts/health_check.sh
#
# 退出码：0 = 服务健康；非 0 = 不可达或返回错误状态。
# 因此它既能在命令行手动查看，也能直接嵌进 CI 或部署脚本做门禁判断。
#
# 注意探的是新架构的 /health（src/api/main.py），
# 它只检查 Milvus；注册登录等不依赖 Milvus 的功能即使返回 degraded 也仍可用。
# ============================================================
set -euo pipefail
# set -euo pipefail 说明同 start.sh

# -f  服务器返回 4xx/5xx 时让 curl 以非 0 退出 —— 这是本脚本能当门禁用的关键；
#     不加的话 curl 对错误响应也算成功，探活就失去意义了
# -s  静默模式，不显示进度条，保持输出只有响应体
# -S  出错时仍显示错误信息（配合 -s 使用，否则静默会把错误也吞掉，难排查）
curl -fsS "${BASE_URL:-http://127.0.0.1:8902}/health"
