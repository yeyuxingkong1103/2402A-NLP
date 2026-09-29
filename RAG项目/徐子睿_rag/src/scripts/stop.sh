#!/usr/bin/env bash
# ============================================================
# scripts/stop.sh —— 停止新架构服务
#
# 用法：bash scripts/stop.sh
#
# 做法是按命令行特征匹配并杀掉 uvicorn 进程。
# 与 start.sh 成对使用。
# ============================================================
set -euo pipefail
# set -euo pipefail 说明同 start.sh：
#   -e 失败即退、-u 禁止未定义变量、-o pipefail 管道任一环失败即整体失败

# pkill -f 按**完整命令行**匹配（不加 -f 只匹配进程名，进程名只是 "python"，
# 一杀就会把机器上所有 python 进程都杀掉，非常危险）。
# 这里匹配 'uvicorn src.api.main:app' 这个特征串，因此只会命中本项目的服务。
#
# 末尾的 `|| true` 是必需的：没找到匹配进程时 pkill 返回 1，
# 配合 set -e 会让脚本以失败退出。而"本来就没在跑"是一种正常情况，
# 停止脚本此时应该成功返回，而不是报错。
pkill -f 'uvicorn src.api.main:app' || true
