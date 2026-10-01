#!/usr/bin/env bash
# 结束脚本：按 pid 文件停止 app 与重排服务
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/lib.sh

# 兜底匹配串只在 pidfile 丢失时才会用到（脚本被强杀过的情况）
stop_by_pidfile "app" "logs/uvicorn.pid" "uvicorn app.main:app"

# 重排实例一个常驻 ~2.2GB，而 RERANK_BASE_URL 只指向一个地址；不管起在哪个端口
# 都一并收掉，免得留下没人管的孤儿实例把 WSL 内存吃满，连带拖垮 PyCharm 的终端。
# 这里刻意按进程名匹配而不是按 pidfile：多个端口各有一个 pidfile，扫全比逐个列举可靠。
if pkill -f "uvicorn rerank_service.server:app" 2>/dev/null; then
    echo "已停止 重排服务"
else
    echo "重排服务 未在运行"
fi
rm -f logs/rerank-*.pid logs/rerank.pid
