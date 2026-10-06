#!/usr/bin/env bash
# 把 deploy/nginx.conf 装成系统 nginx 的站点配置并 reload（负载均衡 + 反向代理）。
#
# 需要 root：写 /etc/nginx/ 要权限。用法：
#   sudo bash scripts/setup_nginx.sh
#
# 为什么要有这个脚本：deploy/ 下原本只有一份 nginx.conf，没有任何安装/加载它的自动化，
# 所以那份配置从没被真正跑起来过。这里把「装配置 → 语法检查 → reload → 验证」固化下来。
#
# 网络约束（本机实测）：这台机器对 archive.ubuntu.com 的 **HTTP 下载会被重置连接**，
# 所以 apt 源若还是 http:// 会卡在无限重试（目录页能开，一碰 .deb 就断）。
# 要装/升级系统包请先把源换成 https://：
#   sudo sed -i 's|http://archive.ubuntu.com|https://archive.ubuntu.com|' /etc/apt/sources.list.d/ubuntu.sources
# （nginx 与 default-jre-headless 本机已装好，本脚本不需要联网。）

set -euo pipefail
cd "$(dirname "$0")/.."

CONF_SRC="deploy/nginx.conf"
CONF_DST="/etc/nginx/conf.d/rag.conf"
# 与 deploy/nginx.conf 的 upstream 保持一致；8001 被 BGE 重排服务占用，不能用
PORTS="${PORTS:-8000 8002 8003}"

if [[ "$(id -u)" != "0" ]]; then
    echo "❌ 需要 root：sudo bash scripts/setup_nginx.sh"
    exit 1
fi

if ! command -v nginx >/dev/null 2>&1; then
    echo "❌ 没装 nginx。本机 HTTP 源被重置，先换 https 源再装："
    echo "   sudo sed -i 's|http://archive.ubuntu.com|https://archive.ubuntu.com|' /etc/apt/sources.list.d/ubuntu.sources"
    echo "   sudo apt install -y nginx"
    exit 1
fi

# 上游没起来的话，nginx 能启动但请求会 502——提前说清楚，别让人以为是 nginx 的错
echo "检查上游 worker 是否在跑（$PORTS）："
for p in $PORTS; do
    if timeout 3 bash -c "cat < /dev/null > /dev/tcp/127.0.0.1/$p" 2>/dev/null; then
        echo "  ✅ :$p 在监听"
    else
        echo "  ❌ :$p 没有监听 —— 先起 worker：bash scripts/run_workers.sh"
    fi
done

echo "安装配置：$CONF_SRC -> $CONF_DST"
cp "$CONF_SRC" "$CONF_DST"

# Ubuntu 默认站点也监听 80（default_server）。我们的 location 用 server_name localhost
# 精确匹配，按 Host 头优先于 default_server，所以两者可以共存、不必禁用默认站点。
if [[ -e /etc/nginx/sites-enabled/default ]]; then
    echo "ℹ️  默认站点仍启用（它监听 80 default_server）。本配置用 server_name localhost 精确匹配，"
    echo "   带 Host: localhost 的请求会走到本配置，不影响默认站点。"
fi

echo "语法检查："
nginx -t

echo "reload："
if pgrep -x nginx >/dev/null 2>&1; then
    nginx -s reload
    echo "  已 reload"
else
    nginx
    echo "  已启动"
fi

sleep 1
echo
echo "验证（经 nginx 而不是直连 worker）："
if curl -sf -m 10 http://127.0.0.1/health >/dev/null 2>&1; then
    echo "  ✅ http://127.0.0.1/health -> $(curl -s -m 10 http://127.0.0.1/health | head -c 120)"
else
    echo "  ❌ http://127.0.0.1/health 不通，查 /var/log/nginx/error.log"
fi

# 「通」只说明 nginx 起来了，不说明流量真被分到多个 worker。deploy/nginx.conf 回了
# X-Upstream（真正处理请求的 127.0.0.1:800X），打十几发看落在几个 worker 上。
#
# 为什么这里并发发而不串行：nginx 默认 worker_processes auto，**每个 worker 的轮询指针
# 各自独立、都从 0 号上游起步**。串行发请求时每条新连接很可能被不同 worker 接走，于是
# 全部命中 0 号上游（实测串行 6 发 6 个 :8000），看着像没轮询。并发时同一 worker 会连续
# 处理多条，立刻均匀。deploy/nginx.conf 里配了 `zone` 后轮询状态跨 worker 共享，串行也轮转。
echo
echo "轮询分发验证（并发 12 发，统计 X-Upstream 落在几个 worker 上）："
tmp="$(mktemp)"
for _ in $(seq 1 12); do
    ( curl -s -m 10 -D - -o /dev/null http://127.0.0.1/health \
      | tr -d '\r' | awk -F': ' 'tolower($1)=="x-upstream"{print $2}' >> "$tmp" ) &
done
wait
sort "$tmp" | uniq -c | sed 's/^/    /'
distinct="$(sort -u "$tmp" | grep -c .)"
rm -f "$tmp"
if [[ "$distinct" -eq 0 ]]; then
    echo "  ⚠️  12 发都没取到 X-Upstream —— /etc/nginx/conf.d/rag.conf 里没有 add_header X-Upstream"
    echo "     （是旧版配置没重装？），或响应被别的 server 块接管了"
elif [[ "$distinct" -ge 2 ]]; then
    echo "  ✅ 12 发落到 $distinct 个不同 worker —— 负载均衡生效"
else
    echo "  ⚠️  12 并发只落到 $distinct 个 worker；若 upstream 只有 1 个端口属正常，"
    echo "     否则查 deploy/nginx.conf 的 upstream 是否与本脚本的 PORTS=$PORTS 一致"
fi
echo
echo "下一步（压测走 nginx 才测到负载均衡）："
echo "  /home/zhongfeiyu/tools/apache-jmeter-5.6.3/bin/jmeter -n -t deploy/jmeter/rag-roleplay-load.jmx \\"
echo "      -JPORT=80 -JTHREADS=10 -JRAMP=5 -JLOOPS=1 -l deploy/jmeter/result.jtl -e -o deploy/jmeter/report"
