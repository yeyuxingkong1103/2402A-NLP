#!/usr/bin/env bash
# 验证 nginx 负载均衡是否**真的配好了、且真的在分发**。
#
# 为什么要单独有它：光有 deploy/nginx.conf 不算数，本机实测踩过一个静默失效——
# upstream 不配 `zone` 时，默认 worker_processes auto（本机十几个 worker）
# **每个 worker 各自维护轮询指针、都从第一台上游起步**。
#
# 为什么判据是「配置里有没有 zone」而不是「实测有没有连庄」：
#   连庄（连着几发都落同一台）确实是无 zone 的症状，但**不稳定、不能当判据**。
#   同一套无 zone 的 nginx 实例，本机实测两次结果相反：
#       串行 9 发  -> 8000 连庄 7 次（症状明显）
#       串行 12 发 -> 零连庄、5/4/3（症状消失）
#   原因是 curl 每次都新建连接，内核把这些连接分给哪个 worker 不确定：连接若集中落到
#   同一个 worker，该 worker 自己的指针会推进，连庄就看不见了。所以「连庄」只能当**参考
#   输出**打出来看，不能用来判成败——否则会随机放过坏配置 / 冤枉好配置。
#   zone 的有无是**配置事实**，静态检查 100% 确定，所以判据用它。
#
# 判据（任一不满足即非 0 退出）：
#   1. 响应里必须带 X-Upstream —— 没有就说明没走 nginx（或 add_header 被改掉了）；
#   2. 多 worker 时 upstream 块必须配 zone —— 静态检查，见上；
#   3. 实际必须分发到 ≥2 台上游，且落点都在 upstream 清单里 —— 动态检查，防「只配了一台」
#      和「打到了没配的机器」（曾经踩过：上游写成 8000/8001/8002，8001 其实是 BGE 重排服务）。
#
# 用法：bash scripts/verify_lb.sh [base_url] [conf_path]
#   默认 http://127.0.0.1 与 deploy/nginx.conf
# 输出同时落盘到 logs/verify_lb.out
set -uo pipefail
cd "$(dirname "$0")/.."

BASE_URL="${1:-http://127.0.0.1}"
CONF="${2:-deploy/nginx.conf}"
OUT="logs/verify_lb.out"
SERIAL_N=12   # 串行发几发（取上游台数的整数倍，轮转是否规律一眼可辨）
CONC_N=30     # 并发发几发
mkdir -p logs

exec > >(tee "$OUT") 2>&1

echo "=== nginx 负载均衡验证 $(date '+%F %T') ==="
echo "入口: $BASE_URL    配置: $CONF"

# 容忍缩进：配置若被放进 http{} 内嵌（而不是 conf.d 下顶格），upstream/} 都会带前导空格
UPSTREAM_BLOCK="$(awk '/^[[:space:]]*upstream[[:space:]]+rag_backend/{f=1}
                       f && /^[[:space:]]*}/{print; exit}
                       f' "$CONF" 2>/dev/null)"

# 预期上游清单：既用来判定「打到了没配的机器」，也用来对表。
# 先去注释再抓端口——本文件的行内注释里就写着 ":8000"，不清掉会把说明文字当上游。
mapfile -t EXPECTED < <(
    printf '%s\n' "$UPSTREAM_BLOCK" | sed 's/#.*//' | grep -oE '127\.0\.0\.1:[0-9]+' | sort -u
)
if [[ ${#EXPECTED[@]} -eq 0 ]]; then
    echo "❌ 没能从 $CONF 解析出 upstream 上游清单——配置文件改过？"
    exit 1
fi
echo "预期上游(${#EXPECTED[@]} 台): ${EXPECTED[*]}"

# 均衡方法：least_conn / ip_hash 下连庄是预期行为，参考输出要说明
if printf '%s\n' "$UPSTREAM_BLOCK" | grep -qE '^\s*(least_conn|ip_hash)\s*;'; then
    METHOD="$(printf '%s\n' "$UPSTREAM_BLOCK" | grep -oE '^\s*(least_conn|ip_hash)' | tr -d '[:space:]')"
else
    METHOD="round-robin"
fi
echo "均衡方法: $METHOD"

# 取一条响应的 X-Upstream。注意 -D - 把响应头打到 stdout，body 丢弃
probe() {
    curl -s --max-time 10 -o /dev/null -D - "$BASE_URL/health" 2>/dev/null \
        | awk 'BEGIN{IGNORECASE=1} /^x-upstream:/ {gsub(/\r/,""); print $2}'
}

# ---------------- 1) 可达性 + 单发 ----------------
echo
echo "--- 1) 入口与 X-Upstream 头 ---"
first="$(probe)"
if [[ -z "$first" ]]; then
    echo "❌ 没拿到 X-Upstream —— 请求没走 nginx？先确认："
    echo "     curl -s -D - $BASE_URL/health | head"
    echo "   nginx 没起: sudo nginx -t && sudo nginx"
    echo "   配置没装:   bash scripts/setup_nginx.sh"
    exit 1
fi
echo "✅ $BASE_URL/health 经 nginx -> $first"

# ---------------- 2) 串行（参考输出：看轮转规律，不参与判成败）----------------
echo
echo "--- 2) 串行发 $SERIAL_N 发 ---"
echo "    （参考输出——轮转规律好看，但连庄与否不稳定，故不作判据，详见本脚本头部注释）"
serial=()
for _ in $(seq 1 "$SERIAL_N"); do
    u="$(probe)"
    serial+=("${u:-<无>}")
done
printf '    %s\n' "$(IFS=' '; echo "${serial[*]}")"

streak=0; longest=1; cur=1
for ((i = 1; i < ${#serial[@]}; i++)); do
    if [[ "${serial[i]}" == "${serial[i-1]}" ]]; then
        streak=$((streak + 1)); cur=$((cur + 1)); (( cur > longest )) && longest=$cur
    else
        cur=1
    fi
done

# ---------------- 3) 并发 ----------------
echo
echo "--- 3) 并发发 $CONC_N 发 ---"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
for i in $(seq 1 "$CONC_N"); do
    probe > "$tmp/$i.up" &
done
wait
conc=()
for f in "$tmp"/*.up; do
    u="$(cat "$f")"
    conc+=("${u:-<无>}")
done

# 把一串上游名按出现次数汇总成条形图（输出风格对齐 deploy/jmeter/summarize.py）
tally() {
    printf '%s\n' "$@" | sort | uniq -c | sort -rn | awk '
        {printf "    %-18s %4d  %s\n", $2, $1, substr("########################################", 1, $1*4)}
    '
}

echo
echo "--- 4) 分发统计 ---"
echo "  串行 $SERIAL_N 发（连庄 $streak 次，最长连庄 $longest）："
tally "${serial[@]}"
echo "  并发 $CONC_N 发："
tally "${conc[@]}"

n_serial="$(printf '%s\n' "${serial[@]}" | sort -u | grep -c . || true)"
n_conc="$(printf '%s\n' "${conc[@]}" | sort -u | grep -c . || true)"
echo "  → 串行落在 $n_serial 台，并发落在 $n_conc 台（共配置 ${#EXPECTED[@]} 台）"

# ---------------- 5) 判定 ----------------
echo
echo "--- 5) 判定 ---"
fail=0

echo "  [1/3] X-Upstream 存在"
echo "        ✅ 有（$first）"

# nginx worker 数：无法精确归属到「正在服务 $BASE_URL 的那个实例」，只能数全机。
# 多实例并存时会把数放大——放大只会让本检查更容易触发（偏保守），不会漏判。
WORKERS="$(ps -eo args= 2>/dev/null | awk '/^nginx: worker process/ {n++} END {print n+0}')"
echo "  [2/3] 多 worker 时 upstream 必须配 zone"
if printf '%s\n' "$UPSTREAM_BLOCK" | grep -qE '^\s*zone\s+'; then
    echo "        ✅ 已配（$(printf '%s\n' "$UPSTREAM_BLOCK" | grep -oE '^\s*zone[^;]*;' | tr -s ' ')）"
elif [[ "$WORKERS" -le 1 ]]; then
    echo "        ⏭ 全机只有 $WORKERS 个 nginx worker，单 worker 不需要 zone"
else
    echo "        ❌ upstream 块没配 zone（当前全机 $WORKERS 个 worker）"
    echo "           每个 worker 会各自维护轮询指针、都从第一台上游起步。"
    echo "           修：在 upstream 块里加  zone rag_backend 64k;  然后 sudo nginx -s reload"
    fail=1
fi

echo "  [3/3] 实际分发到 ≥2 台，且落点都在配置清单里"
if [[ ${#EXPECTED[@]} -lt 2 ]]; then
    echo "        ⏭ 只配了 1 台上游，横向扩展无从谈起"
elif [[ "$n_serial" -lt 2 || "$n_conc" -lt 2 ]]; then
    echo "        ❌ 串行 $n_serial 台 / 并发 $n_conc 台 —— 只落一台，负载均衡没生效"
    fail=1
else
    bad=()
    while read -r u; do
        [[ -z "$u" || "$u" == "<无>" ]] && continue
        ok=0
        for e in "${EXPECTED[@]}"; do [[ "$u" == "$e" ]] && ok=1; done
        [[ "$ok" -eq 0 ]] && bad+=("$u")
    done < <(printf '%s\n' "${serial[@]}" "${conc[@]}" | sort -u)
    if [[ ${#bad[@]} -gt 0 ]]; then
        echo "        ❌ 出现未配置的上游: ${bad[*]}"
        echo "           （曾经踩过：上游写成 8000/8001/8002，8001 其实是 BGE 重排服务）"
        fail=1
    else
        echo "        ✅ 串行 $n_serial 台 / 并发 $n_conc 台，全部命中 ${EXPECTED[*]}"
    fi
fi

echo
if [[ "$fail" -eq 0 ]]; then
    echo "✅ 负载均衡生效：$(IFS=' '; echo "${EXPECTED[*]}") 均在分发中"
else
    echo "❌ 负载均衡验证未通过（详见上面标 ❌ 的项）"
fi
exit "$fail"
