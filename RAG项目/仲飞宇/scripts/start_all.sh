#!/usr/bin/env bash
# 一键启动：依赖自检 → 能自动拉起的自动拉起 → 起应用 → 校验 /health
#
#   bash scripts/start_all.sh             # 常规启动（组件不可用时降级继续）
#   bash scripts/start_all.sh --strict    # 任一组件不可用即退出非 0（CI/交付演示用）
#   bash scripts/start_all.sh --no-start  # 只做依赖自检，不起应用
#
# 退出码：0 正常 / 1 应用没起来 / 2 参数错 / 3 --strict 下有降级项
#
# 与 run.sh 的关系：run.sh 只管 uvicorn 一个进程；本脚本额外负责它的外部依赖
#（Ollama / MySQL / Redis，以及 RERANKER=bge 时的重排服务），最后仍调 run.sh。
# Milvus Lite 不需要单独起——它跟着应用进程内嵌启动（前提：没有别的进程占着 db 文件）。
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/lib.sh

STRICT=0
DO_START=1
for a in "$@"; do
    case "$a" in
        --strict)   STRICT=1 ;;
        --no-start) DO_START=0 ;;
        -h|--help)  sed -n '2,11p' "$0"; exit 0 ;;
        *) echo "未知参数: $a（-h 看用法）" >&2; exit 2 ;;
    esac
done

# ---------- 输出 ----------
if [[ -t 1 ]]; then
    G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; B=$'\e[36m'; D=$'\e[2m'; N=$'\e[0m'
else
    G=; Y=; R=; B=; D=; N=
fi
ok()   { echo "  ${G}✅${N} $*"; }
warn() { echo "  ${Y}⚠️ ${N} $*"; }
bad()  { echo "  ${R}❌${N} $*"; }
info() { echo "  ${B}ℹ️ ${N} $*"; }
step() { echo; echo "${B}▸ $*${N}"; }

DEGRADED=()
mark_degraded() { DEGRADED+=("$1"); }

# ---------- 读 .env（不 source，避免值里的特殊字符被执行） ----------
# 分层顺序与 app/core/config.py 一致：先 .env，再用 .env.${APP_ENV} 覆盖。
# 曾经这里**只**读 .env，于是 `APP_ENV=prod bash scripts/start_all.sh --strict`
# （README 与 deploy/README.md 里写的生产启动命令）会静默错配：
#   脚本按 .env 的 dev 依赖做自检，而应用（经 run.sh 把 APP_ENV 继承给 python）
#   实际按 .env.prod 连生产组件 —— 自检全绿，连的却是另一套东西。
#   本机实测过：APP_ENV=prod 跑自检，报的是 `ollama`/`172.27.224.1:3306`（dev 的值），
#   而不是 .env.prod 里的 `openai_compat`/`127.0.0.1:3306`。
env_files() {
    printf '%s\n' ".env"
    [[ -n "${APP_ENV:-}" ]] && printf '%s\n' ".env.${APP_ENV}"
    return 0
}

env_get() {
    local key="$1" default="${2-}" line f val=""
    # 真实环境变量优先级最高，与应用同序（app/core/config.py：env > .env.{APP_ENV} > .env）。
    # 少了这一条，本函数返回的文件值被赋给**已导出**的同名变量后会静默顶掉调用者传的环境变量
    # （变量已导出，赋值照样传给子进程）。实测（2026-09-21）：
    #   EMBED_MODEL=bge-m3 APP_ENV=prod bash scripts/start_all.sh
    # 里那个 bge-m3 被 .env.prod 的 BAAI/bge-m3 覆盖，应用 /v1/embeddings 请求 404，
    # /health 报 embedding unavailable——而 seed 是直接继承 shell 环境，反而生效，
    # 于是「灌库成功、检索却报向量化不可用」这种自相矛盾的现象。
    # 只影响下面列出的那 11 个键（EMBED_BASE_URL 之类不在其中，故此前侥幸没被顶掉）。
    val="${!key-}"
    if [[ -n "$val" ]]; then printf '%s' "$val"; return 0; fi
    val=""
    while IFS= read -r f; do
        line="$(grep -E "^[[:space:]]*${key}=" "$f" 2>/dev/null | tail -1 || true)"
        # 用 if 而不是 `[[ ]] && …`：后者在值为空时整条返回 1，脚本开着 set -e，
        # 虽因处于 && 列表而豁免，但把它留给「最后一行是短路列表」这种坑不值当
        if [[ -n "$line" ]]; then val="${line#*=}"; fi
    done < <(env_files)
    [[ -n "$val" ]] || { printf '%s' "$default"; return; }
    line="$val"                # 已去掉 `KEY=` 前缀，别再去一次——值里可能含 `=`（如密码）
    line="${line#"${line%%[![:space:]]*}"}"          # 去前导空格
    line="${line%$'\r'}"                              # 去 CRLF
    if [[ "$line" == \"*\" || "$line" == \'*\' ]]; then line="${line:1:${#line}-2}"; fi
    printf '%s' "$line"
}

# ---------- 探测工具（用 python3 带超时，避免 /dev/tcp 在丢包端口上挂死） ----------
url_hostport() { # <url> -> "host port"（缺省端口按 scheme 补）
    python3 - "$1" <<'PY'
import sys
from urllib.parse import urlparse
u = urlparse(sys.argv[1])
default = {"redis": 6379, "mysql": 3306, "http": 80, "https": 443}.get(u.scheme, 0)
print(u.hostname or "localhost", u.port or default)
PY
}

tcp_probe() { # <host> <port> [timeout]
    python3 - "$1" "$2" "${3:-3}" <<'PY'
import socket, sys
s = socket.socket(); s.settimeout(float(sys.argv[3]))
try:
    s.connect((sys.argv[1], int(sys.argv[2])))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
}

redis_ping() { # <redis url> -> 0 表示通了且回 PONG
    python3 - "$1" <<'PY'
import socket, sys
from urllib.parse import urlparse
u = urlparse(sys.argv[1]); host = u.hostname or "localhost"; port = u.port or 6379
s = socket.socket(); s.settimeout(3)
try:
    s.connect((host, port)); s.sendall(b"PING\r\n")
    if b"PONG" not in s.recv(64): sys.exit(1)
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
}

http_ok() { curl -sf -m 6 -o /dev/null "$1"; }

# ---------- 拉起 Windows 侧的依赖（WSL interop） ----------
#
# 为什么值得做：Ollama / MySQL / Docker Desktop 都跑在 Windows 上，而使用者双击的是
# WSL 里的脚本。它们没起来时，人看到的只是一句「Ollama 不可达」，很容易判定成"项目坏了"。
# 能拉起的就拉起来——下面对每个依赖都是先探测、不通才尝试、失败照旧降级并给出人工步骤。
#
# 全是 Windows 专有路径（/mnt/c/...、cmd.exe、sc.exe、net.exe），在 Linux 云主机上
# 这些判断会自然失败并返回 1，调用方回落成原来的提示，不影响部署。

# 返回第一个存在的候选可执行文件（WSL 里能直接执行 .exe）
windows_exe() {
    local p
    for p in "$@"; do
        [[ -x "$p" ]] && { printf '%s' "$p"; return 0; }
    done
    return 1
}

# Windows 的 %LOCALAPPDATA% 转成 WSL 路径。
# **不能拼死路径**：WSL 用户名（zhongfeiyu）和 Windows 用户名（15277）常常不是同一个。
win_localappdata() {
    local d
    d="$(cmd.exe /c echo '%LOCALAPPDATA%' 2>/dev/null | tr -d '\r\n' || true)"
    [[ -n "$d" && "$d" != "%LOCALAPPDATA%" ]] || return 1
    wslpath -u "$d" 2>/dev/null
}

# 用 cmd 的 start 拉起 Windows 程序。
# 不直接执行 .exe：那样它会挂在脚本的进程组上，脚本退出或被 Ctrl-C 时可能被连带杀掉。
start_windows_app() {
    cmd.exe /c start "" "$(wslpath -w "$1")" >/dev/null 2>&1
}

# 拉起 Ollama 并等它应答。<root> 形如 http://172.27.224.1:11434
start_ollama_and_wait() {
    local root="$1" exe local_app
    local_app="$(win_localappdata || true)"
    exe="$(windows_exe \
        "${local_app:+$local_app/Programs/Ollama/ollama app.exe}" \
        "/mnt/c/Program Files/Ollama/ollama.exe" \
        "/mnt/c/Program Files/Ollama/ollama app.exe" || true)"
    if [[ -z "$exe" ]]; then
        info "没找到 Windows 侧的 Ollama 可执行文件，无法自动拉起"
        return 1
    fi
    info "尝试拉起 Ollama：${exe}"
    start_windows_app "$exe"
    # 托盘程序从起来到能应答本机实测约 3-8 秒；给 30 秒余量（首次加载模型列表稍慢）
    local i
    for i in $(seq 1 30); do
        http_ok "${root}/api/tags" && { ok "Ollama 已拉起: ${root}"; return 0; }
        sleep 1
    done
    return 1
}

# 找出 Windows 服务里名字像 MySQL 的那个（MySQL80 / MySQL57 / MySQL ...）
windows_mysql_service() {
    local n
    n="$(/mnt/c/Windows/System32/sc.exe query type= service state= all 2>/dev/null \
        | tr -d '\r' \
        | awk '/^SERVICE_NAME:/ && tolower($2) ~ /mysql/ {print $2; exit}' || true)"
    [[ -n "$n" ]] && printf '%s' "$n"
    return 0
}

# 拉起 Docker Desktop 并等引擎应答。
# 注意它启动要 30-90 秒（要先把 Linux 虚拟机拉起来），比 Ollama 慢得多。
start_docker_desktop_and_wait() { # <docker 可执行文件，用调用方已发现的那个>
    local docker_bin="$1" exe="/mnt/c/Program Files/Docker/Docker/Docker Desktop.exe" i
    [[ -x "$exe" ]] || { info "没找到 Docker Desktop.exe，无法自动拉起"; return 1; }
    info "尝试拉起 Docker Desktop：${exe}"
    start_windows_app "$exe"
    for i in $(seq 1 90); do
        if timeout 25 "$docker_bin" version --format '{{.Server.Version}}' >/dev/null 2>&1; then
            ok "Docker 引擎已就绪"
            return 0
        fi
        sleep 1
    done
    return 1
}

# 把 URL 里的 localhost 换成 WSL 的 Windows 主机 IP（WSL2 里 localhost 连不到 Windows 侧服务）
host_ip_swap() { # <url> -> 换掉主机名后的 url，叫不出主机 IP 就原样返回
    local ip
    ip="$(ip route show default 2>/dev/null | awk '{print $3; exit}')" || true
    [[ -n "${ip:-}" ]] || { printf '%s' "$1"; return; }
    printf '%s' "$1" | sed -E "s#(//)(localhost|127\.0\.0\.1)([:/])#\1${ip}\3#"
}

# ---------- 配置 ----------
LLM_PROVIDER="$(env_get LLM_PROVIDER ollama)"
EMBED_PROVIDER="$(env_get EMBED_PROVIDER ollama)"
OLLAMA_BASE_URL="$(env_get OLLAMA_BASE_URL http://localhost:11434/v1)"
LLM_MODEL="$(env_get LLM_MODEL '')"
EMBED_MODEL="$(env_get EMBED_MODEL '')"
SQL_URL="$(env_get SQL_URL 'sqlite:///./data/app.db')"
MILVUS_URI="$(env_get MILVUS_DB_URI './data/milvus.db')"
MEMORY_BACKEND="$(env_get MEMORY_BACKEND memory)"
REDIS_URL="$(env_get REDIS_URL 'redis://localhost:6379/0')"
RERANKER="$(env_get RERANKER score_fusion)"
RERANK_BASE_URL="$(env_get RERANK_BASE_URL 'http://127.0.0.1:8001/v1')"

HOST="${HOST:-$(env_get HOST 0.0.0.0)}"
PORT="${PORT:-$(env_get PORT 8000)}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"

echo "RAG 角色扮演系统 - 一键启动"
echo "${D}项目目录: $(pwd)${N}"
# 把生效的配置层打出来：否则「自检报的值和应用实际连的不是一套」这种错配没人看得出来
if [[ -n "${APP_ENV:-}" ]]; then
    if [[ -f ".env.${APP_ENV}" ]]; then
        echo "${D}环境: APP_ENV=${APP_ENV}（.env ← 被 .env.${APP_ENV} 覆盖）${N}"
    else
        echo "${Y}环境: APP_ENV=${APP_ENV}，但 .env.${APP_ENV} 不存在 —— 只会用 .env${N}"
    fi
else
    echo "${D}环境: 默认（只读 .env；要切环境用 APP_ENV=dev/test/prod）${N}"
fi

# ---------- 1. Python / venv ----------
step "[1/6] Python 环境"
if [[ ! -d .venv ]]; then
    bad "未找到 .venv —— 先跑安装脚本： bash scripts/install.sh"
    exit 1
fi
ok "已找到 .venv（$(.venv/bin/python --version 2>&1)）"

# ---------- 2. LLM / Embedding（Ollama 或在线 API） ----------
step "[2/6] 大模型 / 向量化（${LLM_PROVIDER} / ${EMBED_PROVIDER}）"
if [[ "$LLM_PROVIDER" == "dummy" && "$EMBED_PROVIDER" == "dummy" ]]; then
    warn "dummy 离线模式：不调模型，走回显答案（仅用于冒烟）"
elif [[ "$LLM_PROVIDER" == "ollama" || "$EMBED_PROVIDER" == "ollama" ]]; then
    OLLAMA_ROOT="${OLLAMA_BASE_URL%/v1}"
    # 不通就先试着拉起（Windows 侧），再走原来的可达性判断——这样成功路径与从前一字不差
    if ! http_ok "${OLLAMA_ROOT}/api/tags"; then
        start_ollama_and_wait "$OLLAMA_ROOT" || true
    fi
    if http_ok "${OLLAMA_ROOT}/api/tags"; then
        tags="$(curl -s -m 6 "${OLLAMA_ROOT}/api/tags" || true)"
        want_models=()
        if [[ "$LLM_PROVIDER" == "ollama" && -n "$LLM_MODEL" ]]; then want_models+=("$LLM_MODEL"); fi
        if [[ "$EMBED_PROVIDER" == "ollama" && -n "$EMBED_MODEL" ]]; then want_models+=("$EMBED_MODEL"); fi
        for m in ${want_models[@]+"${want_models[@]}"}; do
            # Ollama 会把不带 tag 的名字补成 :latest（bge-m3 -> bge-m3:latest），两种都算命中
            if grep -qE "\"name\"[[:space:]]*:[[:space:]]*\"${m}(:latest)?\"" <<<"$tags"; then
                ok "模型在位: ${m}"
            else
                mark_degraded "模型 ${m} 未拉取"
                bad "缺少模型 ${m} —— 在 Windows 侧执行: ollama pull ${m}"
            fi
        done
        ok "Ollama 可达: ${OLLAMA_ROOT}"
    else
        mark_degraded "Ollama 不可达"
        bad "Ollama 不可达: ${OLLAMA_ROOT}"
        info "先在 Windows 侧启动 Ollama；若 Ollama 只在 Windows 上跑，"
        info "WSL 里要用主机 IP 而不是 localhost（本机 IP: $(ip route show default | awk '{print $3; exit}')）"
    fi
else
    info "在线 API 模式（${LLM_PROVIDER}），跳过本地探测"
fi

# ---------- 3. 关系库 ----------
step "[3/6] 关系库"
if [[ "$SQL_URL" == sqlite* ]]; then
    db_path="${SQL_URL#sqlite:///}"
    if [[ -f "$db_path" ]]; then
        ok "SQLite: ${db_path}"
    else
        info "SQLite 文件还没建（${db_path}），首次启动时会自动建表"
    fi
else
    read -r sql_host sql_port <<<"$(url_hostport "$SQL_URL")"
    # 不通先试着把 Windows 服务拉起来（MySQL80/MySQL57/MySQL…）
    if ! tcp_probe "$sql_host" "$sql_port"; then
        sql_svc="$(windows_mysql_service)"
        if [[ -n "$sql_svc" ]]; then
            info "MySQL 没通，尝试启动 Windows 服务 ${sql_svc}…"
            # net start 对**已在运行**的服务返回 0，对需要管理员权限的会失败（那时回落成人工步骤）
            if /mnt/c/Windows/System32/net.exe start "$sql_svc" >/dev/null 2>&1; then
                for _ in $(seq 1 30); do
                    tcp_probe "$sql_host" "$sql_port" && break
                    sleep 1
                done
            else
                info "启动服务失败（多半需要管理员权限），回落成人工步骤"
            fi
        fi
    fi
    if tcp_probe "$sql_host" "$sql_port"; then
        ok "MySQL 可达: ${sql_host}:${sql_port}"
    else
        mark_degraded "关系库不可达"
        bad "MySQL 不可达: ${sql_host}:${sql_port}"
        info "确认 Windows 侧的 MySQL 服务已启动；WSL 里连本机服务要用主机 IP"
        if [[ -n "$(windows_mysql_service)" ]]; then
            # 实测：非管理员下 `net start` 直接返回「拒绝访问」(exit 2)，所以这一步只能给人工命令，
            # 自动拉起做不到（除非弹 UAC——那对一个"一键启动"脚本是更糟的体验）。
            info "启动它需要管理员权限（实测非管理员会「拒绝访问」）。三选一："
            info "  1) Windows 管理员 PowerShell： Start-Service $(windows_mysql_service)"
            info "  2) 普通 PowerShell 弹 UAC： powershell -Command \"Start-Process net -ArgumentList 'start $(windows_mysql_service)' -Verb RunAs\""
            info "  3) 打开「服务」面板，启动 $(windows_mysql_service)"
        fi
    fi
fi

# ---------- 4. 短期记忆（Redis） ----------
step "[4/6] 短期记忆（${MEMORY_BACKEND}）"
if [[ "$MEMORY_BACKEND" != "redis" ]]; then
    info "MEMORY_BACKEND=${MEMORY_BACKEND}，用进程内内存，跳过"
elif redis_ping "$REDIS_URL"; then
    ok "Redis 可达: ${REDIS_URL}"
else
    info "Redis 没通，先尝试自动拉起…"
    HINTS=()

    # 路径一：本机装了 redis-server 就直接起
    if command -v redis-server >/dev/null 2>&1; then
        rp="$(url_hostport "$REDIS_URL" | awk '{print $2}')"
        info "本机有 redis-server，拉起（端口 ${rp}）…"
        redis-server --daemonize yes --port "$rp" --save '' >/dev/null 2>&1 || true
        for _ in 1 2 3 4 5; do redis_ping "$REDIS_URL" && break; sleep 1; done
    fi

    # 路径二：本项目的 Redis 一直是 Docker 容器 redis-memory（跑在 Windows 侧 Docker Desktop 里）。
    # WSL 的 localhost 不转发到 Windows，所以顺带用主机 IP 试一次。
    if ! redis_ping "$REDIS_URL"; then
        swapped="$(host_ip_swap "$REDIS_URL")"
        if [[ "$swapped" != "$REDIS_URL" ]] && redis_ping "$swapped"; then
            HINTS+=("Redis 其实活着，只是地址不对——把 .env 改成 REDIS_URL=${swapped}（WSL2 的 localhost 不转发到 Windows）")
        fi
    fi

    # 路径三：找 docker.exe 看容器状态。WSL 里的 docker 命令要开了 WSL Integration 才有，
    # 所以优先直接调 Windows 上的 docker.exe——它也是判断 Docker Desktop 开没开的判据。
    if ! redis_ping "$REDIS_URL"; then
        DOCKER_EXE=""
        for c in docker.exe "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"; do
            if command -v "$c" >/dev/null 2>&1 || [[ -x "$c" ]]; then DOCKER_EXE="$c"; break; fi
        done
        if [[ -z "$DOCKER_EXE" ]]; then
            HINTS+=("本机没有 docker.exe，可以直接装： sudo apt install -y redis-server && sudo service redis-server start")
        else
            ver_out="$(timeout 25 "$DOCKER_EXE" version --format '{{.Server.Version}}' 2>&1 || true)"
            if grep -qiE 'pipe|daemon is running|cannot connect' <<<"$ver_out"; then
                # 引擎没起——重启后最常见的情形（Docker Desktop 不随开机自动跑）。
                # 以前这里只留一句"请手动启动 Docker Desktop"，现在直接拉起它：
                # 这是三条依赖里等待最久的（30-90 秒，要先把 Linux 虚拟机带起来）。
                if start_docker_desktop_and_wait "$DOCKER_EXE"; then
                    # 容器设了 --restart unless-stopped，引擎一回来通常自己就起了；
                    # 这里再显式 start 一次兜底（没起来时它是幂等的）。
                    timeout 30 "$DOCKER_EXE" start redis-memory >/dev/null 2>&1 || true
                    for _ in 1 2 3 4 5; do redis_ping "$REDIS_URL" && break; sleep 1; done
                fi
                if ! redis_ping "$REDIS_URL"; then
                    HINTS+=("根因：Windows 侧的 Docker Desktop 没启动，容器 redis-memory 随之消失")
                    HINTS+=("判据：docker.exe 能跑、但连不上 named pipe —— 不是代码故障，也不是 WSL Integration 开关的问题")
                    HINTS+=("修：在 Windows 侧启动 Docker Desktop（容器设了 --restart unless-stopped，会自动回来），再重跑本脚本")
                fi
            else
                st="$(timeout 25 "$DOCKER_EXE" ps -a --filter name=^redis-memory$ --format '{{.Status}}' 2>/dev/null | head -1 || true)"
                if [[ -z "$st" ]]; then
                    HINTS+=("Docker 活着，但没有 redis-memory 容器。建一个：")
                    HINTS+=("  docker.exe run -d --name redis-memory --restart unless-stopped -p 6379:6379 --appendonly yes redis:7-alpine")
                elif grep -qi '^Up' <<<"$st"; then
                    HINTS+=("容器 redis-memory 在跑（${st}）但仍连不上 ${REDIS_URL}")
                    HINTS+=("查端口映射 docker.exe port redis-memory，以及 Windows 防火墙")
                else
                    info "容器 redis-memory 没在跑（${st}），自动拉起…"
                    timeout 30 "$DOCKER_EXE" start redis-memory >/dev/null 2>&1 || true
                    for _ in 1 2 3 4 5; do redis_ping "$REDIS_URL" && break; sleep 1; done
                    if ! redis_ping "$REDIS_URL"; then
                        HINTS+=("容器已 start 但仍连不上 ${REDIS_URL}：查端口映射与 Windows 防火墙")
                    fi
                fi
            fi
        fi
    fi

    if redis_ping "$REDIS_URL"; then
        ok "Redis 已拉起: ${REDIS_URL}"
    else
        mark_degraded "Redis 不可达（记忆降级为进程内）"
        bad "Redis 不可达: ${REDIS_URL}"
        for h in ${HINTS[@]+"${HINTS[@]}"}; do info "$h"; done
        info "不修也能跑：对话正常，只是上下文只活在进程里，重启即丢"
    fi
fi

# ---------- 5. 重排服务 ----------
step "[5/6] 重排服务（RERANKER=${RERANKER}）"
if [[ "$RERANKER" == "bge" ]]; then
    bash scripts/run_rerank.sh

    # 必须等重排真的能应答，不能起完就走人。
    # 重排冷启动要 import torch + 从盘上加载 ~2GB 权重（实测 1–2 分钟），而下面 [6/6]
    # 只轮询 app 自己的 /health——app 起来比重排快得多，于是启动后第一个 /chat 会吃
    # Connection refused，并**静默降级成 score_fusion**：只在 app.log 留一条 WARNING，
    # /health 照样全绿，从接口层完全看不出精排没生效。宁可在这一步多等，也别让它假绿。
    RERANK_ROOT="${RERANK_BASE_URL%/v1}"     # http://127.0.0.1:8001/v1 -> http://127.0.0.1:8001
    RERANK_HEALTH="${RERANK_ROOT%/}/health"
    info "等待重排服务就绪（冷启动要加载权重，通常 1–2 分钟）…"
    if wait_health "$RERANK_HEALTH" 60 "" 3; then
        ok "重排服务已就绪: ${RERANK_ROOT}"
    else
        mark_degraded "重排服务未就绪（精排会降级为融合分数）"
        bad "重排服务约 3 分钟未就绪: ${RERANK_HEALTH}"
        info "看进程是否还在: ps -eo pid,args | grep '[u]vicorn rerank_service'"
        info "别用日志判活——logs/rerank-*.out 走块缓冲，空白不代表没在跑"
    fi
else
    info "score_fusion 融合分数，不需要独立重排服务"
fi

# ---------- 6. 起应用 + 校验 ----------
if [[ "$DO_START" == "0" ]]; then
    step "[6/6] --no-start：跳过启动"
    if [[ ${#DEGRADED[@]} -gt 0 ]]; then
        echo; bad "降级项: ${DEGRADED[*]}"
        [[ "$STRICT" == "0" ]] || exit 3
    fi
    exit 0
fi

step "[6/6] 启动应用"
if [[ -n "$MILVUS_URI" && "$MILVUS_URI" != http* ]]; then
    info "Milvus Lite（${MILVUS_URI}）随应用进程内嵌启动，无需单独起"
fi
bash scripts/run.sh

info "等待 ${HEALTH_URL} 就绪…"
# 响应体顺手落盘：下面打印组件表格要用（wait_health 的第 3 个参数）
if ! wait_health "$HEALTH_URL" 30 logs/.health.json; then
    bad "健康检查超时（30s）—— 应用没起来"
    echo "  ${D}---- logs/uvicorn.out 末尾 ----${N}"
    tail -20 logs/uvicorn.out 2>/dev/null | sed 's/^/  /' || true
    exit 1
fi

echo
echo "${B}──────── 启动结果 ────────${N}"
python3 - <<PY
import json
names = {"llm": "大模型", "embedding": "向量化", "milvus": "向量库", "sql": "关系库", "memory": "短期记忆"}
try:
    d = json.load(open("logs/.health.json"))
except Exception as e:
    print(f"  解析 /health 失败: {e}"); raise SystemExit(1)
status = d.get("status", "unknown")
for k, v in (d.get("components") or {}).items():
    icon = "✅" if v == "ok" else ("⚠️ " if v == "degraded" else "❌")
    print(f"  {icon} {names.get(k, k):<8} {v}")
print(f"  总体: {status}")
PY

pid="$(cat logs/uvicorn.pid 2>/dev/null || echo '?')"
echo
echo "  应用:   http://localhost:${PORT}   ${D}(pid ${pid}, 日志 logs/uvicorn.out)${N}"
echo "  健康:   curl -s ${HEALTH_URL}"
echo "  对话:   curl -X POST http://localhost:${PORT}/chat -H 'Content-Type: application/json' \\"
echo "            -d '{\"question\":\"试用期最长可以约定多久？\",\"role_id\":\"lawyer\"}'"
echo "  停止:   bash scripts/shutdown.sh"

# 数据自检：库里没知识时提示 seed（seed 前必须先停服务，Milvus Lite 是单进程锁）
docs="$(curl -s -m 8 "http://127.0.0.1:${PORT}/knowledge/list" | python3 -c 'import sys,json; print(len(json.load(sys.stdin).get("documents",[])))' 2>/dev/null || echo -1)"
roles="$(curl -s -m 8 "http://127.0.0.1:${PORT}/role/list" | python3 -c 'import sys,json; print(len(json.load(sys.stdin)))' 2>/dev/null || echo -1)"
if [[ "$docs" == "0" ]]; then
    echo
    warn "知识库是空的（0 篇文档），检索会没有资料可召回"
    info "先停服务再灌数据（Milvus Lite 单进程锁，不停会 DataDirLockedError）："
    info "  bash scripts/shutdown.sh && .venv/bin/python scripts/seed.py && bash scripts/start_all.sh"
elif [[ "$docs" != "-1" ]]; then
    echo
    ok "已入库 ${docs} 篇文档 / ${roles} 个角色"
fi

if [[ ${#DEGRADED[@]} -gt 0 ]]; then
    echo
    warn "降级项: ${DEGRADED[*]}"
    [[ "$STRICT" == "0" ]] || exit 3
fi
exit 0
