#!/bin/bash
# 工单编号：人工智能NLP-RAG-金融问答系统部署
#
# 部署验收测试：把工单的三条验收标准逐条跑成可复现的检查。
#
# 用法：bash 测试/验收测试.sh
# 需要先在 工单10 目录下完成构建、灌模型，并配好 部署/.env。
# 脚本自己负责清理它创建的临时容器，不会动 rag-app 与数据卷。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
COMPOSE="${ROOT}/部署/docker-compose.yml"
ENVFILE="${ROOT}/部署/.env"
IMAGE="rag-finance:1.0"
APP="rag-app"
RTMP="rtmp-server"
PORT="${APP_PORT:-7860}"
# 验收用的临时容器换个宿主机端口：rag-app 正占着 APP_PORT，
# 同端口起第二个容器会直接失败，换成 +1 各跑各的。
VERIFY_PORT="$((PORT+1))"
LOG="${HERE}/验收结果.log"

PASS=0; FAIL=0
: > "${LOG}"

say()  { echo "$*" | tee -a "${LOG}"; }
ok()   { PASS=$((PASS+1)); say "  [通过] $*"; }
bad()  { FAIL=$((FAIL+1)); say "  [失败] $*"; }
head_() { say ""; say "════ $* ════"; }

say "工单编号：人工智能NLP-RAG-金融问答系统部署"
say "验收时间：$(date '+%Y-%m-%d %H:%M:%S')"
say "Docker  ：$(docker version --format '{{.Server.Version}}' 2>/dev/null)"

# ─────────────────────────────────────────────
head_ "验收一：容器启动与运行"

if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    ok "镜像 ${IMAGE} 存在（$(docker image inspect ${IMAGE} --format '{{.Size}}' | awk '{printf "%.2f GB", $1/1024/1024/1024}')）"
else
    bad "镜像 ${IMAGE} 不存在，请先 docker build"
    say "后续检查无法进行，退出。"; exit 1
fi

# 1.1 单容器 docker run（不依赖 compose）—— 验收原文就是 docker run
say "  · 用纯 docker run 起一个临时容器验证（宿主机端口 ${VERIFY_PORT}）"
docker rm -f rag-verify >/dev/null 2>&1
if docker run -d --name rag-verify --gpus all -p "${VERIFY_PORT}:7860" \
        -v rag-finance_model-data:/models:ro \
        -v rag-finance_kb-data:/kb \
        -e DEEPSEEK_BASE_URL="$(grep -E '^DEEPSEEK_BASE_URL=' "${ENVFILE}" | cut -d= -f2-)" \
        -e DEEPSEEK_API_KEY="$(grep -E '^DEEPSEEK_API_KEY=' "${ENVFILE}" | cut -d= -f2-)" \
        "${IMAGE}" >/dev/null 2>&1; then
    ok "docker run 启动成功"
else
    bad "docker run 启动失败"
    docker logs rag-verify 2>&1 | tail -20 | tee -a "${LOG}"
fi

# 1.2 等待端口提供服务
say "  · 等待服务就绪（最多 240 秒，首次要加载 BGE-M3）"
READY=0
for i in $(seq 1 48); do
    if curl -fsS "http://127.0.0.1:${VERIFY_PORT}/" >/dev/null 2>&1; then READY=1; break; fi
    sleep 5
done
if [ "${READY}" = "1" ]; then
    ok "指定端口 ${VERIFY_PORT} 已提供服务（第 $((i*5)) 秒就绪）"
else
    bad "端口 ${VERIFY_PORT} 在 240 秒内未提供服务"
    docker logs rag-verify 2>&1 | tail -30 | tee -a "${LOG}"
fi

# 1.3 容器状态与健康检查
STATE="$(docker inspect rag-verify --format '{{.State.Status}}' 2>/dev/null)"
[ "${STATE}" = "running" ] && ok "容器状态 running" || bad "容器状态为 ${STATE}"

# 1.4 无异常日志
say "  · 检查容器日志有无异常"
LOGS="$(docker logs rag-verify 2>&1)"
echo "${LOGS}" > "${HERE}/rag-verify.log"
if echo "${LOGS}" | grep -qiE "traceback|exception|error|critical|失败|错误"; then
    bad "日志中发现异常关键字："
    echo "${LOGS}" | grep -iE "traceback|exception|error|critical|失败|错误" | head -10 | tee -a "${LOG}"
else
    ok "日志无异常关键字（全文见 测试/rag-verify.log）"
fi

# ─────────────────────────────────────────────
head_ "验收二：容器数据管理"

# 2.1 卷存在
for v in rag-finance_model-data rag-finance_kb-data; do
    if docker volume inspect "${v}" >/dev/null 2>&1; then
        ok "数据卷 ${v} 存在"
    else
        bad "数据卷 ${v} 不存在"
    fi
done

# 2.2 知识库落在卷里
# 路径包进 sh -c '...' 里：Git Bash 会把以 / 开头的参数转成 Windows 路径
# （`/kb/x` → `C:/Program Files/Git/kb/x`），包进字符串就不会被转。
KBFILES="$(docker exec rag-verify sh -c 'ls /kb/ccf_competition' 2>/dev/null | tr '\n' ' ')"
if echo "${KBFILES}" | grep -q "chunks.json"; then
    ok "知识库在数据卷内：/kb/ccf_competition（${KBFILES}）"
else
    bad "数据卷里没有知识库：${KBFILES}"
fi

# 2.3 持久化：删掉容器重建，数据仍在
say "  · 删除容器后重建，验证数据未丢失"
FEEDBACK_MARK="persist-check-$(date +%s)"
docker exec rag-verify sh -c "echo '${FEEDBACK_MARK}' > /kb/persist_probe.txt" 2>/dev/null
docker rm -f rag-verify >/dev/null 2>&1
docker run -d --name rag-verify --gpus all -p "${VERIFY_PORT}:7860" \
    -v rag-finance_model-data:/models:ro -v rag-finance_kb-data:/kb \
    -e DEEPSEEK_BASE_URL="$(grep -E '^DEEPSEEK_BASE_URL=' "${ENVFILE}" | cut -d= -f2-)" \
    -e DEEPSEEK_API_KEY="$(grep -E '^DEEPSEEK_API_KEY=' "${ENVFILE}" | cut -d= -f2-)" \
    "${IMAGE}" >/dev/null 2>&1
sleep 5
if [ "$(docker exec rag-verify sh -c 'cat /kb/persist_probe.txt' 2>/dev/null)" = "${FEEDBACK_MARK}" ]; then
    ok "容器重建后数据仍在（卷持久化生效）"
else
    bad "容器重建后数据丢失"
fi
docker exec rag-verify sh -c "rm -f /kb/persist_probe.txt" 2>/dev/null

# 2.4 容器间数据共享
say "  · 起第二个容器挂同一个卷，验证共享"
SHARED="$(docker run --rm -v rag-finance_kb-data:/kb alpine:3.20 \
    sh -c 'ls /kb/ccf_competition/chunks.json 2>/dev/null && echo SHARED_OK' 2>/dev/null)"
if echo "${SHARED}" | grep -q "SHARED_OK"; then
    ok "另一容器可读到同一个卷里的知识库（容器间数据共享生效）"
else
    bad "另一容器读不到共享卷内容"
fi

# ─────────────────────────────────────────────
head_ "验收三：网络配置"

# 3.1 网络存在且两个服务都在上面
NET="$(docker network ls --format '{{.Name}}' | grep -E 'rag-finance.*rag-net|rag-net' | head -1)"
if [ -n "${NET}" ]; then
    ok "自定义网络 ${NET} 存在"
else
    bad "未找到自定义网络 rag-net"
fi

# 3.2 rag-app 能解析并连通 rtmp-server
say "  · 从 rag-app 容器内探测 RTMP 服务（容器名 + 端口）"
# 探测对象是 compose 部署的 rag-app，不是临时容器 rag-verify ——
# rag-verify 没接入 rag-net，本来就解析不到 rtmp-server，拿它探是探错了对象。
PROBER="${APP}"
if docker ps --format '{{.Names}}' | grep -q "^${RTMP}$"; then
    ok "RTMP 服务容器 ${RTMP} 在运行"
    # 用 python 做 TCP 连通性探测（应用镜像里有 python）
    PROBE="$(docker exec ${PROBER} python -c "
import socket
for host,port,name in [('${RTMP}',1935,'RTMP'),('${RTMP}',80,'HTTP状态页')]:
    s=socket.socket(); s.settimeout(5)
    try:
        s.connect((host,port)); print(f'{name} {host}:{port} 可达')
    except Exception as e:
        print(f'{name} {host}:{port} 不可达 {e}')
    finally:
        s.close()
" 2>&1)"
    echo "${PROBE}" | tee -a "${LOG}"
    if echo "${PROBE}" | grep -q "RTMP ${RTMP}:1935 可达"; then
        ok "rag-app → rtmp-server:1935 网络连通"
    else
        bad "rag-app 无法连通 rtmp-server:1935（两容器需在同一网络；本检查要求用 compose 起过服务）"
    fi
else
    bad "RTMP 容器未运行，请先 docker compose up -d"
fi

# 3.3 DNS 解析
DNS="$(docker exec ${PROBER} python -c "
import socket
try: print('解析到', socket.gethostbyname('${RTMP}'))
except Exception as e: print('ERR', e)
" 2>&1)"
if echo "${DNS}" | grep -q "解析到"; then
    ok "容器名 DNS 解析正常（${DNS}）"
else
    bad "容器名解析失败：${DNS}"
fi

# ─────────────────────────────────────────────
head_ "功能：金融问答服务"

say "  · 在容器内跑一次完整 RAG 链路（BGE-M3 检索 + 大模型生成）"
QA="$(docker exec rag-verify python -c "
import sys; sys.path.insert(0,'/app')
import app
app.load_kb('ccf_competition')
ans, ctx, meta, _ = app.ask('平安银行2019年末的拨备覆盖率是多少？', True, True,
                             'RAG', 'hybrid', 0.5, 'none', [])
print('ANSWER:', ans[:200].replace(chr(10),' '))
print('META:', meta[:200].replace(chr(10),' '))
" 2>&1)"
echo "${QA}" | tee -a "${LOG}"
if echo "${QA}" | grep -q "183.12"; then
    ok "金融问答正确（问到拨备覆盖率 183.12%）"
elif echo "${QA}" | grep -q "ANSWER:"; then
    bad "服务有回答但内容不含预期数值 183.12"
else
    bad "问答链路执行失败"
fi

# ─────────────────────────────────────────────
head_ "清理"
docker rm -f rag-verify >/dev/null 2>&1 && say "  临时容器 rag-verify 已删除"

say ""
say "════════ 汇总 ════════"
say "通过 ${PASS} 项，失败 ${FAIL} 项"
say "完整日志：${LOG}"
[ "${FAIL}" -eq 0 ] && exit 0 || exit 1
