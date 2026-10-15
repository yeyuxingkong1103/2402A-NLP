#!/usr/bin/env bash
# 工单编号：人工智能NLP-RAG-金融问答系统部署
# 脚本职责：对部署结果做自动化验收，逐条对应工单验收标准并输出 通过/失败
# 使用方式：bash scripts/verify_deploy.sh
# 说明：需在工单十根目录或其上层执行，脚本会自动定位到工单十根目录

set -u

# 定位工单根目录（脚本位于 <root>/scripts/ 下）
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

IMAGE="rag-finance-qa:1.0.0"
CNAME="rag-verify-api"
PORT="18000"
PASS=0
FAIL=0

# 打印步骤标题
step() { printf "\n===== %s =====\n" "$1"; }
# 记录通过项
ok()   { printf "  [通过] %s\n" "$1"; PASS=$((PASS + 1)); }
# 记录失败项
bad()  { printf "  [失败] %s\n" "$1"; FAIL=$((FAIL + 1)); }
# 简单断言
check() { if [ "$1" = "0" ]; then ok "$2"; else bad "$2"; fi; }

# 清理上一次验收遗留的容器与卷
cleanup() {
  docker rm -f "$CNAME" >/dev/null 2>&1
}

step "一、容器启动与运行（验收标准 2.1）"
cleanup

# 1. 镜像必须已构建
docker image inspect "$IMAGE" >/dev/null 2>&1
check $? "镜像 $IMAGE 存在"

# 2. docker run 能成功启动并在指定端口提供服务（不依赖 compose）
docker run -d --name "$CNAME" \
  -p "${PORT}:8000" \
  -e DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}" \
  -e EMBED_ENABLED=false \
  -v rag-verify-data:/app/data \
  -v rag-verify-logs:/app/logs \
  "$IMAGE" >/dev/null 2>&1
check $? "docker run 启动容器"

# 3. 等待服务就绪（最长 120 秒）
READY=1
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then READY=0; break; fi
  sleep 2
done
check $READY "服务在端口 ${PORT} 就绪（/health 可访问）"

# 4. /health 返回内容正确
HEALTH="$(curl -fsS "http://127.0.0.1:${PORT}/health" 2>/dev/null)"
case "$HEALTH" in
  *'"status":"ok"'*) ok "/health 返回 status=ok（$HEALTH）" ;;
  *) bad "/health 返回异常：$HEALTH" ;;
esac

# 5. 容器运行日志中不得出现异常堆栈（验收标准 2.1）
if docker logs "$CNAME" 2>&1 | grep -qE "Traceback \(most recent call last\)|ERROR:"; then
  bad "容器日志存在异常输出"
  docker logs "$CNAME" 2>&1 | grep -E "Traceback|ERROR:" | head -5
else
  ok "容器运行过程中无异常日志输出"
fi

step "二、容器数据管理（验收标准 2.2）"
cleanup

# 1. 用命名卷启动，触发索引构建并写入数据卷
docker run -d --name "$CNAME" -p "${PORT}:8000" \
  -e EMBED_ENABLED=false \
  -v rag-verify-data:/app/data "$IMAGE" >/dev/null 2>&1
for _ in $(seq 1 60); do
  curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS "http://127.0.0.1:${PORT}/stats" >/dev/null 2>&1

# 2. 卷中确实产生了数据文件
VOLFILES="$(docker run --rm -v rag-verify-data:/d alpine:latest ls /d 2>/dev/null | tr '\n' ' ')"
if [ -n "$VOLFILES" ]; then ok "数据卷 rag-verify-data 已持久化文件：$VOLFILES"
else bad "数据卷中没有任何文件"; fi

# 3. 容器删除后数据仍在（验证持久化）
docker rm -f "$CNAME" >/dev/null 2>&1
VOLFILES2="$(docker run --rm -v rag-verify-data:/d alpine:latest ls /d 2>/dev/null | tr '\n' ' ')"
[ "$VOLFILES" = "$VOLFILES2" ]
check $? "容器删除后数据卷内容不变（数据不丢失）"

# 4. 两个容器挂载同一命名卷，可互相看到对方写入的文件（容器间数据共享）
docker run --rm -v rag-verify-data:/shared alpine:latest \
  sh -c 'echo hello-from-shared-volume > /shared/__share_probe.txt' >/dev/null 2>&1
docker run --rm -v rag-verify-data:/shared alpine:latest \
  sh -c 'grep -q hello-from-shared-volume /shared/__share_probe.txt' >/dev/null 2>&1
check $? "容器间数据共享（同一命名卷可读写）"
docker run --rm -v rag-verify-data:/shared alpine:latest \
  rm -f /shared/__share_probe.txt >/dev/null 2>&1

step "三、网络配置（验收标准 2.3）"
# 自建网络并验证两个容器可通过服务名互相访问
docker network inspect rag-verify-net >/dev/null 2>&1 || docker network create rag-verify-net >/dev/null 2>&1
check $? "创建自定义桥接网络 rag-verify-net"

docker rm -f rag-verify-peer >/dev/null 2>&1
docker run -d --name rag-verify-peer --network rag-verify-net \
  --network-alias peer-service alpine:latest \
  sh -c 'while true; do nc -l -p 1935 -e echo ok 2>/dev/null || sleep 1; done' >/dev/null 2>&1
docker run -d --name "$CNAME" --network rag-verify-net -p "${PORT}:8000" \
  -e EMBED_ENABLED=false -v rag-verify-data:/app/data "$IMAGE" >/dev/null 2>&1
for _ in $(seq 1 60); do
  docker exec "$CNAME" python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3)" >/dev/null 2>&1 && break
  sleep 2
done

# 服务容器内通过容器名访问对端服务：DNS 能解析即证明自定义网络生效
docker exec "$CNAME" python -c "
import socket, sys
socket.gethostbyname('peer-service')
sys.exit(0)
" >/dev/null 2>&1
check $? "服务容器可通过容器名解析到对端服务（自定义网络 DNS 生效）"

# 反向：对端容器也能解析到服务容器
docker exec rag-verify-peer sh -c 'nc -z -w 3 '"$CNAME"' 8000' >/dev/null 2>&1
check $? "对端容器可访问服务容器端口 8000"

step "四、服务接口测试（验收标准 1）"
# 仅在提供了 API Key 时测试完整问答（会调用大模型产生费用）
if [ -n "${DEEPSEEK_API_KEY:-}" ]; then
  CUR="$(curl -fsS -X POST "http://127.0.0.1:${PORT}/ask" \
        -H 'Content-Type: application/json' \
        -d '{"question":"平安银行2019年的不良贷款率是多少？","mode":"bm25"}' 2>/dev/null)"
  case "$CUR" in
    *'"answer"'*) ok "/ask 接口问答成功" ;;
    *) bad "/ask 接口返回异常：${CUR:0:200}" ;;
  esac
else
  printf "  [跳过] 未设置 DEEPSEEK_API_KEY，跳过 /ask 完整问答测试\n"
fi

# 检索接口不依赖大模型，始终可测
RET="$(curl -fsS -X POST "http://127.0.0.1:${PORT}/retrieve" \
      -H 'Content-Type: application/json' \
      -d '{"question":"不良贷款率","mode":"bm25"}' 2>/dev/null)"
case "$RET" in
  *'"contexts"'*) ok "/retrieve 接口检索成功" ;;
  *) bad "/retrieve 接口返回异常：${RET:0:200}" ;;
esac

step "五、清理"
cleanup
docker rm -f rag-verify-peer >/dev/null 2>&1
docker network rm rag-verify-net >/dev/null 2>&1
docker volume rm rag-verify-data rag-verify-logs >/dev/null 2>&1
printf "  已清理验收用容器、网络与卷\n"

printf "\n================ 验收汇总 ================\n"
printf "  通过：%d 项\n" "$PASS"
printf "  失败：%d 项\n" "$FAIL"
if [ "$FAIL" -eq 0 ]; then
  printf "  结论：全部验收项通过\n"
  exit 0
else
  printf "  结论：存在未通过项，请检查上方 [失败] 条目\n"
  exit 1
fi
