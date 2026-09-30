#!/usr/bin/env bash
# 多 worker 启动（配合 nginx 负载均衡）。
#
# ⚠️ 前提：必须 `RAGLORA_VECTOR_STORE=milvus`
# ---------------------------------------------------------------
# 本项目**原本无法多 worker** —— Qdrant 嵌入式对存储目录持有独占文件锁，
# 第二个 worker 打开同一目录会直接失败。切到 Milvus 后该限制才解除。
# 本脚本会先校验，向量库不是 milvus 就直接拒绝启动。
#
# ⚠️ 内存是真瓶颈
# ---------------------------------------------------------------
# **每个 worker 会各自加载一份 bge-m3 + 精排（约 3.4GB）** ——
# 模型实例是进程内的，不跨进程共享。本机 15.7GB：
#
#     2 个 worker  ≈ 6.8GB
#     3 个 worker  ≈ 10.2GB
#     再叠加 Milvus（2.5G）+ Dify  ≈ 爆
#
# 因此本脚本**默认以 RAGLORA_WARMUP=0 启动**（模型首次使用时才加载），
# 并把上面这段话打出来，避免有人在内存不足时起一堆 worker 然后困惑于 OOM。
#
# 用法:
#   bash backend/run_multi.sh 2          # 启动 2 个 worker（8000/8001）
#   bash backend/run_multi.sh 3 8010     # 启动 3 个，从 8010 起
set -u

ROOT_DIR="D:/桌面/RAGLoRA"
BACKEND_DIR="$ROOT_DIR/backend"
PY="D:/anaconda3/envs/rag_env/python.exe"

N="${1:-2}"
BASE_PORT="${2:-8000}"

if [ ! -x "$PY" ]; then
  echo "[错误] 找不到 Python 环境: $PY" >&2
  exit 1
fi

STORE="${RAGLORA_VECTOR_STORE:-milvus}"
if [ "$STORE" != "milvus" ]; then
  cat >&2 <<EOF
[拒绝启动] 当前 VECTOR_STORE=$STORE，多 worker 只支持 milvus。

  原因：Qdrant 嵌入式模式对存储目录持有**独占文件锁**，第二个 worker
  打开同一目录会失败。这不是脚本限制，是 Qdrant 嵌入式模式的固有约束。

  改用：RAGLORA_VECTOR_STORE=milvus bash backend/run_multi.sh $N
  并确保 Milvus 已启动：bash tools/demo/up.sh milvus
EOF
  exit 2
fi

cat <<EOF
============================================================
 多 worker 启动 | worker 数 $N | 端口 ${BASE_PORT}..$((BASE_PORT+N-1))
============================================================
 ⚠️ 每个 worker 各加载一份 bge-m3+精排（约 3.4GB）
    $N 个 worker 约需 $((N * 34 / 10)).$((N * 34 % 10))GB 常驻内存
    已默认 RAGLORA_WARMUP=0（模型按需加载）以缓解
============================================================
EOF

# 先停掉单 worker 启动的实例，避免端口冲突
bash "$BACKEND_DIR/shutdown.sh" >/dev/null 2>&1 || true
sleep 2

PIDS=()
for i in $(seq 0 $((N - 1))); do
  PORT=$((BASE_PORT + i))
  echo "==> worker $i -> 端口 $PORT"
  RAGLORA_WARMUP=0 powershell -NoProfile -Command \
    "Start-Process -FilePath '$PY' -ArgumentList '-m','uvicorn','app.main:app','--host','0.0.0.0','--port','$PORT' -WorkingDirectory 'D:\\桌面\\RAGLoRA\\backend' -WindowStyle Hidden -RedirectStandardOutput 'D:\\桌面\\RAGLoRA\\backend\\logs\\uvicorn_$PORT.log' -RedirectStandardError 'D:\\桌面\\RAGLoRA\\backend\\logs\\uvicorn_$PORT.err.log'"
done

echo
echo "等待就绪..."
for i in $(seq 0 $((N - 1))); do
  PORT=$((BASE_PORT + i))
  for _ in $(seq 1 30); do
    netstat -ano 2>/dev/null | grep -qE ":${PORT}\s.*LISTENING" && break
    sleep 1
  done
  if netstat -ano 2>/dev/null | grep -qE ":${PORT}\s.*LISTENING"; then
    echo "  ✓ $PORT 已监听"
  else
    echo "  ✗ $PORT 未监听 —— 看 logs/uvicorn_$PORT.err.log"
  fi
done

echo
echo "下一步：起 nginx 做负载均衡"
echo "  docker compose -f docker/docker-compose.nginx.yml up -d"
echo "压测（指向 nginx 的 8080）："
echo "  python backend/scripts/loadtest.py --url http://127.0.0.1:8080 --scenario health"
