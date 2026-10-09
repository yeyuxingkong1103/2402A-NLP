#!/bin/bash
# 工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
# 启动脚本：起压测服务 + 跑完整的性能对比流水线
#
# 流程：起服务(带预热) → 串行基准 → JMeter 并发梯度 → 出报告与截图
set -e

ENV_NAME="${ENV_NAME:-rag_gd}"
PY="D:/Anaconda/envs/${ENV_NAME}/python.exe"
JMETER="${JMETER_HOME:-/d/apache-jmeter-5.6.3}/bin/jmeter"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../研发" && pwd)"
TEST_DIR="$(cd "${SCRIPT_DIR}/../测试" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
PID_FILE="${LOG_DIR}/api.pid"

mkdir -p "${LOG_DIR}"
if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "服务已在运行（PID $(cat "${PID_FILE}")），如需重启先执行 部署/stop.sh"
    exit 0
fi
[ -n "${DEEPSEEK_API_KEY}" ] || echo "[警告] 未设置 DEEPSEEK_API_KEY" >&2

echo "==> 启动压测服务（启动时自动预热）"
cd "${PROJECT_DIR}"
nohup "${PY}" -u api.py --port 8000 > "${LOG_DIR}/api.log" 2>&1 &
echo $! > "${PID_FILE}"

for i in $(seq 1 40); do
    curl -s -m 3 http://127.0.0.1:8000/health 2>/dev/null | grep -q '"ready": *true' && break
    sleep 3
done
curl -s -m 3 http://127.0.0.1:8000/health | grep -q '"ready": *true' \
    || { echo "[错误] 服务未就绪"; tail -20 "${LOG_DIR}/api.log" >&2; exit 1; }
echo "    服务就绪：http://127.0.0.1:8000"

echo "==> ① 串行基准（出分阶段延迟）"
"${PY}" "${TEST_DIR}/bench.py" --n 20 --warmup 3 --tag after

echo "==> ② JMeter 并发梯度"
for U in 1 4 8 16; do
    R=$(( U > 4 ? 4 : 1 ))
    printf "    并发 %-2s " $U
    "${JMETER}" -n -f -t "${TEST_DIR}/jmeter/rag_bench.jmx" \
        -Jusers=$U -Jloops=5 -Jramp=$R \
        -Jqfile="${TEST_DIR}/jmeter/questions.csv" \
        -l "${TEST_DIR}/results/jmeter_after_u${U}.jtl" \
        -j "${LOG_DIR}/jmeter_u${U}.log" 2>&1 | grep -E "^summary =" | sed 's/^summary =//'
done

echo "==> ③ 生成报告与截图"
"${PY}" "${TEST_DIR}/report.py"
"${PY}" "${TEST_DIR}/截图.py"

echo
echo "完成。看 测试/性能报告.md 与 测试/截图/"
echo "想跑优化前的对比：PERF_OPT=off LLM_REASONING_EFFORT= bash 部署/start.sh"
