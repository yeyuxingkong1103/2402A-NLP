#!/bin/bash
# 工单编号：人工智能NLP-RAG-解决API服务并发瓶颈与资源泄漏
# 跑 JMeter 压测，结果写到 测试/results/
#
# JMeter 装在 D:\apache-jmeter-5.6.3（JM 变量可覆盖）。
# RAGFlow 的 id / token 从工单14 的 .ragflow_ids.json 与 .api_token 读，
# 也可以命令行覆盖（换知识库时用得上）。
#
# 用法：
#   bash jmeter_run.sh a baseline             # 场景A 基准（20并发 × 10轮）
#   bash jmeter_run.sh b baseline             # 场景B 基准
#   bash jmeter_run.sh a after 20 20          # 场景A，20并发 × 20轮
set -e

JM="${JM:-/d/apache-jmeter-5.6.3/bin/jmeter}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JMX_DIR="${HERE}/jmeter"
OUT_DIR="${HERE}/results"
mkdir -p "${OUT_DIR}"

SCENARIO="${1:?用法: bash jmeter_run.sh <a|b> <tag> [users] [loops]}"
TAG="${2:?缺 tag，如 baseline / after}"
USERS="${3:-}"
LOOPS="${4:-}"

# ---- RAGFlow 连接参数 ----
IDS="${HERE}/../../工单14/测试/.ragflow_ids.json"
TOKF="${HERE}/../../工单14/测试/.api_token"
TOKEN="${TOKEN:-$(cat "${TOKF}")}"
CHAT_ID="${CHAT_ID:-$(python -c "import json;print(json.load(open(r'${IDS}'))['chat_id'])")}"
DS_ID="${DS_ID:-$(python -c "import json;print(json.load(open(r'${IDS}'))['ds_id'])")}"

if [ "$SCENARIO" = "a" ]; then
    JMX="${JMX_DIR}/scenario_a_chat.jmx"
    USERS="${USERS:-20}"; LOOPS="${LOOPS:-30}"      # 20并发 × 30轮 ≈ 10 分钟
    EXTRA=(-Jusers="${USERS}" -Jloops="${LOOPS}")
else
    JMX="${JMX_DIR}/scenario_b_mixed.jmx"
    USERS="${USERS:-10}"; LOOPS="${LOOPS:-8}"
    EXTRA=(-Jusers_chat=5 -Jusers_upload=5 -Jloops="${LOOPS}")
fi

OUT="${OUT_DIR}/jmeter_${SCENARIO}_${TAG}.jtl"
LOG="${OUT_DIR}/jmeter_${SCENARIO}_${TAG}.log"

echo "==> 场景 ${SCENARIO}  并发 ${USERS}  每线程 ${LOOPS} 轮"
echo "==> 结果 ${OUT}"
cd "${JMX_DIR}"
"${JM}" -n -t "${JMX}" \
    -Jhost=localhost -Jport=80 \
    -Jtoken="${TOKEN}" -Jchat_id="${CHAT_ID}" -Jds_id="${DS_ID}" \
    -Jqfile="${JMX_DIR}/questions.csv" -Jpdfdir="${JMX_DIR}/pdfs" \
    "${EXTRA[@]}" -Jramp=5 \
    -l "${OUT}" -j "${LOG}" 2>&1 | grep -E "^summary|Err:|error" || true

echo "==> 完成"
