#!/bin/bash
# 工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
# 按并发梯度跑 JMeter，输出到 测试/results/jmeter_<tag>_u<N>.jtl
# 用法：bash jmeter_run.sh <tag> [loops] [并发列表...]
set -e
TAG="${1:-run}"; LOOPS="${2:-5}"; shift 2 2>/dev/null || true
LEVELS=("${@:-1 4 8 16}")
JM=/d/apache-jmeter-5.6.3/bin/jmeter
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for U in $LEVELS; do
    echo "===== 并发 ${U}（每线程 ${LOOPS} 次）====="
    "$JM" -n -t "${HERE}/jmeter/rag_bench.jmx" \
        -Jusers="${U}" -Jloops="${LOOPS}" -Jramp="$(( U > 4 ? 4 : 1 ))" \
        -l "${HERE}/results/jmeter_${TAG}_u${U}.jtl" \
        -j "${HERE}/results/jmeter_${TAG}_u${U}.log" 2>&1 | grep -E "^summary" || true
done
