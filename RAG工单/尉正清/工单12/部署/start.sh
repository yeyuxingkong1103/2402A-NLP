#!/bin/bash
# 工单编号：人工智能NLP-RAG项目-LightRAG优化
# 启动脚本：跑完整对比流水线（建库 → 两路问答 → RAGAS → 出报告）
#
# 本工单不是常驻服务，所以 "start" = 把整条流水线拉起来。
# 首次全跑约 1.5 小时，瓶颈是两处：
#   ① 图表多模态解析 74 页 × 约 18 秒 ≈ 25 分钟（有缓存，重跑跳过）
#   ② LightRAG 建图 494 块，每块一次大模型抽取 ≈ 15 分钟
set -e

RAG_ENV="${RAG_ENV:-rag_gd}"
LR_ENV="${LR_ENV:-rag_gd1}"
PY_RAG="D:/Anaconda/envs/${RAG_ENV}/python.exe"
PY_LR="D:/Anaconda/envs/${LR_ENV}/python.exe"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../研发" && pwd)"
TEST_DIR="$(cd "${SCRIPT_DIR}/../测试" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
PID_FILE="${LOG_DIR}/pipeline.pid"

mkdir -p "${LOG_DIR}"

if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "流水线已在运行（PID $(cat "${PID_FILE}")），如需中止请先执行 部署/stop.sh"
    exit 0
fi

[ -n "${DEEPSEEK_API_KEY}" ] || echo "[警告] 未设置 DEEPSEEK_API_KEY，大模型调用会失败" >&2
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# neo4j：先看有没有现成容器，有就启动，没有才创建（踩过重复拉镜像的坑）
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx neo4j; then
    echo "==> neo4j 已在运行"
else
    echo "==> 启动 neo4j"
    docker start neo4j >/dev/null 2>&1 || docker run -d --name neo4j \
        -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/neo4j123 neo4j:latest
    sleep 15
fi

echo "==> 启动对比流水线，日志 ${LOG_DIR}/pipeline.log"
nohup bash -c '
    set -e
    echo "===== $(date "+%F %T") ① 建 RAG 向量知识库 ====="
    cd "'"${PROJECT_DIR}"'" && "'"${PY_RAG}"'" -u build_kb.py
    echo "===== $(date "+%F %T") ② 建 LightRAG 知识图谱 ====="
    cd "'"${PROJECT_DIR}"'" && "'"${PY_LR}"'" -u build_graph.py
    echo "===== $(date "+%F %T") ③ RAG 侧跑 16 题 ====="
    cd "'"${PROJECT_DIR}"'" && "'"${PY_RAG}"'" -u run_qa.py --kb rag
    echo "===== $(date "+%F %T") ④ LightRAG 侧跑 16 题 ====="
    cd "'"${PROJECT_DIR}"'" && "'"${PY_LR}"'" -u run_qa.py --kb lightrag
    echo "===== $(date "+%F %T") ⑤ RAGAS 评估 ====="
    cd "'"${TEST_DIR}"'" && "'"${PY_LR}"'" -u ragas_eval.py
    echo "===== $(date "+%F %T") ⑥ 生成对比报告 ====="
    cd "'"${TEST_DIR}"'" && "'"${PY_LR}"'" -u compare.py
    echo "===== $(date "+%F %T") 全部完成 ====="
' > "${LOG_DIR}/pipeline.log" 2>&1 &

echo $! > "${PID_FILE}"
sleep 5
if kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "==> 已启动（PID $(cat "${PID_FILE}")）"
    echo "    查看进度：tail -f ${LOG_DIR}/pipeline.log"
    echo "    图谱浏览：http://localhost:7474 （neo4j / neo4j123）"
    echo "    中止任务：bash 部署/stop.sh"
else
    echo "[错误] 启动失败，日志如下：" >&2
    tail -20 "${LOG_DIR}/pipeline.log" >&2
    rm -f "${PID_FILE}"
    exit 1
fi
