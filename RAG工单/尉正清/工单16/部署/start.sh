#!/bin/bash
# 工单编号：人工智能NLP-RAG-微调专用视觉语言模型
# 启动脚本：后台跑完整微调流程，日志写入 logs/pipeline.log
#
# 本工单是**一次性训练任务**，没有常驻服务，所以 "start" = 把整条流水线拉起来：
#   ① 数据转换 → ② 微调前基线评估 → ③ LoRA 微调 → ④ 微调后评估 → ⑤ 对比报告
#
# 全程首次约 2 小时（6GB 显存上 600 样本 × 2 轮约 70 分钟，两次评估各约 10 分钟，
# 数据转换要渲染 200 多页 PDF 约 5 分钟）。
set -e

ENV_NAME="${ENV_NAME:-rag_gd2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/../研发"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/pipeline.log"
PID_FILE="${LOG_DIR}/pipeline.pid"

mkdir -p "${LOG_DIR}"

if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "[警告] 流水线已在运行（PID $(cat "${PID_FILE}")），先执行 bash stop.sh" >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "[错误] 未找到 conda" >&2
    exit 1
fi
eval "$(conda shell.bash hook)"

echo "==> 日志：${LOG_FILE}"
echo "==> 后台启动流水线"

nohup bash -c "
    set -e
    cd '${PROJECT_DIR}'

    echo \"[\$(date '+%F %T')] ① 数据转换\"
    conda run -n '${ENV_NAME}' python prepare_data.py \\
        --out data --eval-docs 60 --max-train 600 --ragflow --shuffle-options

    echo \"[\$(date '+%F %T')] ② 微调前基线评估\"
    conda run -n '${ENV_NAME}' python evaluate.py --tag base

    echo \"[\$(date '+%F %T')] ③ LoRA 微调\"
    conda run -n '${ENV_NAME}' python finetune.py --config finetune_config.json

    echo \"[\$(date '+%F %T')] ④ 微调后评估\"
    conda run -n '${ENV_NAME}' python evaluate.py --tag finetuned \\
        --adapter output/qwen2vl-2b-lora-imdr

    echo \"[\$(date '+%F %T')] ⑤ 对比报告\"
    conda run -n '${ENV_NAME}' python ../测试/compare.py

    echo \"[\$(date '+%F %T')] 全部完成\"
" > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "==> 已启动（PID $(cat "${PID_FILE}")）"
echo "==> 看进度：tail -f ${LOG_FILE}"
echo "==> 要中止：bash stop.sh"
