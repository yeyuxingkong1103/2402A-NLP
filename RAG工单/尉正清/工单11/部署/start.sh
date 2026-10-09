#!/bin/bash
# 工单编号：人工智能NLP-RAG项目-Embedding 模型微调任务
# 启动脚本：后台跑完整微调流程，日志写入 logs/pipeline.log
#
# 本工单是一次性训练任务，没有常驻服务，所以 "start" = 把整条流水线拉起来：
#   ① 数据集生成 → ② 微调前基线评估 → ③ 大模型生成问答对
#   → ④ 微调 → ⑤ 微调后评估 → ⑥ 生成评估报告
# 全过程首次约 1 小时（瓶颈是给 57638 篇语料编码，每次十几分钟）。
set -e

ENV_NAME="${ENV_NAME:-rag_gd}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../研发" && pwd)"
TEST_DIR="$(cd "${SCRIPT_DIR}/../测试" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
PID_FILE="${LOG_DIR}/pipeline.pid"

# 生成问答对的目标条数（每条约 1 次大模型调用，5000 条约 10 分钟）
N_PAIRS="${N_PAIRS:-5000}"

mkdir -p "${LOG_DIR}"

# ---------- 检查是否已在运行 ----------
if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "流水线已在运行（PID $(cat "${PID_FILE}")），如需中止请先执行 部署/stop.sh"
    exit 0
fi

# ---------- 检查环境变量 ----------
if [ -z "${DEEPSEEK_API_KEY}" ]; then
    echo "[警告] 未设置 DEEPSEEK_API_KEY，第 ③ 步生成问答对会失败" >&2
fi
# 国内直连 huggingface.co 超时，必须走镜像，否则第 ① 步下载数据集会卡死
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ---------- 激活环境 ----------
if ! command -v conda >/dev/null 2>&1; then
    echo "[错误] 未找到 conda" >&2
    exit 1
fi
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# ---------- 流水线（写成子 shell 丢后台，便于 stop.sh 整组结束） ----------
cd "${PROJECT_DIR}"
echo "==> 启动微调流水线，日志 ${LOG_DIR}/pipeline.log"

nohup bash -c '
    set -e
    cd "'"${PROJECT_DIR}"'"
    echo "===== $(date "+%F %T") ① 生成数据集 ====="
    python -u build_dataset.py
    echo "===== $(date "+%F %T") ② 微调前基线评估 ====="
    python -u evaluate.py --model base
    echo "===== $(date "+%F %T") ③ 生成问答对 ====="
    python -u gen_qa.py --n "'"${N_PAIRS}"'"
    echo "===== $(date "+%F %T") ④ 微调 ====="
    # --freeze-layers 6：训练样本只有约 5.6k，全参数微调会把预训练学到的
    #   通用语义带偏、指标低于基线；冻住底部只训顶部才是正的（问题 9）。
    # --keep-last：只用 batch 内负例时 dev loss 会随训练单调上涨、与检索质量
    #   脱钩，不能拿它当早停信号（问题 10），所以固定训满 2 轮、保留最后一版。
    python -u finetune.py --use-gen --batch 24 --epochs 2 --freeze-layers 6 \
           --keep-last --patience 1000
    echo "===== $(date "+%F %T") ⑤ 微调后评估 ====="
    python -u evaluate.py --model finetuned
    echo "===== $(date "+%F %T") ⑥ 生成评估报告 ====="
    cd "'"${TEST_DIR}"'"
    python -u compare.py
    echo "===== $(date "+%F %T") 全部完成 ====="
' > "${LOG_DIR}/pipeline.log" 2>&1 &

echo $! > "${PID_FILE}"

sleep 5
if kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "==> 已启动（PID $(cat "${PID_FILE}")）"
    echo "    查看进度：tail -f ${LOG_DIR}/pipeline.log"
    echo "    中止任务：bash 部署/stop.sh"
else
    echo "[错误] 启动失败，日志如下：" >&2
    tail -20 "${LOG_DIR}/pipeline.log" >&2
    rm -f "${PID_FILE}"
    exit 1
fi
