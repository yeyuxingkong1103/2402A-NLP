#!/usr/bin/env bash
# 多臂 LoRA 对照评测驱动：**同一个 vLLM 实例、同一时间窗、每臂全新会话**跑完整套题。
#
# 为什么要有这个脚本（三个真踩过的坑，都写进代码了）：
#   1. **跨窗口比不出东西**：一条臂换时间窗重测能差 3 题（本项目抖动地板 4 题），
#      所以所有臂必须在一次运行里跑完（服务只起一次）。
#   2. **`eval_answers.py` 防覆盖闸门**：同 tag 的结果文件已存在且非空时它**直接 exit 2**
#      （怕两个进程写同一文件把结果写坏）⇒ 重跑必须给 `--force`，否则会静默"什么都没跑"，
#      看起来像模型突然不会回答了。这个坑真的踩过：四臂各 exit=2、用时 0 秒。
#   3. **LoRA rank 上限**：vLLM 默认 `--max-lora-rank 16`，挂了 r=32 的 adapter 会
#      `ValueError: LoRA rank 32 is greater than max_lora_rank 16` 直接把服务起崩。
#
# 另外三件"看起来多余但必须做"的事：
#   * **停服务顺序**：先 API 后 vLLM（反了会去抢 Milvus Lite 的单进程锁）；
#   * **采 GPU 显存**：报告要"评测过程的显存占用"，闲时/评测中各采一次并落盘；
#   * **收尾恢复生产配置**：跑完把 27B 生产服务拉起来（否则下次有人来测发现是 4B 在答）。
#
# 用法（云端）：
#   bash scripts/run_lora_ab.sh
#   ARMS="qwen4b v1 v5a v5b v5c" TAG_PREFIX=arm2 SESSION_PREFIX=fresh \
#       MAX_LORA_RANK=32 bash scripts/run_lora_ab.sh
#
# 可覆盖的环境变量（默认值 = 本次实跑用的那一套）：
#   APP=/root/autodl-tmp/legal-rag          PY=/root/miniconda3/envs/rag/bin/python
#   E=/root/miniconda3/envs/rag             BASE=/root/autodl-tmp/models/Qwen--Qwen3-4B
#   LORA_DIR=/root/autodl-tmp/lora          PORT=30001  API_PORT=18080
#   ARMS="qwen4b v1 v5a v5b v5c"            LORA_MODULES="v1 v5a v5b v5c"
#     （若 adapter 目录名与**服务名**不一致，用 `名字=路径` 写，例：
#       LORA_MODULES="v1=$LORA_DIR/legal-v1 v5a v5b v5c"。写错会在起服务前就被拦下。）
#   TAG_PREFIX=arm2                         SESSION_PREFIX=fresh
#   ACCT=eval-4b-arm                        PASSWORD='EvalAccount#2026'
#   MAX_LORA_RANK=32                        DO_RESTORE=1
#   QA_FILE=eval/perturb-routing.jsonl      # 换题集（默认 eval/qa_set.jsonl 冻结 107 题）
#   SKIP_DONE=1   跳过已经有结果文件的臂（**被关机/断线打断后用来续跑**，只补没跑完的）
set -u

APP=${APP:-/root/autodl-tmp/legal-rag}
PY=${PY:-/root/miniconda3/envs/rag/bin/python}
E=${E:-/root/miniconda3/envs/rag}
BASE=${BASE:-/root/autodl-tmp/models/Qwen--Qwen3-4B}
LORA_DIR=${LORA_DIR:-/root/autodl-tmp/lora}
PORT=${PORT:-30001}
API_PORT=${API_PORT:-18080}
ARMS=${ARMS:-"qwen4b v1 v5a v5b v5c"}
LORA_MODULES=${LORA_MODULES:-"v1 v5a v5b v5c"}
TAG_PREFIX=${TAG_PREFIX:-arm2}
SESSION_PREFIX=${SESSION_PREFIX:-fresh}
ACCT=${ACCT:-eval-4b-arm}
PASSWORD=${PASSWORD:-'EvalAccount#2026'}
MAX_LORA_RANK=${MAX_LORA_RANK:-32}
DO_RESTORE=${DO_RESTORE:-1}
#: 题集路径。默认是**冻结的 107 题**；做"扰动复测"时指向派生探针集
#: （`eval/perturb-routing.jsonl`，由 `scripts/perturb_questions.py` 生成）——
#: 注意那是**探针**，不是把评测集扩大了。
QA_FILE=${QA_FILE:-eval/qa_set.jsonl}
#: 1 = 跳过已有结果的臂。**为什么需要**：2026-09-28 六臂评测跑到一半被关机打断，
#: 有了它续跑只补缺的那几臂，而不是把 6 臂 40 分钟全重来一遍。
SKIP_DONE=${SKIP_DONE:-0}

LOG=${LOG:-/root/autodl-tmp/logs/lora-ab.log}
VRAM_JSON=${VRAM_JSON:-/root/autodl-tmp/logs/vram-inference-$TAG_PREFIX.json}
CU13=$E/lib/python3.10/site-packages/nvidia/cu13/lib
CU12=$E/lib/python3.10/site-packages/nvidia/cuda_runtime/lib

cd "$APP" || { echo "!! 应用目录不存在：$APP"; exit 9; }

# 按名字调用的脚本必须存在：改名了要在这里就炸，而不是在计费的 GPU 机器上跑到一半才炸
for script in scripts/run_api.py scripts/eval_answers.py scripts/rescore_answers.py; do
    [ -f "$script" ] || { echo "!! 缺脚本：$script"; exit 9; }
done

vram() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; }

stop_api() {
    pkill -TERM -f 'run_api\.py' 2>/dev/null || true
    for i in $(seq 1 20); do pgrep -f 'run_api\.py' >/dev/null 2>&1 || break; sleep 1; done
    pkill -KILL -f 'run_api\.py' 2>/dev/null || true
}

stop_vllm() {
    pkill -TERM -f 'vllm serve' 2>/dev/null || true
    for i in $(seq 1 40); do pgrep -f 'vllm serve' >/dev/null 2>&1 || break; sleep 2; done
    pkill -KILL -f 'vllm serve' 2>/dev/null || true
}

{
    echo "======== $(date -Is) LoRA 多臂评测（$ARMS）========"
    echo "--- 0) 停 API 与 vLLM（顺序：API -> vLLM），再起 4B + LoRA"
    stop_api
    stop_vllm
    sleep 3
    echo "  起服务前 GPU：$(vram)"

    LORA_ARGS=""
    MISSING="0"
    # 每个条目支持两种写法：
    #   * `v5a`                 ⇒ 路径取 $LORA_DIR/v5a（多数情况）
    #   * `v1=$LORA_DIR/legal-v1` ⇒ 显式给路径（**目录名跟服务名不一样时必须这么写**：
    #     服务名 v1 对应的目录是 legal-v1，按名字拼路径会让 vLLM 报
    #     `Loading lora v1 failed: No adapter found for <path>`）
    for module in $LORA_MODULES; do
        case "$module" in
            *=*) name=${module%%=*}; path=${module#*=} ;;
            *)   name=$module; path=$LORA_DIR/$module ;;
        esac
        if [ ! -e "$path" ]; then
            echo "!! adapter 路径不存在：$path（服务名 $name）"
            MISSING="1"
        fi
        LORA_ARGS="$LORA_ARGS $name=$path"
    done
    if [ "$MISSING" != "0" ]; then
        echo "!! 先补好 adapter 路径再起服务（省得白等一次 vLLM 启动）；"
        echo "   提示：v1 的目录名是 legal-v1，写 LORA_MODULES=\"v1=$LORA_DIR/legal-v1 v5a v5b ...\""
        exit 9
    fi

    export LD_LIBRARY_PATH="$CU13:$CU12:${LD_LIBRARY_PATH:-}"
    export PATH="$E/bin:$PATH"
    export VLLM_USE_FLASHINFER_SAMPLER=0
    export TMPDIR=${TMPDIR:-/root/autodl-tmp/tmp}
    setsid nohup "$E/bin/vllm" serve "$BASE" \
        --served-model-name qwen4b \
        --host 127.0.0.1 --port "$PORT" \
        --dtype auto --gpu-memory-utilization "${GPU_UTIL:-0.80}" \
        --max-model-len "${MAX_MODEL_LEN:-8192}" --max-num-seqs 8 --trust-remote-code \
        --max-lora-rank "$MAX_LORA_RANK" \
        --enable-lora --lora-modules $LORA_ARGS \
        >> /root/autodl-tmp/logs/vllm-4b.log 2>&1 </dev/null &
    echo "  等 4B 就绪（adapter：$LORA_MODULES，max-lora-rank=$MAX_LORA_RANK）……"
    for i in $(seq 1 72); do
        curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
        sleep 5
    done
    curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | tr ',' '\n' | grep '"id"' | head -8
    IDLE_VRAM=$(vram)
    echo "  服务就绪空闲 GPU：$IDLE_VRAM"
    : >"$VRAM_JSON"
    echo "{\"idle\": \"$IDLE_VRAM\"," >>"$VRAM_JSON"

    for ARM in $ARMS; do
        echo
        echo "--- 臂 $ARM $(date -Is)"
        ARM_RESULT="eval/results/$TAG_PREFIX-$ARM.jsonl"
        if [ "$SKIP_DONE" = "1" ] && [ -s "$ARM_RESULT" ]; then
            echo "  SKIP_DONE=1 且 $ARM_RESULT 已有内容（$(wc -l < "$ARM_RESULT") 行）⇒ 跳过"
            continue
        fi
        stop_api
        sleep 2
        # 每臂用**全新会话前缀**：复用会话名会把上一臂的问答留在窗口里，系统性拉平两臂
        bash -c "source /root/env_4b.sh; export ARM_MODEL=$ARM; cd $APP && setsid nohup \
            $PY scripts/run_api.py --host \"\$API_HOST\" --port \"\$API_PORT\" \
            >> /root/autodl-tmp/logs/api-4b.log 2>&1 </dev/null &"
        for i in $(seq 1 40); do
            curl -s -m 3 "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1 && break
            sleep 5
        done
        curl -s -m 10 "http://127.0.0.1:$API_PORT/health" | head -c 120
        echo
        ARM_START=$(date +%s)
        # --force：结果文件已存在时**必须显式覆盖**，否则 eval_answers.py 直接 exit 2
        "$PY" -u scripts/eval_answers.py --auth --auth-user "$ACCT" --auth-password "$PASSWORD" \
            --user "$ACCT" --role lawyer --session "$SESSION_PREFIX$ARM" \
            --qa-file "$QA_FILE" \
            --tag "$TAG_PREFIX-$ARM" --out eval/results --force
        echo "$TAG_PREFIX-$ARM exit=$? 用时 $(( $(date +%s) - ARM_START ))s"
        echo "  评测中 GPU：$(vram)"
        echo "\"$ARM\": \"$(vram)\"," >>"$VRAM_JSON"
    done
    echo "\"done\": true}" >>"$VRAM_JSON"

    echo
    echo "--- 对比 $(date -Is)"
    IN_ARGS=""
    for ARM in $ARMS; do
        IN_ARGS="$IN_ARGS --in eval/results/$TAG_PREFIX-$ARM.jsonl"
    done
    "$PY" -u scripts/rescore_answers.py $IN_ARGS \
        --compare --out "eval/results/$TAG_PREFIX-compare.json"
    echo "compare exit=$?"

    if [ "$DO_RESTORE" = "1" ]; then
        echo
        echo "--- 收尾：恢复 27B 生产配置 $(date -Is)"
        stop_api
        stop_vllm
        sleep 3
        bash /root/start_vllm.sh
        bash /root/wait_vllm.sh
        bash /root/run_api_cloud.sh
        sleep 25
        curl -s -m 10 "http://127.0.0.1:$API_PORT/health" | head -c 160
        echo
        echo "27B 生产 GPU：$(vram)"
    fi
    echo "LORA-AB-DONE $(date -Is)"
} >>"$LOG" 2>&1
echo "（后台跑着，日志：$LOG；单臂约 6 分钟）"
sleep 10
tail -4 "$LOG" 2>/dev/null || echo "(还没输出)"
