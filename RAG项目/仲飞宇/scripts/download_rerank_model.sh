#!/usr/bin/env bash
# 下载 BAAI/bge-reranker-v2-m3（2.29GB）到 D 盘模型目录，供 rerank_service 使用。
#
# 为什么不用 huggingface_hub / hf-mirror：
#   1) hf-mirror 对该 repo 走 xet 存储，CAS 重建接口 cas-server.xethub.hf.co 返回
#      401 Unauthorized，下载反复中断（2026-09-19 实测，重试 3 次均失败）；
#   2) hf_hub 的临时文件名带随机 uuid，进程一旦被强杀，已下部分下次无法复用
#      （huggingface_hub/file_download.py:2008 明确写了 "could not be reused anyway"）。
# 所以改用 modelscope 直链 + curl -C -：断点续传跨进程有效，中断了重跑就接着下。
#
# 实测速度（2026-09-19 22:00 CST）：modelscope 808KB/s、hf-mirror 570KB/s、huggingface.co 不通。
# 全量约 47 分钟。
#
# 日志: logs/rerank_model_download.log
set -uo pipefail

DEST="${RERANK_MODEL_DIR:-/mnt/d/models/bge-reranker-v2-m3}"
BASE="https://www.modelscope.cn/api/v1/models/BAAI/bge-reranker-v2-m3/repo?Revision=master&FilePath="

# 文件名 + 期望字节数（取自 modelscope repo files 接口，用于校验下全没下全）
FILES=(
    "config.json:795"
    "model.safetensors:2271071852"
    "sentencepiece.bpe.model:5069051"
    "special_tokens_map.json:964"
    "tokenizer.json:17098273"
    "tokenizer_config.json:1173"
)

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/logs/rerank_model_download.log"

mkdir -p "$DEST" "$ROOT/logs"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

size_of() { stat -c%s "$1" 2>/dev/null || echo 0; }

for attempt in $(seq 1 60); do
    log "=== 尝试 $attempt ==="
    pending=0
    for entry in "${FILES[@]}"; do
        f="${entry%:*}"; want="${entry##*:}"
        have="$(size_of "$DEST/$f")"
        if [[ "$have" == "$want" ]]; then
            continue
        fi
        pending=$((pending + 1))
        log "  ↓ $f ($have/$want) 续传中..."
        # -C - 断点续传；--retry 应付单次连接抖动
        curl -L -C - --retry 5 --retry-delay 10 --retry-all-errors \
             --connect-timeout 30 --speed-time 60 --speed-limit 10240 \
             -o "$DEST/$f" "${BASE}${f}" >> "$LOG" 2>&1
        log "  → $f 现在 $(size_of "$DEST/$f") 字节"
    done

    if [[ "$pending" == 0 ]]; then
        log "=== 全部文件校验通过，下载完成：$DEST ==="
        echo "✅ 下载完成：$DEST"
        exit 0
    fi
    log "仍有 $pending 个文件未完成，10s 后重试"
    sleep 10
done

log "=== 60 轮仍未完成，放弃 ==="
echo "❌ 下载未完成，详见 $LOG"
exit 1
