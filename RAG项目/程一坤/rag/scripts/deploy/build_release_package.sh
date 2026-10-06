#!/usr/bin/env bash
set -euo pipefail

# 从项目根目录构建可直接上传的发布包，并在压缩前后执行完整性断言。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PACKAGE_DATE="$(date +%Y%m%d)"
PACKAGE_NAME="legalrag-release-${PACKAGE_DATE}.tar.gz"
OUTPUT_PATH="${PROJECT_ROOT}/${PACKAGE_NAME}"
STAGING_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "${STAGING_DIR}"
}
trap cleanup EXIT

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

copy_tree() {
  local source_path="$1"
  local target_path="${STAGING_DIR}/${source_path}"
  mkdir -p "$(dirname "${target_path}")"
  tar -C "${PROJECT_ROOT}" \
    --exclude='.env' \
    --exclude='frontend/.env*' \
    --exclude='frontend/.next' \
    --exclude='frontend/node_modules' \
    --exclude='.workbuddy' \
    --exclude='*/.workbuddy' \
    --exclude='__pycache__' \
    --exclude='*/__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='*/.pytest_cache' \
    --exclude='*.log' \
    --exclude='*.bak' \
    --exclude='*.backup-*' \
    --exclude='*/.next' \
    --exclude='*/node_modules' \
    --exclude='*/tmp_*' \
    --exclude='*/temp_*' \
    -cf - "${source_path}" | tar -C "${STAGING_DIR}" -xf -
}

cd "${PROJECT_ROOT}"
rm -f "${OUTPUT_PATH}"

for required_path in backend frontend evaluation scripts docs data/labor_law_processed data/labor_law_raw data/evaluation data/legal_synonyms_v1.json backend/requirements.txt .env.development .env.test .env.production README.md; do
  [ -e "${required_path}" ] || fail "缺少交付项：${required_path}"
done

for required_path in backend frontend evaluation scripts docs data/labor_law_processed data/labor_law_raw data/evaluation; do
  copy_tree "${required_path}"
done

for required_file in data/legal_synonyms_v1.json backend/requirements.txt .env.development .env.test .env.production README.md; do
  copy_tree "${required_file}"
done

find "${PROJECT_ROOT}/data/labor_law_processed" -mindepth 1 -maxdepth 1 -type d \
  \( -iname '*_old_*' -o -iname '*_bak*' -o -iname '*_backup*' \) -print -quit | grep -q . && \
  fail 'data/labor_law_processed 下存在旧目录（*_old_* / *_bak* / *_backup*）'

python - "${STAGING_DIR}" <<'PY'
import json
import sys
from pathlib import Path

staging = Path(sys.argv[1])
processed = staging / "data/labor_law_processed"
counts = {"documents": 0, "chunks": 0, "parent": 0, "child": 0}
for manifest_path in sorted(processed.glob("*/manifest.json")):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts["documents"] += 1
    counts["chunks"] += int(manifest["record_counts"]["document_chunks.jsonl"])
    chunks_path = manifest_path.parent / "document_chunks.jsonl"
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        counts[record["chunk_type"]] += 1
if counts != {"documents": 11, "chunks": 1557, "parent": 486, "child": 1071}:
    raise SystemExit(f"语料断言失败：{counts}")
PY

REQUIREMENTS_COUNT="$(grep -Ev '^[[:space:]]*(#|$)' "${STAGING_DIR}/backend/requirements.txt" | wc -l | tr -d ' ')"
PACKAGE_FILE_COUNT="$(find "${STAGING_DIR}" -type f | wc -l | tr -d ' ')"
MANIFEST_PATH="${STAGING_DIR}/RELEASE_MANIFEST.txt"
cat > "${MANIFEST_PATH}" <<EOF
Legal RAG Release Manifest
==========================
打包时间：$(date '+%Y-%m-%d %H:%M:%S %z')
包名：${PACKAGE_NAME}
包内文件数：$((PACKAGE_FILE_COUNT + 1))
依赖：backend/requirements.txt 非注释条数 ${REQUIREMENTS_COUNT}
语料：11 部 / 1557 块（parent 486 + child 1071）/ 486 摘要
部署目录：/opt/legal-rag（install.sh 默认 INSTALL_DIR；第 c/e 步统一使用该变量）
提示词 md5：backend/app/chat/prompt_builder.py = f43ee63672ff2b7e136a968441bac640
评测基线：predeploy_baseline_v3
Recall@5：0.9529
MRR@10：0.7554
引用正确率：1.0
拒答准确率：0.9333
EOF
PACKAGE_FILE_COUNT="$(find "${STAGING_DIR}" -type f | wc -l | tr -d ' ')"

tar -C "${STAGING_DIR}" -czf "${OUTPUT_PATH}" .

printf '包路径：%s\n' "${OUTPUT_PATH}"
printf '包大小：%s\n' "$(du -h "${OUTPUT_PATH}" | cut -f1)"

ARCHIVE_LIST="$(tar -tzf "${OUTPUT_PATH}")"
for forbidden_pattern in \
  '*/.env' '.env' 'frontend/.env*' '*.next/*' '*/node_modules/*' \
  '*/__pycache__/*' '*/.pytest_cache/*' '*.log' '*.backup-*' '*.bak' '.workbuddy/*'; do
  printf '%s\n' "${ARCHIVE_LIST}" | grep -Eq "(^|/)${forbidden_pattern//\*/.*}($|/)" && \
    fail "包内命中禁止项：${forbidden_pattern}"
done

printf '%s\n' "${ARCHIVE_LIST}" | grep -q '^./data/labor_law_processed/' || fail '包内缺少 data/labor_law_processed'
INSTALL_SCRIPT_CONTENT="$(tar -xOzf "${OUTPUT_PATH}" ./scripts/deploy/install.sh)"
printf '%s\n' "${INSTALL_SCRIPT_CONTENT}" | grep -q 'MINICONDA_URL=.*mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh' || fail '包内 install.sh 未配置清华 Miniconda 下载源'
printf '%s\n' "${INSTALL_SCRIPT_CONTENT}" | grep -q 'MINICONDA_FALLBACK_URL=.*repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh' || fail '包内 install.sh 未配置官方 Miniconda 备选源'
printf '%s\n' "${INSTALL_SCRIPT_CONTENT}" | grep -q 'anaconda/archive/Anaconda3-latest' && fail '包内 install.sh 仍包含已失效的 Anaconda latest 地址'
printf '%s\n' "${INSTALL_SCRIPT_CONTENT}" | grep -q '第 %s 步.*执行失败' || fail '包内 install.sh 缺少失败步骤提示'
printf '%s\n' "${INSTALL_SCRIPT_CONTENT}" | grep -q '\[失败命令\]' || fail '包内 install.sh 缺少失败命令提示'
printf '%s\n' "${INSTALL_SCRIPT_CONTENT}" | grep -q '\[退出码\]' || fail '包内 install.sh 缺少退出码提示'
printf '%s\n' "${INSTALL_SCRIPT_CONTENT}" | grep -q '\[修复提示\]' || fail '包内 install.sh 缺少修复提示'
ARCHIVE_CHUNK_COUNT="$(tar -xOzf "${OUTPUT_PATH}" --wildcards '*/data/labor_law_processed/*/document_chunks.jsonl' 2>/dev/null | awk 'NF {count++} END {print count + 0}')"
[ "${ARCHIVE_CHUNK_COUNT}" = 1557 ] || fail "包内 document_chunks.jsonl 合计为 ${ARCHIVE_CHUNK_COUNT}，期望 1557"
printf '%s\n' "${ARCHIVE_LIST}" | grep -Eq '(^|/)data/labor_law_processed/[^/]+/([^/]*/)?(_old_|_bak)' && fail '包内存在旧语料目录'

printf '文件清单（前 30 行）：\n'
printf '%s\n' "${ARCHIVE_LIST}" | sed -n '1,30p'
printf '总文件数：%s\n' "${PACKAGE_FILE_COUNT}"
printf '交付包校验：通过\n'
