#!/bin/bash
# 工单编号：人工智能NLP-RAG项目-LightRAG优化
# 安装脚本：准备**两个** conda 环境 + neo4j 容器 + 基座模型
#
# 为什么要两个环境（详见 优化/过程问题记录.md 问题 2）：
#   rag_gd   RAG 基座那一路。里面的 FlagEmbedding 1.3.3 在元数据里硬锁了
#            transformers==4.44.2 和 datasets==2.19.0，gradio 4.44.1 又要求
#            huggingface-hub<1.0 —— 这一组**不能动**，动了整条链路就废。
#   rag_gd1  LightRAG + neo4j + RAGAS。ragas 会把 huggingface-hub 升到 2.x，
#            装进 rag_gd 必然顶翻上面那一组（实测过，整条链路全废）。
#
# 踩过的教训：装包前先 pip freeze 快照。本次就是靠快照把 rag_gd 完整还原的。
set -e

RAG_ENV="${RAG_ENV:-rag_gd}"
LR_ENV="${LR_ENV:-rag_gd1}"
PY_VERSION="3.11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../研发" && pwd)"
MODEL_ROOT="${MODEL_ROOT:-$(cd "${PROJECT_DIR}/../.." && pwd)/models}"

NEO4J_NAME="${NEO4J_NAME:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-neo4j123}"

echo "==> 项目目录：${PROJECT_DIR}"
echo "==> 环境：${RAG_ENV}（RAG 基座） / ${LR_ENV}（LightRAG + RAGAS）"

command -v conda >/dev/null 2>&1 || { echo "[错误] 未找到 conda" >&2; exit 1; }
eval "$(conda shell.bash hook)"

# ---------- 1. rag_gd：RAG 基座 ----------
# 若已存在就**只做检查、绝不升级**：这个环境的版本组合是被上游锁死的，
# 一旦被 pip 顺手升级，工单 01-10 的整条链路就废了。
if conda env list | grep -qE "^${RAG_ENV}\s"; then
    echo "==> 环境 ${RAG_ENV} 已存在，跳过安装（不升级，版本被上游锁死）"
else
    echo "==> 创建 ${RAG_ENV} 并安装 RAG 基座依赖"
    conda create -n "${RAG_ENV}" "python=${PY_VERSION}" -y
    conda activate "${RAG_ENV}"
    pip install -r "${PROJECT_DIR}/requirements-rag.txt"
    pip install torch --index-url https://download.pytorch.org/whl/cu126
fi

# ---------- 2. rag_gd1：LightRAG + neo4j + RAGAS ----------
if conda env list | grep -qE "^${LR_ENV}\s"; then
    echo "==> 环境 ${LR_ENV} 已存在，跳过创建"
else
    echo "==> 创建 ${LR_ENV}"
    conda create -n "${LR_ENV}" "python=${PY_VERSION}" -y
fi
conda activate "${LR_ENV}"
echo "==> 安装 LightRAG / neo4j / RAGAS"
pip install --upgrade pip
pip install -r "${PROJECT_DIR}/requirements-lightrag.txt"
pip install torch --index-url https://download.pytorch.org/whl/cu126
# ragas 0.4.3 硬导入 langchain_community.chat_models.vertexai，
# 而 langchain-community 0.4 把这个模块删了（迁到 langchain-google-vertexai）。
# 降到 0.3.x 才有这个路径 —— 详见 优化/过程问题记录.md 问题 3。
pip install "langchain-community<0.4"

# ---------- 3. neo4j 容器 ----------
# ⚠️ 先看有没有现成的，不要重复拉镜像/建容器（踩过，白占 986MB）。
if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "${NEO4J_NAME}"; then
    echo "==> neo4j 容器 ${NEO4J_NAME} 已存在，直接启动"
    docker start "${NEO4J_NAME}"
else
    echo "==> 创建 neo4j 容器"
    docker run -d --name "${NEO4J_NAME}" \
        -p 7474:7474 -p 7687:7687 \
        -e NEO4J_AUTH="neo4j/${NEO4J_PASSWORD}" \
        neo4j:latest
fi
sleep 15
docker ps --filter "name=${NEO4J_NAME}" --format "    {{.Names}} {{.Status}} {{.Ports}}"

# ---------- 4. 基座模型 ----------
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
for m in bge-m3 bge-reranker-v2-m3; do
    if [ -d "${MODEL_ROOT}/${m}" ]; then
        echo "==> ${m} 已就绪"
    else
        echo "[提示] 未找到 ${MODEL_ROOT}/${m}，请先下载（各约 2GB / 1GB）"
    fi
done

cat <<EOF

安装完成。后续步骤：
  1) 配置大模型环境变量：
       export DEEPSEEK_BASE_URL=https://api.deepseek.com
       export DEEPSEEK_API_KEY=sk-xxxxxxxx
  2) 建两个知识库（首次约 40 分钟，瓶颈是图表多模态解析）：
       D:/Anaconda/envs/${RAG_ENV}/python.exe 研发/build_kb.py     # RAG 向量库
       D:/Anaconda/envs/${LR_ENV}/python.exe  研发/build_graph.py  # neo4j 图谱
  3) 跑对比：bash 部署/start.sh
EOF
