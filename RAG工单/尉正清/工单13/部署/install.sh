#!/bin/bash
# 工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
# 安装脚本：在**既有** rag_gd 环境里补两个性能分析工具，并检查压测工具链
#
# ⚠️ 本工单不新建 conda 环境 —— 复用工单 01-07 的 rag_gd。
#    那里的版本组合被上游硬锁，**绝不能升级既有包**。
#    装之前先快照，装完 diff 确认是纯新增。
set -e

ENV_NAME="${ENV_NAME:-rag_gd}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../研发" && pwd)"
SNAPSHOT="${SCRIPT_DIR}/logs/pip_before.txt"

command -v conda >/dev/null 2>&1 || { echo "[错误] 未找到 conda" >&2; exit 1; }
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
mkdir -p "${SCRIPT_DIR}/logs"

# ---------- 1. 快照（出事能还原）----------
echo "==> 快照当前环境 -> ${SNAPSHOT}"
pip freeze > "${SNAPSHOT}"
echo "    $(wc -l < "${SNAPSHOT}") 个包"

# ---------- 2. 只装新增的两个工具 ----------
echo "==> 安装 snakeviz / py-spy（纯增量）"
pip install -r "${PROJECT_DIR}/requirements.txt"

# ---------- 3. diff 确认没有改动既有包 ----------
echo "==> 校验：应当只有新增，没有版本变化"
pip freeze > "${SCRIPT_DIR}/logs/pip_after.txt"
python - <<PY
def load(p):
    d = {}
    for ln in open(p, encoding="utf-8"):
        ln = ln.strip()
        if "==" in ln:
            n, v = ln.split("==", 1)
            d[n.lower().replace("_", "-")] = (n, v)
    return d
a = load(r"${SNAPSHOT}"); b = load(r"${SCRIPT_DIR}/logs/pip_after.txt")
add = sorted(set(b) - set(a)); rm = sorted(set(a) - set(b))
ch = sorted(k for k in set(a) & set(b) if a[k][1] != b[k][1])
print("  新增:", [b[k][0] for k in add] or "无")
print("  丢失:", [a[k][0] for k in rm] or "无")
print("  版本变化:", [(k, a[k][1], b[k][1]) for k in ch] or "无")
print("  ✅ 纯新增，既有包一个没动" if not (rm or ch) else "  ⚠️ 有改动，检查上面")
PY

# ---------- 4. 检查 JMeter ----------
JMETER="${JMETER_HOME:-/d/apache-jmeter-5.6.3}"
if [ -x "${JMETER}/bin/jmeter" ]; then
    echo "==> JMeter 就绪：${JMETER}"
else
    echo "[提示] 未找到 JMeter（${JMETER}）。它是独立的 Java 程序，"
    echo "       下载解压后设 JMETER_HOME 即可，不需要 pip 安装。"
fi
command -v java >/dev/null 2>&1 && echo "==> Java：$(java -version 2>&1 | head -1)" \
    || echo "[提示] 未找到 java，JMeter 跑不起来"

# ---------- 5. 知识库 ----------
KB="${PROJECT_DIR}/kb_cache/ccf_competition"
if [ -f "${KB}/chunks.json" ]; then
    echo "==> 知识库就绪：${KB}"
else
    echo "[提示] 知识库不存在。可从工单 10 复制："
    echo "       cp -r ../../工单10/研发/kb_cache ${PROJECT_DIR}/"
    echo "       或重建：python 研发/build_kb.py（约 25 分钟）"
fi

cat <<EOF

安装完成。后续：
  1) 配大模型环境变量：
       export DEEPSEEK_BASE_URL=https://api.deepseek.com
       export DEEPSEEK_API_KEY=sk-xxxxxxxx
  2) 跑完整性能对比：bash 部署/start.sh
EOF
