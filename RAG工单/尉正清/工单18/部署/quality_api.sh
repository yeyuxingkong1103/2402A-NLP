#!/usr/bin/env bash
# 工单编号：人工智能NLP-RAG-文档质量评估 Skill 与 RAGFlow 集成
#
# 质检 API 调用示例。两个端点：
#   POST /v1/document/quality-inspection          跑质检，返回 JSON（给程序看）
#   GET  /v1/document/quality-inspection/report   取上次的 HTML 简报（给人看）
#
# 三个参数演示：
#   path    要质检的**容器内**路径（目录或单文件）。宿主机路径必须先挂进容器，
#           否则端点直接报「路径不存在」——这是最容易踩的一个坑。
#   limit   只处理前 N 份，试跑用；不传或 0 = 全量（1700 份约 1 分钟）。
#   format  想要什么形态的结果：json（POST 的响应）| html（GET report 存的简报文件）。
#           一个 HTTP 响应不能既是 JSON 又是 HTML，所以拆成两个端点、用 format 切换。
#   另外还有 refresh（丢弃解析缓存重解析）和 out（报告落盘目录）两个可选参数。
#
# 认证：Authorization: Bearer <API_KEY>。
# **API Key 只从环境变量读，不写死在脚本里，也不要提交进 git。**
#
# 用法：
#   export API_KEY=ragflow-xxxxxxxx
#   bash quality_api.sh                    # 默认：json + limit=3 试跑
#   LIMIT=0 bash quality_api.sh            # 全量
#   FORMAT=html bash quality_api.sh        # 取 HTML 简报，落到 ./quality_report.html
#   DOC_PATH=/data/其他目录 bash quality_api.sh
#   bash quality_api.sh --new-key          # 拿登录 token 换一个 API Key（需 LOGIN_TOKEN）
#   bash quality_api.sh --create-workflow  # 创建 document_ingestion_workflow（只需一次）
#   bash quality_api.sh --examples         # 打印所有示例命令
#   bash quality_api.sh --examples         # 只打印示例命令，不发请求
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_URL="http://localhost"
# 容器内的路径（不是宿主机的！）。挂载点见 docker-compose.yml
DOC_PATH="${DOC_PATH:-/ragflow/document_quality_assessment}"
LIMIT="${LIMIT:-3}"
FORMAT="${FORMAT:-json}"
OUT_DIR="${OUT_DIR:-/ragflow/quality_report}"
HTML_OUT="${HTML_OUT:-${SCRIPT_DIR}/quality_report.html}"

ACTION="${1:-run}"

# ------------------------------------------------------------------ 示例速查
print_examples() {
    cat <<'EOF'
==> 质检 API 示例命令（三参数各演示一遍）
#    下面的地址按默认部署写死成 http://localhost（宿主 80 端口 → nginx → 容器 9380）。
#    换了地址就把 http://localhost 替换掉，或者直接用 BASE_URL=... bash quality_api.sh

# 0) 认证头：所有请求都要带，API Key 从环境变量来
#    export API_KEY=ragflow-xxxxxxxx

# 1) 最小请求 —— 只给 path（必填），全量质检，返回 JSON
curl -sS -X POST "http://localhost/v1/document/quality-inspection" \
     -H "Authorization: Bearer ${API_KEY}" \
     -H 'Content-Type: application/json' \
     -d '{"path": "/ragflow/document_quality_assessment"}'

# 2) 加 limit —— 只跑前 3 份，试跑用（大目录全量要跑几十秒到几分钟）
curl -sS -X POST "http://localhost/v1/document/quality-inspection" \
     -H "Authorization: Bearer ${API_KEY}" \
     -H 'Content-Type: application/json' \
     -d '{"path": "/ragflow/document_quality_assessment", "limit": 3}'

# 3) 加 refresh + out —— 丢缓存重解析，报告落到自定义目录
curl -sS -X POST "http://localhost/v1/document/quality-inspection" \
     -H "Authorization: Bearer ${API_KEY}" \
     -H 'Content-Type: application/json' \
     -d '{"path": "/ragflow/document_quality_assessment", "limit": 3, "refresh": true, "out": "/ragflow/quality_report"}'

# 4) format=html —— 取人看的 HTML 简报，存成文件
curl -sS -G "http://localhost/v1/document/quality-inspection/report" \
     --data-urlencode "out=/ragflow/quality_report" \
     -H "Authorization: Bearer ${API_KEY}" \
     -o quality_report.html

# 5) 换个数据集 / 容器内实际挂进去的目录
#    DOC_PATH=/data/manuals bash quality_api.sh
EOF
}

if [ "${ACTION}" = "--examples" ]; then
    print_examples
    exit 0
fi

# ------------------------------------------------------------------ 换 API Key
if [ "${ACTION}" = "--new-key" ]; then
    # 登录 token 从 RAGFlow 登录接口拿（浏览器 F12 → Network，或 /v1/user/login 的响应）
    LOGIN_TOKEN="${LOGIN_TOKEN:?先 export LOGIN_TOKEN=<登录 token> 再跑 --new-key}"
    echo "==> POST ${BASE_URL}/v1/api/new_token"
    curl -sS -X POST "${BASE_URL}/v1/api/new_token" \
         -H "Authorization: Bearer ${LOGIN_TOKEN}" \
         -H 'Content-Type: application/json'
    echo
    echo "    把返回里的 api_key 存进环境变量：export API_KEY=ragflow-xxxx"
    exit 0
fi

# ------------------------------------------------------------------ 建工作流
if [ "${ACTION}" = "--create-workflow" ]; then
    # 把 研发/workflows/document_ingestion_workflow.json 里的 dsl 提交给 RAGFlow。
    # 纯 DSL 创建，不改 RAGFlow 代码。
    #
    # canvas_category 必须是 agent_canvas：dataflow 画布的执行器
    # （rag/flow/pipeline.py）只按下游线性推进，**不支持条件分支**，
    # 而本工作流靠 Switch 分流。
    WFDIR="${SCRIPT_DIR}/../研发/workflows"
    WF_JSON="${WFDIR}/document_ingestion_workflow.json"
    [ -f "${WF_JSON}" ] || { echo "[错误] 找不到工作流 DSL：${WF_JSON}" >&2; exit 1; }

    # 抽 dsl 字段并转成单行 JSON（curl -d 里塞多行容易出问题）
    PAYLOAD="$(python -c "
import json, sys
wf = json.load(open(sys.argv[1], encoding='utf-8'))
print(json.dumps({'title': wf['title'],
                  'description': wf.get('description', ''),
                  'canvas_category': 'agent_canvas',
                  'dsl': wf['dsl']}, ensure_ascii=False))
" "${WF_JSON}" 2>/dev/null || python3 -c "
import json, sys
wf = json.load(open(sys.argv[1], encoding='utf-8'))
print(json.dumps({'title': wf['title'],
                  'description': wf.get('description', ''),
                  'canvas_category': 'agent_canvas',
                  'dsl': wf['dsl']}, ensure_ascii=False))
" "${WF_JSON}")"

    API_KEY="${API_KEY:?先设置 API_KEY，见本脚本头部说明}"
    echo "==> POST ${BASE_URL}/v1/canvas/set  （创建工作流 document_ingestion_workflow）"
    curl -sS -X POST "${BASE_URL}/v1/canvas/set" \
         -H "Authorization: Bearer ${API_KEY}" \
         -H 'Content-Type: application/json' \
         -d "${PAYLOAD}"
    echo
    echo "    建好后在 RAGFlow 画布界面能看到它；Webhook 地址形如"
    echo "    ${BASE_URL}/api/v1/agents/webhook/<agent_id>"
    exit 0
fi

# API Key 只从环境变量读，不写死
API_KEY="${API_KEY:?请先设置 API_KEY，例如：export API_KEY=ragflow-xxxx（无 Key 时先跑 bash quality_api.sh --new-key）}"

POST_URL="${BASE_URL}/v1/document/quality-inspection"
GET_URL="${BASE_URL}/v1/document/quality-inspection/report"

# 输出 JSON 的排版器：能找着 python 就美化，找不着就直接打
pretty() {
    if command -v python3 >/dev/null 2>&1; then
        python3 -m json.tool
    elif command -v python >/dev/null 2>&1; then
        python -m json.tool
    else
        cat
    fi
}

echo "==> 目标：${BASE_URL}    容器内路径：${DOC_PATH}    limit=${LIMIT}    format=${FORMAT}"

case "${FORMAT}" in
json)
    # ---- 演示 1/2：path + limit（POST 返回 JSON） ----
    BODY="{\"path\": \"${DOC_PATH}\", \"limit\": ${LIMIT}}"
    echo "==> POST ${POST_URL}"
    echo "    body: ${BODY}"
    RESP="$(curl -sS -X POST "${POST_URL}" \
                 -H "Authorization: Bearer ${API_KEY}" \
                 -H 'Content-Type: application/json' \
                 -d "${BODY}")"

    # RAGFlow 的响应外层是 {"code":0,"data":{...},"message":"..."}，code 非 0 就是失败
    echo "${RESP}" | pretty

    if command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1; then
        PY="$(command -v python3 || command -v python)"
        echo "${RESP}" | "${PY}" -c '
import json, sys
try:
    r = json.load(sys.stdin)
except Exception:
    sys.exit(0)
if r.get("code") not in (0, None):
    print("    [失败] code=%s message=%s" % (r.get("code"), r.get("message")), file=sys.stderr)
    sys.exit(0)
s = (r.get("data") or {}).get("summary") or {}
if s:
    print("    --- 摘要 ---")
    print("    文件总数      : %s（%s 字节）" % (s.get("total_files"), s.get("total_bytes")))
    print("    PDF 类型分布  : %s" % (s.get("pdf"),))
    print("    扫描件占比    : %s" % (s.get("pdf_scanned_ratio"),))
    print("    解析失败      : %s" % (s.get("parse_failed"),))
    print("    完全重复组    : %s（冗余副本 %s）" % (s.get("exact_dup_groups"), s.get("redundant_copies")))
    print("    近似重复对    : %s" % (s.get("similar_pairs"),))
    print("    敏感信息命中  : %s" % (s.get("sensitive_hits"),))
    print("    主路由分布    : %s" % (s.get("by_primary_route"),))
'
        echo "    HTML 简报     : GET ${GET_URL}?out=${OUT_DIR}   （FORMAT=html bash quality_api.sh）"
    fi
    ;;
html)
    # ---- 演示 2/2：format=html（先跑质检确保有报告，再 GET 简报） ----
    echo "==> 先 POST 跑一次质检（否则可能还没有报告）"
    curl -sS -X POST "${POST_URL}" \
         -H "Authorization: Bearer ${API_KEY}" \
         -H 'Content-Type: application/json' \
         -d "{\"path\": \"${DOC_PATH}\", \"limit\": ${LIMIT}}" \
         -o /dev/null -w '    HTTP %{http_code}\n'

    echo "==> GET ${GET_URL}?out=${OUT_DIR}  ->  ${HTML_OUT}"
    CODE="$(curl -sS -G "${GET_URL}" \
                 --data-urlencode "out=${OUT_DIR}" \
                 -H "Authorization: Bearer ${API_KEY}" \
                 -o "${HTML_OUT}" -w '%{http_code}')"
    echo "    HTTP ${CODE}"
    if [ "${CODE}" = "200" ]; then
        echo "    简报已存：${HTML_OUT}（$(wc -c < "${HTML_OUT}" | tr -d ' ') 字节）"
        echo "    浏览器打开就能看，或者：less ${HTML_OUT}"
    else
        echo "    [失败] 拿不到简报。检查 out 目录对不对（默认 ${OUT_DIR}），"
        echo "           以及上一次 POST 是不是成功过。"
        head -c 300 "${HTML_OUT}" 2>/dev/null || true
        echo
    fi
    ;;
*)
    echo "[错误] FORMAT 只能是 json 或 html，收到：${FORMAT}" >&2
    exit 2
    ;;
esac

echo
echo "==> 看全部示例命令：bash quality_api.sh --examples"
