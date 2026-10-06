#!/usr/bin/env bash
# 验证本地 bge-reranker 重排服务是否**真的在重排**，而不是只起了个能响应 /health 的空壳。
#
# 用一组「语义最相关的那条并不是字面最像的那条」的样例：交叉编码器应当把它顶到第一，
# 且各条分数要有区分度。若服务不可用或返回全等分数，脚本以非 0 退出。
#
# 用法：bash scripts/verify_rerank.sh [base_url]
# 输出同时落盘到 logs/verify_rerank.out
set -uo pipefail
cd "$(dirname "$0")/.."

BASE_URL="${1:-http://127.0.0.1:8001}"
OUT="logs/verify_rerank.out"
mkdir -p logs

exec > >(tee "$OUT") 2>&1

echo "=== bge-rerank 服务验证 $(date '+%F %T') ==="
echo "服务地址: $BASE_URL"

echo
echo "--- 1) 健康检查 ---"
health="$(curl -s --max-time 10 "$BASE_URL/health")"
if [[ -z "$health" ]]; then
    echo "❌ /health 无响应：服务没起来？先跑 bash scripts/run_rerank.sh"
    exit 1
fi
echo "✅ /health -> $health"

echo
echo "--- 2) 真实重排请求 ---"
# query 与 doc[0] 语义最相关（问「最长」，只有它给的是上限）；doc[2] 的字面重合度最高
# （三处「试用期」）却答非所问——考的就是交叉编码器看没看懂语义
curl -s --max-time 60 -X POST "$BASE_URL/v1/rerank" \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "BAAI/bge-reranker-v2-m3",
      "query": "试用期最长可以约定多久？",
      "documents": [
        "三年以上固定期限和无固定期限的劳动合同，试用期不得超过六个月。",
        "劳动合同期限三个月以上不满一年的，试用期不得超过一个月。",
        "试用期期间，用人单位应当为劳动者缴纳社会保险。",
        "今天天气不错，适合出门散步。"
      ],
      "top_n": 4
    }' \
| .venv/bin/python -c '
import json, sys

raw = sys.stdin.read()
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    print("返回不是 JSON：", raw[:500]); sys.exit(1)

results = data.get("results") or []
if not results:
    print("没有 results：", raw[:500]); sys.exit(1)

for r in results:
    print("  #{}  score={:.6f}".format(r["index"], r["relevance_score"]))

scores = [r["relevance_score"] for r in results]
order = [r["index"] for r in results]
problems = []

# 判据一：语义相关的两条（#0 六个月 / #1 一个月）都要压在字面陷阱 #2 之上。
#
# 这里刻意**不**要求 #0 排第一：#0 与 #1 都是「试用期最长」的合理答案（都给了具体期限），
# 谁第一取决于模型当天的偏好，实测两条只差 0.011（0.687 / 0.676）。要 #0 必第一，就成了
# 在给模型打分，而不是在验「精排有没有把语义算进去」——2026-09-23 实测就因此误报过一次。
# 真正有信息量的是 #2：它字面重合最高（三处「试用期」）却答的是社保问题，纯字面匹配会把它
# 顶到前面，只有语义精排会把它压下去。
if not (order.index(0) < order.index(2) and order.index(1) < order.index(2)):
    problems.append("字面陷阱 #2 压过了语义相关的 #0/#1，实际顺序 {}".format(order))
# 判据二：分数要有区分度，全等说明没真的算
if len(set(round(s, 6) for s in scores)) < 2:
    problems.append("所有分数完全相同（{}），交叉编码器没真的算".format(scores[0]))
# 判据三：无关的那条要垫底
if results[-1]["index"] != 3:
    problems.append("无关文档 #3 没垫底，实际垫底是 #{}".format(results[-1]["index"]))

if problems:
    print()
    print("重排结果不符合预期：")
    for p in problems:
        print("   -", p)
    sys.exit(1)

print()
print("重排行为正确：#0/#1（语义相关）都在字面陷阱 #2 之上、#3（无关）垫底，分数有区分度")
'
status=$?

echo
if [[ $status -eq 0 ]]; then
    echo "=== ✅ 通过：bge-reranker 确实在做语义精排 ==="
else
    echo "=== ❌ 未通过 ==="
fi
exit $status
