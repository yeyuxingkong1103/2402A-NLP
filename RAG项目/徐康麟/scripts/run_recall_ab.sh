#!/usr/bin/env bash
#
# 召回 A/B 驱动（部署在云端 Ubuntu 上跑；本机 Windows 跑不了：要 GPU、要 Linux bash）。
#
# 历史（读之前先看这段，否则会重复踩坑）
# ---------------------------------------------------------------------------
# * 第一版（2026-09-26/27）：只跑一份历史（labor），产出 `recall-{off,on}.jsonl`。
#   正文保留在 git：`git show cedceeb:scripts/run_recall_ab.sh`
# * 第二轮（2026-09-27 晚）：账号/会话/标签/**历史味道**改成环境变量可覆盖（默认值 = 第一版行为），
#   并新增"停 API 期间导出记忆原文 + 回音检查"一步（D5 口径 ②）。
# * 第三轮（2026-09-27 深夜）：**修掉一个把结论带偏的缺陷** ——
#   `scripts/eval_answers.py` 用的是"每题一会话"（`<session>-<item.id>`），
#   而两臂传的是**同一个** `--session` ⇒ **臂 B 的会话窗口里躺着臂 A 的那一轮问答**
#   （Redis 实证：同一个 key 里有"同一个问题问了两遍"两轮）。
#   于是臂 B 不只是"多了记忆分区"，还多了"自己刚才答过同一题"这个上下文 ⇒
#   模型倾向复述上一轮 ⇒ **系统性地把两臂拉平**。这正好能解释"两臂指标几乎一样"。
#   现在两臂各用**独立会话前缀**（`<SESS>off` / `<SESS>on`），并从结构上禁止共用
#   （见 `tests/test_recall_ab_driver.py::test_arms_do_not_share_a_session`）。
#
# 只在**云端**跑（写死了 /root/autodl-tmp/legal-rag、conda env `rag`）。
#
# 用法（例：跑「用户贴过法条」那份历史）：
#     ACCT=eval-recall-st SESS=recallst FLAVOR=statute \
#     TAG_OFF=recall-off-st TAG_ON=recall-on-st \
#     LOG=/root/autodl-tmp/logs/recall-ab-st.log \
#     nohup bash scripts/run_recall_ab.sh >/dev/null 2>&1 &
#     # 进度：tail -f /root/autodl-tmp/logs/recall-ab-st.log
#
# 退出码：长任务驱动，不靠退出码表达结果；每步退出码都打进日志，
# 跑完看三个哨兵：ARM-OFF-DONE / ARM-ON-DONE / RECALL-AB-DONE。
set -u

# ==========================================================================
# 可覆盖参数
# ==========================================================================
ACCT=${ACCT:-eval-recall-ab}          # 评测账号（**一份历史一个账号**：召回按用户维度取）
SESS=${SESS:-recallab}                # 会话**前缀**（⚠️ 别用被占过的 eval- 前缀，会 403）
SESS_OFF=${SESS_OFF:-${SESS}off}      # 臂 A 自己的会话前缀（**不许与臂 B 相同**）
SESS_ON=${SESS_ON:-${SESS}on}         # 臂 B 自己的会话前缀
FLAVOR=${FLAVOR:-labor}               # 历史性质：labor / statute / daily
TURNS=${TURNS:-10}                    # 灌几轮背景对话
TAG_OFF=${TAG_OFF:-recall-off}        # 召回**关**那一臂的产物标签
TAG_ON=${TAG_ON:-recall-on}           # 召回**开**那一臂的产物标签
DO_MEMORY_AUDIT=${DO_MEMORY_AUDIT:-1} # 1 = 停 API 期间导出记忆原文 + 回音检查
DO_SEED=${DO_SEED:-1}                 # 1 = 先养记忆；**0 = 复用已有记忆**（重跑同一份历史时用）
MIN_ECHO_CHARS=${MIN_ECHO_CHARS:-30}  # 回音检查阈值（最长公共子串 >= 它即判可疑）
ENV_FILE=${ENV_FILE:-/root/env_cloud.sh}

PY=/root/miniconda3/envs/rag/bin/python
APP=/root/autodl-tmp/legal-rag
LOG=${LOG:-/root/autodl-tmp/logs/recall-ab.log}
BASE=${BASE:-http://127.0.0.1:18080}

if [ "$SESS_OFF" = "$SESS_ON" ]; then
    echo "!! SESS_OFF 与 SESS_ON 相同（$SESS_OFF）：两臂会共用会话、臂 B 会看到臂 A 的问答" >&2
    echo "   这正是 2026-09-27 修掉的缺陷，拒绝继续。" >&2
    exit 2
fi

cd "$APP" || exit 9

# ⚠️ 所有 python 都带 `-u`：不带的话 stdout 是**块缓冲**，长任务跑到一半日志还是空的，
#    会让人误判成"卡住了"（2026-09-27 真被这个骗过一次，白等了十几分钟）。
{
  echo "======== $(date -Is) 召回 A/B 开始 ========"
  echo "账号=$ACCT  历史性质=$FLAVOR  轮数=$TURNS"
  echo "两臂会话前缀：关=$SESS_OFF  开=$SESS_ON（必须不同）"
  echo "两臂标签：关=$TAG_OFF  开=$TAG_ON"

  echo "--- 0) 等 API（27B 生产配置：AUTH_REQUIRED=true / LONGTERM_ENABLED=true / 召回默认关）"
  for i in $(seq 1 60); do
      curl -s -m 3 "$BASE/health" >/dev/null 2>&1 && break
      sleep 10
  done
  curl -s -m 10 "$BASE/health" | "$PY" -u -c '
import json,sys
d=json.load(sys.stdin)
print("  status   =", d.get("status"))
print("  longterm =", json.dumps(d.get("longterm"), ensure_ascii=False))
print("  engine   =", (d.get("engine") or {}).get("status"))
'
  pid=$(pgrep -f 'scripts/run_api.py' | head -1)
  echo "  API pid=$pid"
  tr '\0' '\n' < "/proc/$pid/environ" | grep -aE '^(AUTH_REQUIRED|LONGTERM_ENABLED|LONGTERM_RECALL_ENABLED|SESSION_WINDOW|LLM_MODEL|OPENAI_COMPAT_BASE_URL)=' | sed 's/^/  /'

  if [ "$DO_SEED" = "1" ]; then
      echo "--- 1) 养记忆（$ACCT 的历史性质=$FLAVOR，$TURNS 轮）$(date -Is)"
      "$PY" -u scripts/seed_eval_memory.py --username "$ACCT" --turns "$TURNS" --flavor "$FLAVOR"
      echo "seed exit=$?"
  else
      # 复跑同一份历史时不要再灌一遍：**位置寻址**下同样的内容会拿到新的 seq
      # ⇒ 变成新记录 ⇒ 记忆里堆近似重复，召回被自己的旧答案稀释（体检脚本会报出来）。
      echo "--- 1) 跳过养记忆（DO_SEED=0，复用 $ACCT 已有记忆）$(date -Is)"
  fi

  # 上一轮那半截坏结果不能留着：同 tag 非空会被拒绝覆盖（403 那次的教训）
  if [ -f "eval/results/${TAG_OFF}.jsonl" ]; then
      mv "eval/results/${TAG_OFF}.jsonl" "eval/results/${TAG_OFF}.aborted.jsonl"
      echo "  已把上一轮半截结果改名：${TAG_OFF}.aborted.jsonl"
  fi

  echo "--- 2) 臂 A：召回**关**（107 题，session=$SESS_OFF）$(date -Is)"
  "$PY" -u scripts/eval_answers.py --auth --auth-user "$ACCT" --auth-password 'EvalAccount#2026' \
      --user "$ACCT" --role lawyer --session "$SESS_OFF" --tag "$TAG_OFF" --out eval/results
  echo "arm-off exit=$?"
  echo "ARM-OFF-DONE $(date -Is)"

  echo "--- 3) 切到召回**开**：重启 API $(date -Is)"
  pkill -TERM -f 'run_api\.py' 2>/dev/null || true
  for i in $(seq 1 20); do pgrep -f 'run_api\.py' >/dev/null 2>&1 || break; sleep 1; done
  pkill -KILL -f 'run_api\.py' 2>/dev/null || true
  sleep 3
  bash -c "source $ENV_FILE; export LONGTERM_RECALL_ENABLED=true; \
      cd $APP && setsid nohup $PY \
      scripts/run_api.py --host \"\$API_HOST\" --port \"\$API_PORT\" \
      >> /root/autodl-tmp/logs/api.log 2>&1 </dev/null &"
  for i in $(seq 1 60); do
      curl -s -m 3 "$BASE/health" >/dev/null 2>&1 && break
      sleep 5
  done
  pid=$(pgrep -f 'scripts/run_api.py' | head -1)
  echo "  新 API pid=$pid"
  tr '\0' '\n' < "/proc/$pid/environ" | grep -aE '^(AUTH_REQUIRED|LONGTERM_ENABLED|LONGTERM_RECALL_ENABLED|SESSION_WINDOW|LLM_MODEL)=' | sed 's/^/  /'

  echo "--- 4) 臂 B：召回**开**（107 题，session=$SESS_ON）$(date -Is)"
  "$PY" -u scripts/eval_answers.py --auth --auth-user "$ACCT" --auth-password 'EvalAccount#2026' \
      --user "$ACCT" --role lawyer --session "$SESS_ON" --tag "$TAG_ON" --out eval/results
  echo "arm-on exit=$?"
  echo "ARM-ON-DONE $(date -Is)"

  echo "--- 5) 对比 $(date -Is)"
  "$PY" -u scripts/rescore_answers.py --in "eval/results/${TAG_OFF}.jsonl" \
      --in "eval/results/${TAG_ON}.jsonl" --compare --out "eval/results/${TAG_ON}-compare.json"
  echo "compare exit=$?"

  echo "--- 6) 确认召回真的发生了（看日志里的召回计数）"
  grep -ac '长期记忆召回成功' /root/autodl-tmp/logs/api.log || true
  grep -a '长期记忆召回' /root/autodl-tmp/logs/api.log | tail -3 | cut -c1-200 || true

  echo "--- 7) 停 API（下面两步要用 Milvus Lite 的单进程锁）$(date -Is)"
  pkill -TERM -f 'run_api\.py' 2>/dev/null || true
  for i in $(seq 1 20); do pgrep -f 'run_api\.py' >/dev/null 2>&1 || break; sleep 1; done
  pkill -KILL -f 'run_api\.py' 2>/dev/null || true
  sleep 3

  if [ "$DO_MEMORY_AUDIT" = "1" ]; then
      DUMP="eval/results/memory-duplicates-${FLAVOR}.json"
      ECHO_OUT="eval/results/memory-echo-${FLAVOR}.json"
      echo "--- 7b) 导出记忆原文（--user $ACCT -> $DUMP）$(date -Is)"
      ( source "$ENV_FILE" 2>/dev/null; "$PY" -u scripts/audit_memory_duplicates.py \
            --user "$ACCT" --out "$DUMP" )
      echo "  audit exit=$?"
      echo "--- 7c) 回音检查（D5 口径 ②）"
      # ⚠️ 退出码语义（v2）：1 = 有**记忆正文**被逐字照抄（待人复核）；
      #    命中引擎固定文案**不算**（那是设计如此）。所以这里不因 1 中断流程。
      "$PY" -u scripts/check_memory_echo.py --in "eval/results/${TAG_OFF}.jsonl" \
            --in "eval/results/${TAG_ON}.jsonl" --memory "$DUMP" \
            --min-chars "$MIN_ECHO_CHARS" --out "$ECHO_OUT"
      echo "  echo exit=$?（1 = 有记忆正文重合待人复核）"
  fi

  echo "--- 8) 恢复生产配置（召回关）$(date -Is)"
  bash -c "source $ENV_FILE; cd $APP && setsid nohup \
      $PY scripts/run_api.py --host \"\$API_HOST\" \
      --port \"\$API_PORT\" >> /root/autodl-tmp/logs/api.log 2>&1 </dev/null &"
  for i in $(seq 1 60); do
      curl -s -m 3 "$BASE/health" >/dev/null 2>&1 && break
      sleep 5
  done
  echo "生产配置已恢复"
  echo "RECALL-AB-DONE $(date -Is)"
} >>"$LOG" 2>&1
echo "（后台跑着，日志：$LOG）"
