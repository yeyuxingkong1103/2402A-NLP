# -*- coding: utf-8 -*-
"""M3+M4 验收：对话链路（改写→混合检索→精排→人格→流式生成）+ 短期记忆 + 隔离。

用法（服务需已启动）：
    D:\\anaconda3\\envs\\rag_env\\python.exe scripts\\test_m3_m4.py
"""
import json
import secrets
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
BASE = "http://127.0.0.1:8000/api"

ok_count = fail_count = 0


def check(label, cond, detail=""):
    global ok_count, fail_count
    if cond:
        ok_count += 1
        print(f"  [PASS] {label}" + (f"  {detail}" if detail else ""))
    else:
        fail_count += 1
        print(f"  [FAIL] {label}  {detail}")


def login(c, prefix):
    u = f"{prefix}_{int(time.time())}"
    pw = secrets.token_hex(8)   # 临时用户密码每次随机生成，不写死
    r = c.post(f"{BASE}/auth/register", json={"username": u, "password": pw})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}, u


def main():
    c = httpx.Client(timeout=300)
    H, uname = login(c, "m3")
    chars = {ch["slug"]: ch for ch in c.get(f"{BASE}/characters", headers=H).json()}

    print("=" * 74)
    print("① 医生角色：非流式问答")
    print("=" * 74)
    conv = c.post(f"{BASE}/conversations", headers=H,
                  json={"character_id": chars["doctor"]["id"]}).json()
    q1 = "高血压患者能吃腌菜吗？"
    t0 = time.time()
    r = c.post(f"{BASE}/chat/completions", headers=H,
               json={"conversation_id": conv["id"], "question": q1})
    dt = time.time() - t0
    res = r.json()
    print(f"  Q: {q1}")
    print(f"  A: {res['answer'][:180]}")
    print(f"  耗时 {dt:.1f}s | trace={json.dumps(res['trace'], ensure_ascii=False)}")
    print(f"  来源 {len(res['sources'])} 条:")
    for s in res["sources"]:
        print(f"     [{s['idx']}] {s['source']}  rerank={s['rerank_score']}")
    print()
    check("返回了回答", len(res["answer"]) > 20, f"{len(res['answer'])} 字")
    check("精排分数已计算", all(s["rerank_score"] is not None for s in res["sources"]))
    check("来源来自 kb_medical", all(s["collection"] == "kb_medical" for s in res["sources"]))
    check("精排精选出 ≤5 条", len(res["sources"]) <= 5, f"{len(res['sources'])} 条")
    check("trace 含召回→精排明细", res["trace"]["recall"] >= res["trace"]["reranked"])
    print()

    print("=" * 74)
    print("② 短期记忆（Redis）：写入 + 多轮指代消解")
    print("=" * 74)
    mem = c.get(f"{BASE}/conversations/{conv['id']}/memory", headers=H).json()
    check("Redis 可用", mem["redis_available"])
    check("记忆已写入（1 轮 = 2 条）", len(mem["redis"]) == 2, f"{len(mem['redis'])} 条")
    for m in mem["redis"]:
        print(f"     {m['role']:<10} {m['content'][:56]}")
    print()

    q2 = "那每天最多能吃多少？"
    t0 = time.time()
    r2 = c.post(f"{BASE}/chat/completions", headers=H,
                json={"conversation_id": conv["id"], "question": q2}).json()
    rw = r2["trace"]["rewritten_query"]
    print(f"  Q: {q2}")
    print(f"  改写后: {rw}")
    print(f"  A: {r2['answer'][:150]}")
    print(f"  耗时 {time.time()-t0:.1f}s")
    print()
    check("指代被消解（改写引入了原文没有的主语）",
          rw != q2 and ("高血压" in rw or "盐" in rw or "钠" in rw), f"-> {rw}")
    check("回答含具体数值（指南里的 5 克）", "5" in r2["answer"] or "克" in r2["answer"])
    mem2 = c.get(f"{BASE}/conversations/{conv['id']}/memory", headers=H).json()
    check("记忆增长到 2 轮（4 条）", len(mem2["redis"]) == 4, f"{len(mem2['redis'])} 条")
    if mem2["redis"]:
        check("记忆按时间正序", mem2["redis"][0]["role"] == "user")
    else:
        check("记忆按时间正序", False, "记忆为空，无法判断（Redis 是否可用？）")
    print()

    print("=" * 74)
    print("③ 律师角色：知识库自动路由到 kb_legal")
    print("=" * 74)
    conv2 = c.post(f"{BASE}/conversations", headers=H,
                   json={"character_id": chars["lawyer"]["id"]}).json()
    q3 = "在网上买到假货，能要求退一赔三吗？"
    r3 = c.post(f"{BASE}/chat/completions", headers=H,
                json={"conversation_id": conv2["id"], "question": q3}).json()
    print(f"  Q: {q3}")
    print(f"  A: {r3['answer'][:220]}")
    print(f"  来源:")
    for s in r3["sources"]:
        print(f"     [{s['idx']}] {s['source']}  rerank={s['rerank_score']}")
    print()
    check("来源全部来自 kb_legal", all(s["collection"] == "kb_legal" for s in r3["sources"]))
    check("命中消保法条文（退一赔三在消保法第55条）",
          any("消费者权益保护法" in (s["law_name"] or "") for s in r3["sources"]),
          str([s["law_name"] for s in r3["sources"]]))
    check("回答引用了来源编号", "[" in r3["answer"])
    print()

    print("=" * 74)
    print("④ 检索调试台：纯检索（不生成）")
    print("=" * 74)
    sr = c.post(f"{BASE}/search", headers=H,
                json={"query": "违约责任有哪些承担方式", "collection": "kb_legal",
                      "top_k": 5, "use_rerank": True}).json()
    print(f"  Q: 违约责任有哪些承担方式")
    print(f"  召回 {sr['recall_count']} → 精排 {len(sr['hits'])} | 耗时 {sr['elapsed_ms']}ms")
    for h in sr["hits"][:3]:
        print(f"     {h['rerank_score']:+.3f}  {h['source_label']}")
        print(f"        {h['text'][:72]}…")
    print()
    check("纯检索返回结果", len(sr["hits"]) == 5)
    check("未调用大模型（无 answer 字段）", "answer" not in sr)
    check("检索历史已存", len(c.get(f"{BASE}/search/history", headers=H).json()) >= 1)
    print()

    print("=" * 74)
    print("⑤ SSE 流式对话")
    print("=" * 74)
    conv3 = c.post(f"{BASE}/conversations", headers=H,
                   json={"character_id": chars["doctor"]["id"]}).json()
    events = []
    deltas = []
    t_first = None
    t0 = time.time()
    with c.stream("POST", f"{BASE}/chat/stream", headers=H,
                  json={"conversation_id": conv3["id"], "question": "高血压的危害有哪些？"}) as resp:
        check("SSE 状态码 200", resp.status_code == 200)
        check("Content-Type 为 text/event-stream",
              "text/event-stream" in resp.headers.get("content-type", ""))
        cur_event = None
        for line in resp.iter_lines():
            if line.startswith("event: "):
                cur_event = line[7:].strip()
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                events.append(cur_event)
                if cur_event == "delta":
                    if t_first is None:
                        t_first = time.time() - t0
                    deltas.append(data["text"])
    print(f"  事件序列: {[e for i, e in enumerate(events) if i == 0 or events[i-1] != e]}")
    print(f"  事件总数 {len(events)} | delta {len(deltas)} 段")
    print(f"  首 token 延迟 {t_first:.2f}s | 总耗时 {time.time()-t0:.1f}s")
    print(f"  拼接结果: {''.join(deltas)[:150]}")
    print()
    check("事件顺序正确 trace→sources→delta→done",
          events[0] == "trace" and events[1] == "sources"
          and events[-1] == "done" and "delta" in events)
    check("首 token 延迟 < 3s", t_first is not None and t_first < 3, f"{t_first:.2f}s")
    check("流式内容非空", len("".join(deltas)) > 30)
    print()

    print("=" * 74)
    print("⑥ 多用户隔离（会话不可越权访问）")
    print("=" * 74)
    H2, uname2 = login(c, "bob")
    r = c.get(f"{BASE}/conversations/{conv['id']}/messages", headers=H2)
    check("他人会话返回 404（不泄露存在性）", r.status_code == 404, f"HTTP {r.status_code}")
    r = c.post(f"{BASE}/chat/completions", headers=H2,
               json={"conversation_id": conv["id"], "question": "偷看"})
    check("他人会话无法对话 (404)", r.status_code == 404, f"HTTP {r.status_code}")
    lst1 = c.get(f"{BASE}/conversations", headers=H).json()
    lst2 = c.get(f"{BASE}/conversations", headers=H2).json()
    check(f"会话列表按用户隔离（{uname}:{len(lst1)} 个 / {uname2}:{len(lst2)} 个）",
          len(lst1) >= 3 and len(lst2) == 0)

    import pymysql
    from app.core.config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB
    conn = pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
                           password=MYSQL_PASSWORD, database=MYSQL_DB, charset="utf8mb4")
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*), SUM(role='user'), SUM(role='assistant') FROM messages")
        total, nu, na = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM messages WHERE sources_json IS NOT NULL")
        with_src = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM messages WHERE trace_json IS NOT NULL")
        with_trace = cur.fetchone()[0]
    conn.close()
    check("消息已落库", total >= 8, f"{total} 条（用户 {nu} / 助手 {na}）")
    check("助手消息含来源快照", with_src >= 4, f"{with_src} 条")
    check("助手消息含链路追溯", with_trace >= 4, f"{with_trace} 条")
    print()

    print("=" * 74)
    print(f"结果: {ok_count} 通过 / {fail_count} 失败")
    print("=" * 74)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
