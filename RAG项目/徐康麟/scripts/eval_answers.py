#!/usr/bin/env python3
"""跑自建评测集：逐题打 API（或只判分已有结果），产出 JSONL 明细 + Markdown 报告。

用法（云端，链路已起）：
    python scripts/eval_answers.py --base-url http://127.0.0.1:18080 \
        --role lawyer --out eval/results --tag v1

也支持「只判分」：``--score-only eval/results/v1.jsonl``（换了判分逻辑不想重跑时用）。

设计取舍：
* **逐题串行**：并发会抢同一张卡的 KV cache，TTFT/ITL 不再可比；要压测用
  ``.pytmp/cloud/bench_vllm.py``。
* **每题都落盘**（JSONL 追加 + 完整答案），跑一半崩了不至于全丢。
* **不静默失败**：单题异常照常记录 `error` 字段并继续，最后在报告里点名。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_rag.eval_set import (  # noqa: E402
    QAItem, corpus_law_names, load_qa_set, score_item, summarize,
)
from legal_rag.logging_setup import console_safe  # noqa: E402


def _one_turn(base_url: str, *, role: str, user: str, session_id: str, message: str,
              top_k: int, timeout: float, cookie: str = "") -> tuple[dict, float]:
    """发一条消息，返回 ``(响应, 秒)``。

    ``cookie`` 非空时带上登录态：**交付配置（`AUTH_REQUIRED=true`）下不带 cookie 的
    `/chat` 一律 401**，整套评测会全题失败（不是模型的问题，别误判成"答不出来"）。
    """
    body: dict = {
        "user_id": user, "role_id": role, "session_id": session_id,
        "message": message, "stream": False,
    }
    if top_k > 0:                      # 0 = 不覆盖（别把 0 发下去，那会被当成"要 0 条"）
        body["top_k"] = top_k
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if cookie:
        headers["Cookie"] = cookie
    req = urllib.request.Request(f"{base_url.rstrip('/')}/chat", data=payload,
                                 headers=headers)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace")), time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        return ({"_error": f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:200]}"},
                time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001 - 单题失败不该中断整轮
        return {"_error": f"{type(exc).__name__}: {exc}"}, time.perf_counter() - started


def _auth_cookie(base_url: str, username: str, password: str, timeout: float) -> str:
    """注册（或已存在则登录）评测专用账号，返回可直接放进 ``Cookie`` 头的字符串。

    * **注册优先**：全新 tag 的评测账号名不会撞车；撞了就说明跑过一次，改走登录；
    * **失败即抛**：拿不到登录态时**绝不能**继续跑（否则 107 题全变 401，
      报告会显示"全部答不出"，把一个环境问题误读成模型能力问题）。
    """
    base = base_url.rstrip("/")

    def _post(path: str, payload: dict) -> tuple[int, dict, str]:
        req = urllib.request.Request(
            base + path, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8", "replace")), \
                    resp.headers.get("Set-Cookie", "")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:200]
            return exc.code, {"_error": body}, ""

    status, _body, set_cookie = _post("/auth/register",
                                      {"username": username, "password": password})
    ok = status == 201                       # 注册成功即已登录（201 + Set-Cookie）
    if not ok:
        status, _body, set_cookie = _post("/auth/login",
                                          {"username": username, "password": password})
        ok = status == 200                    # 账号已存在时走登录
    if not ok or not set_cookie:
        raise RuntimeError(f"评测账号登录失败（register/login 最后返回 {status}）：{_body}")
    return set_cookie.split(";", 1)[0]


def ask(base_url: str, item: QAItem, *, role: str, user: str, session: str,
        timeout: float, top_k: int = 0, cookie: str = "") -> dict:
    """把一道题问完（**多轮题按顺序在同一会话里发**），返回最后一轮的响应。

    多轮只对**最后一轮**判分（``question`` 是首问、``turns`` 是完整消息序列），
    并保留每一轮的答案（``_turns``）便于人工看指代是否跟对。
    """
    session_id = f"{session}-{item.id}"
    last: dict = {}
    answers: list[str] = []
    total = 0.0
    for message in item.messages:
        last, seconds = _one_turn(base_url, role=role, user=user, session_id=session_id,
                                  message=message, top_k=top_k, timeout=timeout,
                                  cookie=cookie)
        total += seconds
        answers.append(str(last.get("answer") or last.get("_error") or ""))
        if last.get("_error"):
            break
    result = dict(last)
    result["_seconds"] = round(total, 3)
    result["_turns"] = answers
    return result


def run_round(args) -> list[dict]:
    items = load_qa_set(args.qa_file)
    if args.limit:
        items = items[: args.limit]
    known = corpus_law_names(args.corpus)
    print(f"题集 {args.qa_file}：{len(items)} 题；语料法名 {len(known)} 个；"
          f"目标 {args.base_url} role={args.role}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / f"{args.tag}.jsonl"
    # **拒绝覆盖已有结果**（真机踩到，2026-09-24）：同一 tag 被两个进程同时写，行会**交错**，
    # 结果是文件里出现 `Extra data` / 非法 UTF-8 字节、几十条记录报废（v6c 掉了 13 条）。
    # 事前拦一道，比事后从字节流里捞数据便宜得多。
    if detail_path.exists() and detail_path.stat().st_size > 0 and not args.force:
        print(f"!! {detail_path} 已存在（{detail_path.stat().st_size} 字节）—— **拒绝覆盖**："
              f"同一 tag 被两个进程同时写会让结果交错损坏。请换 --tag（推荐）或显式 --force。",
              file=sys.stderr)
        return []          # 与 run_round 的返回类型一致（调用方按"没有结果"处理并 exit 2）

    results: list[dict] = []
    cookie = ""
    if getattr(args, "auth", False):
        cookie = _auth_cookie(args.base_url, args.auth_user, args.auth_password,
                              args.timeout)
        print(f"  已取得登录态（评测账号 {args.auth_user}），/chat 会带上 Cookie")
    denied = 0          # 连续被 401/403 拒绝的题数（身份/归属问题，不是模型问题）
    aborted = False
    with detail_path.open("w", encoding="utf-8") as fh:
        for index, item in enumerate(items, 1):
            resp = ask(args.base_url, item, role=args.role, user=args.user,
                       session=args.session, timeout=args.timeout, top_k=args.top_k,
                       cookie=cookie)
            answer = str(resp.get("answer") or "")
            record = {
                "id": item.id, "category": item.category, "expect": item.expect,
                "question": item.question, "notes": item.notes,
                "sources": list(item.sources), "claims": list(item.claims),
                # 多轮题：turns 是完整消息序列，turns_answers 是每一轮的答案（人工看指代用）
                "turns": list(item.turns), "turns_answers": resp.get("_turns") or [answer],
                "answer": answer, "citations_raw": resp.get("citations") or [],
                "provider": resp.get("provider"), "degraded": resp.get("degraded"),
                "seconds": resp.get("_seconds"), "error": resp.get("_error"),
            }
            if record["error"]:
                # ⚠️ 出错题也必须写**同一套字段**（真机踩到，2026-09-24）：以前这里只写
                # `{"id","ok","error"}`，于是 `summarize()` 里 `r["expect"]` 直接 KeyError
                # ⇒ **整轮报告一行都没写**，而 jsonl 看着是完整的（"跑完了没报告"）。
                record["score"] = {
                    "id": item.id, "category": item.category, "expect": item.expect,
                    "ok": False, "source_hit": None, "claims_ok": None,
                    "missing_claims": list(item.claims), "citations": [],
                    "unknown_citations": [], "citations_from_evidence": [],
                    "citations_not_a_law": [], "citations_known": [],
                    "refused": False, "warned": False, "cites_law": False,
                    "answer_chars": 0, "error": record["error"],
                }
            else:
                record["score"] = score_item(item, answer, resp.get("citations"), known)
            results.append(record)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            # **连撞 401/403 就立刻停**（真机踩到，2026-09-26）：身份/归属问题会让
            # 107 题**全部**在几毫秒内失败，报告看着像"模型一问三不知"，其实是环境问题。
            # 早停 + 明说原因，比跑完 107 题再让人去猜便宜得多。
            if record["error"] and str(record["error"]).startswith(("HTTP 401", "HTTP 403")):
                denied += 1
                if denied >= 2:
                    print(
                        f"!! 连续 {denied} 题被拒绝（401/403）—— 这是**身份/归属**问题，"
                        f"不是模型答不出来。两种常见原因：\n"
                        f"   (a) 服务端 AUTH_REQUIRED=true 但本工具没加 --auth（会全 401）；\n"
                        f"   (b) --session={args.session!r} 这个前缀与**别的账号**建过的会话撞了"
                        f"（会话归属校验会 403「该会话不属于当前用户」）—— 换个 --session 再跑。\n"
                        f"   最后一次错误：{record['error'][:200]}\n"
                        f"   半截结果将改名为 {detail_path.with_suffix('.failed.jsonl').name}"
                        f"（**不作为有效臂**）",
                        file=sys.stderr)
                    aborted = True
                    break
            else:
                denied = 0
            flag = "[OK]" if record["score"].get("ok") else "[FAIL]"
            turns = f"[{len(item.turns)}轮]" if item.turns else ""
            # ⚠️ 答案来自模型 ⇒ 里面可能有 emoji。GBK 控制台上那会 **UnicodeEncodeError**：
            #    整行丢失、脚本退出码变 1（看着像"评测自己失败了"）。所以过一道 console_safe。
            print(f"  [{index}/{len(items)}] {flag} {item.id}{turns} {record['seconds']}s "
                  f"{console_safe(answer[:60].replace(chr(10), ' '))}")
    if aborted:
        # 改名必须在文件句柄关闭之后做：Windows 上**不允许重命名打开着的文件**
        detail_path.replace(detail_path.with_suffix(".failed.jsonl"))
        return []
    return [r["score"] for r in results]


def write_report(results: list[dict], detail_path: Path, report_path: Path) -> dict:
    summary = summarize(results)
    detail = {r["id"]: r for r in results}

    lines = [
        "# 自建评测集报告",
        "",
        f"- 题数：{summary['total']}　整体通过率：{summary['ok_rate']}",
        f"- **来源命中率**：{summary['source_hit_rate']}　（有据题里期望来源被引用/召回的比例）",
        f"- **要点覆盖率**：{summary['claim_coverage']}　（期望关键词齐备的比例）",
        f"- **引用可信度**：{summary['citation_fidelity']}　（没有引用语料外法名的比例）",
        f"- **分流准确率**：{summary['route_accuracy']}　（拒答题真拒答 / 日常题不引法条）",
        "",
        f"- 引用语料外法名的题：{', '.join(summary['unknown_citation_items']) or '无'}",
        f"- 未通过的题：{', '.join(summary['failed_items']) or '无'}",
        "",
        "> 判据是**词面**的：只能证伪不能证真，逐题答案见同名 `.jsonl`，务必人工复核。",
        "",
        "## 逐题",
        "",
        "| id | 期望 | 通过 | 来源命中 | 要点 | 未知引用 | 秒 | 答案摘录 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rid, rec in detail.items():
        answer = (rec.get("answer") or "").replace("|", "／").replace("\n", " ")[:80]
        lines.append(
            f"| {rid} | {rec.get('expect')} | {'✅' if rec.get('ok') else '❌'} | "
            f"{rec.get('source_hit')} | {rec.get('claims_ok')} | "
            f"{','.join(rec.get('unknown_citations') or []) or '-'} | "
            f"{rec.get('seconds', '-')} | {answer} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="自建评测集：跑题 + 判分 + 报告")
    parser.add_argument("--qa-file", default="eval/qa_set.jsonl")
    parser.add_argument("--corpus", default="knowledge/lawyer", help="用于识别语料外法名")
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--role", default="lawyer")
    parser.add_argument("--user", default="eval-bot")
    parser.add_argument("--session", default="eval")
    parser.add_argument("--out", default="eval/results")
    parser.add_argument("--tag", default="run1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--top-k", type=int, default=0,
                        help="覆盖最终喂给模型的条数（0 = 用服务端配置）")
    parser.add_argument("--score-only", default="", help="只判分已有 JSONL，不重新打 API")
    parser.add_argument("--force", action="store_true",
                        help="允许覆盖已存在的 <tag>.jsonl（默认**拒绝**：同 tag 双写会让结果交错损坏）")
    parser.add_argument("--auth", action="store_true",
                        help="交付配置（AUTH_REQUIRED=true）下必须加：先注册/登录评测账号再打 /chat")
    parser.add_argument("--auth-user", default="", help="评测账号名（默认按 --tag 自动生成）")
    parser.add_argument("--auth-password", default="EvalAccount#2026",
                        help="评测账号口令（≥10 位）")
    args = parser.parse_args()
    if args.auth and not args.auth_user:
        args.auth_user = f"eval-{args.tag}"[:48]

    out_dir = Path(args.out)
    if args.score_only:
        path = Path(args.score_only)
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        known = corpus_law_names(args.corpus)
        # 优先用**当前题集**里的期望（改了 sources/claims 后不必重新打 API），
        # 题集里没有的 id 再退回记录里带的字段。
        by_id = {item.id: item for item in load_qa_set(args.qa_file)}
        results = []
        for rec in records:
            item = by_id.get(rec["id"]) or QAItem.from_dict({
                "id": rec["id"], "category": rec["category"], "expect": rec["expect"],
                "question": rec["question"], "sources": rec.get("sources") or [],
                "claims": rec.get("claims") or [],
            })
            results.append(score_item(item, rec.get("answer") or "",
                                      rec.get("citations_raw"), known))
        report_path = path.with_suffix(".report.md")
        summary = write_report(results, path, report_path)
    else:
        results = run_round(args)
        if not results:
            print("!! 本轮没有产出任何判分（见上面的原因，例如拒绝覆盖已有 tag）", file=sys.stderr)
            return 2
        detail_path = out_dir / f"{args.tag}.jsonl"
        report_path = out_dir / f"{args.tag}.report.md"
        summary = write_report(results, detail_path, report_path)
        expected = len(load_qa_set(args.qa_file)) if not args.limit else args.limit
        if len(results) != expected:
            # 写完再核一遍条数：少了就说明中途出错/被打断，别让报告看起来"跑完了"
            print(f"!! 完成条数 {len(results)} != 预期 {expected}（结果可能不完整，请核对日志）",
                  file=sys.stderr)

    print("\n=== 汇总 ===")
    for key in ("total", "ok_rate", "source_hit_rate", "claim_coverage",
                "citation_fidelity", "route_accuracy"):
        print(f"  {key}: {summary[key]}")
    print(f"  未通过: {summary['failed_items']}")
    print(f"  引用语料外法名: {summary['unknown_citation_items']}")
    print(f"报告已写：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
