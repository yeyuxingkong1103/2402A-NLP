"""交付前自检：**鉴权面 + 跨用户隔离**（真跑真服务，A/B 两个账号）。

为什么写它（两条都是真机教训）：
1. `handoff/ISOLATION-audit.md` 记过一个真实缺口 **G1**：`POST /sessions` 缺归属校验，
   B 用自己身份 + A 的 `session_id` 就能**改掉 A 的标题**，而且响应谎报归属 ——
   当时"只报告未修"。文档说后来按 `T75-B1` 修过，但**文档不等于实现**，交付前必须实跑复现一遍。
2. 交付要求 `AUTH_REQUIRED=true`（默认 false 时接口接受任意 `user_id`）—— 这条不能靠"我记得设了"。

覆盖（每条都有明确 PASS/FAIL/SKIP 与证据）：
* 开关与新鲜度：`/health` 可用、`auth_required=true`；
* 凭据面：注册 → 登录（Cookie）→ 未带凭据/伪造凭据一律 401 → 登出后 401；
* 归属面：B 不能建/改名/删/读 A 的会话（G1 回归），且"别人的"与"不存在的"响应**不可区分**；
* 身份面：B 带 A 的 `user_id` 调 `/chat` 必须 403（`user_mismatch`）；
* 口径面：用户名不存在与密码错误**状态码+文案逐字一致**；
* 存储面：SQLite 只读核对行归属、Redis 只读 SCAN 核对 key 是否带用户维度。

用法（云端）::

    python scripts/verify_delivery.py --base-url http://127.0.0.1:18080 \\
        --db index/legal_rag.db --out eval/results/delivery-check.json

退出码：0 = 全通过（SKIP 不算失败）；1 = 有 FAIL。
"""
from __future__ import annotations

import argparse
import http.cookiejar
import json
import secrets
import sqlite3
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legal_rag.logging_setup import console_safe  # noqa: E402

PASSWORD = "DelivCheck#2026"
CHECKS: list[dict[str, Any]] = []


class Client:
    """一个账号一个 cookie jar（服务端把令牌只放 Cookie，脚本用 jar 最贴近真实浏览器）。"""

    def __init__(self, base_url: str, *, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar))

    def call(self, method: str, path: str, payload: dict | None = None,
             *, headers: dict | None = None) -> tuple[int, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(f"{self.base_url}{path}", data=data, method=method)
        request.add_header("Content-Type", "application/json; charset=utf-8")
        for key, value in (headers or {}).items():
            request.add_header(key, value)
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", "replace")
                return response.status, _decode(body)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            return exc.code, _decode(body)
        except Exception as exc:  # noqa: BLE001 - 连不上也要看得见
            return 0, {"_error": f"{type(exc).__name__}: {exc}"}


def _decode(body: str) -> Any:
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body[:400]


def record(name: str, status: str, detail: str, **evidence: Any) -> None:
    CHECKS.append({"check": name, "status": status, "detail": detail, **evidence})


def expect(condition: bool, name: str, ok_detail: str, fail_detail: str, **evidence: Any) -> bool:
    record(name, "PASS" if condition else "FAIL", ok_detail if condition else fail_detail, **evidence)
    return condition


def main() -> int:
    parser = argparse.ArgumentParser(description="交付前自检（鉴权面 + 跨用户隔离）")
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--db", default="index/legal_rag.db")
    parser.add_argument("--redis-prefix", default="legal_rag:")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    suffix = secrets.token_hex(3)
    name_a, name_b = f"deliv-a-{suffix}", f"deliv-b-{suffix}"

    # ---------- 0) 服务与开关 ----------
    anon = Client(args.base_url)
    status, health = anon.call("GET", "/health")
    record("health", "PASS" if status == 200 else "FAIL",
           f"GET /health -> {status}", body=str(health)[:200])
    if status != 200:
        print(json.dumps(CHECKS, ensure_ascii=False, indent=2))
        return 1
    # ⚠️ `auth_required` **不在** `/health` 里（真机踩到：我第一版查 health，得到假 FAIL）。
    # 它在 `/auth/login`、`/auth/register`、`/auth/me` 的响应里；这里用登录响应判定，
    # 并用"无凭据问问题是否 401"作为**行为侧**的交叉验证（两条都要成立）。
    auth_required: bool | None = None

    # ---------- 1) 注册 / 登录 ----------
    client_a, client_b = Client(args.base_url), Client(args.base_url)
    status_a, body_a = client_a.call("POST", "/auth/register",
                                     {"username": name_a, "password": PASSWORD})
    status_b, body_b = client_b.call("POST", "/auth/register",
                                     {"username": name_b, "password": PASSWORD})
    expect(status_a == 201 and status_b == 201, "register",
           f"两个账号注册成功（{status_a}/{status_b}）", f"注册失败：{status_a}/{status_b}",
           body_a=str(body_a)[:200], body_b=str(body_b)[:200])

    if not (status_a == 201 and status_b == 201):
        _flush(args.out)
        return 1

    _, me_a = client_a.call("GET", "/auth/me")
    _, me_b = client_b.call("GET", "/auth/me")
    user_a = str((me_a or {}).get("user_id") or "") if isinstance(me_a, dict) else ""
    user_b = str((me_b or {}).get("user_id") or "") if isinstance(me_b, dict) else ""
    expect(bool(user_a and user_b), "login_via_register_cookie",
           f"注册即登录，/auth/me 拿到 user_id（{user_a[:18]}… / {user_b[:18]}…）",
           f"/auth/me 异常：{me_a} / {me_b}")

    # 开关判定（用登录/注册响应里的 auth_required 字段，而不是 /health）
    if isinstance(me_a, dict):
        auth_required = me_a.get("auth_required")
    if auth_required is None:
        _, login_probe = Client(args.base_url).call("POST", "/auth/login",
                                                    {"username": name_a, "password": PASSWORD})
        auth_required = (login_probe or {}).get("auth_required")
    expect(bool(auth_required), "auth_required_on",
           "已开启（交付要求）", "**未开启**：接口会接受任意 user_id（交付阻塞项）",
           auth_required=auth_required)

    # ---------- 0.5) 页面面（交付时前端也得能打开；未登录访问 /ui 应 302 到 /login） ----------
    # ⚠️ 口径说明：urllib **默认跟随 302**，所以未登录时这里看到 200 很可能是"跟随到 /login 之后"
    #    —— 判据取三者之一即可，并在说明里如实写明，不把它读成"直接 200"。
    status_ui, _ = anon.call("GET", "/ui")
    status_login, login_body = anon.call("GET", "/login")
    record("pages_served",
           "PASS" if status_ui in (200, 302, 307) and status_login == 200 else "FAIL",
           f"/ui -> {status_ui}（未登录本应 302/307；urllib 跟随重定向，故 200 = 跟随到 /login 后），"
           f"/login -> {status_login}",
           login_chars=len(login_body) if isinstance(login_body, str) else None)

    # ---------- 2) 未带/伪造凭据 ----------
    status, body = Client(args.base_url).call("POST", "/chat", {
        "user_id": user_a, "role_id": "lawyer", "session_id": f"s-{suffix}",
        "message": "你好", "stream": False})
    expect(status == 401, "unauth_chat_401",
           "无凭据问问题 -> 401", f"无凭据竟然不是 401（{status}）", body=str(body)[:200])

    status, body = Client(args.base_url).call(
        "POST", "/chat", {"user_id": user_a, "role_id": "lawyer",
                          "session_id": f"s-{suffix}", "message": "你好", "stream": False},
        headers={"Authorization": "Bearer not-a-real-token"})
    expect(status == 401, "forged_token_401",
           "伪造令牌 -> 401", f"伪造令牌未被拒（{status}）", body=str(body)[:200])

    # ---------- 3) A 建会话；B 尝试越权 ----------
    session_a = f"{user_a}-lawyer-{suffix}"
    status, body = client_a.call("POST", "/sessions",
                                 {"user_id": user_a, "role_id": "lawyer",
                                  "session_id": session_a, "title": "A的会话"})
    expect(status == 200 and isinstance(body, dict) and body.get("user_id") == user_a,
           "a_create_session", "A 建会话成功且归属正确",
           f"建会话异常（{status}）：{body}", body=str(body)[:200])

    status, body = client_b.call("POST", "/sessions",
                                 {"user_id": user_b, "role_id": "lawyer",
                                  "session_id": session_a, "title": "B-覆盖尝试"})
    expect(status == 403, "g1_b_cannot_touch_a_session",
           "B 用 A 的 session_id 建会话 -> 403（G1 已修）",
           f"**G1 仍在**：B 未被拒（{status}）—— 跨用户写入", body=str(body)[:200])

    status, body = client_b.call("PATCH", f"/sessions/{session_a}",
                                 {"user_id": user_b, "title": "B-改名"})
    expect(status == 403, "b_patch_a_session_403",
           "B 改名 A 的会话 -> 403", f"未被拒（{status}）", body=str(body)[:200])

    status, body = client_b.call(
        "DELETE", f"/sessions/{session_a}?user_id={urllib.parse.quote(user_b)}")
    expect(status == 403, "b_delete_a_session_403",
           "B 删 A 的会话 -> 403", f"未被拒（{status}）", body=str(body)[:200])

    # ---------- 4) 读面 + 不泄露存在性 ----------
    status_b_real, body_b_real = client_b.call(
        "GET", f"/sessions/{session_a}/messages?user_id={urllib.parse.quote(user_b)}")
    ghost = f"{user_a}-lawyer-ghost-{suffix}"
    status_b_ghost, body_b_ghost = client_b.call(
        "GET", f"/sessions/{ghost}/messages?user_id={urllib.parse.quote(user_b)}")
    same = (status_b_real == status_b_ghost and json.dumps(body_b_real, sort_keys=True)
            == json.dumps(body_b_ghost, sort_keys=True))
    expect(status_b_real in (403, 404) and same, "no_existence_leak",
           f"读别人的={status_b_real} 与 读不存在的={status_b_ghost} **不可区分**",
           f"可区分或未拒绝：{status_b_real} vs {status_b_ghost}", body=str(body_b_real)[:200])

    status, body = client_b.call("GET", f"/sessions?user_id={urllib.parse.quote(user_b)}")
    listed = [row.get("session_id") for row in body] if isinstance(body, list) else []
    expect(session_a not in listed, "b_list_excludes_a",
           "B 的会话列表里没有 A 的会话", f"B 看到了 A 的会话：{listed[:5]}")

    status, body = client_b.call("POST", "/chat", {
        "user_id": user_a, "role_id": "lawyer", "session_id": session_a,
        "message": "你好", "stream": False})
    expect(status == 403, "b_chat_as_a_403",
           "B 带 A 的 user_id 提问 -> 403 user_mismatch",
           f"身份闸门失效（{status}）：{str(body)[:160]}")

    # ---------- 4.5) A 真问一轮（有真实数据后，再验存储侧与"有数据也不泄露"） ----------
    status_a_chat, body_a_chat = client_a.call("POST", "/chat", {
        "user_id": user_a, "role_id": "lawyer", "session_id": session_a,
        "message": "劳动争议申请仲裁的时效是多久", "stream": False})
    answer = str((body_a_chat or {}).get("answer") or "") if isinstance(body_a_chat, dict) else ""
    citations = (body_a_chat or {}).get("citations") if isinstance(body_a_chat, dict) else None
    expect(status_a_chat == 200 and bool(answer), "a_chat_ok",
           f"A 正常问答成功（答案 {len(answer)} 字，引用 {len(citations or [])} 条）",
           f"A 问答失败（{status_a_chat}）：{str(body_a_chat)[:200]}")

    status_b_real2, body_b_real2 = client_b.call(
        "GET", f"/sessions/{session_a}/messages?user_id={urllib.parse.quote(user_b)}")
    _, body_b_ghost2 = client_b.call(
        "GET", f"/sessions/{ghost}/messages?user_id={urllib.parse.quote(user_b)}")
    same2 = (json.dumps(body_b_real2, sort_keys=True, ensure_ascii=False)
             == json.dumps(body_b_ghost2, sort_keys=True, ensure_ascii=False))
    expect(status_b_real2 in (403, 404) and same2, "no_leak_after_a_chatted",
           "A 有真实历史后，B 读它仍 403 且与读不存在的不可区分",
           f"A 有历史后仍可区分/放行：{status_b_real2}", body=str(body_b_real2)[:200])

    # ---------- 5) 登录失败口径一致 ----------
    anon1, anon2 = Client(args.base_url), Client(args.base_url)
    s1, b1 = anon1.call("POST", "/auth/login", {"username": name_a, "password": "wrong-password-x"})
    s2, b2 = anon2.call("POST", "/auth/login",
                        {"username": f"no-such-user-{suffix}", "password": "wrong-password-x"})
    expect(s1 == s2 == 401 and json.dumps(b1, sort_keys=True, ensure_ascii=False)
           == json.dumps(b2, sort_keys=True, ensure_ascii=False),
           "login_failure_uniform",
           "密码错与用户不存在：状态码与文案逐字一致",
           f"口径不一致：{s1}/{b1} vs {s2}/{b2}")

    # ---------- 6) 存储面（只读） ----------
    db_path = Path(args.db)
    if db_path.is_file():
        try:
            connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            row = connection.execute("SELECT user_id, title FROM sessions WHERE session_id = ?",
                                     (session_a,)).fetchone()
            own_ok = bool(row) and row["user_id"] == user_a
            title_ok = bool(row) and row["title"] == "A的会话"
            cross = connection.execute(
                "SELECT COUNT(*) AS n FROM sessions WHERE session_id = ? AND user_id = ?",
                (session_a, user_b)).fetchone()["n"]
            connection.close()
            expect(own_ok and cross == 0, "sqlite_row_scoped",
                   f"会话行归属正确（title={row['title'] if row else None}）且带 user_id 查不到 B 的行",
                   f"SQLite 归属异常：own={own_ok} title_ok={title_ok} cross={cross}")
        except Exception as exc:  # noqa: BLE001
            record("sqlite_row_scoped", "SKIP", f"读不了库（{type(exc).__name__}: {exc}）")
    else:
        record("sqlite_row_scoped", "SKIP", f"库文件不存在：{db_path}")

    try:
        import redis  # noqa: PLC0415

        client = redis.Redis(decode_responses=True)
        keys = [key for key in client.scan_iter(match=f"{args.redis_prefix}*", count=500)][:2000]
        ours = [key for key in keys if suffix in key or session_a in key]
        unscoped = [key for key in ours if user_a not in key and user_b not in key]
        mixed = [key for key in keys if user_a in key and user_b in key]
        expect(bool(ours) and not unscoped and not mixed, "redis_keys_user_scoped",
               f"本次会话相关 key {len(ours)} 个**全部带用户维度**（扫描总数 {len(keys)}），无跨用户 key",
               f"key 维度异常：相关 {len(ours)} 个，其中无用户维度 {unscoped[:3]}，跨用户 {mixed[:3]}",
               scanned=len(keys), ours=ours[:5])
    except Exception as exc:  # noqa: BLE001
        record("redis_keys_user_scoped", "SKIP", f"Redis 不可用/未装（{type(exc).__name__}: {exc}）")

    # ---------- 7) 登出后失效 ----------
    status, _ = client_a.call("POST", "/auth/logout")
    after_status, after_body = client_a.call("GET", "/auth/me")
    expect(status in (200, 204) and after_status == 401, "logout_invalidates",
           "登出后 /auth/me -> 401", f"登出未失效：logout={status} me={after_status}",
           body=str(after_body)[:200])

    _flush(args.out)
    failures = [c for c in CHECKS if c["status"] == "FAIL"]
    return 1 if failures else 0


def _flush(out: str) -> None:
    passed = [c for c in CHECKS if c["status"] == "PASS"]
    skipped = [c for c in CHECKS if c["status"] == "SKIP"]
    failed = [c for c in CHECKS if c["status"] == "FAIL"]
    print(f"{'检查项':32s} 结果   说明")
    for item in CHECKS:
        # ⚠️ 控制台标记必须 **GBK 安全**（[PASS]/[FAIL]/[SKIP]）：Windows 控制台是 GBK，
        #    写 ✅/❌ 会让**整行丢失**并把退出码变成 1（见 tests/test_console_encoding.py）。
        #    JSON 产物里仍是 "PASS"/"FAIL"/"SKIP"（机器读的那个，不受影响）。
        mark = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}.get(item["status"], "?")
        print(f"{item['check']:32s} {mark}    {console_safe(item['detail'])[:110]}")
    print(f"\n合计：PASS {len(passed)} / FAIL {len(failed)} / SKIP {len(skipped)}")
    if out:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"checks": CHECKS, "passed": len(passed),
                                      "failed": len(failed), "skipped": len(skipped)},
                                     ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"证据已写：{target}")


if __name__ == "__main__":
    raise SystemExit(main())
