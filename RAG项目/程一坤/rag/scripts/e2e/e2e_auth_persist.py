"""批次5 认证落库 e2e 验收（对应验收项 a~e）。

流程：
  a) 起后端 → 注册新账号 → 从 MySQL 记下 user_key
  b) 真实杀掉后端进程并重启 → 同一账号登录成功（核心验收）
  c) 用该账号发一轮真实问答（SSE）→ 会话列表/历史能读到
  d) 注册另一账号访问该会话 → 404
  e) 贴出库里 password_hash 前缀（非明文证据）
运行方式：单脚本 subprocess.Popen 全程管理后端进程（起 → 验收 → 停）。
"""
import os
import subprocess
import sys
import time
from pathlib import Path

# 项目根（scripts/e2e → scripts → rag），供下面导入 scripts/_env 使用
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pymysql
import requests

from scripts._env import load_project_env, mysql_config

# 读项目根 .env（顺带剔除 http(s)_proxy）：数据库口令等机密只存在于 .env
load_project_env(PROJECT_ROOT)

SESSION = requests.Session()
SESSION.trust_env = False  # 绕过代理环境变量
BASE = "http://127.0.0.1:8123"
BACKEND_DIR = r"C:\Users\92842\Desktop\rag\backend"
PY = r"C:\Users\92842\anaconda3\python.exe"
import uuid
_RUN = uuid.uuid4().hex[:8]
EMAIL_A = f"e2e_auth_persist_a_{_RUN}@qq.com"
EMAIL_B = f"e2e_auth_persist_b_{_RUN}@qq.com"
PASSWORD = "E2ePass2026"
SESSION_KEY = f"session_e2e_auth_persist_{_RUN}"

# 数据库连接参数从 .env 读（MYSQL_HOST/PORT/USER/PASSWORD/DATABASE），源码零口令
DB = mysql_config()


def backend_env() -> dict:
    env = os.environ.copy()
    env.pop("http_proxy", None)
    env.pop("https_proxy", None)
    # 集成测试专用环境：ENVIRONMENT=test 才会启用固定验证码旁路
    # （仅用于集成测试，生产禁止启用；见 app/auth/service.py 旁路说明）
    env["ENVIRONMENT"] = "test"
    return env


def start_backend() -> subprocess.Popen:
    # 输出重定向到日志文件：PIPE 无人消费会被日志写满，导致后端进程整体卡死
    log_handle = open(r"C://Users//92842//Desktop//rag//e2e_backend.log", "ab", buffering=0)
    proc = subprocess.Popen(
        [PY, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8123"],
        cwd=BACKEND_DIR,
        env=backend_env(),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            r = SESSION.get(f"{BASE}/health/ready", timeout=3)
            if r.status_code == 200:
                return proc
        except requests.RequestException:
            pass
        time.sleep(1)
    proc.kill()
    raise RuntimeError("后端 60 秒内未就绪，见 e2e_backend.log")


def stop_backend(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    # 等端口释放
    time.sleep(2)
    print("[OK] 后端进程已停止（真实重启前）")


def db_user_row(email: str) -> dict:
    conn = pymysql.connect(**DB)
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(
        "SELECT user_key, email, LEFT(password_hash, 12) AS hash_prefix, "
        "CHAR_LENGTH(password_hash) AS hash_len, is_admin, is_active "
        "FROM users WHERE email=%s",
        (email,),
    )
    row = cur.fetchone()
    conn.close()
    return row


def register(email: str) -> str:
    r = SESSION.post(
        f"{BASE}/api/v1/auth/register/code",
        json={"email": email, "purpose": "register"},
        timeout=30,
    )
    body = r.json()
    # 集成测试旁路（ENVIRONMENT=test）：固定码放行，不再依赖验证码回显
    # （register/code 端点已按生产口径不回显；固定码仅在 test 环境被接受）
    code = "000000"
    r = SESSION.post(
        f"{BASE}/api/v1/auth/register",
        json={"email": email, "password": PASSWORD, "code": code},
        timeout=30,
    )
    assert r.status_code == 200, f"注册失败: {r.status_code} {r.text}"
    return r.json()["data"]["access_token"]


def login(email: str, password: str):
    return SESSION.post(
        f"{BASE}/api/v1/auth/login",
        json={"email": email, "password": password},
        timeout=30,
    )


def run_qa(token: str) -> None:
    """发一轮真实问答（SSE 流），等到 message_end。"""
    payload = {
        "session_id": SESSION_KEY,
        "message": "试用期最长可以约定多久？",
    }
    r = SESSION.post(
        f"{BASE}/api/v1/chat/stream",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        stream=True,
        timeout=180,
    )
    assert r.status_code == 200, f"问答流失败: {r.status_code} {r.text[:300]}"
    events = []
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("event:"):
            events.append(line.split(":", 1)[1].strip())
        if events and events[-1] == "message_end":
            break
    r.close()
    print(f"[OK] c) 问答流完成，事件序列: {events[0]}...{events[-1]} 共 {len(events)} 个事件")
    assert "message_start" in events and "message_end" in events
    assert events.count("token") >= 1, "没有收到增量 token，不是真流式"


def main() -> None:
    results = {}

    # ============ 第一次启动：注册 ============
    proc = start_backend()
    print("[OK] 后端第一次启动（端口 8123）")

    # a) 注册并记录 user_key
    token_a = register(EMAIL_A)
    row_a = db_user_row(EMAIL_A)
    assert row_a, "注册后 users 表查不到该用户"
    results["user_key"] = row_a["user_key"]
    print(f"[OK] a) 注册成功 user_key={row_a['user_key']} email={row_a['email']}")

    # e) 密码哈希形态（只看前缀）
    print(f"[OK] e) password_hash 前缀={row_a['hash_prefix']}… 长度={row_a['hash_len']}")
    assert row_a["hash_prefix"] != PASSWORD

    stop_backend(proc)

    # ============ 第二次启动（真实重启）：登录 ============
    proc = start_backend()
    print("[OK] 后端已真实重启（新进程）")

    r = login(EMAIL_A, PASSWORD)
    assert r.status_code == 200, f"重启后登录失败: {r.status_code} {r.text}"
    token_a2 = r.json()["data"]["access_token"]
    print(f"[OK] b) 重启后登录成功（核心验收通过），新令牌前 8 位={token_a2[:8]}…")

    # 错误密码仍被拒
    r = login(EMAIL_A, "WrongPass999")
    assert r.status_code == 401, f"错误密码未被拒绝: {r.status_code}"
    print("[OK] b2) 重启后错误密码被拒（401）")

    # c) 一轮真实问答 + 会话列表/历史
    run_qa(token_a2)
    r = SESSION.get(
        f"{BASE}/api/v1/sessions",
        headers={"Authorization": f"Bearer {token_a2}"},
        timeout=30,
    )
    items = r.json()["data"]["items"]
    assert any(s["session_id"] == SESSION_KEY for s in items), "会话列表缺本次会话"
    print(f"[OK] c) 会话列表可读到本次会话（共 {len(items)} 条）")
    r = SESSION.get(
        f"{BASE}/api/v1/sessions/{SESSION_KEY}/messages",
        headers={"Authorization": f"Bearer {token_a2}"},
        timeout=30,
    )
    msgs = r.json()["data"]["items"]
    assert len(msgs) >= 2, f"历史消息不足: {len(msgs)}"
    assert msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant"
    print(f"[OK] c) 历史可读：{len(msgs)} 条消息（user+assistant 成对）")

    # 落库归属核对：chat_sessions.user_id == user_key
    conn = pymysql.connect(**DB)
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM chat_sessions WHERE session_key=%s", (SESSION_KEY,))
    owner = cur.fetchone()[0]
    conn.close()
    assert owner == results["user_key"], f"归属不符: {owner} != {results['user_key']}"
    print(f"[OK] c) 落库归属核对：chat_sessions.user_id == user_key == {owner}")

    # d) 另一账号访问该会话 → 404
    register(EMAIL_B)
    token_b = login(EMAIL_B, PASSWORD).json()["data"]["access_token"]
    r = SESSION.get(
        f"{BASE}/api/v1/sessions/{SESSION_KEY}/messages",
        headers={"Authorization": f"Bearer {token_b}"},
        timeout=30,
    )
    assert r.status_code == 404, f"跨用户未返回 404: {r.status_code}"
    print("[OK] d) 跨用户访问该会话 → 404（隔离成立）")

    stop_backend(proc)
    print("\nE2E_ALL_PASS")
    print("user_key =", results["user_key"])


if __name__ == "__main__":
    main()
