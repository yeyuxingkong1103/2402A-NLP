"""批次7-1 create_admin CLI e2e 验收。

验收项：
a) 对未注册邮箱执行 → 报错提示（退出码 1）
b) 对已注册邮箱执行 → 变成管理员（is_admin=1）
c) 用该账号调 GET /api/v1/admin/documents?status=pending_review → 200
d) 重复执行 → 幂等提示（退出码 0）
"""

import json
import os
import subprocess
import time
import uuid

import requests

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

BASE = "http://127.0.0.1:8123"
RAG = r"C:\Users\92842\Desktop\rag"
PY = r"C:\Users\92842\anaconda3\python.exe"
_RUN = uuid.uuid4().hex[:8]
EMAIL = f"e2e_admin_cli_{_RUN}@qq.com"
GHOST = f"e2e_ghost_cli_{_RUN}@qq.com"
PASSWORD = "E2ePass2026"
LOG = os.path.join(RAG, "e2e_cli_backend.log")


def run_cli(email: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PY, "-m", "app.cli.create_admin", "--email", email],
        cwd=os.path.join(RAG, "backend"),
        capture_output=True, text=True, timeout=60,
    )


def main() -> None:
    log_handle = open(LOG, "ab", buffering=0)
    # 集成测试专用环境：ENVIRONMENT=test 才会启用固定验证码旁路
    # （仅用于集成测试，生产禁止启用；见 app/auth/service.py 旁路说明）。
    # 只注入给后端子进程，不改动本进程环境（CLI 子进程仍按默认环境加载配置）。
    backend_env = {**os.environ, "ENVIRONMENT": "test"}
    proc = subprocess.Popen(
        [PY, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8123"],
        cwd=os.path.join(RAG, "backend"),
        env=backend_env,
        stdout=log_handle, stderr=subprocess.STDOUT,
    )
    try:
        s = requests.Session(); s.trust_env = False
        for _ in range(60):
            try:
                if s.get(f"{BASE}/health/ready", timeout=3).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("后端 60 秒内未就绪，见 e2e_cli_backend.log")

        # 准备：注册一个新账号（ENVIRONMENT=test 下用固定验证码旁路）
        r = s.post(f"{BASE}/api/v1/auth/register/code",
                   json={"email": EMAIL, "purpose": "register"}, timeout=30)
        code = "000000"
        r = s.post(f"{BASE}/api/v1/auth/register",
                   json={"email": EMAIL, "password": PASSWORD, "code": code}, timeout=30)
        assert r.status_code == 200, r.text
        print(f"[OK] 准备：注册测试账号 {EMAIL}")

        # a) 未注册邮箱 → 报错
        res = run_cli(GHOST)
        print(f"[OK] a) 未注册邮箱：退出码={res.returncode}，stderr={res.stderr.strip()}")
        assert res.returncode == 1 and "尚未注册" in res.stderr

        # b) 已注册邮箱 → 升级
        res = run_cli(EMAIL)
        print(f"[OK] b) 已注册邮箱：退出码={res.returncode}，stdout={res.stdout.strip()}")
        assert res.returncode == 0 and "is_admin=1" in res.stdout

        # c) 该账号调管理员接口 → 200
        r = s.post(f"{BASE}/api/v1/auth/login",
                   json={"email": EMAIL, "password": PASSWORD}, timeout=30)
        token = r.json()["data"]["access_token"]
        r = s.get(f"{BASE}/api/v1/admin/documents?status=pending_review&page=1&page_size=10",
                  headers={"Authorization": f"Bearer {token}"}, timeout=30)
        print(f"[OK] c) 管理员接口状态码={r.status_code}，响应={r.text[:180]}")
        assert r.status_code == 200
        body = r.json()
        assert "data" in body

        # d) 重复执行 → 幂等
        res = run_cli(EMAIL)
        print(f"[OK] d) 重复执行：退出码={res.returncode}，stdout={res.stdout.strip()}")
        assert res.returncode == 0 and "已是管理员" in res.stdout

        print("E2E_CREATE_ADMIN_ALL_PASS")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
