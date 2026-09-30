"""JMeter 替代：locust 压测脚本（loadtest/）。

用环境变量 LOCUST_TARGET 选择加载哪个 User（abstract 隔离，避免另一 User 被实例化报错）：
  # /health 健康检查（无 LLM）
  LOCUST_TARGET=health .venv/bin/locust -f loadtest/locustfile.py --host=http://127.0.0.1:8000 \
    --headless -u 50 -r 10 -t 30s --only-summary --csv=loadtest/results_health_50

  # /chat 轻量压测（调 DeepSeek）
  LOCUST_TARGET=chat .venv/bin/locust -f loadtest/locustfile.py --host=http://127.0.0.1:8000 \
    --headless -u 3 -r 1 -t 120s --only-summary --csv=loadtest/results_chat_3

详见 docs/压力测试报告.md。
"""
import os

from locust import HttpUser, between, task

# /chat 固定请求体（role 依赖 MySQL roles 表已初始化「高血压医生」）
CHAT_BODY = {
    "message": "血压多少算高",
    "role": "高血压医生",
    "session_id": "loadtest",
}

# abstract=True 的 User 不会被 locust 实例化；用环境变量选目标场景
TARGET = os.getenv("LOCUST_TARGET", "health")


class HealthUser(HttpUser):
    """GET /health，纯健康检查，不调 LLM。错误率由 locust 默认记录（>=400 记失败）。"""

    abstract = TARGET != "health"
    wait_time = between(0.1, 0.5)

    @task
    def health(self):
        self.client.get("/health")


class ChatUser(HttpUser):
    """POST /chat，真实调用 DeepSeek。固定 session_id 共享会话，多轮会触发 query 改写。"""

    abstract = TARGET != "chat"
    wait_time = between(5, 10)

    @task
    def chat(self):
        with self.client.post("/chat", json=CHAT_BODY, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"期望 200，实际 {resp.status_code}")
