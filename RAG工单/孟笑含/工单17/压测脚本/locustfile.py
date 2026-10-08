# -*- coding: utf-8 -*-
from locust import HttpUser, task, between, events
import random
import time

QUESTIONS = [
    "注册资本是多少？",
    "公司成立于哪一年？",
    "法定代表人是谁？",
    "武汉兴图新科的注册资本",
    "公司的主营业务是什么？",
]


class RAGUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(3)
    def chat(self):
        q = random.choice(QUESTIONS)
        with self.client.post(
            "/api/chat",
            json={"query": q, "chat_id": "test_chat", "stream": False},
            catch_response=True,
            name="/api/chat",
        ) as resp:
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if not data.get("answer"):
                        resp.failure("empty answer")
                except Exception as e:
                    resp.failure(f"parse error: {e}")
            else:
                resp.failure(f"status {resp.status_code}")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print(f"\n=== 压测开始 @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print(f"\n=== 压测结束 @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
