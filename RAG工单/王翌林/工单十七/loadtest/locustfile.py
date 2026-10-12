# -*- coding: utf-8 -*-
"""Locust 压测脚本：场景A 高频问答 / 场景B 混合负载（问答+文档解析）。
运行：
  locust -f loadtest/locustfile.py --host http://127.0.0.1:8017
  # 场景A：20 用户，孵化 2/s，运行 10 分钟
  # 场景B：10 用户（5 问答 + 5 上传解析）
"""
import random
from locust import HttpUser, task, between, tag

QUESTIONS = [
    "什么是淬火工艺？", "公差与配合的作用？", "链条传动适合什么场合？",
    "带式输送机由哪些部件组成？", "轧制带材时为什么调节张力？", "水洗洗涤塔的作用？",
]


class RAGUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @tag("chat")
    @task(3)
    def chat(self):
        q = random.choice(QUESTIONS)
        with self.client.post("/api/v1/chats_openai/c1/chat/completions",
                              json={"question": q, "chat_id": "c1"},
                              name="/chat/completions", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"status {r.status_code}")

    @tag("search")
    @task(1)
    def health(self):
        self.client.get("/health", name="/health")


class ParseUser(HttpUser):
    """场景B：文档解析（异步入队）。"""
    wait_time = between(0.5, 1.5)

    @tag("parse")
    @task
    def parse(self):
        with self.client.post("/api/v1/documents/parse?filename=demo.pdf",
                              name="/documents/parse", catch_response=True) as r:
            if r.status_code not in (200, 503):
                r.failure(f"status {r.status_code}")
