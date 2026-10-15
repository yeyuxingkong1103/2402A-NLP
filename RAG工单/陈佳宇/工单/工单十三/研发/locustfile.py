# -*- coding: utf-8 -*-
# locustfile.py 工单13 locust负载测试脚本
from locust import HttpUser, task
from rag_perf_main import rag_full_pipeline

class RAGUser(HttpUser):
    host = "http://127.0.0.1"
    test_queries = [
        "武汉力源本次募集资金投资哪些项目？",
        "销售部有几个部门？",
        "关联方持股比例是多少？"
    ]

    @task
    def rag_task(self):
        for q in self.test_queries:
            rag_full_pipeline(q)
