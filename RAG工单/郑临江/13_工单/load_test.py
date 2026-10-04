# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
负载测试：模拟生产流量，压力下暴露瓶颈。
提供 Locust（Python）脚本与 k6（JS）脚本。
"""
import config

# 测试查询复用
QUERIES = config.TEST_QUERIES


def locust_script():
    """返回 Locust 负载测试脚本内容（保存为 locustfile.py 运行）。"""
    return '''# -*- coding: utf-8 -*-
# 运行：locust -f locustfile.py --host=http://localhost:5000
from locust import HttpUser, task, between


class RAGUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def query(self):
        self.client.post("/query", json={"query": "武汉兴图新科电子股份有限公司法定代表人是谁？"})
'''


def k6_script():
    """返回 k6 负载测试脚本内容（保存为 load.js 运行）。"""
    return '''// 运行：k6 run load.js
import http from 'k6/http';
import { check } from 'k6';

export const options = {
  stages: [
    { duration: '30s', target: 20 },  // 爬升到 20 并发
    { duration: '1m', target: 20 },   // 保持
    { duration: '10s', target: 0 },   // 回落
  ],
};

export default function () {
  const res = http.post('http://localhost:5000/query',
    JSON.stringify({ query: '武汉力源信息技术股份有限公司本次发行股数是多少？' }),
    { headers: { 'Content-Type': 'application/json' } });
  check(res, { 'status 200': (r) => r.status === 200 });
}
'''


if __name__ == "__main__":
    with open("locustfile.py", "w", encoding="utf-8") as f:
        f.write(locust_script())
    with open("load.js", "w", encoding="utf-8") as f:
        f.write(k6_script())
    print("已生成 locustfile.py（Locust）与 load.js（k6）负载测试脚本")
