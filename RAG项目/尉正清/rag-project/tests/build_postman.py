# -*- coding: utf-8 -*-
"""从运行中的服务生成 Postman 集合，供导入 Postman 或用 newman 命令行执行。

    python run.py -d                                   # 先启动服务
    .venv/bin/python tests/build_postman.py            # 生成 jmeter/../postman_collection.json
    newman run tests/postman_collection.json           # 命令行执行
    newman run tests/postman_collection.json -r htmlextra --reporter-htmlextra-export report.html

集合覆盖全部接口，每个请求都带断言：HTTP 状态码、响应结构、业务字段。
"""
import argparse
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

OUT_DEFAULT = os.path.join(BASE, "tests", "postman_collection.json")

# 每个接口的请求定义：路径、方法、示例请求体、断言
ITEMS = [
    ("系统", [
        ("服务概览", "GET", "/", None,
         ["pm.response.to.have.status(200)",
          "pm.expect(pm.response.json()).to.have.property('name')"]),
        ("健康检查", "GET", "/api/health", None,
         ["pm.response.to.have.status(200)",
          "const d = pm.response.json().data;",
          "pm.expect(d.mysql, 'MySQL 不可用').to.be.true;",
          "pm.expect(d.redis, 'Redis 不可用').to.be.true;",
          "pm.expect(d.milvus, 'Milvus 不可用').to.be.true;",
          "pm.expect(d.llm, '大模型未配置').to.be.true;"]),
        ("业务统计", "GET", "/api/stats?refresh=true", None,
         ["pm.response.to.have.status(200)",
          "const d = pm.response.json().data;",
          "pm.expect(d.collections.rag_knowledge).to.be.above(0);",
          "pm.expect(Object.keys(d.knowledge_per_role).length).to.eql(3);"]),
    ]),
    ("角色", [
        ("角色列表", "GET", "/api/roles", None,
         ["pm.response.to.have.status(200)",
          "const roles = pm.response.json().data;",
          "pm.expect(roles.length).to.be.above(0);",
          "pm.expect(roles.map(r => r.role_key)).to.include('lawyer');"]),
        ("角色详情", "GET", "/api/roles/lawyer", None,
         ["pm.response.to.have.status(200)",
          "const d = pm.response.json().data;",
          "pm.expect(d.persona, '缺少人设').to.be.a('string').and.not.empty;",
          "pm.expect(d.disclaimer, '缺少免责声明').to.be.a('string').and.not.empty;"]),
        ("角色不存在应 404", "GET", "/api/roles/not_exist_role", None,
         ["pm.response.to.have.status(404)"]),
        ("角色热度", "GET", "/api/roles/hot?k=5", None,
         ["pm.response.to.have.status(200)",
          "pm.expect(pm.response.json().data).to.be.an('array');"]),
    ]),
    ("问答", [
        ("角色问答", "POST", "/api/chat/ask",
         {"question": "酒驾撞人要判多久？", "role_key": "lawyer", "user_id": 1},
         ["pm.response.to.have.status(200)",
          "const d = pm.response.json().data;",
          "pm.expect(d.answer, '回答为空').to.be.a('string').and.not.empty;",
          "pm.expect(d.session_id).to.be.a('string').and.not.empty;",
          "pm.expect(d.sources, '没有检索到资料').to.be.an('array').and.not.empty;",
          "// 免责声明应自动追加",
          "pm.expect(d.answer).to.include('不构成');",
          "pm.collectionVariables.set('sessionId', d.session_id);"]),
        ("多轮追问", "POST", "/api/chat/ask",
         {"question": "那要是逃逸了呢？", "role_key": "lawyer", "user_id": 1,
          "session_id": "{{sessionId}}"},
         ["pm.response.to.have.status(200)",
          "const d = pm.response.json().data;",
          "pm.expect(d.session_id).to.eql(pm.collectionVariables.get('sessionId'));",
          "// 追问应被改写，检索用的问题不再是原句",
          "pm.expect(d.search_query).to.not.eql('那要是逃逸了呢？');"]),
        ("空问题应 422", "POST", "/api/chat/ask",
         {"question": "", "role_key": "lawyer", "user_id": 1},
         ["pm.response.to.have.status(422)"]),
        ("未知角色应 400", "POST", "/api/chat/ask",
         {"question": "测试", "role_key": "no_such_role", "user_id": 1},
         ["pm.response.to.have.status(400)"]),
    ]),
    ("会话", [
        ("会话列表", "GET", "/api/sessions?user_id=1&limit=5", None,
         ["pm.response.to.have.status(200)",
          "pm.expect(pm.response.json().data).to.be.an('array');"]),
        ("会话历史", "GET", "/api/sessions/{{sessionId}}/messages", None,
         ["pm.response.to.have.status(200)",
          "const d = pm.response.json().data;",
          "pm.expect(d.messages.length).to.be.above(0);"]),
        ("会话记忆", "GET", "/api/sessions/{{sessionId}}/memory?user_id=1&role_key=lawyer",
         None,
         ["pm.response.to.have.status(200)",
          "const d = pm.response.json().data;",
          "pm.expect(d).to.have.property('short_term');"]),
        ("会话不存在应 404", "GET", "/api/sessions/no_such_session/messages", None,
         ["pm.response.to.have.status(404)"]),
    ]),
    ("用户", [
        ("登录", "POST", "/api/users/login",
         {"username": "demo", "password": "demo123"},
         ["pm.response.to.have.status(200)",
          "pm.expect(pm.response.json().data.username).to.eql('demo');"]),
        ("密码错误应 401", "POST", "/api/users/login",
         {"username": "demo", "password": "wrong_password"},
         ["pm.response.to.have.status(401)"]),
    ]),
    ("知识入库与更新", [
        ("未知角色入库应 400", "POST", "/api/ingest/dataset",
         {"role_key": "no_such_role"}, ["pm.response.to.have.status(400)"]),
        ("更新不存在的文档应 400", "POST", "/api/update/document",
         {"file_path": "/tmp/not_here.jsonl", "role_key": "lawyer"},
         ["pm.response.to.have.status(400)"]),
        ("删除不存在的来源是幂等的", "DELETE", "/api/update/document",
         {"source": "not_a_real_file.jsonl", "role_key": "lawyer"},
         ["pm.response.to.have.status(200)"]),
    ]),
]


def build():
    items = []
    for group, cases in ITEMS:
        children = []
        for name, method, path, body, tests in cases:
            url_path = path.split("?")[0].lstrip("/")
            query = []
            if "?" in path:
                for kv in path.split("?", 1)[1].split("&"):
                    k, _, v = kv.partition("=")
                    query.append({"key": k, "value": v})

            request = {
                "method": method,
                "header": [{"key": "Content-Type", "value": "application/json"}],
                "url": {
                    "raw": "{{baseUrl}}/" + path,
                    "host": ["{{baseUrl}}"],
                    "path": url_path.split("/"),
                },
            }
            if query:
                request["url"]["query"] = query
            if body is not None:
                request["body"] = {
                    "mode": "raw",
                    "raw": json.dumps(body, ensure_ascii=False),
                    "options": {"raw": {"language": "json"}},
                }

            script = ["pm.test('%s', function () {" % name] + \
                     ["    " + t for t in tests] + ["});"]
            children.append({
                "name": name,
                "request": request,
                "event": [{"listen": "test",
                           "script": {"type": "text/javascript", "exec": script}}],
            })
        items.append({"name": group, "item": children})

    return {
        "info": {
            "name": "RAG 角色扮演系统",
            "description": "覆盖全部接口的回归集合，含状态码、结构、业务字段断言。",
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
        },
        "variable": [
            {"key": "baseUrl", "value": "http://127.0.0.1:8000"},
            {"key": "sessionId", "value": ""},
        ],
        "item": items,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    collection = build()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(collection, f, ensure_ascii=False, indent=2)

    total = sum(len(g["item"]) for g in collection["item"])
    print("已生成 Postman 集合: %s" % args.out)
    print("  分组 %d 个，请求 %d 个" % (len(collection["item"]), total))
    print()
    print("执行方式：")
    print("  newman run %s" % os.path.relpath(args.out))
    print("  newman run %s -r cli,json --reporter-json-export tests/postman_report.json"
          % os.path.relpath(args.out))


if __name__ == "__main__":
    main()
