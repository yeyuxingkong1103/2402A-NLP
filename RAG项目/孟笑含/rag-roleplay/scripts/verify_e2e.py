# -*- coding: utf-8 -*-
"""端到端验证脚本：注册 → 角色 → 对话（非流式/流式）→ 历史。"""
import json
# 解析：JSON 打印

import httpx
# 解析：HTTP 客户端

BASE = "http://127.0.0.1:8000"
# 解析：服务地址


# 打印验证区块（标题+JSON 摘要）
def show(title, obj):
    # 解析：打印验证区块
    print(f"\n===== {title} =====")
    # 解析：区块标题
    print(json.dumps(obj, ensure_ascii=False, indent=2)[:1200])
    # 解析：JSON 格式化输出（截断 1200 字符防刷屏）


with httpx.Client(base_url=BASE, timeout=120) as c:
    # 解析：HTTP 客户端（120 秒超时——真实 LLM 可能慢）
    # 1. 注册
    r = c.post("/api/users/register", json={"username": "tester", "password": "test123456"})
    # 解析：注册（重复运行会 400，后面用登录兜底）
    if r.status_code == 400:
        # 解析：已注册过
        r = c.post("/api/users/login", json={"username": "tester", "password": "test123456"})
        # 解析：改登录
    r.raise_for_status()
    # 解析：非 200 抛异常
    token = r.json()["token"]
    # 解析：取 token
    headers = {"X-Token": token}
    # 解析：鉴权头
    show("注册/登录", {"status": r.status_code, "username": r.json()["username"],
                       "token": token[:12] + "..."})
    # 解析：打印摘要（token 只显示前 12 位）

    # 2. 角色列表
    r = c.get("/api/roles", headers=headers)
    # 解析：查角色
    r.raise_for_status()
    # 解析：校验
    roles = r.json()
    # 解析：角色列表
    show("角色列表", [{"id": x["id"], "name": x["name"], "category": x["category"]} for x in roles])
    # 解析：打印精简列表
    yang = next(x for x in roles if x["name"] == "小阳")
    # 解析：找小阳
    doctor = next(x for x in roles if x["name"] == "林医生")
    # 解析：找林医生

    # 3. 非流式对话（虚拟朋友小阳）
    r = c.post("/api/chat", headers=headers, json={"role_id": yang["id"], "content": "你好小阳！今天心情不太好"})
    # 解析：与小阳对话
    r.raise_for_status()
    # 解析：校验
    show("非流式对话·小阳", r.json())
    # 解析：打印

    # 4. 非流式对话（医生林医生，验证角色隔离）
    r = c.post("/api/chat", headers=headers, json={"role_id": doctor["id"], "content": "医生你好，我最近血压有点高"})
    # 解析：与林医生对话（同一用户不同角色）
    r.raise_for_status()
    # 解析：校验
    show("非流式对话·林医生", r.json())
    # 解析：打印

    # 5. 流式对话（验证多轮记忆：小阳应记得之前聊过心情不好）
    print("\n===== 流式对话·小阳（验证短期记忆） =====")
    # 解析：标题
    full = []
    # 解析：收集流式分块
    with c.stream(
        # 解析：流式请求
        "POST", "/api/chat/stream",
        # 解析：SSE 接口
        headers=headers,
        # 解析：鉴权
        json={"role_id": yang["id"], "content": "还记得我刚才说我心情不好吗？"},
        # 解析：问记忆问题
    ) as resp:
        resp.raise_for_status()
        # 解析：校验
        print("content-type:", resp.headers["content-type"])
        # 解析：打印媒体类型
        for line in resp.iter_lines():
            # 解析：逐行读 SSE
            if line.startswith("data:") and "[DONE]" not in line:
                # 解析：数据事件且非结束
                delta = json.loads(line[5:])["delta"]
                # 解析：解析 delta
                full.append(delta)
                # 解析：收集
                print(delta, end="", flush=True)
                # 解析：逐字打印（打字机效果）
    print("\n（流式完整回复）:", "".join(full))
    # 解析：完整回复

    # 6. 历史记录
    r = c.get(f"/api/chat/history?role_id={yang['id']}", headers=headers)
    # 解析：查小阳历史
    r.raise_for_status()
    # 解析：校验
    show(
        # 解析：打印
        "历史记录·小阳",
        # 解析：标题
        [{"sender": m["sender"], "content": m["content"][:40]} for m in r.json()],
        # 解析：精简字段
    )
    r = c.get(f"/api/chat/history?role_id={doctor['id']}", headers=headers)
    # 解析：查林医生历史
    show(
        # 解析：打印
        "历史记录·林医生（应与小阳隔离）",
        # 解析：标题
        [{"sender": m["sender"], "content": m["content"][:40]} for m in r.json()],
        # 解析：精简字段
    )

print("\n全部验证通过 ✔")
# 解析：结论
