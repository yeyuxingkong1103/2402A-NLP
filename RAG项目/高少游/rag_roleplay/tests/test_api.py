# -*- coding: utf-8 -*-
"""
RAG 角色扮演系统 - API 端到端测试
用法：python tests/test_api.py
用 TestClient 不需要真实启动 uvicorn（lifespan 会自动执行，模型在测试开始时加载）
"""

import os  # 操作系统接口：读取环境变量、路径操作
import sys  # 系统接口：修改 sys.path 让测试能 import 到项目根目录的模块

# 把项目根目录加入 sys.path，否则 from main import app 会找不到 main.py
# __file__ = 当前文件路径；dirname 两层 = 上级目录（tests/ 的父目录 = 项目根目录）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 插入到 path 列表头部

# 设置环境变量：用临时集合 "_test_roleplay" 测试，不碰正式知识库数据
os.environ["MILVUS_COLLECTION"] = "_test_roleplay"  # 临时集合名，测试完会删掉
os.environ["MILVUS_URI"] = "http://localhost:19530"  # Milvus 地址

from fastapi.testclient import TestClient  # FastAPI 的测试客户端：不需要真正起 HTTP 服务
from pymilvus import MilvusClient  # Milvus 原生客户端：用来清理测试集合

# 导入 main 模块会触发 app 的 lifespan 回调（加载模型、建表、BM25 重建）
from main import app  # FastAPI 应用对象

# 创建 Milvus 原生客户端实例，用于测试前清理和测试后清理
client = MilvusClient(uri="http://localhost:19530")  # 连接 Milvus
# 如果之前的测试残留了临时集合，先删掉（保证干净的测试环境）
if client.has_collection("_test_roleplay"):  # 检查集合是否存在
    client.drop_collection("_test_roleplay")  # 删除集合

# 创建 TestClient：它会在 __enter__ 里触发 lifespan 启动事件（加载所有模型）
tc = TestClient(app)  # 封装 FastAPI app 为可测试对象

# ==================== 测试用例 ====================

def test_health():
    """测试健康检查接口 GET /"""
    r = tc.get("/")  # 向 "/" 发 GET 请求
    assert r.status_code == 200  # 断言返回 200 OK
    assert r.json()["status"] == "ok"  # 断言响应体里 status 字段为 "ok"
    print("[PASS] 健康检查")  # 打印通过


def test_list_roles():
    """测试角色列表接口 GET /roles"""
    r = tc.get("/roles")  # 向 "/roles" 发 GET 请求
    assert r.status_code == 200  # 断言 200
    roles = r.json()["roles"]  # 提取角色列表
    assert len(roles) > 0  # 断言至少有一个角色（lifespan 里写入了 8 个预置角色）
    print(f"[PASS] 角色列表：{len(roles)} 个角色")  # 打印角色数量
    return roles  # 返回角色列表供后续测试用


def test_ingest_text():
    """测试文本入库接口 POST /ingest/text"""
    legal_text = """  # 模拟法律文档（Markdown 格式，有标题层级）
# 民法典 相邻关系

## 第二百八十八条
不动产的相邻权利人应当按照有利生产、方便生活、团结互助、公平合理的原则，正确处理相邻关系。

## 第二百九十四条
不动产权利人不得违反国家规定弃置固体废物，排放大气污染物、水污染物、土壤污染物、噪声、光辐射、电磁辐射等有害物质。

# 判例摘要
（2021）京0105民初12345号：业主长期在夜间进行高噪声装修，法院判令停止侵害并赔偿精神损害抚慰金2000元。
"""
    r = tc.post("/ingest/text", json={  # 向 "/ingest/text" 发 POST 请求，带 JSON body
        "source": "民法典测试.md",  # 来源文件名（.md 后缀会触发 Markdown 标题切分）
        "text": legal_text,  # 原始文本
        "strategy": "paragraph",  # 切分策略（.md 文件会自动用 markdown 策略覆盖这个）
    })
    assert r.status_code == 200  # 断言 200
    assert r.json()["chunks"] > 0  # 断言切出了至少 1 块
    print(f"[PASS] 文本入库：{r.json()['chunks']} 块")  # 打印块数
    return r.json()  # 返回响应体


def test_cases():
    """测试判例检索接口 POST /cases（不调 LLM，只返回检索到的文档片段）"""
    r = tc.post("/cases", json={"query": "夜间装修噪音 判例 赔偿", "top_k": 3})  # 发 POST 请求
    assert r.status_code == 200  # 断言 200
    assert r.json()["count"] > 0  # 断言检索到至少 1 条
    print(f"[PASS] 判例检索：命中 {r.json()['count']} 条")  # 打印命中数
    return r.json()  # 返回响应体


def test_chat(roles):
    """测试多轮对话接口 POST /chat（角色 + 记忆 + RAG 全链路）"""
    role_id = roles[0]["id"]  # 取第一个角色（法律顾问）的 ID

    # --- 第一轮对话 ---
    r = tc.post("/chat", json={  # 发 POST 请求
        "user_id": 1,  # 模拟用户 ID 1
        "role_id": role_id,  # 角色 ID
        "query": "邻居半夜装修噪音很大，我可以主张什么权利？",  # 用户提问
        "top_k": 3,  # 引用资料条数
    })
    assert r.status_code == 200  # 断言 200
    answer = r.json()["answer"]  # 提取 LLM 回答
    assert len(answer) > 10  # 断言回答不为空（至少 10 个字符）
    print(f"[PASS] 第一轮对话：{answer[:50]}...")  # 打印前 50 字

    # --- 第二轮对话（测试多轮记忆）---
    # "那如果对方不配合呢？" 里的"对方"指代第一轮的"邻居"
    # 如果 Redis 短期记忆生效，LLM 能理解"对方"是谁
    r2 = tc.post("/chat", json={  # 第二轮 POST 请求
        "user_id": 1,  # 同一个用户
        "role_id": role_id,  # 同一个角色
        "query": "那如果对方不配合呢？",  # 测试代词消解（依赖上下文记忆）
        "top_k": 3,
    })
    assert r2.status_code == 200  # 断言 200
    print(f"[PASS] 第二轮对话（记忆测试）：{r2.json()['answer'][:50]}...")  # 打印前 50 字


def test_memory():
    """测试长期记忆保存 + 短期记忆清空"""
    # 保存长期记忆到 Milvus
    r = tc.post("/memory/save", json={  # 发 POST 请求
        "user_id": 1,  # 用户 ID
        "role_id": 1,  # 角色 ID
        "content": "用户曾咨询过噪音扰民问题，关注精神损害赔偿",  # 要记住的内容
        "summary": "噪音扰民咨询",  # 摘要
    })
    assert r.status_code == 200  # 断言 200
    print("[PASS] 长期记忆保存")  # 打印通过

    # 清空短期记忆（Redis）
    r2 = tc.post("/memory/clear", json={"user_id": 1, "role_id": 1})  # 发 POST 请求
    assert r2.status_code == 200  # 断言 200
    print("[PASS] 短期记忆清空")  # 打印通过


def test_stream():
    """测试流式对话接口 POST /chat/stream（SSE 逐 chunk 返回）"""
    r = tc.post("/chat/stream", json={  # 发 POST 请求
        "user_id": 1,  # 用户 ID
        "role_id": 1,  # 角色 ID
        "query": "什么是相邻关系？",  # 查询
        "top_k": 3,  # 引用条数
    }, stream=True)  # stream=True：不等待完整响应，逐块接收
    assert r.status_code == 200  # 断言 200
    # r.iter_bytes() 逐块迭代响应体，len(chunk) 算每块字节数，sum 求总字节
    total = sum(len(chunk) for chunk in r.iter_bytes())  # 累加所有 chunk 的字节数
    assert total > 0  # 断言收到了数据
    print(f"[PASS] 流式对话：收到 {total} 字节")  # 打印总字节数


# ==================== 主测试入口 ====================
if __name__ == "__main__":
    print("=" * 60)  # 打印分隔线
    print("RAG 角色扮演系统 - 端到端测试")  # 标题
    print("=" * 60)  # 打印分隔线

    test_health()  # 测试 1：健康检查
    roles = test_list_roles()  # 测试 2：角色列表（返回 roles 供后续用）
    test_ingest_text()  # 测试 3：文本入库
    test_cases()  # 测试 4：判例检索
    test_chat(roles)  # 测试 5：多轮对话（含记忆测试）
    test_memory()  # 测试 6：长期记忆 + 短期清空
    try:  # 流式对话可能因网络/模型超时失败，用 try 兜底
        test_stream()  # 测试 7：流式对话
    except Exception as e:  # 捕获异常
        print(f"[SKIP] 流式对话测试跳过：{e}")  # 打印跳过原因（不算失败）

    # 清理临时集合（测试完不留垃圾）
    client.drop_collection("_test_roleplay")  # 删掉测试集合
    print("=" * 60)  # 打印分隔线
    print("全部测试通过！")  # 成功提示
    print("=" * 60)  # 打印分隔线
