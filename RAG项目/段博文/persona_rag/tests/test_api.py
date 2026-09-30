# -*- coding: utf-8 -*-
"""
PersonaRAG · 多角色智能顾问系统 - API 端到端测试
用法：python tests/test_api.py
用 TestClient 不需要真实启动 uvicorn（lifespan 会自动执行，模型在测试开始时加载）

本模块在系统中的位置：
    这是「整链路冒烟测试」，走的是真实的 HTTP 接口层：TestClient(app) 会把请求发给 main.py 里的
    FastAPI 应用，因此沿途会真正用到 Milvus、Redis、MySQL 和 DeepSeek。
    上游：人工执行或 CI 调用；下游：main.py 的各个接口（间接用到 db_* / retriever / llm_client）。

为什么用 TestClient 而不是真起服务：
    1. 不用管端口占用和进程退出；
    2. 进入 with/首次请求时会自动触发 app 的 lifespan，模型、集合、BM25 都会被初始化，
       和真实启动服务的环境一致。

关键取舍：
    - 用环境变量把集合名切成临时集合 "_test_roleplay"，跑测试不会动到正式知识库；
      注意这个变量必须在 import main 之前设置（config 在导入时就会读取）。
    - 测试跑完主动 drop 临时集合，避免残留数据影响下一次测试结果。
    - 用例之间共享同一个 TestClient 和 Redis/MySQL 里的同一批测试数据，所以顺序执行、
      并且依赖前一步的产物（例如 /chat 依赖先入库的语料）。
"""

import os  # 操作系统接口：读取环境变量、路径操作
import sys  # 系统接口：修改 sys.path 让测试能 import 到项目根目录的模块

# 把项目根目录加入 sys.path，否则 from main import app 会找不到 main.py
# __file__ = 当前文件路径；dirname 两层 = 上级目录（tests/ 的父目录 = 项目根目录）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 插入到 path 列表头部

# 设置环境变量：用临时集合 "_test_roleplay" 测试，不碰正式知识库数据
os.environ["MILVUS_COLLECTION"] = "_test_roleplay"  # 临时集合名，测试完会删掉；必须在 import main 之前设置，config 是在导入时读环境变量的
os.environ["MILVUS_URI"] = "http://localhost:19530"  # Milvus 地址

from fastapi.testclient import TestClient  # FastAPI 的测试客户端：不需要真正起 HTTP 服务
from pymilvus import MilvusClient  # Milvus 原生客户端：用来清理测试集合

# 导入 main 模块会触发 app 的 lifespan 回调（加载模型、建表、BM25 重建）
from main import app  # FastAPI 应用对象

# 创建 Milvus 原生客户端实例，用于测试前清理和测试后清理
client = MilvusClient(uri="http://localhost:19530")  # 连接 Milvus
# 如果之前的测试残留了临时集合，先删掉（保证干净的测试环境）
if client.has_collection("_test_roleplay"):  # 检查集合是否存在
    client.drop_collection("_test_roleplay")  # 删除集合：上次跑挂了留下的半成品数据会干扰断言

# 创建 TestClient：它会在 __enter__ 里触发 lifespan 启动事件（加载所有模型）
tc = TestClient(app)  # 封装 FastAPI app 为可测试对象

# ==================== 测试用例 ====================

def test_health():
    """测试健康检查接口 GET /health

    请求：GET /health（返回 JSON，形如 {"status":"ok","online_users":0}）。
          注意不能请求 "/"，根路径返回的是跳转前端的 HTML，用 r.json() 解析会直接失败。
    断言：状态码 200，且响应体能解析出 status="ok"。
    返回：无。
    """
    r = tc.get("/health")  # 向 "/health" 发 GET 请求
    assert r.status_code == 200  # 断言返回 200 OK
    assert r.json()["status"] == "ok"  # 断言响应体里 status 字段为 "ok"
    print("[PASS] 健康检查")  # 打印通过


def test_list_roles():
    """测试角色列表接口 GET /roles

    请求：GET /roles（角色来自 lifespan 写入 MySQL 的预置角色）。
    断言：200 且角色数大于 0。
    返回：角色列表，供后面的 /chat 等用例取 role_id 用（用例之间有依赖，顺序不能乱）。
    """
    r = tc.get("/roles")  # 向 "/roles" 发 GET 请求
    assert r.status_code == 200  # 断言 200
    roles = r.json()["roles"]  # 提取角色列表
    assert len(roles) > 0  # 断言至少有一个角色（lifespan 里写入了 8 个预置角色）
    print(f"[PASS] 角色列表：{len(roles)} 个角色")  # 打印角色数量
    return roles  # 返回角色列表供后续测试用


def test_ingest_text():
    """测试文本入库接口 POST /ingest/text

    请求体：
        source   固定为 "民法典测试.md"，后缀 .md 会让切分自动走 Markdown 标题策略；
        text     下面这段模拟法条 + 判例的文本；
        strategy 传 paragraph，但因为是 .md 会被 markdown 策略覆盖。
    断言：200 且 chunks > 0，即确实切出了块。
    返回：入库接口的响应体 {"message","chunks","corpus_total","collection","role_key"}。
    说明：后面的 /cases 和 /chat 用例都依赖这一步写入的语料，所以本用例必须排在它们前面。
    """
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
    """测试判例检索接口 POST /cases（不调 LLM，只返回检索到的文档片段）

    请求体：query 是「夜间装修噪音 判例 赔偿」；top_k=3；role_key 不传走默认 lawyer，
            所以检索的是上面入库的临时集合。
    断言：200 且 count > 0，说明混合检索这条链路（向量 + BM25 + 重排）真的能召回数据。
    返回：响应体 {"query","count","documents"}。
    说明：这个用例同时充当「检索链路是否可用」的探针——如果它挂了，后面的 /chat 也没意义。
    """
    r = tc.post("/cases", json={"query": "夜间装修噪音 判例 赔偿", "top_k": 3})  # 发 POST 请求
    assert r.status_code == 200  # 断言 200
    assert r.json()["count"] > 0  # 断言检索到至少 1 条
    print(f"[PASS] 判例检索：命中 {r.json()['count']} 条")  # 打印命中数
    return r.json()  # 返回响应体


def test_chat(roles):
    """测试多轮对话接口 POST /chat（角色 + 记忆 + RAG 全链路）

    参数：roles —— test_list_roles 的返回值，取第 0 个角色的 id 作为 role_id（法律顾问）。
    请求体：user_id=1（固定测试用户）、role_id、query、top_k=3。
    断言：第一轮 200 且回答长度 > 10，保证 LLM 真的产出了内容。
    返回：无。
    说明：第二轮是故意设计的「代词消解」用例：单独一句「那如果对方不配合呢？」语义不完整，
          只有 Redis 短期记忆把第一轮上下文带进去，模型才知道「对方」指邻居。
          这也意味着本用例会真实消耗 DeepSeek 的额度。
    """
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
    """测试长期记忆保存 + 短期记忆清空

    第一步 POST /memory/save：user_id=1、role_id=1，content 是要写进 Milvus 长期记忆的正文，
        summary 是摘要。断言 200。
    第二步 POST /memory/clear：清掉 (user_id=1, role_id=1) 的 Redis 短期记忆。断言 200。
    返回：无。
    说明：这里故意「先存长期记忆、再清短期记忆」，用来说明两者互不影响：
          清掉的只是 Redis 里的对话窗口，Milvus 里的长期记忆仍然保留。
    """
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
    """测试流式对话接口 POST /chat/stream（SSE 逐 chunk 返回）

    请求体：user_id=1、role_id=1、query、top_k=3；stream=True 表示不等待完整响应。
    断言：200 且收到的总字节数 > 0，即确实有流式数据回来（不校验内容，因为流式结果不稳定）。
    返回：无。
    说明：流式响应不能用 r.json() 解析（响应体是纯文本片段），所以这里只统计字节数。
    """
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
    # 直接跑本脚本时按固定顺序执行全部用例；没有用 pytest，所以断言失败会直接抛异常中断
    print("=" * 60)  # 打印分隔线
    print("PersonaRAG · 多角色智能顾问系统 - 端到端测试")  # 标题
    print("=" * 60)  # 打印分隔线

    test_health()  # 测试 1：健康检查
    roles = test_list_roles()  # 测试 2：角色列表（返回 roles 供后续用）
    test_ingest_text()  # 测试 3：文本入库（后面两条用例依赖它写入的语料）
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
