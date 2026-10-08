"""多轮对话的最小可运行示例。**不依赖 Milvus，也不调用真实模型。**

    D:/zg6_Project/9/med_rag/rag/python.exe -m examples.chat_demo

---

## 这个示例演示什么、不演示什么

**演示**：五个接口的完整生命周期（建会话 → 发消息 → 读历史 → 改名 → 删除），
以及一个容易忽略的事实 —— 会话**不建消息 List**，第一次发言才建。

**不演示**：检索与生成的**质量**。它把 `retrieve` 打桩成"没命中"，
因此每一轮都走拒答路径。理由：

  1. 真实的检索需要 Milvus + BGE-M3（约 10 秒权重加载 + 2.3 GB 内存），
     而本示例的意图是让人**在两秒内看到接口怎么用**；
  2. 拒答是**多轮场景下的高频路径**（省略主语的追问本就容易检索落空），
     把它演示出来比演示一次成功问答更有代表性。

要看真实检索，跑 `backend.serve` 然后照 `specs/010-multiturn-chat/quickstart.md` 做。

## 为什么用 `TestClient` 而不是 `requests`

本示例要能在**没启动服务**的情况下跑。`TestClient` 直接驱动 ASGI 应用，
不需要监听端口 —— 于是"照着示例跑一遍"不会因为"忘了先启动服务"而失败。

真实部署下把 `client` 换成 `httpx.Client(base_url=...)` 即可，
请求与响应的形状一模一样。
"""

from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore")

import fakeredis.aioredis  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.api.app import create_app  # noqa: E402
from backend.api.config import AppConfig  # noqa: E402
from backend.chat import dialogue  # noqa: E402
from backend.chat.service import ChatService  # noqa: E402
from backend.chat.store import ChatStore  # noqa: E402


def build_client() -> TestClient:
    """装配一个可直接用的应用。**用内存版 Redis，不碰真实存储。**"""

    app = create_app(AppConfig())

    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    app.state.chat_client = fake
    app.state.chat_service = ChatService(ChatStore(fake, ttl=1800, max_history=50))

    # 打桩：不检索、不调模型（见模块文档）
    dialogue.retrieve = lambda *a, **k: None  # type: ignore[assignment]
    import backend.api.chat_routes as chat_routes

    chat_routes.capture_question = lambda q, aid: [0.0] * 1024  # type: ignore[assignment]

    return TestClient(app)


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """把 SSE 文本解析成 (事件名, 载荷) 列表。"""

    events: list[tuple[str, dict]] = []
    name = None
    for line in text.splitlines():
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            events.append((name, json.loads(line[5:].strip())))
    return events


def main() -> None:
    with build_client() as client:
        # ---- 1. 建会话（**不传 user_id**，让服务端生成）----
        print("① 建会话")
        body = client.post("/api/chat/session", json={}).json()
        session_id = body["session_id"]
        print(f"   session_id = {session_id[:16]}…")
        print(f"   user_id    = {body['user_id']}   ← 自动生成，可改")
        print(f"   过期时长   = {body['expires_in']} 秒")
        print()

        # ---- 2. 未发言时没有历史 ----
        print("② 读历史（还没说过话）")
        print("  ", client.get(f"/api/chat/history/{session_id}").json())
        print("   注：消息 List 是第一次发言时才创建的 —— 空列表不占一条 Key")
        print()

        # ---- 3. 发一条消息 ----
        print("③ 发消息（SSE，逐帧打印事件名）")
        resp = client.post(
            "/api/chat/message",
            json={"session_id": session_id, "content": "高血压平时要注意些什么"},
        )
        events = parse_sse(resp.text)
        print("   事件序列：", " → ".join(name for name, _ in events))

        done = events[-1][1]
        print(f"   is_refusal = {done['is_refusal']}   ← 本示例打桩成不检索，故为拒答")
        print(f"   answer_text 末 20 字：…{done['answer_text'][-20:]}")
        print()

        # ---- 4. 历史里现在有两条 ----
        print("④ 再读历史")
        history = client.get(f"/api/chat/history/{session_id}").json()
        print(f"   total = {history['total']}")
        for message in history["messages"]:
            print(f"   [{message['role']:9}] {message['content'][:24]}…")
        print()

        # ---- 5. 改名字 ----
        print("⑤ 改发起者标识（**不刷新 TTL**）")
        renamed = client.patch(
            f"/api/chat/session/{session_id}", json={"user_id": "我的昵称"}
        ).json()
        print(f"   user_id = {renamed['user_id']}")
        print()

        # ---- 6. 删除（幂等）----
        print("⑥ 删除会话（删两次都返回 200 —— DELETE 是幂等的）")
        print("   第一次：", client.delete(f"/api/chat/session/{session_id}").json())
        print("   第二次：", client.delete(f"/api/chat/session/{session_id}").json())
        print("   再读历史：", client.get(f"/api/chat/history/{session_id}").status_code)
        print()

        print("完成。真实部署下把 TestClient 换成 httpx.Client(base_url=...) 即可。")


if __name__ == "__main__":
    main()
