"""批次 14 e2e 验收：长期记忆 写入 → 检索 → 注入提示词 → 删除 → 开关（真实链路）。

验收项（用户批次 14 清单）：
a) 写入 → 检索 → 注入提示词（贴提示词里记忆段落的真实内容）
b) 跨用户隔离（A 的记忆在 B 的检索里查不到）
c) 删除后立即从列表与检索中消失
d) 关闭后不再写入（贴开关前后对比）
e) 去重生效：同一事实问两遍只留一条（相似则更新）

运行（项目根）：python scripts/e2e/e2e_long_term_memory.py
注意：USER_A/USER_B/SESSION 为固定值，重复运行前需清理 Milvus 中对应 user_id 的旧记忆
（详见 scripts/e2e/README.md）。
"""
import sys
import time
from pathlib import Path

# 项目根与 backend（脚本已从项目根移入 scripts/e2e/，故向上三级定位 rag/）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from scripts._env import load_project_env  # noqa: E402

# 加载 .env（与 run_eval.prepare_env 同口径），顺带剔除 http(s)_proxy
load_project_env(PROJECT_ROOT)

from app.core.config import settings  # noqa: E402
# MemorySettingsStore 在批次19 已拆到 memory_settings.py（用户级开关独立成模块）
from app.memory.long_term import LongTermMemoryStore  # noqa: E402
from app.memory.memory_settings import MemorySettingsStore  # noqa: E402
from app.models.embedding import SiliconFlowEmbeddingClient  # noqa: E402
from pymilvus import MilvusClient  # noqa: E402

USER_A = "e2e" + "a" * 28  # 32 位，独立于真实用户
USER_B = "e2e" + "b" * 28
SESSION = "e2e_session_mem"


def section(title: str) -> None:
    print()
    print("=" * 32, title, "=" * 32)


def main() -> None:
    # ---------- 装配（真实 Milvus + SiliconFlow Embedding + 真实 LLM） ----------
    milvus = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
    store = LongTermMemoryStore(
        milvus_client=milvus,
        embedding_client=SiliconFlowEmbeddingClient(
            api_url=settings.embedding_api_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
        ),
        collection_name=settings.milvus_long_term_collection_name,
        dimension=settings.embedding_dimension,
        dedup_threshold=settings.long_term_memory_dedup_threshold,
    )
    store.ensure_collection()
    from app.chat.chat_runtime import build_default_session_factory
    from app.db.sql_models import User

    factory = build_default_session_factory()
    settings_store = MemorySettingsStore(factory)
    # e2e 专用测试用户（开关门卫要查 users 表；不存在会被正确拒绝读写）
    with factory() as session:
        for uid, email in ((USER_A, "e2e-mem-a@test.local"), (USER_B, "e2e-mem-b@test.local")):
            existing = session.query(User).filter(User.user_key == uid).one_or_none()
            if existing is None:
                session.add(User(user_key=uid, email=email, password_hash="e2e-only"))
        session.commit()
    settings_store.set_enabled(USER_A, True)
    settings_store.set_enabled(USER_B, True)

    # 预清理：上一轮 e2e 残留
    for uid in (USER_A, USER_B):
        for item in store.list_memories(user_id=uid, include_deleted=True)["items"]:
            store.soft_delete(user_id=uid, memory_id=item["memory_id"])

    # 真实问答服务：包一层记录提示词的 LLM 客户端
    # build_default_chat_service 在批次 19 已拆到 app.chat.bootstrap（原在 app.chat.service）
    from app.chat.bootstrap import build_default_chat_service

    service = build_default_chat_service()
    real_llm = service.llm_client

    class RecordingLlm:
        """记录发给大模型的 user prompt，其余全部委托真实客户端。"""

        def __init__(self, inner):
            self.inner = inner
            self.user_prompts: list[str] = []

        def chat(self, system_prompt, user_prompt):
            self.user_prompts.append(user_prompt)
            return self.inner.chat(system_prompt, user_prompt)

        def stream_chat(self, system_prompt, user_prompt):
            self.user_prompts.append(user_prompt)
            yield from self.inner.stream_chat(system_prompt, user_prompt)

    recording = RecordingLlm(real_llm)
    service.llm_client = recording
    service.memory_write_async = False  # e2e：同步写，时序确定可断言

    # SiliconFlow 瞬时超时频发：给记忆存储的 Embedding 包一层重试（仅 e2e）
    import time as _t

    class RetryEmbedding:
        def __init__(self, inner):
            self.inner = inner

        def embed(self, texts):
            for attempt in range(8):
                try:
                    return self.inner.embed(texts)
                except Exception:
                    if attempt == 7:
                        raise
                    _t.sleep(3)

    retry_embedding = RetryEmbedding(service.long_term_memory.embedding_client)
    service.long_term_memory.embedding_client = retry_embedding
    store.embedding_client = retry_embedding

    def memory_section(prompt: str) -> str:
        """从 user prompt 里抠出记忆段落（便于贴真实内容）。"""
        if "# 用户记忆" not in prompt:
            return "（本次提示词未注入记忆段落）"
        start = prompt.index("# 用户记忆")
        end = prompt.index("# 法源清单")
        return prompt[start:end].strip()

    # ================= a) 写入 → 检索 → 注入提示词 =================
    section("a) 写入 → 检索 → 注入提示词")
    result1 = service.chat(
        "我在一家公司做程序员，入职一年了一直没签书面劳动合同，有什么后果？",
        user_id=USER_A,
        session_id=SESSION,
    )
    print("[第 1 问] 回答（前 200 字）:", (result1.answer or "")[:200].replace("\n", " "))
    print("[第 1 问] guardrails:", result1.guardrail_applied)
    items = store.list_memories(user_id=USER_A)["items"]
    print(f"[写入] 用户 A 当前记忆 {len(items)} 条：")
    for item in items:
        print(f"   - summary: {item['summary']} | importance: {item['importance']}")
    # Milvus 写入→可检索存在最终一致性窗口：轮询等可见（上限 30s）
    for _ in range(30):
        try:
            if store.search(USER_A, "不签合同能要求双倍工资吗"):
                break
        except Exception:
            pass  # Embedding 瞬时超时，重试
        time.sleep(2)
    print("[可见性] 等待后 search 可见:", bool(store.search(USER_A, "不签合同能要求双倍工资吗")))

    # Embedding 瞬时超时会走"检索失败→不注入"兜底；轮询等一次成功，拿到真实记忆块
    block = None
    for _ in range(30):
        try:
            block = service._memory_block(USER_A, "那不签合同能要求双倍工资吗？")
        except Exception:
            pass
        if block:
            break
        time.sleep(2)
    print("[第 2 问前] 记忆块（真实内容）:")
    print("   " + (block or "（仍不可用）").replace("\n", "\n   "))
    p0 = len(recording.user_prompts)  # chat 会先调"回答"（下标 p0）再调"摘要"（p0+1）
    result2 = service.chat("那不签合同能要求双倍工资吗？", user_id=USER_A, session_id=SESSION)
    print()
    print("[第 2 问] 提示词里注入的记忆段落（真实内容，取回答生成调用）:")
    print("   " + memory_section(recording.user_prompts[p0]).replace("\n", "\n   "))
    print("[第 2 问] 回答（前 120 字）:", (result2.answer or "")[:120].replace("\n", " "))

    # ================= b) 跨用户隔离 =================
    section("b) 跨用户隔离")
    p1 = len(recording.user_prompts)
    service.chat("试用期最长不能超过多久？", user_id=USER_B, session_id=SESSION)
    b_prompt = recording.user_prompts[p1]
    print("[用户 B 提示词注入记忆段落]:", memory_section(b_prompt).replace("\n", " | "))
    hits_b = store.search(USER_B, "程序员")
    foreign = [h for h in hits_b if h.user_id != USER_B]
    a_summary_leaked = "始终未与其签订书面劳动合同" in b_prompt
    print(f"[用户 B 向量检索 '程序员'] 命中 {len(hits_b)} 条，全部属于 B 自己:", all(h.user_id == USER_B for h in hits_b))
    print("隔离判定:", "❌ 泄漏" if foreign or a_summary_leaked else "✅ B 只能看到自己的记忆，检索不到 A 的")

    # ================= e) 去重：同一事实问两遍只留一条 =================
    section("e) 去重生效")
    before = len(store.list_memories(user_id=USER_A)["items"])
    service.chat("对了我再确认下，不签书面合同的双倍工资最多能拿几个月？", user_id=USER_A, session_id=SESSION)
    after_items = store.list_memories(user_id=USER_A)["items"]
    print(f"[A 记忆条数] 问前 {before} → 问后 {len(after_items)}（相似事实应更新原记录而非新增）")
    for item in after_items:
        print(f"   - summary: {item['summary']} | updated_at: {item['updated_at']}")

    # ================= c) 删除后立即消失 =================
    section("c) 删除后立即从列表与检索中消失")
    target = after_items[0]["memory_id"]
    print("[软删除] memory_id:", target)
    ok = store.soft_delete(user_id=USER_A, memory_id=target)
    print("soft_delete 返回:", ok)
    print("[删除后] 列表条数:", len(store.list_memories(user_id=USER_A)["items"]), f"（删前 {len(after_items)} 条）")
    remaining = [r["memory_id"] for r in after_items if r["memory_id"] != target]
    hit_after = store.search(USER_A, "双倍工资")
    print("[删除后] 向量检索命中:", len(hit_after), "，被删 memory_id 是否还在:", any(h.memory_id == target for h in hit_after))
    assert target not in [r["memory_id"] for r in store.list_memories(user_id=USER_A, include_deleted=False)["items"]]
    # 非本人删除 → False（404 语义）：B 尝试删 A 剩下的那条
    if remaining:
        print("[越权删除] B 尝试删 A 的记录返回:", store.soft_delete(user_id=USER_B, memory_id=remaining[0]))

    # ================= d) 关闭后不再写入 =================
    section("d) 关闭后不再写入（开关前后对比）")
    settings_store.set_enabled(USER_A, False)
    print("[开关] 用户 A long_term_memory_enabled = False")
    count_before = len(store.list_memories(user_id=USER_A, include_deleted=True)["items"])
    p2 = len(recording.user_prompts)
    service.chat("年假有几天？", user_id=USER_A, session_id=SESSION)
    d_prompt = recording.user_prompts[p2]
    count_after = len(store.list_memories(user_id=USER_A, include_deleted=True)["items"])
    print("[开关关] 提示词记忆段落:", memory_section(d_prompt).replace("\n", " | "))
    print(f"[开关关] 记忆条数 {count_before} → {count_after}（应不变）")
    settings_store.set_enabled(USER_A, True)
    print("[开关] 恢复 True")
    stamps_before = {r["memory_id"]: r["updated_at"] for r in store.list_memories(user_id=USER_A, include_deleted=True)["items"]}
    service.chat("我上周提了离职，公司押着上个月工资不发怎么办？", user_id=USER_A, session_id=SESSION)
    items_re = store.list_memories(user_id=USER_A, include_deleted=True)["items"]
    stamps_after = {r["memory_id"]: r["updated_at"] for r in items_re}
    changed = any(stamps_after[k] != v for k, v in stamps_before.items()) or len(items_re) > len(stamps_before)
    print(f"[开关重开] 记忆条数 {len(stamps_before)} → {len(items_re)}，有新增或更新:", changed, "（恢复写入）")
    for r in items_re:
        print(f"   - summary: {r['summary']} | updated_at: {r['updated_at']}")

    print()
    print("e2e 完成。")


if __name__ == "__main__":
    main()
