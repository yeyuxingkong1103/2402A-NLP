"""批次 15 任务 1：清理 e2e 探针/测试账号及其从属数据。

删除条件（写死，不许按"看起来像测试"挑）：
  password_hash 不是合法格式（urlsafe-base64 解码后应为 48 字节 = 16 字节盐 + 32 字节摘要）
  OR email 形如 e2e*_*@*（SQL: email LIKE 'e2e%@%'）

删除顺序（保持引用完整）：
  ① Milvus legal_long_term_memory 按实体 delete（filter=user_id == uid），不 drop 集合
  ② chat_messages（按会话外键）
  ③ chat_sessions
  ④ users 行

运行（项目根）：python scripts/e2e/cleanup_e2e_probe_accounts.py
（连接参数从项目根 .env 读取，见 scripts/_env.py）
"""
import base64
import sys
from pathlib import Path

# 项目根与 backend（脚本已从项目根移入 scripts/e2e/，故向上三级定位 rag/）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from scripts._env import load_project_env  # noqa: E402

# 加载 .env，顺带剔除 http(s)_proxy（本机请求会被沙箱代理劫持）
load_project_env(PROJECT_ROOT)

from app.core.config import settings  # noqa: E402
from app.chat.chat_runtime import build_default_session_factory  # noqa: E402
from app.db.chat_models import ChatMessage, ChatSession  # noqa: E402
from app.db.sql_models import User  # noqa: E402
from pymilvus import MilvusClient  # noqa: E402

VALID_HASH_LEN = 48  # 16 字节盐 + 32 字节 PBKDF2 摘要


def is_valid_hash(value: str) -> bool:
    try:
        raw = base64.urlsafe_b64decode(value.encode())
    except Exception:  # noqa: BLE001
        return False
    return len(raw) == VALID_HASH_LEN


def main() -> None:
    factory = build_default_session_factory()
    milvus = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
    collection = settings.milvus_long_term_collection_name
    milvus.load_collection(collection)

    with factory() as session:
        users = session.query(User).order_by(User.id).all()
        print(f"users 表总行数：{len(users)}")
        targets = [u for u in users if (not is_valid_hash(u.password_hash)) or u.email.startswith("e2e")]
        kept = [u for u in users if u not in targets]

        print("\n========== ① 待删清单（死条件命中）==========")
        if not targets:
            print("（无命中行）")
        for u in targets:
            reason = []
            if not is_valid_hash(u.password_hash):
                reason.append("password_hash 无效")
            if u.email.startswith("e2e"):
                reason.append("email 形如 e2e_*@*")
            mem_ids = milvus.query(collection, filter=f'user_id == "{u.user_key}"', output_fields=["memory_id"])
            sess = session.query(ChatSession).filter(ChatSession.user_id == u.user_key).all()
            msg_count = (
                session.query(ChatMessage)
                .join(ChatSession, ChatMessage.session_id == ChatSession.id)
                .filter(ChatSession.user_id == u.user_key)
                .count()
            )
            print(
                f"- id={u.id} user_key={u.user_key} email={u.email} "
                f"created_at={u.created_at} 命中={'+'.join(reason)}"
            )
            print(f"    名下长期记忆 {len(mem_ids)} 条 / chat_sessions {len(sess)} 条 / chat_messages {msg_count} 条")

        print("\n========== ② 保留用户（不应动）==========")
        for u in kept:
            print(f"- id={u.id} user_key={u.user_key} email={u.email}")

        print("\n========== ③ 开始删除（记忆向量 → 消息 → 会话 → 用户行）==========")
        for u in targets:
            milvus.delete(collection, filter=f'user_id == "{u.user_key}"')
            print(f"[Milvus] user_id={u.user_key} 的记忆实体已 delete（未 drop 集合）")
            session.query(ChatMessage).filter(
                ChatMessage.session_id.in_(
                    session.query(ChatSession.id).filter(ChatSession.user_id == u.user_key)
                )
            ).delete(synchronize_session=False)
            session.query(ChatSession).filter(ChatSession.user_id == u.user_key).delete(synchronize_session=False)
            session.query(User).filter(User.id == u.id).delete(synchronize_session=False)
            session.commit()
            print(f"[MySQL] user id={u.id} ({u.email}) 名下消息/会话/用户行已删除")

        print("\n========== ④ 删后核验 ==========")
        remain_users = session.query(User).order_by(User.id).all()
        remain_sess = session.query(ChatSession).count()
        remain_msgs = session.query(ChatMessage).count()
        remain_mem = milvus.query(collection, filter="deleted == false or deleted == true", output_fields=["user_id"])
        from collections import Counter

        mem_by_user = Counter(r["user_id"] for r in remain_mem)
        print(f"users 表剩余 {len(remain_users)} 行 / chat_sessions 剩余 {remain_sess} 行 / chat_messages 剩余 {remain_msgs} 行")
        print(f"Milvus {collection} 剩余 {len(remain_mem)} 条实体，按 user 分布：{dict(mem_by_user)}")
        # 核验：目标 user_key 应全部消失
        leaked = [r for r in remain_mem if r["user_id"] in {t.user_key for t in targets}]
        orphan_sess = (
            session.query(ChatSession).filter(ChatSession.user_id.in_([t.user_key for t in targets])).count()
        )
        print(f"Milvus 残留测试用户记忆：{len(leaked)} 条；测试用户残留会话：{orphan_sess} 条")
        print("\n保留用户数据核对：")
        for u in remain_users:
            sess = session.query(ChatSession).filter(ChatSession.user_id == u.user_key).count()
            print(f"- {u.email} (user_key={u.user_key}) 名下会话 {sess} 条 / 记忆 {mem_by_user.get(u.user_key, 0)} 条")


if __name__ == "__main__":
    main()
