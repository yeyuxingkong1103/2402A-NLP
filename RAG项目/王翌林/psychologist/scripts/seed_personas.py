"""幂等写入三个心理医生角色与系统角色。

用法：
    python scripts/seed_personas.py
"""
import os
import sys

# 把项目根目录插入 sys.path 首位，保证从任意目录执行都能 import 到 src.* 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.logging import get_logger, setup_logging  # noqa: E402
from src.db.mysql import init_db, session_scope  # noqa: E402
from src.db.redis import invalidate_persona  # noqa: E402
from src.services import persona_service  # noqa: E402

logger = get_logger("scripts.seed_personas")


def main() -> None:
    setup_logging()
    # 先确保表结构存在（幂等），再写入角色数据
    init_db()
    # session_scope 是 SQLAlchemy 会话上下文管理器，退出时自动 commit/rollback 并释放连接
    with session_scope() as db:
        # seed_roles_and_personas 内部按角色 code 判重，重复执行不会产生重复记录（幂等）
        mapping = persona_service.seed_roles_and_personas(db)
        for persona in persona_service.list_personas(db, only_active=False):
            # 角色有变化时清除 Redis 中的缓存，避免读到旧的角色信息
            invalidate_persona(persona.id)
            logger.info("角色：%s(%s) 流派=%s 状态=%s",
                        persona.name, persona.persona_code, persona.therapy_type, persona.status)
        logger.info("角色初始化完成：%s", mapping)


if __name__ == "__main__":
    main()