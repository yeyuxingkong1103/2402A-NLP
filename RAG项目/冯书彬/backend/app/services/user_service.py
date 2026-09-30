import logging

from backend.app.api.v1.chat import chat_service
from backend.app.services.auth_service import get_auth_store
from backend.app.services.export_service import delete_export_data_for_user
from backend.app.services.feedback_service import delete_feedback_data_for_user
from backend.app.services.memory_service import delete_all_memories_for_user

logger = logging.getLogger(__name__)


def delete_user_account(user_id: str) -> bool:
    # 账号注销删除可关联个人数据；审计日志由审计服务保留脱敏记录。
    deleted_conversations = 0
    deleted_messages = 0
    for conversation_id, conversation in list(chat_service.conversations.items()):
        if conversation.user_id != user_id:
            continue
        chat_service.conversations.pop(conversation_id, None)
        deleted_conversations += 1
        for message_id in conversation.message_ids:
            if chat_service.messages.pop(message_id, None) is not None:
                deleted_messages += 1
    for message_id, message in list(chat_service.messages.items()):
        if message.user_id == user_id:
            chat_service.messages.pop(message_id, None)
            deleted_messages += 1
    deleted_memories = delete_all_memories_for_user(user_id)
    deleted_feedback = delete_feedback_data_for_user(user_id)
    deleted_exports = delete_export_data_for_user(user_id)
    get_auth_store().delete_user(user_id)
    logger.info(
        "user account deleted",
        extra={
            "user_id": user_id,
            "deleted_conversations": deleted_conversations,
            "deleted_messages": deleted_messages,
            "deleted_memories": deleted_memories,
            "deleted_feedback": deleted_feedback["feedback"],
            "deleted_alerts": deleted_feedback["alerts"],
            "deleted_export_jobs": deleted_exports["jobs"],
            "deleted_export_tokens": deleted_exports["tokens"],
        },
    )
    return True
