import logging

from backend.app.api.v1.chat import chat_service
from backend.app.services.memory_service import delete_exclusive_memories_for_conversation

logger = logging.getLogger(__name__)


def delete_conversation_and_exclusive_memories(user_id: str, conversation_id: str) -> list[str]:
    # 删除前先查找并校验归属，禁止通过会话 ID 删除他人会话。
    conversation = chat_service.conversations.get(conversation_id)
    if conversation is None or conversation.user_id != user_id:
        logger.warning("conversation delete denied", extra={"user_id": user_id, "conversation_id": conversation_id})
        raise PermissionError("无权删除该会话")
    chat_service.conversations.pop(conversation_id, None)
    deleted_messages = 0
    # 逐条删除内存消息，避免残留可关联聊天内容。
    for message_id in conversation.message_ids:
        if chat_service.messages.pop(message_id, None) is not None:
            deleted_messages += 1
    # 共享记忆由 memory_service 保留。
    deleted_memories = delete_exclusive_memories_for_conversation(user_id, conversation_id)
    logger.info("conversation deleted with exclusive memories", extra={"user_id": user_id, "conversation_id": conversation_id, "deleted_messages": deleted_messages, "deleted_memories": len(deleted_memories)})
    return deleted_memories
