"""Persistent conversation and message history routes."""
import json
import time

from fastapi import APIRouter, Depends, HTTPException

from src import config
from .auth import _auth_db, get_current_user

router = APIRouter(tags=["history"])


def _create_conversation(user_id: int | None, title: str, type_: str = "qa") -> int:  # 新建一条历史对话
    now = time.time()  # 当前时间
    with _auth_db() as conn:  # 打开数据库
        cur = conn.execute(  # 插入对话记录
            "INSERT INTO conversations (user_id, title, type, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, title.strip()[: config.HISTORY_TITLE_CHARS] or "新对话", type_, now, now),
        )
        conversation_id = cur.lastrowid  # 拿到自增 ID
        conn.execute(  # 清理无主对话的残留消息（安全兜底）
            "DELETE FROM messages WHERE conversation_id NOT IN (SELECT id FROM conversations)"
        )
        if user_id is not None:  # 仅对已登录用户做数量限制
            conn.execute(  # 删除超出保留上限的最旧对话
                """DELETE FROM conversations WHERE user_id = ? AND id NOT IN (
                    SELECT id FROM conversations WHERE user_id = ?
                    ORDER BY updated_at DESC LIMIT ?
                )""",
                (user_id, user_id, config.HISTORY_LIMIT),
            )
            conn.execute(  # 同步删除被清理对话的消息
                "DELETE FROM messages WHERE conversation_id NOT IN (SELECT id FROM conversations)"
            )
    return conversation_id  # 返回对话 ID


def _resolve_conversation(
    user_id: int | None,
    conversation_id: int | None,
    title: str,
    type_: str = "qa",
) -> int:
    """校验并复用已有会话；不存在或不属于当前用户时新建。"""
    if conversation_id is not None:  # 前端传了会话 ID
        with _auth_db() as conn:  # 打开数据库
            row = conn.execute(  # 校验会话存在且属于当前用户
                "SELECT id FROM conversations WHERE id = ? "
                "AND ((? IS NULL AND user_id IS NULL) OR user_id = ?)",
                (conversation_id, user_id, user_id),
            ).fetchone()
        if row is not None:  # 会话有效
            return conversation_id  # 直接复用
    return _create_conversation(user_id, title, type_)  # 否则新建


def _append_message(  # 往对话里追加一条消息
    conversation_id: int, role: str, content: str, images: str | None = None
) -> None:
    now = time.time()  # 当前时间
    with _auth_db() as conn:  # 打开数据库
        conn.execute(  # 插入消息（images 为 JSON 数组，存上传的图片/文件）
            "INSERT INTO messages (conversation_id, role, content, images, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, content, images, now),
        )
        conn.execute(  # 更新对话的最后活跃时间
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )


def _list_conversations(user_id: int, limit: int) -> list[dict]:  # 查询某用户的历史对话列表
    with _auth_db() as conn:  # 打开数据库
        rows = conn.execute(  # 按活跃时间倒序取最近 N 条，附带消息数
            """SELECT c.id, c.title, c.type, c.created_at, c.updated_at,
                      (SELECT COUNT(*) FROM messages m
                        WHERE m.conversation_id = c.id) AS message_count
               FROM conversations c
               WHERE c.user_id = ?
               ORDER BY c.updated_at DESC
               LIMIT ?""",
            (user_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]  # 转成字典列表返回


def _get_conversation_messages(
    user_id: int, conversation_id: int
) -> dict | None:
    with _auth_db() as conn:  # 打开数据库
        conv = conn.execute(  # 查对话基本信息
            "SELECT id, title, type, user_id FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if conv is None or conv["user_id"] != user_id:  # 不存在或不是本人的
            return None  # 返回无记录
        rows = conn.execute(  # 按时间顺序查全部消息
            "SELECT role, content, images, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
    messages = []  # 消息列表
    for row in rows:  # 逐条组装
        item = {  # 基础字段
            "role": row["role"],  # 角色
            "content": row["content"],  # 文字内容
            "created_at": row["created_at"],  # 时间
        }
        try:  # 解析附件 JSON（旧数据可能没有 images 列或为空）
            item["images"] = json.loads(row["images"]) if row["images"] else []
        except Exception:  # 解析失败
            item["images"] = []  # 给空列表
        messages.append(item)  # 加入
    return {  # 组装返回数据
        "id": conv["id"],  # 对话 ID
        "title": conv["title"],  # 对话标题
        "type": conv["type"],  # 类型（问答/图片/报告）
        "messages": messages,  # 消息列表（含 images 附件）
    }


@router.get("/api/history")  # 历史对话列表接口
def history_list(_user: dict | None = Depends(get_current_user)):  # 依赖注入获取当前用户
    """当前登录账号的历史对话列表（未登录返回空列表）。"""
    if _user is None:  # 未登录
        return {"items": []}  # 返回空列表
    return {"items": _list_conversations(_user["id"], config.HISTORY_LIMIT)}  # 返回该用户的历史


@router.get("/api/history/{conversation_id}")
def history_detail(
    conversation_id: int,
    _user: dict | None = Depends(get_current_user),
):
    """某次对话的完整消息记录。"""
    if _user is None:  # 未登录
        raise HTTPException(status_code=404, detail="记录不存在")  # 不给查看
    data = _get_conversation_messages(_user["id"], conversation_id)  # 取对话内容
    if data is None:  # 不存在或不是本人的
        raise HTTPException(status_code=404, detail="记录不存在")  # 返回 404
    return data  # 返回消息记录


@router.delete("/api/history")
def history_clear(_user: dict | None = Depends(get_current_user)):
    """清空当前登录账号的全部历史记录。"""
    if _user is None:  # 未登录
        raise HTTPException(status_code=404, detail="记录不存在")  # 不给操作
    with _auth_db() as conn:  # 打开数据库
        deleted = conn.execute(  # 删除该用户全部对话
            "DELETE FROM conversations WHERE user_id = ?", (_user["id"],)
        ).rowcount
        conn.execute(  # 清理孤儿消息
            "DELETE FROM messages WHERE conversation_id NOT IN (SELECT id FROM conversations)"
        )
    return {"ok": True, "deleted": deleted}  # 返回删除数量


@router.delete("/api/history/{conversation_id}")
def history_delete(
    conversation_id: int,
    _user: dict | None = Depends(get_current_user),
):
    """删除某一条历史对话。"""
    if _user is None:  # 未登录
        raise HTTPException(status_code=404, detail="记录不存在")  # 不给操作
    with _auth_db() as conn:  # 打开数据库
        cur = conn.execute(  # 删除该对话（仅限本人）
            "DELETE FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, _user["id"]),
        )
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))  # 删除其消息
    return {"ok": bool(cur.rowcount)}  # 是否真的删到了
