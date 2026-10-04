# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失工单
Redis Stream 消息队列模拟：解析任务触发 → 放入队列 → 任务执行器消费。
"""
import json
import time
import uuid


class RedisStream:
    """极简 Redis Stream 内存实现，忠实复刻“触发-入队-消费”链路。"""

    def __init__(self, name="ragflow_task_queue"):
        self.name = name
        self._entries = []   # 每条：{"id", "payload"}

    def xadd(self, payload):
        entry_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
        self._entries.append({"id": entry_id, "payload": json.dumps(payload, ensure_ascii=False)})
        return entry_id

    def xread(self, count=1):
        """消费者：取出并移除任务。"""
        batch, self._entries = self._entries[:count], self._entries[count:]
        return batch

    def pending(self):
        return len(self._entries)


# 全局任务队列（对应 RAGFlow 的 Task Executor 消费队列）
TASK_QUEUE = RedisStream("ragflow_task_broker")


def build_task(document, parser_id):
    """按 parser_id 构造解析任务（对应 RAGFlow 的 task payload）。"""
    return {
        "task_id": uuid.uuid4().hex,
        "document": document,
        "parser_id": parser_id,
        "parser_config": {"chunk_token_num": 512, "delimiter": "\n"},
        "created": time.time(),
    }


def enqueue_parse(document, parser_id):
    """文档上传后触发解析：构造任务并放入 Redis Stream。"""
    task = build_task(document, parser_id)
    TASK_QUEUE.xadd(task)
    return task["task_id"]
