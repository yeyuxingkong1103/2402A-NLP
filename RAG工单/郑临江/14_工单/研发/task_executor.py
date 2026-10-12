# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失工单
do_handle_task 任务处理函数：消费队列任务，完成解析→分块→向量化→索引的完整流程。
"""
import hashlib
import json

import config
from pdf_parser import extract_pdf
from queue import TASK_QUEUE


class TaskExecutor:
    def __init__(self):
        self.kb = {}   # 知识库：chunk_id -> {"text", "page", "vector", "doc"}

    def _vector(self, text):
        """轻量向量化：字符 bigram 哈希向量（离线可用）。"""
        text = text.replace("\n", " ")
        grams = [text[i:i + 2] for i in range(len(text) - 1)]
        vec = {}
        for g in grams:
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16) % 4096
            vec[h] = vec.get(h, 0) + 1
        return vec

    def do_handle_task(self, task):
        """处理单个任务：解析 → 分块 → 向量化 → 索引。"""
        parser_id = task["parser_id"]
        document = task["document"]
        chunks = extract_pdf(document, parser_id=parser_id)
        for i, c in enumerate(chunks):
            cid = f"{document}#{parser_id}#{i}"
            self.kb[cid] = {
                "text": c["text"], "page": c.get("page"),
                "kind": c.get("kind"), "vector": self._vector(c["text"]),
                "doc": document,
            }
        return len(chunks)

    def consume(self, count=1):
        """从 Redis Stream 消费任务并处理。"""
        done = 0
        for entry in TASK_QUEUE.xread(count):
            task = json.loads(entry["payload"])
            n = self.do_handle_task(task)
            done += n
        return done
