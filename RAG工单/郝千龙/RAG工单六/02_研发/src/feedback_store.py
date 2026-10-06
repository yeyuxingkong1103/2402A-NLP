# -*- coding: utf-8 -*-
# 【用户反馈存储 · feedback_store.py】持久化用户点击/采纳/不采纳反馈，供自适应重排形成反馈闭环
# 工单编号：人工智能NLP-RAG-混合检索任务

"""反馈闭环数据层。

Streamlit 界面每条证据可点“采纳 / 不采纳”，点击行为通过本模块追加写入
JSONL 文件（进程重启不丢失）；``FeedbackAdaptiveReranker`` 读取历史反馈，
把“被采纳块”的词项分布与“相似问题的采纳块”转化为重排加权信号，形成
``检索 → 展示 → 用户反馈 → 自适应重排 → 再检索`` 的闭环。

每条记录字段：
- ``ts``：时间戳；``query``：用户原问题；``chunk_id``：被反馈的块编号；
- ``action``：click(点击) / adopt(采纳) / reject(不采纳)；
- ``simulated``：是否脚本模拟数据（真实用户操作恒为 False，模拟数据强制 True）；
- ``source``：数据来源标注（web / simulate 等）。
"""
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class FeedbackRecord:
    """单条用户反馈记录。"""

    query: str               # 用户原问题
    chunk_id: int            # 被反馈的检索块编号
    action: str              # click / adopt / reject
    ts: float                # 时间戳
    simulated: bool = False  # 是否脚本模拟（真实用户恒为 False）
    source: str = "web"      # 来源标注

    def to_dict(self) -> dict:
        """序列化为 JSONL 行字典。"""
        return {"query": self.query, "chunk_id": self.chunk_id,
                "action": self.action, "ts": self.ts,
                "simulated": self.simulated, "source": self.source}


class FeedbackStore:
    """JSONL 追加式反馈存储（读时全量载入，写时加锁追加）。"""

    def __init__(self, file_path: str) -> None:
        """指定持久化文件路径（目录不存在时自动创建）。

        :param file_path: feedback.jsonl 绝对路径
        """
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.records: List[FeedbackRecord] = []
        self._load()

    def _load(self) -> None:
        """启动时从 JSONL 文件全量载入历史反馈。"""
        if not os.path.exists(self.file_path):
            return
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    self.records.append(FeedbackRecord(
                        query=d["query"], chunk_id=int(d["chunk_id"]),
                        action=d.get("action", "click"),
                        ts=float(d.get("ts", time.time())),
                        simulated=bool(d.get("simulated", False)),
                        source=d.get("source", "web")))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue  # 坏行跳过，不影响服务启动

    def _append(self, record: FeedbackRecord) -> None:
        """追加一条反馈到内存列表与磁盘文件。

        :param record: 反馈记录
        """
        self.records.append(record)
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def record_click(self, query: str, chunk_id: int,
                     simulated: bool = False, source: str = "web") -> None:
        """记录点击行为。

        :param query: 用户原问题
        :param chunk_id: 被点击块编号
        :param simulated: 是否模拟数据
        :param source: 来源标注
        """
        self._append(FeedbackRecord(query, int(chunk_id), "click",
                                    time.time(), simulated,
                                    "simulate" if simulated else source))

    def record_adopt(self, query: str, chunk_id: int,
                     simulated: bool = False, source: str = "web") -> None:
        """记录“采纳”正反馈。

        :param query: 用户原问题
        :param chunk_id: 被采纳块编号
        :param simulated: 是否模拟数据
        :param source: 来源标注
        """
        self._append(FeedbackRecord(query, int(chunk_id), "adopt",
                                    time.time(), simulated,
                                    "simulate" if simulated else source))

    def record_reject(self, query: str, chunk_id: int,
                      simulated: bool = False, source: str = "web") -> None:
        """记录“不采纳”负反馈。

        :param query: 用户原问题
        :param chunk_id: 被否定块编号
        :param simulated: 是否模拟数据
        :param source: 来源标注
        """
        self._append(FeedbackRecord(query, int(chunk_id), "reject",
                                    time.time(), simulated,
                                    "simulate" if simulated else source))

    def adopted_ids(self) -> Set[int]:
        """全部被采纳过的块编号集合（去重）。"""
        return {r.chunk_id for r in self.records if r.action == "adopt"}

    def rejected_ids(self) -> Set[int]:
        """全部被“不采纳”过的块编号集合（去重）。"""
        return {r.chunk_id for r in self.records if r.action == "reject"}

    def feedback_by_query(self) -> Dict[str, Dict[str, Set[int]]]:
        """按问题聚合正负反馈，供相似问题触发。

        :return: {query: {"adopt": {块id}, "reject": {块id}}}
        """
        grouped: Dict[str, Dict[str, Set[int]]] = defaultdict(
            lambda: {"adopt": set(), "reject": set()})
        for r in self.records:
            if r.action in ("adopt", "reject"):
                grouped[r.query][r.action].add(r.chunk_id)
        return dict(grouped)

    def count(self) -> int:
        """返回反馈总条数。"""
        return len(self.records)

    def simulated_count(self) -> int:
        """返回模拟反馈条数（报告中需与真实反馈区分）。"""
        return sum(1 for r in self.records if r.simulated)
