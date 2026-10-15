"""工单编号：人工智能NLP-RAG-解决API服务并发瓶颈与资源泄漏。"""

import asyncio, collections, time


class TTLCache:
    def __init__(self, max_entries=128, ttl=300):
        self.entries = collections.OrderedDict()
        self.max_entries = max_entries
        self.ttl = ttl

    def get(self, key):
        now = time.monotonic()
        while self.entries and next(iter(self.entries.values()))[0] <= now:
            self.entries.popitem(last=False)
        value = self.entries.get(key)
        if not value or value[0] <= now:
            return None
        self.entries.move_to_end(key)
        return value[1]

    def put(self, key, value):
        self.entries[key] = (time.monotonic() + self.ttl, value)
        self.entries.move_to_end(key)
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)


class Capacity:
    def __init__(self, active=4, waiting=32):
        self.sem = asyncio.Semaphore(active)
        self.waiting = 0
        self.max_waiting = waiting

    async def __aenter__(self):
        if self.waiting >= self.max_waiting:
            raise OverflowError("request_queue_full")
        self.waiting += 1
        try:
            await asyncio.wait_for(self.sem.acquire(), timeout=5)
        finally:
            self.waiting -= 1

    async def __aexit__(self, *args):
        self.sem.release()
