"""工单17：索引后台重载与查询并行，不能读到空/错配的中间索引。"""
import sys,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'研发'))
from app.pipeline import Pipeline
def test_query_during_atomic_reload():
    p=Pipeline()
    def rebuild():
        for _ in range(5):p.reload()
    def read():
        for _ in range(40):
            hits=p.retrieve('发明人是谁？','CN100342976C.pdf',True)
            assert hits and hits[0]['document']=='CN100342976C.pdf'
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        tasks=[pool.submit(rebuild),pool.submit(read),pool.submit(read)]
        for task in tasks:task.result()
