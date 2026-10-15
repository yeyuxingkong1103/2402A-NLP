"""工单编号：人工智能NLP-RAG-解决API服务并发瓶颈与资源泄漏"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'研发'))
from app.resources import TTLCache
def test_ttl_cache_bound():
    c=TTLCache(2,60);c.put('a',1);c.put('b',2);c.put('c',3);assert c.get('a') is None and c.get('c')==3
